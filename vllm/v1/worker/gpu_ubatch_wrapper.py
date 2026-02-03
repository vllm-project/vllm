# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

import vllm.envs as envs
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_ep_group
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.forward_context import (
    AFDMetadata,
    DPMetadata,
    create_forward_context,
    get_forward_context,
    override_forward_context,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import has_deep_gemm
from vllm.v1.worker.ubatching import UBatchContext, make_ubatch_contexts

logger = init_logger(__name__)


@dataclass
class UbatchMetadata:
    context: UBatchContext
    input_ids: torch.Tensor
    positions: torch.Tensor
    inputs_embeds: torch.Tensor | None
    intermediate_tensors: IntermediateTensors | None
    num_tokens: int


@dataclass
class CUDAGraphMetaData:
    cudagraph: torch.cuda.CUDAGraph
    ubatch_metadata: UbatchMetadata
    outputs: Any | None = None


class SMControlContextManager:
    def __init__(
        self,
        comm_sms: int,
        set_comm_sms: Callable[[int], None],
        set_compute_sms: Callable[[int], None],
    ):
        """
        Context manager for controlling SM (Streaming Multiprocessor)
        allocation. Upon entering the context, it sets the number of SMs
        allocated for communication and computation to comm_sms and
        total_sms - comm_sms respectively. Upon exiting, it restores the
        allocation to use all available SMs (i.e. total_sms).

        Args:
            comm_sms (int): The number of SMs to allocate for communication.
                (The remainder will be used for computation.)
            set_comm_sms (Callable[[int], None]):
                A function that sets the number of SMs for communication.
            set_compute_sms (Callable[[int], None]):
                A function that sets the number of SMs for computation.
        """

        assert current_platform.is_cuda(), (
            "SM control is currently only supported on CUDA"
        )

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        total_sms = props.multi_processor_count

        assert comm_sms < total_sms
        self.total_sms = total_sms
        self.compute_sms = total_sms - comm_sms
        self.comm_sms = comm_sms
        self.set_comm_sms = set_comm_sms
        self.set_compute_sms = set_compute_sms

    def __enter__(self):
        self.set_comm_sms(self.comm_sms)
        self.set_compute_sms(self.compute_sms)

    def __exit__(self, exc_type, exc_value, traceback):
        self.set_comm_sms(self.total_sms)
        self.set_compute_sms(self.total_sms)


class UBatchWrapper:
    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        device: torch.cuda.device,
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.comm_stream = torch.cuda.Stream(device=device)
        # Ubatch threads plus the main thread
        self.ready_barrier = threading.Barrier(
            self.vllm_config.parallel_config.num_ubatches + 1
        )

        self.cudagraphs: dict[int, CUDAGraphMetaData] = {}

        self.cudagraph_wrapper = None
        self.graph_pool = None
        if runtime_mode is not CUDAGraphMode.NONE:
            self.cudagraph_wrapper = CUDAGraphWrapper(
                runnable, vllm_config, runtime_mode=runtime_mode
            )
            self.graph_pool = current_platform.get_global_graph_pool()

        self.sm_control = self._create_sm_control_context(vllm_config)
        self.device = device

    @staticmethod
    def _create_sm_control_context(vllm_config: VllmConfig):
        comm_sms: int = envs.VLLM_DBO_COMM_SMS

        set_comm_sms = lambda sms: None
        if (
            vllm_config.parallel_config.enable_expert_parallel
            and not vllm_config.afd_config
        ):
            # Currently only DeepEP highthroughput supports SM control so this
            # only affects that case.
            ep_group = get_ep_group()
            device_communicator = ep_group.device_communicator
            all2all_manager = None
            if device_communicator is not None:
                all2all_manager = device_communicator.all2all_manager

            if all2all_manager is not None:
                max_sms_used = all2all_manager.max_sms_used()
                if max_sms_used is not None:
                    comm_sms = min(comm_sms, max_sms_used)

            if comm_sms > 0 and all2all_manager is not None:
                set_comm_sms = lambda sms: all2all_manager.set_num_sms(sms)

        # TODO(lucas): support other kernels besides DeepGEMM
        set_compute_sms = lambda sms: None
        if has_deep_gemm() and comm_sms > 0:
            import deep_gemm as dg

            set_compute_sms = lambda sms: dg.set_num_sms(sms)

        return SMControlContextManager(
            comm_sms=comm_sms,
            set_comm_sms=set_comm_sms,
            set_compute_sms=set_compute_sms,
        )

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(
            f"Attribute {key} not exists in the runnable of "
            f"cudagraph wrapper: {self.runnable}"
        )

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def _capture_ubatches(self, ubatch_metadata, model) -> torch.Tensor:
        """
        Capture a cudagraph for a microbatched run.

        The logic here is somewhat complicated because we need to make sure that
        each of the ubatch threads initialize the cuda context before we start
        the graph capture.

        The flow is as follows:
        1. The main thread starts up each ubatch thread. Each thread will
        initialize its cuda context (torch.cuda.current_blas_handle())
        before going to sleep upon entering the ubatch_context.

        2. The main thread starts the graph capture and wakes up the first
        ubatch thread.

        3. Each ubatch thread runs the model to completion and returns the
        completed output tensors back to the main thread.

        4. The main thread stores the captured cudagraph along with its metadata
        and returns
        """

        @torch.inference_mode()
        def _capture_ubatch_thread(results, ubatch_metadata):
            torch.cuda.set_device(self.device)
            ubatch_context = ubatch_metadata.context
            with torch.cuda.stream(ubatch_context.compute_stream):
                _ = torch.cuda.current_blas_handle()
            with torch.cuda.stream(ubatch_context.comm_stream):
                _ = torch.cuda.current_blas_handle()
            with ubatch_context:
                model_output = model(
                    input_ids=ubatch_metadata.input_ids,
                    positions=ubatch_metadata.positions,
                    intermediate_tensors=ubatch_metadata.intermediate_tensors,
                    inputs_embeds=ubatch_metadata.inputs_embeds,
                )

            results.append((ubatch_metadata.context.id, model_output))

        results: list[tuple[int, torch.Tensor]] = []
        compute_stream = ubatch_metadata[0].context.compute_stream
        num_tokens = ubatch_metadata[0].num_tokens + ubatch_metadata[1].num_tokens

        # Ubatches will manually manage the forward context, so we override
        # it to None here so we can have it restored correctly later
        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=_capture_ubatch_thread,
                    args=(
                        results,
                        metadata,
                    ),
                )
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()  # Wait for both threads to be ready

            # Capture the cudagraph
            cudagraph_metadata = CUDAGraphMetaData(
                cudagraph=torch.cuda.CUDAGraph(),
                ubatch_metadata=ubatch_metadata,
            )
            if self.graph_pool is not None:
                set_graph_pool_id(self.graph_pool)
            else:
                set_graph_pool_id(current_platform.graph_pool_handle())
            with torch.cuda.graph(
                cudagraph_metadata.cudagraph,
                stream=compute_stream,
                pool=self.graph_pool,
            ):
                ubatch_metadata[0].context.cpu_wait_event.set()
                for thread in ubatch_threads:
                    thread.join()
                sorted_results = [value for position, value in sorted(results)]
                result = torch.cat(sorted_results, dim=0)
                cudagraph_metadata.outputs = result
            self.cudagraphs[num_tokens] = cudagraph_metadata
        return cudagraph_metadata.outputs

    def _run_ubatches(self, ubatch_metadata, model) -> torch.Tensor:
        @torch.inference_mode()
        def _ubatch_thread(results, model, ubatch_metadata):
            with ubatch_metadata.context:
                model_output = model(
                    input_ids=ubatch_metadata.input_ids,
                    positions=ubatch_metadata.positions,
                    intermediate_tensors=ubatch_metadata.intermediate_tensors,
                    inputs_embeds=ubatch_metadata.inputs_embeds,
                )
            results.append((ubatch_metadata.context.id, model_output))

        results: list[tuple[int, torch.Tensor]] = []

        # Ubatch threads will manually manage the forward context, so we
        # override it to None here so we can have it restored correctly
        # after both threads have finished
        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=_ubatch_thread,
                    args=(
                        results,
                        model,
                        metadata,
                    ),
                )
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()  # Wait for both threads to be ready
            ubatch_metadata[0].context.cpu_wait_event.set()
            for thread in ubatch_threads:
                thread.join()
        sorted_results = [value for position, value in sorted(results)]
        result = torch.cat(sorted_results, dim=0)
        return result

    def _make_ubatch_metadata(
        self,
        ubatch_slices,
        attn_metadata,
        slot_mapping,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
        compute_stream,
        dp_metadata,
        batch_descriptor,
        cudagraph_runtime_mode,
        afd_metadata,
    ) -> list[UbatchMetadata]:
        # Create one forward context per ubatch
        forward_contexts = []
        # slot_mapping can be None, an empty dict (from create_forward_context
        # converting None to {}), or a list of dicts (one per ubatch)
        has_slot_mapping = slot_mapping and isinstance(slot_mapping, list)
        for i, ubatch_slice in enumerate(ubatch_slices):
            afd_metadata_clone = afd_metadata.clone()
            afd_metadata_clone.afd_stage_idx = i
            logger.info(f"jcz _make_ubatch_metadata afd_metadata_clone.afd_stage_idx:{afd_metadata_clone.afd_stage_idx}")
            forward_contexts.append(
                create_forward_context(
                    attn_metadata[i] if attn_metadata is not None else None,
                    self.vllm_config,
                    dp_metadata=dp_metadata[i],
                    batch_descriptor=batch_descriptor,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
<<<<<<< HEAD
                    slot_mapping=slot_mapping[i] if has_slot_mapping else None,
                    afd_metadata=afd_metadata,
=======
                    afd_metadata=afd_metadata_clone,
>>>>>>> efbc0d799 (forward use ubatch multithread)
                )
            )

        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        ubatch_metadata: list[UbatchMetadata] = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            (
                sliced_input_ids,
                sliced_positions,
                sliced_inputs_embeds,
                sliced_intermediate_tensors,
            ) = self._slice_model_inputs(
                ubatch_slice.token_slice,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
            )
            ubatch_metadata.append(
                UbatchMetadata(
                    context=ubatch_ctxs[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.token_slice.stop
                    - ubatch_slice.token_slice.start,
                )
            )

        return ubatch_metadata

    def _slice_model_inputs(
        self,
        tokens_slice: slice,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
    ):
        sliced_input_ids = input_ids[tokens_slice]
        # if we are using mrope. Mrope adds an additional dimension to the
        # positions tensor
        if positions.ndim == 2:
            sliced_positions = positions[:, tokens_slice]
        else:
            sliced_positions = positions[tokens_slice]
        sliced_inputs_embeds = inputs_embeds[tokens_slice] if inputs_embeds else None
        sliced_intermediate_tensors = (
            intermediate_tensors[tokens_slice] if intermediate_tensors else None
        )

        return (
            sliced_input_ids,
            sliced_positions,
            sliced_inputs_embeds,
            sliced_intermediate_tensors,
        )

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        ubatch_slices = forward_context.ubatch_slices
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
        afd_metadata = forward_context.afd_metadata

        attn_metadata = forward_context.attn_metadata
        input_ids = kwargs["input_ids"]
        positions = kwargs["positions"]
        intermediate_tensors = kwargs["intermediate_tensors"]
        inputs_embeds = kwargs["inputs_embeds"]
        compute_stream = torch.cuda.current_stream()

        dp_metadata = forward_context.dp_metadata

        # If there's no ubatching, just run the runnable object
        if ubatch_slices is None:
            # This is to account for the case where ubatching was aborted.
            # When we capture full graphs we only capture one graph per shape,
            # meaning that if we have a ubatched  cudagraph for the current
            # num_tokens, we don't have a non-ubatched one. Without this
            # check, the cudagraph wrapper will try to capture a cudagraph
            # for this shape during a normal run.

            if cudagraph_runtime_mode is CUDAGraphMode.FULL:
                assert batch_descriptor is not None
                if batch_descriptor.num_tokens in self.cudagraphs:
                    cudagraph_runtime_mode = CUDAGraphMode.NONE

            if cudagraph_runtime_mode in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE):
                return self.runnable(*args, **kwargs)
            else:
                assert self.cudagraph_wrapper is not None
                return self.cudagraph_wrapper(*args, **kwargs)

        attn_metadata = forward_context.attn_metadata
        slot_mapping = forward_context.slot_mapping
        num_tokens = sum(ubatch_slice.num_tokens for ubatch_slice in ubatch_slices)
        input_ids = kwargs["input_ids"]
        positions = kwargs["positions"]
        intermediate_tensors = kwargs["intermediate_tensors"]
        inputs_embeds = kwargs["inputs_embeds"]
        compute_stream = torch.cuda.current_stream()

        dp_metadata = forward_context.dp_metadata

        # We shouldn't be here unless we are running with multiple DP ranks
        assert dp_metadata is not None
        ubatch_dp_metadata = []
        for ubatch_slice in ubatch_slices:
            dp_size = self.vllm_config.parallel_config.data_parallel_size
            ubatch_num_tokens_across_dp = torch.tensor(
                [ubatch_slice.num_tokens] * dp_size, device="cpu", dtype=torch.int32
            )
            ubatch_dp_metadata.append(
                DPMetadata.make(
                    self.vllm_config.parallel_config,
                    ubatch_slice.num_tokens,
                    ubatch_num_tokens_across_dp,
                )
            )

        if (
            num_tokens not in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                slot_mapping=slot_mapping,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=ubatch_dp_metadata,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                afd_metadata=afd_metadata,
            )
            with self.sm_control:
                return self._capture_ubatches(ubatch_metadata, self.model)
        elif (
            num_tokens in self.cudagraphs
            and cudagraph_runtime_mode is CUDAGraphMode.FULL
        ):
            cudagraph_metadata = self.cudagraphs[num_tokens]
            cudagraph_metadata.cudagraph.replay()
            return cudagraph_metadata.outputs
        else:
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                slot_mapping=slot_mapping,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=ubatch_dp_metadata,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                afd_metadata=afd_metadata,
            )
            with self.sm_control:
                return self._run_ubatches(ubatch_metadata, self.model)
