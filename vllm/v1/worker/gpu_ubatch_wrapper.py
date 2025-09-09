# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import threading
from typing import Any, Callable, Optional

import torch

import vllm.envs as envs
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (create_forward_context, get_forward_context,
                                  override_forward_context, DPMetadata)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.ubatching import UBatchContext, make_ubatch_contexts
from vllm.utils import round_up

from vllm.v1.worker.ubatch_utils import UBatchSlices, UbatchSlice
from vllm.distributed.parallel_state import is_global_first_rank

logger = init_logger(__name__)


@dataclasses.dataclass
class UbatchMetadata:
    context: UBatchContext
    input_ids: torch.Tensor
    positions: torch.Tensor
    inputs_embeds: Optional[torch.Tensor]
    intermediate_tensors: Optional[IntermediateTensors]
    num_tokens: int


@dataclasses.dataclass
class CUDAGraphMetaData:
    cudagraph: torch.cuda.CUDAGraph
    ubatch_metadata: UbatchMetadata
    outputs: Optional[Any] = None


class UBatchWrapper:

    def __init__(self, runnable: Callable, vllm_config: VllmConfig,
                 runtime_mode: CUDAGraphMode, device: torch.cuda.device):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.comm_stream = torch.cuda.Stream(device=device)
        self.ready_barrier = threading.Barrier(3)

        self.cudagraphs: dict[int, CUDAGraphMetaData] = {}

        self.cudagraph_wrapper = None
        self.graph_pool = None
        if runtime_mode is not CUDAGraphMode.NONE:
            self.cudagraph_wrapper = CUDAGraphWrapper(
                runnable, vllm_config, runtime_mode=runtime_mode)
            self.graph_pool = current_platform.get_global_graph_pool()
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not exists in the runnable of "
                             f"cudagraph wrapper: {self.runnable}")

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
        num_tokens = ubatch_metadata[0].num_tokens + \
            ubatch_metadata[1].num_tokens

        # Ubatches will manually manage the forward context, so we override
        # it to None here so we can have it restored correctly later
        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(target=_capture_ubatch_thread,
                                          args=(
                                              results,
                                              metadata,
                                          ))
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()  # Wait for both threads to be ready

            # Capture the cudagraph
            cudagraph_metadata = \
                CUDAGraphMetaData(
                            cudagraph=torch.cuda.CUDAGraph(),
                            ubatch_metadata=ubatch_metadata,
                        )
            with torch.cuda.graph(cudagraph_metadata.cudagraph,
                                  stream=compute_stream,
                                  pool=self.graph_pool):
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
                thread = threading.Thread(target=_ubatch_thread,
                                          args=(
                                              results,
                                              model,
                                              metadata,
                                          ))
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()  # Wait for both threads to be ready
            ubatch_metadata[0].context.cpu_wait_event.set()
            for thread in ubatch_threads:
                thread.join()
        sorted_results = [value for position, value in sorted(results)]
        result = torch.cat(sorted_results, dim=0)
        return result

    def _make_ubatch_metadata(self, ubatch_slices, attn_metadata, input_ids,
                              positions, inputs_embeds, intermediate_tensors,
                              compute_stream, dp_metadata, batch_descriptor,
                              cudagraph_runtime_mode) -> list[UbatchMetadata]:

        # Create one forward context per ubatch
        forward_contexts = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            forward_contexts.append(
                create_forward_context(
                    attn_metadata[i] if attn_metadata is not None else None,
                    self.vllm_config,
                    dp_metadata=dp_metadata,
                    batch_descriptor=batch_descriptor,
                    cudagraph_runtime_mode=cudagraph_runtime_mode))

        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier)

        ubatch_metadata: list[UbatchMetadata] = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            sliced_input_ids, sliced_positions, sliced_inputs_embeds, \
            sliced_intermediate_tensors = \
                self._slice_model_inputs(
                    ubatch_slice.token_slice, input_ids, positions,
                    inputs_embeds, intermediate_tensors)
            ubatch_metadata.append(
                UbatchMetadata(
                    context=ubatch_ctxs[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.token_slice.stop -
                    ubatch_slice.token_slice.start))

        return ubatch_metadata

    def _slice_model_inputs(self, tokens_slice: slice, input_ids, positions,
                            inputs_embeds, intermediate_tensors):
        sliced_input_ids = input_ids[tokens_slice]
        # if we are using mrope
        if positions.ndim == 2:
            sliced_positions = positions[:, tokens_slice]
        else:
            sliced_positions = positions[tokens_slice]
        sliced_inputs_embeds = inputs_embeds[
            tokens_slice] if inputs_embeds else None
        sliced_intermediate_tensors = intermediate_tensors[
            tokens_slice] if intermediate_tensors else None

        return (sliced_input_ids, sliced_positions, sliced_inputs_embeds,
                sliced_intermediate_tensors)

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        ubatch_slices = forward_context.ubatch_slices
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        # If there's no ubatching, just run the runnable object
        if ubatch_slices is None:
            if cudagraph_runtime_mode in (CUDAGraphMode.NONE,
                                          CUDAGraphMode.PIECEWISE):
                return self.runnable(*args, **kwargs)
            else:
                assert self.cudagraph_wrapper is not None
                return self.cudagraph_wrapper(*args, **kwargs)

        attn_metadata = forward_context.attn_metadata
        num_tokens = (ubatch_slices[0].token_slice.stop -
                      ubatch_slices[0].token_slice.start) * 2
        input_ids = kwargs['input_ids']
        positions = kwargs['positions']
        intermediate_tensors = kwargs['intermediate_tensors']
        inputs_embeds = kwargs['inputs_embeds']
        compute_stream = torch.cuda.current_stream()

        dp_metadata = forward_context.dp_metadata

        # We shouldn't be here unless we are running with multiple DP ranks
        assert dp_metadata is not None

        if num_tokens not in self.cudagraphs \
            and cudagraph_runtime_mode is CUDAGraphMode.FULL:
            if is_global_first_rank():
                logger.info(f"Capturing ubatched cudagraph for {num_tokens} tokens")
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=dp_metadata,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE)

            return self._capture_ubatches(ubatch_metadata, self.model)
        elif num_tokens in self.cudagraphs:
            if is_global_first_rank():
                logger.info(f"Replaying ubatched cudagraph for {num_tokens} tokens")
            cudagraph_metadata = self.cudagraphs[num_tokens]
            cudagraph_metadata.cudagraph.replay()
            return cudagraph_metadata.outputs
        else:
            if is_global_first_rank():
                logger.info(f"Running ubatched for {num_tokens} tokens")
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                compute_stream=compute_stream,
                dp_metadata=dp_metadata,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE)
            return self._run_ubatches(ubatch_metadata, self.model)

def should_ubatch_with_num_tokens(
    should_ubatch: bool,
    orig_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    vllm_config: VllmConfig,
) -> tuple[bool, Optional[torch.Tensor]]:
    dp_size = vllm_config.parallel_config.data_parallel_size
    dp_rank = vllm_config.parallel_config.data_parallel_rank
    return DPMetadata.should_ubatch_across_dp(
        should_ubatch, orig_num_tokens_per_ubatch,
        padded_num_tokens_per_ubatch, dp_size, dp_rank)


def get_dp_padding_ubatch(
    num_tokens_unpadded: int, 
    num_tokens_padded: int,
    should_attempt_ubatching: bool,
    vllm_config: VllmConfig
) -> tuple[bool, int, Optional[torch.Tensor]]:
    assert num_tokens_padded >= num_tokens_unpadded
    dp_size = vllm_config.parallel_config.data_parallel_size
    if dp_size == 1:
        # Early exit.
        return False, 0, None

    if not should_attempt_ubatching:
        (should_ubatch,
            num_tokens_across_dp) = should_ubatch_with_num_tokens(
                False, 0, 0, vllm_config)
        assert should_ubatch is False
        assert num_tokens_across_dp is None
        return should_ubatch, 0, num_tokens_across_dp

    num_tokens_padded = round_up(num_tokens_padded, 2)
    num_tokens_per_ubatch = num_tokens_padded // 2
    should_ubatch = True
    # num_pad_tokens = num_tokens_padded - num_tokens_unpadded

    # Sanity Check that the existing padding isn't giving us an empty second
    # ubatch. Abort if so
    if num_tokens_unpadded <= num_tokens_padded // 2:
        should_ubatch = False

    # Note that we compute the number of padded tokens per ubatch
    (should_ubatch,
        num_tokens_across_dp) = should_ubatch_with_num_tokens(
            should_ubatch, num_tokens_unpadded // 2, num_tokens_per_ubatch, vllm_config)
    if not should_ubatch:
        assert num_tokens_across_dp is None
        return should_ubatch, 0, num_tokens_across_dp

    assert num_tokens_across_dp is not None

    max_tokens_across_dp_cpu = int(torch.max(num_tokens_across_dp).item())
    num_tokens_after_padding = torch.tensor([max_tokens_across_dp_cpu] *
                                            dp_size,
                                            device="cpu",
                                            dtype=torch.int32)
    num_pad_tokens = max_tokens_across_dp_cpu - num_tokens_per_ubatch
    num_pad_tokens = ((num_pad_tokens + num_tokens_per_ubatch) * 2) - \
        num_tokens_unpadded
    assert num_pad_tokens >= 0
    return should_ubatch, num_pad_tokens, num_tokens_after_padding

def ubatch_split(max_num_scheduled_tokens: int,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
        vllm_config: VllmConfig,
    ) -> tuple[Optional[UBatchSlices], int, Optional[torch.Tensor]]:
    parallel_config = vllm_config.parallel_config
    # Don't bother with the should_ubatch handshaking unless microbatching
    # is enabled
    if not parallel_config.enable_microbatching:
        return (None, 0, None)

    # Check preconditions for microbatching
    should_attempt_ubatching = \
        parallel_config.enable_microbatching and \
        num_tokens_unpadded >= \
        parallel_config.microbatching_token_threshold \
        and max_num_scheduled_tokens == 1

    # Don't microbatch unless every other DP worker is also microbatching
    num_pad_tokens = 0
    num_tokens_after_padding = None
    (should_ubatch, num_pad_tokens,
        num_tokens_after_padding) = get_dp_padding_ubatch(num_tokens_unpadded, 
                                                          num_tokens_padded, 
                                                          should_attempt_ubatching, 
                                                          vllm_config)
    if not should_ubatch:
        return (None, 0, None)

    # This doesn't actually pad the ubatch slices. It just initializes the
    # split point to the padded value so that padding can be applied
    # to the second ubatch in pad_out_ubatch_slice after attention
    # metadata creation
    assert num_pad_tokens < num_tokens_unpadded,\
        f"num_pad_tokens {num_pad_tokens} "\
        f"original_num_tokens {num_tokens_unpadded}"
    total_num_tokens_per_ubatch = (num_tokens_unpadded +
                                    num_pad_tokens) // 2
    padded_first_ubatch_slice = slice(0, total_num_tokens_per_ubatch)
    padded_second_ubatch_slice = slice(total_num_tokens_per_ubatch,
                                        num_tokens_unpadded)

    # Note there's an assumption here that there's 1 token per request
    ubatch_slices = [
        UbatchSlice(padded_first_ubatch_slice, padded_first_ubatch_slice),
        UbatchSlice(padded_second_ubatch_slice, padded_second_ubatch_slice)
    ]

    return (ubatch_slices, num_pad_tokens, num_tokens_after_padding)