# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from contextlib import ExitStack
from typing import Any, Callable, Optional
from unittest.mock import patch
import threading

import torch

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (BatchDescriptor, get_forward_context, 
                                  create_forward_context, 
                                  override_forward_context)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import weak_ref_tensors
from vllm.v1.worker.ubatching import UBatchContext, make_ubatch_contexts
from vllm.sequence import IntermediateTensors
from vllm.compilation.cuda_graph import CUDAGraphWrapper



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

    def __init__(self,
                 runnable: Callable,
                 vllm_config: VllmConfig,
                 runtime_mode: CUDAGraphMode,
                 device: torch.cuda.device):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.comm_stream = torch.cuda.Stream()
        self.device = device

        self.cudagraphs = {}

        self.cudagraph_wrapper = None
        self.graph_pool = None
        if runtime_mode is not CUDAGraphMode.NONE:
            self.cudagraph_wrapper = CUDAGraphWrapper(runnable, vllm_config, runtime_mode=runtime_mode)
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

        def _capture_ubatch_thread(results, ubatch_metadata, start_signal):
            # print(f"Starting Request on ubatch: {ubatch_ctx.id}", flush=True)
            context = ubatch_metadata.context
            with torch.cuda.stream(context.compute_stream):
                _ = torch.cuda.current_blas_handle()
            with torch.cuda.stream(context.comm_stream):
                _ = torch.cuda.current_blas_handle()
            with context:
                start_signal.wait()
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
            start_signals = []
            for metadata in ubatch_metadata:
                start_signal = threading.Event()
                thread = threading.Thread(target=_capture_ubatch_thread,
                                          args=(
                                              results,
                                              metadata,
                                              start_signal,
                                          ))
                ubatch_threads.append(thread)
                thread.start()
                start_signals.append(start_signal)

            # DO capture
            cudagraph_metadata = \
                CUDAGraphMetaData(
                            cudagraph=torch.cuda.CUDAGraph(),
                            ubatch_metadata=ubatch_metadata,
                        )
            with torch.cuda.graph(cudagraph_metadata.cudagraph,
                                  stream=compute_stream,
                                  pool=self.graph_pool):
                for start_signal in start_signals:
                    start_signal.set()
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

            ubatch_metadata[0].context.cpu_wait_event.set()
            for thread in ubatch_threads:
                thread.join()
        sorted_results = [value for position, value in sorted(results)]
        result = torch.cat(sorted_results, dim=0)
        return result
    
    def _make_ubatch_metadata(self, ubatch_slices, attn_metadata,
                              input_ids, positions, inputs_embeds, 
                              intermediate_tensors, compute_stream, 
                              num_tokens_across_dp, batch_descriptor, 
                              cudagraph_runtime_mode) -> list[UbatchMetadata]:

        # Create one forward context per ubatch
        forward_contexts = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            num_tokens = (ubatch_slice.token_slice.stop -
                          ubatch_slice.token_slice.start)
            forward_contexts.append(
                create_forward_context(
                    attn_metadata[i] if attn_metadata is not None else None,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    batch_descriptor=batch_descriptor,
                    cudagraph_runtime_mode=cudagraph_runtime_mode))

        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            device=self.device,
            enable_async_comms=self.vllm_config.parallel_config.enable_async_comms)

        ubatch_metadata: list[UbatchMetadata] = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            sliced_input_ids, sliced_positions, sliced_inputs_embeds, \
            sliced_intermediate_tensors = \
                self._slice_model_inputs(
                    ubatch_slice.token_slice, input_ids, positions,
                    inputs_embeds, intermediate_tensors)
            ubatch_metadata.append(
                UbatchMetadata(context=ubatch_ctxs[i],
                               input_ids=sliced_input_ids,
                               positions=sliced_positions,
                               inputs_embeds=sliced_inputs_embeds,
                               intermediate_tensors=sliced_intermediate_tensors,
                               num_tokens=ubatch_slice.token_slice.stop -
                               ubatch_slice.token_slice.start))

        return ubatch_metadata


    def _slice_model_inputs(self, tokens_slice: slice, input_ids, positions, inputs_embeds, intermediate_tensors):
        sliced_input_ids = input_ids[tokens_slice]
        # if we are using mrope
        if positions.ndim == 2:
            sliced_positions = positions[:, tokens_slice]
        else:
            sliced_positions = positions[tokens_slice]
        sliced_inputs_embeds = inputs_embeds[tokens_slice] if inputs_embeds else None
        sliced_intermediate_tensors = intermediate_tensors[tokens_slice] if intermediate_tensors else None

        return (sliced_input_ids, sliced_positions, sliced_inputs_embeds, 
                sliced_intermediate_tensors)

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        ubatch_slices = forward_context.ubatch_slices
        # If there's no ubatching, just run the runnable object
        # Check the cudagraph_runtime_mode
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
        if ubatch_slices is None:
            if cudagraph_runtime_mode is CUDAGraphMode.NONE:
                return self.runnable(*args, **kwargs)
            else:
                assert self.cudagraph_wrapper is not None
                return self.cudagraph_wrapper(*args, **kwargs)

        attn_metadata = forward_context.attn_metadata
        num_tokens = (ubatch_slices[0].token_slice.stop - ubatch_slices[0].token_slice.start) * 2
        input_ids = kwargs['input_ids']
        positions = kwargs['positions']
        intermediate_tensors = kwargs['intermediate_tensors']
        inputs_embeds = kwargs['inputs_embeds']
        compute_stream = torch.cuda.current_stream()

        # TODO (Sage) I'm not sure this is exactly correct.
        # We'll have to figure this out
        dp_metadata = forward_context.dp_metadata
        assert dp_metadata is not None
        num_tokens_across_dp = dp_metadata._num_tokens_across_dp

        if num_tokens not in self.cudagraphs \
            and cudagraph_runtime_mode is CUDAGraphMode.FULL:
            ubatch_metadata = self._make_ubatch_metadata(
                    ubatch_slices=ubatch_slices,
                    attn_metadata=attn_metadata,
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    compute_stream=compute_stream,
                    num_tokens_across_dp=num_tokens_across_dp,
                    batch_descriptor=batch_descriptor,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE)

            return self._capture_ubatches(ubatch_metadata, self.model)
        elif num_tokens in self.cudagraphs \
            and cudagraph_runtime_mode is CUDAGraphMode.FULL:
            cudagraph_metadata = self.cudagraphs[num_tokens]
            cudagraph_metadata.cudagraph.replay()
            return cudagraph_metadata.outputs
        else:
            ubatch_metadata = self._make_ubatch_metadata(
                    ubatch_slices=ubatch_slices,
                    attn_metadata=attn_metadata,
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    compute_stream=compute_stream,
                    num_tokens_across_dp=num_tokens_across_dp,
                    batch_descriptor=batch_descriptor,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE)
            return self._run_ubatches(ubatch_metadata, self.model)
