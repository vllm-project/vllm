# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import graph_capture, is_global_first_rank
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.offloader.base import get_offloader
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup


@dataclass(frozen=True)
class BatchExecutionDescriptor:
    """Batch execution shape descriptor with mode and padded values."""

    cg_mode: CUDAGraphMode
    num_tokens: int
    num_reqs: int
    uniform_token_count: int | None = None


def get_uniform_token_count(
    num_reqs: int,
    num_tokens: int,
    max_query_len: int,
) -> int | None:
    """
    Return the uniform token count if batch is uniform, else None.
    A batch is uniform if all requests have the same number of tokens.
    """
    if (max_query_len == num_tokens // num_reqs) and (
        num_tokens == max_query_len * num_reqs
    ):
        return max_query_len
    return None


def make_num_tokens_across_dp(dp_size: int, num_tokens: int) -> torch.Tensor | None:
    if dp_size == 1:
        return None
    return torch.full((dp_size,), num_tokens, dtype=torch.int32, device="cpu")


class CudaGraphManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        uniform_decode_query_len: int,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None
        self.cudagraph_mode = cudagraph_mode
        self.uniform_decode_query_len = uniform_decode_query_len
        self.dp_size = vllm_config.parallel_config.data_parallel_size

        self.graphs: dict[BatchExecutionDescriptor, torch.cuda.CUDAGraph] = {}
        self._graphs_captured = False
        self.pool = (
            torch.cuda.graph_pool_handle()
            if self.cudagraph_mode != CUDAGraphMode.NONE
            else None
        )
        self._candidates: list[list[BatchExecutionDescriptor]] = []
        self._capture_descs: dict[CUDAGraphMode, list[BatchExecutionDescriptor]] = {}
        self._init_candidates()

    def _init_candidates(self) -> None:
        """Build priority-ordered candidate lists for each token count."""
        if self.cudagraph_mode == CUDAGraphMode.NONE:
            return
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        if not capture_sizes:
            return
        capture_sizes = sorted(capture_sizes)
        max_decode_tokens = self.max_num_reqs * self.uniform_decode_query_len
        decode_mode = self.cudagraph_mode.decode_mode()
        mixed_mode = self.cudagraph_mode.mixed_mode()
        separate_decode_routine = self.cudagraph_mode.separate_routine()

        descs_by_token_count = defaultdict(list)
        descs_by_mode = defaultdict(list)

        for padded in capture_sizes:
            if (
                decode_mode != CUDAGraphMode.NONE
                and self.uniform_decode_query_len <= padded <= max_decode_tokens
            ):
                desc = BatchExecutionDescriptor(
                    cg_mode=decode_mode,
                    num_tokens=padded,
                    num_reqs=padded // self.uniform_decode_query_len,
                    # if separate decode routine, we assume the decode graphs needs
                    # to be uniform (otherwise, a mixed mode graph would be fine)
                    uniform_token_count=(
                        self.uniform_decode_query_len
                        if separate_decode_routine
                        else None
                    ),
                )
                descs_by_mode[decode_mode].append(desc)
                descs_by_token_count[padded].append(desc)

            if mixed_mode != CUDAGraphMode.NONE:
                desc = BatchExecutionDescriptor(
                    cg_mode=mixed_mode,
                    num_tokens=padded,
                    num_reqs=min(padded, self.max_num_reqs),
                )
                descs_by_mode[mixed_mode].append(desc)
                descs_by_token_count[padded].append(desc)

        if not descs_by_token_count:
            return

        sorted_padded = sorted(descs_by_token_count.keys())
        self._candidates = [[] for _ in range(sorted_padded[-1] + 1)]

        current_range_start = 0
        for cg_size in sorted_padded:
            for i in range(current_range_start, cg_size):
                self._candidates[i] = descs_by_token_count[cg_size]
            current_range_start += cg_size

        for mode, descs in descs_by_mode.items():
            descs.sort(key=lambda d: d.num_tokens, reverse=True)
            self._capture_descs[mode] = descs

    def needs_capture(self) -> bool:
        return len(self._capture_descs) > 0

    @torch.inference_mode()
    def capture(
        self,
        capture_fn: Callable,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs by calling capture_fn for each descriptor.

        The callback is invoked with desc.cg_mode indicating the runtime mode:
        - NONE: warmup pass (no capture)
        - PIECEWISE: piecewise cudagraph capture
        - FULL: inside torch.cuda.graph() capture (but mode is NONE)
        """
        with graph_capture(device=self.device):
            # Capture in order: PIECEWISE first, then FULL, PIECEWISE has larger
            # activations so in theory the FULL activations should fit in the already
            # allocated buffers in the graph pool. More experiments are needed to
            # to confirm this.
            for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
                if mode not in self._capture_descs:
                    continue

                descs = self._capture_descs[mode]
                if is_global_first_rank():
                    descs = tqdm(descs, desc=f"{progress_bar_desc} ({mode.name})")
                for desc in descs:
                    # Warmup with cg_mode=NONE
                    warmup_desc = BatchExecutionDescriptor(
                        cg_mode=CUDAGraphMode.NONE,
                        num_tokens=desc.num_tokens,
                        num_reqs=desc.num_reqs,
                        uniform_token_count=desc.uniform_token_count,
                    )
                    capture_fn(warmup_desc)

                    # Capture
                    if desc.cg_mode == CUDAGraphMode.PIECEWISE:
                        capture_fn(desc)
                    else:
                        assert desc not in self.graphs, (
                            f"Graph already captured for {desc}"
                        )
                        graph = torch.cuda.CUDAGraph()
                        # Sync offloader's copy stream before capture.
                        # Ensure any pre-capture prefetches from offloader are complete.
                        get_offloader().sync_prev_onload()
                        with torch.cuda.graph(graph, self.pool):
                            capture_fn(warmup_desc)
                            # Join offloader's copy stream after forward to avoid
                            # unjoined stream error. The last layer's start_prefetch
                            # forks copy_stream, but wait_prefetch only happens in
                            # the next forward pass.
                            get_offloader().join_after_forward()
                        self.graphs[desc] = graph
        self._graphs_captured = True

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
    ) -> BatchExecutionDescriptor:
        """Find matching cudagraph descriptor from priority-ordered candidates."""
        if self._graphs_captured and 0 < num_tokens < len(self._candidates):
            for desc in self._candidates[num_tokens]:
                if desc.uniform_token_count != uniform_token_count:
                    continue
                return desc
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens, num_reqs=num_reqs
        )

    def run_fullgraph(self, desc: BatchExecutionDescriptor):
        """Replay a captured FULL cudagraph."""
        assert desc.cg_mode == CUDAGraphMode.FULL, (
            f"Expected FULL mode, got {desc.cg_mode}"
        )
        assert desc in self.graphs, f"No cudagraph for {desc}"
        # Sync offloader before replay - needed when transitioning from
        # eager/piecewise to full cudagraph (e.g., prefill → decode).
        # The previous eager iteration's start_prefetch may have queued
        # H2D copies on copy_stream that the graph's captured events
        # cannot see. Without this, replay could overwrite static buffers
        # while those copies are still in flight.
        get_offloader().sync_prev_onload()
        self.graphs[desc].replay()


class ModelCudaGraphManager(CudaGraphManager):
    """CudaGraphManager with model-specific capture and hidden state management."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        uniform_decode_query_len: int,
    ):
        super().__init__(vllm_config, device, cudagraph_mode, uniform_decode_query_len)
        self.hidden_states: torch.Tensor | None = None
        self.aux_hidden_states: list[torch.Tensor] = []
        self.use_aux_hidden_state_outputs = False

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        self.use_aux_hidden_state_outputs = use_aux_hidden_state_outputs

        def capture_fn(desc: BatchExecutionDescriptor) -> None:
            num_tokens = desc.num_tokens
            num_tokens_across_dp = make_num_tokens_across_dp(self.dp_size, num_tokens)
            attn_metadata, slot_mappings = prepare_inputs_to_capture(
                desc,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
            )
            batch_descriptor = (
                BatchDescriptor(num_tokens=num_tokens)
                if desc.cg_mode == CUDAGraphMode.PIECEWISE
                else None
            )
            with set_forward_context(
                attn_metadata if desc.cg_mode != CUDAGraphMode.PIECEWISE else None,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                slot_mapping=slot_mappings,
                batch_descriptor=batch_descriptor,
            ):
                model_inputs = {
                    "input_ids": input_buffers.input_ids[:num_tokens],
                    "positions": input_buffers.positions[:num_tokens],
                }
                model_output = model(**model_inputs)
                if self.use_aux_hidden_state_outputs:
                    hidden_states, aux_hidden_states = model_output
                else:
                    hidden_states = model_output
                    aux_hidden_states = []
                if self.hidden_states is None:
                    self.hidden_states = torch.empty_like(hidden_states)
                if self.use_aux_hidden_state_outputs and not self.aux_hidden_states:
                    self.aux_hidden_states = [
                        torch.empty_like(x) for x in aux_hidden_states
                    ]
                self.hidden_states[:num_tokens] = hidden_states
                for i, aux in enumerate(aux_hidden_states):
                    self.aux_hidden_states[i][:num_tokens] = aux

        super().capture(capture_fn, progress_bar_desc)

    def run_fullgraph(
        self, desc: BatchExecutionDescriptor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Replay a captured FULL cudagraph and return hidden states."""
        super().run_fullgraph(desc)
        assert self.hidden_states is not None
        hidden_states = self.hidden_states[: desc.num_tokens]
        if not self.use_aux_hidden_state_outputs:
            return hidden_states
        return hidden_states, [x[: desc.num_tokens] for x in self.aux_hidden_states]


def prepare_inputs_to_capture(
    desc: BatchExecutionDescriptor,
    model_state: ModelState,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    num_reqs, num_tokens = desc.num_reqs, desc.num_tokens
    input_batch = InputBatch.make_dummy(desc, input_buffers)
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    # HACK(woosuk): Special handling for DCP.
    if block_tables.cp_size > 1:
        prepare_dcp_local_seq_lens(
            input_buffers.dcp_local_seq_lens,
            input_batch.seq_lens,
            num_reqs,
            block_tables.cp_size,
            block_tables.cp_rank,
            block_tables.cp_interleave,
        )
        input_batch.dcp_local_seq_lens = input_buffers.dcp_local_seq_lens[:num_reqs]

    attn_metadata = model_state.prepare_attn(
        input_batch,
        input_block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
    )
    return attn_metadata, slot_mappings_by_layer
