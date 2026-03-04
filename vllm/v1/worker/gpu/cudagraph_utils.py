# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
from vllm.v1.worker.gpu.dp_utils import make_num_tokens_across_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup


@dataclass(frozen=True)
class BatchExecutionDescriptor:
    """Batch execution shape descriptor with mode and padded values."""

    cg_mode: CUDAGraphMode
    num_tokens: int
    num_reqs: int
    uniform: bool = False
    # Future LoRA support: num_active_loras: int = 0

    @property
    def is_cudagraph(self) -> bool:
        return self.cg_mode != CUDAGraphMode.NONE

    @property
    def cudagraph_size(self) -> int | None:
        return self.num_tokens if self.is_cudagraph else None


def is_uniform_batch(
    num_reqs: int,
    num_tokens: int,
    max_query_len: int,
    uniform_decode_query_len: int,
) -> bool:
    """Check if a batch qualifies as uniform decode for cudagraph selection."""
    return (max_query_len == uniform_decode_query_len) and (
        num_tokens == max_query_len * num_reqs
    )


# Type alias for capture callback function
CaptureCallback = Callable[
    [
        int,  # num_reqs
        int,  # num_tokens
        dict[str, Any],  # attn_metadata
        dict[str, torch.Tensor],  # slot_mappings
        torch.Tensor,  # num_tokens_across_dp
        CUDAGraphMode,  # cudagraph_runtime_mode
    ],
    None,
]


class CudaGraphManager:
    """CUDA graph manager for callback-based capture.

    Args:
        vllm_config: vLLM configuration
        device: CUDA device
        cudagraph_mode: CUDA graph mode to use
        uniform_decode_query_len: Query length for uniform decode batches
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        uniform_decode_query_len: int,
    ):
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device = device
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None
        self.cudagraph_mode = cudagraph_mode
        self.uniform_decode_query_len = uniform_decode_query_len

        self.graphs: dict[BatchExecutionDescriptor, torch.cuda.CUDAGraph] = {}
        self.pool = (
            torch.cuda.graph_pool_handle()
            if self.cudagraph_mode != CUDAGraphMode.NONE
            else None
        )
        self._candidates: list[list[BatchExecutionDescriptor]] = []
        self._capture_descs: list[
            tuple[CUDAGraphMode, list[BatchExecutionDescriptor]]
        ] = []
        self._init_candidates()

    def _init_candidates(self) -> None:
        """Build priority-ordered candidate lists for each token count."""
        if self.cudagraph_mode == CUDAGraphMode.NONE:
            return
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        if not capture_sizes:
            return
        capture_sizes = sorted(capture_sizes)
        max_uniform_tokens = self.max_num_reqs * self.uniform_decode_query_len
        decode_mode = self.cudagraph_mode.decode_mode()
        mixed_mode = self.cudagraph_mode.mixed_mode()
        separate_decode_routine = self.cudagraph_mode.separate_routine()

        candidates_by_padded_token_count: dict[int, list[BatchExecutionDescriptor]] = {}
        for padded in capture_sizes:
            candidates: list[BatchExecutionDescriptor] = []
            if (
                decode_mode != CUDAGraphMode.NONE
                and self.uniform_decode_query_len <= padded <= max_uniform_tokens
            ):
                candidates.append(
                    BatchExecutionDescriptor(
                        cg_mode=decode_mode,
                        num_tokens=padded,
                        num_reqs=padded // self.uniform_decode_query_len,
                        # if separate decode routine, we assume the decode graphs needs
                        # to be uniform (otherwise, a mixed mode graph would be fine)
                        uniform=separate_decode_routine,
                    )
                )
            if mixed_mode != CUDAGraphMode.NONE:
                candidates.append(
                    BatchExecutionDescriptor(
                        cg_mode=mixed_mode,
                        num_tokens=padded,
                        num_reqs=min(padded, self.max_num_reqs),
                        uniform=False,
                    )
                )
            if candidates:
                candidates_by_padded_token_count[padded] = candidates

        if candidates_by_padded_token_count:
            max_capture_size = max(candidates_by_padded_token_count.keys())
            sorted_padded = sorted(candidates_by_padded_token_count.keys())
            self._candidates = [[] for _ in range(max_capture_size + 1)]
            for num_tokens in range(1, max_capture_size + 1):
                for padded in sorted_padded:
                    if num_tokens <= padded:
                        self._candidates[num_tokens] = candidates_by_padded_token_count[
                            padded
                        ]
                        break

        descs_by_mode: dict[CUDAGraphMode, list[BatchExecutionDescriptor]] = {}
        for candidates in candidates_by_padded_token_count.values():
            for desc in candidates:
                descs_by_mode.setdefault(desc.cg_mode, []).append(desc)
        for mode, descs in descs_by_mode.items():
            descs.sort(key=lambda d: d.num_tokens, reverse=True)
            self._capture_descs.append((mode, descs))

    def needs_capture(self) -> bool:
        return len(self._capture_descs) > 0

    def get_capture_descs(
        self,
    ) -> list[tuple[CUDAGraphMode, list[BatchExecutionDescriptor]]]:
        return self._capture_descs

    def get_cudagraph_desc(
        self,
        num_reqs: int,
        num_tokens: int,
        is_uniform: bool,
    ) -> BatchExecutionDescriptor:
        """Find matching cudagraph descriptor from priority-ordered candidates."""
        if 0 < num_tokens < len(self._candidates):
            for desc in self._candidates[num_tokens]:
                if desc.uniform and not is_uniform:
                    continue
                return desc
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            uniform=is_uniform,
        )

    @torch.inference_mode()
    def capture(
        self,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        capture_fn: CaptureCallback,
        dp_size: int = 1,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs using the provided callback."""
        capture_descs = self.get_capture_descs()
        if not capture_descs:
            return
        with graph_capture(device=self.device):
            for mode, descs in capture_descs:
                if is_global_first_rank():
                    descs = tqdm(descs, desc=f"{progress_bar_desc} ({mode.name})")
                for desc in descs:
                    attn_metadata, slot_mappings = prepare_inputs_to_capture(
                        desc,
                        model_state,
                        input_buffers,
                        block_tables,
                        attn_groups,
                        kv_cache_config,
                    )
                    self._capture_graph(
                        desc, capture_fn, attn_metadata, slot_mappings, dp_size
                    )

    def _capture_graph(
        self,
        desc: BatchExecutionDescriptor,
        capture_fn: CaptureCallback,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        dp_size: int = 1,
    ) -> None:
        """Capture a single CUDA graph using the provided callback."""
        num_tokens, num_reqs = desc.num_tokens, desc.num_reqs
        assert desc.cg_mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        num_tokens_across_dp = make_num_tokens_across_dp(dp_size, num_tokens)

        # Warmup
        capture_fn(
            num_reqs,
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            CUDAGraphMode.NONE,
        )

        # Capture
        if desc.cg_mode == CUDAGraphMode.PIECEWISE:
            capture_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                CUDAGraphMode.PIECEWISE,
            )
        else:
            assert desc not in self.graphs, f"Graph already captured for {desc}"
            graph = torch.cuda.CUDAGraph()
            get_offloader().sync_prev_onload()
            with torch.cuda.graph(graph, self.pool):
                capture_fn(
                    num_reqs,
                    num_tokens,
                    attn_metadata,
                    slot_mappings,
                    num_tokens_across_dp,
                    CUDAGraphMode.NONE,
                )
                get_offloader().join_after_forward()
            self.graphs[desc] = graph

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

    @torch.inference_mode()
    def capture(
        self,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        model: nn.Module,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        dp_size: int = 1,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        from functools import partial

        capture_descs = self.get_capture_descs()
        if not capture_descs:
            return
        self.use_aux_hidden_state_outputs = use_aux_hidden_state_outputs

        def run_capture(
            model_inputs: dict[str, torch.Tensor],
            num_reqs: int,
            num_tokens: int,
            attn_metadata: dict[str, Any],
            slot_mappings: dict[str, torch.Tensor],
            num_tokens_across_dp: torch.Tensor,
            cudagraph_mode: CUDAGraphMode,
        ) -> None:
            batch_descriptor = (
                BatchDescriptor(num_tokens=num_tokens, has_lora=has_lora)
                if cudagraph_mode == CUDAGraphMode.PIECEWISE
                else None
            )
            with set_forward_context(
                attn_metadata if cudagraph_mode != CUDAGraphMode.PIECEWISE else None,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=cudagraph_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                slot_mapping=slot_mappings,
                batch_descriptor=batch_descriptor,
            ):
                model_output = model(**model_inputs)
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
            else:
                hidden_states = model_output
                aux_hidden_states = []
            # Allocate output buffers if not already done.
            if self.hidden_states is None:
                self.hidden_states = torch.empty_like(hidden_states)
            if self.use_aux_hidden_state_outputs and not self.aux_hidden_states:
                self.aux_hidden_states = [
                    torch.empty_like(x) for x in aux_hidden_states
                ]
            # Copy outputs to static buffers.
            self.hidden_states[:num_tokens] = hidden_states
            for i, aux in enumerate(aux_hidden_states):
                self.aux_hidden_states[i][:num_tokens] = aux

        with graph_capture(device=self.device):
            for mode, descs in capture_descs:
                if is_global_first_rank():
                    descs = tqdm(descs, desc=f"{progress_bar_desc} ({mode.name})")
                for desc in descs:
                    num_tokens, num_reqs = desc.num_tokens, desc.num_reqs
                    attn_metadata, slot_mappings = prepare_inputs_to_capture(
                        desc,
                        model_state,
                        input_buffers,
                        block_tables,
                        attn_groups,
                        kv_cache_config,
                    )
                    model_inputs = {
                        "input_ids": input_buffers.input_ids[:num_tokens],
                        "positions": input_buffers.positions[:num_tokens],
                        **model_state.prepare_dummy_inputs(num_reqs, num_tokens),
                    }
                    capture_fn = partial(run_capture, model_inputs)
                    self._capture_graph(
                        desc, capture_fn, attn_metadata, slot_mappings, dp_size
                    )

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
