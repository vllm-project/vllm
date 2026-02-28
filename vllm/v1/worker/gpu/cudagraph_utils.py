# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import graph_capture, is_global_first_rank
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.lora.utils import get_captured_lora_counts
from vllm.model_executor.offloader.base import get_offloader
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.dp_utils import make_num_tokens_across_dp
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states import ModelState
from vllm.v1.worker.utils import AttentionGroup


def _make_graph_key(num_tokens: int, num_active_loras: int = 0) -> tuple[int, int]:
    """Create a unique key for CUDA graph storage (num_tokens, num_active_loras)."""
    return (num_tokens, num_active_loras)


class CudaGraphManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        use_aux_hidden_state_outputs: bool,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.use_aux_hidden_state_outputs = use_aux_hidden_state_outputs
        self.device = device

        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.dp_size = vllm_config.parallel_config.data_parallel_size

        self.uniform_decode_query_len = 1
        spec_config = vllm_config.speculative_config
        if spec_config is not None:
            self.uniform_decode_query_len += spec_config.num_speculative_tokens

        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None
        self.cudagraph_mode = self.compilation_config.cudagraph_mode

        use_uniform_decode_cudagraph = (
            self.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and self.cudagraph_mode.separate_routine()
        )
        self.cudagraph_sizes, self.uniform_decode_cudagraph_sizes = get_cudagraph_sizes(
            self.compilation_config.cudagraph_capture_sizes,
            self.max_num_reqs,
            self.max_num_tokens,
            self.cudagraph_mode,
            self.uniform_decode_query_len,
            use_uniform_decode_cudagraph,
        )

        # Compute LoRA capture cases for cudagraph_specialize_lora.
        lora_config = vllm_config.lora_config
        if (
            lora_config is not None
            and self.compilation_config.cudagraph_specialize_lora
        ):
            specialize_lora_count = getattr(
                lora_config, "specialize_active_lora", False
            )
            captured = get_captured_lora_counts(
                lora_config.max_loras, specialize_lora_count
            )
            # Include 0 for no-LoRA case; filter out 0 from captured (it's added)
            self.lora_capture_cases = [0] + [c for c in captured if c > 0]
        else:
            # When cudagraph_specialize_lora=False, use single LoRA case
            # (max_loras+1) for has_lora batches; 0 for no-LoRA.
            if lora_config is not None:
                self.lora_capture_cases = [0, lora_config.max_loras + 1]
            else:
                self.lora_capture_cases = [0]

        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.pool = None
        if self.cudagraph_mode != CUDAGraphMode.NONE:
            self.pool = torch.cuda.graph_pool_handle()
        self.hidden_states: torch.Tensor | None = None
        self.aux_hidden_states: list[torch.Tensor] = []

    def needs_capture(self) -> bool:
        return len(self.cudagraph_sizes) > 0

    def get_cudagraph_size(
        self, num_tokens: int, uniform_decode: bool = False
    ) -> int | None:
        if uniform_decode and self.uniform_decode_cudagraph_sizes:
            return self.uniform_decode_cudagraph_sizes.get(num_tokens)
        return self.cudagraph_sizes.get(num_tokens)

    def capture_graph(
        self,
        num_tokens: int,
        capture_cg_mode: CUDAGraphMode,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        inputs_embeds: torch.Tensor | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        num_active_loras: int = 0,
        uniform_decode: bool = False,
    ) -> None:
        # select and check capture function
        assert capture_cg_mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL], (
            f"Invalid capture_cudagraph_mode for capture: {capture_cg_mode}"
        )
        if capture_cg_mode == CUDAGraphMode.PIECEWISE:
            capture_fn = self._capture_piecewise_graph
        else:
            capture_fn = self._capture_full_graph
        # prepare inputs
        if uniform_decode:
            num_reqs = min(
                cdiv(num_tokens, self.uniform_decode_query_len),
                self.max_num_reqs,
            )
        else:
            num_reqs = min(num_tokens, self.max_num_reqs)

        model_inputs = {
            "input_ids": input_buffers.input_ids[:num_tokens],
            "positions": input_buffers.positions[:num_tokens],
            "inputs_embeds": (
                inputs_embeds[:num_tokens] if inputs_embeds is not None else None
            ),
            # NOTE: Values returned by `prepare_dummy_inputs` will override the
            # default values above.
            **model_state.prepare_dummy_inputs(num_reqs, num_tokens),
        }

        attn_metadata, slot_mappings = prepare_inputs_to_capture(
            num_reqs,
            num_tokens,
            input_buffers,
            block_tables,
            attn_groups,
            self.max_model_len,
            kv_cache_config,
            uniform_decode_query_len=(
                self.uniform_decode_query_len if uniform_decode else 0
            ),
        )
        num_tokens_across_dp = make_num_tokens_across_dp(self.dp_size, num_tokens)

        # Warm up.
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
        ):
            model_output = model(**model_inputs)
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
            else:
                hidden_states = model_output
                aux_hidden_states = None

        # Allocate output buffers if not already done.
        if self.hidden_states is None:
            self.hidden_states = torch.empty_like(hidden_states)
        if self.use_aux_hidden_state_outputs and not self.aux_hidden_states:
            self.aux_hidden_states = [torch.empty_like(x) for x in aux_hidden_states]

        capture_fn(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            model=model,
            model_inputs=model_inputs,
            num_tokens_across_dp=num_tokens_across_dp,
            attn_metadata=attn_metadata,
            slot_mappings=slot_mappings,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
        )

    def _capture_full_graph(
        self,
        num_tokens: int,
        num_reqs: int,
        model: nn.Module,
        model_inputs: dict[str, torch.Tensor | None],
        num_tokens_across_dp: torch.Tensor,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        has_lora: bool = False,
        num_active_loras: int = 0,
    ) -> None:
        assert attn_metadata is not None
        graph_key = _make_graph_key(num_tokens, num_active_loras)
        assert graph_key not in self.graphs
        graph = torch.cuda.CUDAGraph()

        # Sync offloader's copy stream before capture.
        # Ensure any pre-capture prefetches from offloader are complete.
        get_offloader().sync_prev_onload()

        with (
            set_forward_context(
                attn_metadata=attn_metadata,
                vllm_config=self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                num_tokens_across_dp=num_tokens_across_dp,
                slot_mapping=slot_mappings,
            ),
            torch.cuda.graph(graph, self.pool),
        ):
            model_output = model(**model_inputs)

            # Join offloader's copy stream after forward to avoid unjoined
            # stream error. The last layer's start_prefetch forks copy_stream,
            # but wait_prefetch only happens in the next forward pass.
            get_offloader().join_after_forward()

            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
            else:
                hidden_states = model_output
                aux_hidden_states = None

            # Copy outputs to the output buffers.
            assert self.hidden_states is not None
            self.hidden_states[:num_tokens] = hidden_states
            if self.use_aux_hidden_state_outputs:
                for i, aux_hidden in enumerate(aux_hidden_states):
                    self.aux_hidden_states[i][:num_tokens] = aux_hidden
        self.graphs[graph_key] = graph

    def _capture_piecewise_graph(
        self,
        num_tokens: int,
        num_reqs: int,
        model: nn.Module,
        model_inputs: dict[str, torch.Tensor | None],
        num_tokens_across_dp: torch.Tensor,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        has_lora: bool = False,
        num_active_loras: int = 0,
    ) -> None:
        # create batch descriptor for piecewise cudagraph dispatch key
        batch_descriptor = BatchDescriptor(
            num_tokens=num_tokens,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
        )

        # Capture run - CUDAGraphWrapper inside torch.compile will auto capture.
        with set_forward_context(
            attn_metadata=None,  # piecewise no need attn_metadata
            vllm_config=self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            num_tokens_across_dp=num_tokens_across_dp,
            batch_descriptor=batch_descriptor,
            slot_mapping=slot_mappings,
        ):
            model(**model_inputs)

    @torch.inference_mode()
    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        inputs_embeds: torch.Tensor | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
    ) -> None:
        common_kwargs = dict(
            device=self.device,
            capture_fn=self.capture_graph,
            model=model,
            model_state=model_state,
            input_buffers=input_buffers,
            inputs_embeds=inputs_embeds,
            block_tables=block_tables,
            attn_groups=attn_groups,
            kv_cache_config=kv_cache_config,
            has_lora=has_lora,
            uniform_decode_query_len=self.uniform_decode_query_len,
            max_num_reqs=self.max_num_reqs,
        )

        # Phase 1: Capture for mixed prefill-decode batches if needed.
        mixed_mode = self.cudagraph_mode.mixed_mode()
        if mixed_mode != CUDAGraphMode.NONE:
            capture_graphs(
                cudagraph_sizes=self.cudagraph_sizes,
                capture_cudagraph_mode=mixed_mode,
                desc=f"Capturing CUDA graphs (mixed, {mixed_mode.name})",
                uniform_decode=False,
                lora_capture_cases=self.lora_capture_cases,
                lora_capture_hook=lora_capture_hook,
                **common_kwargs,
            )

        # Phase 2: Capture FULL graphs for uniform decode batches if needed.
        # This is only needed if we use a separate routine for decode batches
        # and the decode_mode is FULL.
        if self.uniform_decode_cudagraph_sizes:
            capture_graphs(
                cudagraph_sizes=self.uniform_decode_cudagraph_sizes,
                capture_cudagraph_mode=CUDAGraphMode.FULL,
                desc="Capturing CUDA graphs (decode, FULL)",
                uniform_decode=True,
                lora_capture_cases=self.lora_capture_cases,
                lora_capture_hook=lora_capture_hook,
                **common_kwargs,
            )

    def get_cudagraph_runtime_mode(
        self,
        num_reqs: int,
        num_tokens: int,
        max_query_len: int,
        num_active_loras: int = 0,
    ) -> tuple[CUDAGraphMode, int | None, int]:
        """Returns (cudagraph_mode, cudagraph_size, effective_num_active_loras)."""
        is_uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
            num_tokens == max_query_len * num_reqs
        )

        cudagraph_size = self.get_cudagraph_size(num_tokens, is_uniform_decode)
        if cudagraph_size is None:
            cudagraph_mode = CUDAGraphMode.NONE
        elif is_uniform_decode:
            cudagraph_mode = self.cudagraph_mode.decode_mode()
        else:
            cudagraph_mode = self.cudagraph_mode.mixed_mode()

        # Resolve effective_num_active_loras for graph lookup.
        effective_num_active_loras = num_active_loras
        if num_active_loras > 0 and self.lora_capture_cases:
            import bisect

            idx = bisect.bisect_left(
                [c for c in self.lora_capture_cases if c > 0], num_active_loras
            )
            captured_with_lora = [c for c in self.lora_capture_cases if c > 0]
            if idx < len(captured_with_lora):
                effective_num_active_loras = captured_with_lora[idx]
            else:
                effective_num_active_loras = (
                    captured_with_lora[-1] if captured_with_lora else 0
                )

        if (
            cudagraph_mode == CUDAGraphMode.FULL
            and cudagraph_size is not None
            and _make_graph_key(cudagraph_size, effective_num_active_loras)
            not in self.graphs
        ):
            # If graph wasn't captured yet, fall back to eager.
            # This might happen when the dummy run is called before capture.
            cudagraph_mode = CUDAGraphMode.NONE
            cudagraph_size = None
        return cudagraph_mode, cudagraph_size, effective_num_active_loras

    def run_fullgraph(
        self, num_tokens: int, num_active_loras: int = 0
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        graph_key = _make_graph_key(num_tokens, num_active_loras)
        assert graph_key in self.graphs, (
            f"No cudagraph for {num_tokens} tokens, num_active_loras={num_active_loras}"
        )
        # Sync offloader before replay - needed when transitioning from
        # eager/piecewise to full cudagraph (e.g., prefill → decode).
        # The previous eager iteration's start_prefetch may have queued
        # H2D copies on copy_stream that the graph's captured events
        # cannot see. Without this, replay could overwrite static buffers
        # while those copies are still in flight.
        get_offloader().sync_prev_onload()
        self.graphs[graph_key].replay()
        assert self.hidden_states is not None
        hidden_states = self.hidden_states[:num_tokens]
        if not self.use_aux_hidden_state_outputs:
            return hidden_states
        return hidden_states, [x[:num_tokens] for x in self.aux_hidden_states]


def get_cudagraph_sizes(
    capture_sizes: list[int] | None,
    max_num_reqs: int,
    max_num_tokens: int,
    cudagraph_mode: CUDAGraphMode,
    uniform_decode_query_len: int = 1,
    uniform_decode_cudagraph: bool = False,
) -> tuple[dict[int, int], dict[int, int]]:
    # Support both FULL and PIECEWISE cudagraph modes
    if cudagraph_mode == CUDAGraphMode.NONE:
        return {}, {}
    if not capture_sizes:
        return {}, {}

    capture_sizes = sorted(capture_sizes)
    if not capture_sizes:
        return {}, {}

    cudagraph_sizes: dict[int, int] = {}
    for i in range(1, capture_sizes[-1] + 1):
        for x in capture_sizes:
            if i <= x:
                cudagraph_sizes[i] = x
                break

    uniform_decode_cudagraph_sizes: dict[int, int] = {}
    if uniform_decode_cudagraph:
        max_num_tokens = max_num_reqs * uniform_decode_query_len
        uniform_decode_cudagraph_sizes = {
            k: v
            for k, v in cudagraph_sizes.items()
            if v <= max_num_tokens and v >= uniform_decode_query_len
        }
    return cudagraph_sizes, uniform_decode_cudagraph_sizes


def capture_graphs(
    cudagraph_sizes: dict[int, int],
    device: torch.device,
    capture_fn: Callable,
    capture_cudagraph_mode: CUDAGraphMode,
    desc: str = "Capturing CUDA graphs",
    lora_capture_cases: list[int] | None = None,
    lora_capture_hook: Callable[[int, int, int], None] | None = None,
    uniform_decode: bool = False,
    **capture_kwargs,
) -> None:
    # Capture larger graphs first.
    sizes_to_capture = sorted(set(cudagraph_sizes.values()), reverse=True)
    lora_cases = lora_capture_cases if lora_capture_cases is not None else [0]

    # Build (size, num_active_loras) pairs for capture.
    capture_pairs = [
        (size, num_active_loras)
        for size in sizes_to_capture
        for num_active_loras in lora_cases
    ]
    if is_global_first_rank():
        capture_pairs = tqdm(capture_pairs, desc=desc)

    uniform_decode_query_len = capture_kwargs.pop("uniform_decode_query_len", 1)
    max_num_reqs = capture_kwargs.pop("max_num_reqs", 256)
    with graph_capture(device=device):
        for size, num_active_loras in capture_pairs:
            if lora_capture_hook is not None:
                num_reqs = (
                    min(cdiv(size, uniform_decode_query_len), max_num_reqs)
                    if uniform_decode
                    else min(size, max_num_reqs)
                )
                lora_capture_hook(num_active_loras, num_reqs, size)
            capture_fn(
                size,
                capture_cudagraph_mode,
                num_active_loras=num_active_loras,
                **capture_kwargs,
            )


def prepare_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    max_model_len: int,
    kv_cache_config: KVCacheConfig,
    uniform_decode_query_len: int = 0,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if uniform_decode_query_len > 0:
        num_tokens_per_req = uniform_decode_query_len
    else:
        num_tokens_per_req = num_tokens // num_reqs

    query_start_loc_np = np.arange(num_reqs + 1, dtype=np.int32) * num_tokens_per_req
    query_start_loc_np[-1] = num_tokens
    query_start_loc_cpu = torch.from_numpy(query_start_loc_np)
    input_buffers.query_start_loc[: num_reqs + 1] = query_start_loc_cpu
    input_buffers.query_start_loc[num_reqs + 1 :] = num_tokens
    query_start_loc = input_buffers.query_start_loc[: num_reqs + 1]

    # HACK(woosuk): For faster warmup, we set seq_lens (GPU) to num_tokens
    # rather than max_model_len.
    input_buffers.seq_lens[:num_reqs] = num_tokens
    input_buffers.seq_lens[num_reqs:] = 0

    input_buffers.dcp_local_seq_lens[:num_reqs] = num_tokens
    input_buffers.dcp_local_seq_lens[num_reqs:] = 0

    input_block_tables = [x[:num_reqs] for x in block_tables.input_block_tables]
    slot_mappings = block_tables.slot_mappings[:, :num_tokens]
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    attn_metadata = build_attn_metadata(
        attn_groups=attn_groups,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        query_start_loc_gpu=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        max_query_len=num_tokens_per_req,
        seq_lens=input_buffers.seq_lens,
        max_seq_len=max_model_len,
        block_tables=input_block_tables,
        slot_mappings=slot_mappings,
        kv_cache_config=kv_cache_config,
        dcp_local_seq_lens=input_buffers.dcp_local_seq_lens,
    )
    return attn_metadata, slot_mappings_by_layer
