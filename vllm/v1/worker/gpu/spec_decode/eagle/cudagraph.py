# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup


class EagleCudaGraphManager(CudaGraphManager):
    """CudaGraphManager for Eagle speculative decoding (FULL mode only)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        draft_tokens: torch.Tensor,
    ):
        # Eagle always uses uniform decode with query_len=1
        super().__init__(vllm_config, device, decode_query_len=1)
        self.draft_tokens = draft_tokens

    def set_cg_mode_and_candidates(self, resolved_mode: CUDAGraphMode) -> None:
        # Eagle only need FULL-DECODE-ONLY mode now.
        if resolved_mode.decode_mode() == CUDAGraphMode.FULL:
            resolved_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            resolved_mode = CUDAGraphMode.NONE
        super().set_cg_mode_and_candidates(resolved_mode)
        # Use a dedicated pool for Eagle to avoid memory overlap with the main
        # model's cudagraph. The base class uses a shared global pool, but Eagle's
        # internal allocations (e.g., gumbel_sample temporaries) can conflict with
        # the main model's allocations when sharing the same pool.
        if self.cudagraph_mode:
            self.pool = torch.cuda.graph_pool_handle()

    def capture(
        self,
        generate_fn: Callable,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs for Eagle speculative decoding (FULL mode only)."""

        def create_forward_fn(
            desc: BatchExecutionDescriptor,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
            attn_metadata, slot_mappings = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
            )

            return lambda cg_mode: generate_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cg_mode,
            )

        super().capture(create_forward_fn, progress_bar_desc)

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor:
        """Replay a captured FULL cudagraph and return draft tokens."""
        super().run_fullgraph(desc)
        return self.draft_tokens
