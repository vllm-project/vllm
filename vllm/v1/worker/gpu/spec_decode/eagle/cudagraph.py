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
    make_num_tokens_across_dp,
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
        cudagraph_mode: CUDAGraphMode,
        draft_tokens: torch.Tensor,
    ):
        assert not cudagraph_mode.has_mode(CUDAGraphMode.PIECEWISE), (
            "EagleCudaGraphManager does not support PIECEWISE mode yet"
        )
        # Eagle always uses uniform decode with query_len=1
        super().__init__(
            vllm_config, device, cudagraph_mode, uniform_decode_query_len=1
        )
        self.draft_tokens = draft_tokens

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

        def capture_fn(desc: BatchExecutionDescriptor) -> None:
            num_reqs, num_tokens = desc.num_reqs, desc.num_tokens
            num_tokens_across_dp = make_num_tokens_across_dp(self.dp_size, num_tokens)
            attn_metadata, slot_mappings = prepare_inputs_to_capture(
                desc,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
            )
            # Eagle only uses FULL mode - cg_mode is NONE for both warmup
            # and inside torch.cuda.graph capture
            generate_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                CUDAGraphMode.NONE,
            )

        super().capture(capture_fn, progress_bar_desc)

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor:
        """Replay a captured FULL cudagraph and return draft tokens."""
        super().run_fullgraph(desc)
        return self.draft_tokens[: desc.num_reqs]
