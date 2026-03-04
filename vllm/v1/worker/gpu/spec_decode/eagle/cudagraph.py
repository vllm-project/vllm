# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CaptureCallback,
    CudaGraphManager,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.dp_utils import make_num_tokens_across_dp
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup


class EagleCudaGraphManager(CudaGraphManager):
    """CudaGraphManager for Eagle speculative decoding with draft token management."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        draft_tokens: torch.Tensor,
    ):
        # Eagle always uses uniform decode with query_len=1
        super().__init__(
            vllm_config, device, cudagraph_mode, uniform_decode_query_len=1
        )
        self.draft_tokens = draft_tokens

    def capture(
        self,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        generate_fn: CaptureCallback,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs for Eagle speculative decoding."""

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
                generate_fn(desc)

        super().capture(capture_fn, progress_bar_desc)

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor:
        """Replay a captured FULL cudagraph and return draft tokens."""
        super().run_fullgraph(desc)
        return self.draft_tokens[: desc.num_reqs]
