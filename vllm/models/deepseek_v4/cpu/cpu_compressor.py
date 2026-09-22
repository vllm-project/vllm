# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU DeepSeek-V4 compressor subclass and dispatch: partial-state-cache
write and the compress -> RMSNorm -> RoPE -> FP8 quant -> KV cache store
step, calling ``torch.ops._C.save_partial_states_cpu``/
``compress_norm_rope_store_cpu``/``compress_norm_rope_store_indexer_cpu``
(``csrc/cpu/sgl-kernels/compressor.cpp``) in place of triton-cpu.

Covers both the head_dim=512 main-attention compressor path and the
head_dim=128 indexer compressor path (fp8 only -- MXFP4, ``use_fp4_cache``,
never occurs on CPU: the indexer's MXFP4 cache requires a Blackwell GPU).
"""

from typing import Any, cast

import torch

from vllm._custom_ops import (
    compress_norm_rope_store_cpu,
    compress_norm_rope_store_indexer_cpu,
    save_partial_states_cpu,
)
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.compressor import CompressorMetadata, DeepseekCompressor
from vllm.models.deepseek_v4.cpu.cpu_utils import map_local_to_global_slots_cpu


class DeepseekV4CPUCompressor(DeepseekCompressor):
    """CPU compressor: same state/weights as the shared base, but
    ``forward``'s cache-write and compress->RMSNorm->RoPE->quant->store
    steps always dispatch straight to the ported CPU kernels above instead
    of through the shared method's platform-dispatch chain. This class
    only ever runs on CPU, and ``use_fp4_cache`` is always ``False`` here
    (the indexer's MXFP4 cache requires a Blackwell GPU), so the shared
    method's MXFP4/two-stage/cutedsl/triton branches never apply.
    """

    _norm_weight_fp32: torch.Tensor

    def cache_norm_weight_fp32(self) -> None:
        """Cache a fp32 contiguous copy of ``norm.weight`` once, right after
        weight loading (see
        ``DeepseekV4CPUAttention.process_weights_after_loading``) -- the CPU
        kernel requires fp32/contiguous but the checkpoint loads this weight
        in bf16, and it never changes after loading."""
        self._norm_weight_fp32 = self.norm.weight.to(torch.float32).contiguous()

    def forward(
        self,
        kv_score: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        state_metadata = attn_metadata[self.state_cache.prefix]
        assert isinstance(state_metadata, CompressorMetadata)
        token_to_req_indices = state_metadata.token_to_req_indices
        assert token_to_req_indices is not None
        slot_mapping = state_metadata.slot_mapping
        num_actual = slot_mapping.shape[0]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        state_cache = self.state_cache.kv_cache

        save_partial_states_cpu(
            kv, score, self.ape, positions, state_cache, slot_mapping
        )

        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = k_cache_layer.kv_cache

        assert self.head_dim in (512, 128), (
            f"unsupported compressor head_dim: {self.head_dim}"
        )
        window = (1 + self.overlap) * self.compress_ratio
        win_offsets = torch.arange(window, device=positions.device)
        win_local = positions.unsqueeze(-1) - window + 1 + win_offsets
        req_idx = token_to_req_indices[:num_actual]
        gather_slots = map_local_to_global_slots_cpu(
            win_local, req_idx, block_table, block_size
        )

        kv_cache_2d = kv_cache.view(kv_cache.shape[0], -1)
        kv_cache_block_size = kv_cache.shape[1]

        store_fn = (
            compress_norm_rope_store_cpu
            if self.head_dim == 512
            else compress_norm_rope_store_indexer_cpu
        )
        store_fn(
            state_cache,
            gather_slots,
            positions,
            k_cache_metadata.slot_mapping,
            self._norm_weight_fp32,
            self.rms_norm_eps,
            cos_sin_cache,
            kv_cache_2d,
            kv_cache_block_size,
            self.compress_ratio,
        )
