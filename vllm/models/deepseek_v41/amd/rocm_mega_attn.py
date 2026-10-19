# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import torch

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.models.deepseek_v41.amd.rocm import (
    DeepseekV41ROCMAiterMLAAttention,
    DeepseekV41ROCMAiterMLASparseBackend,
    DeepseekV41ROCMAiterSparseSWAMetadata,
)
from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLAMetadata
from vllm.platforms import current_platform


class DeepseekV41ROCMAiterMLAMegAttnBackend(DeepseekV41ROCMAiterMLASparseBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_FLASHMLA_MEGA_ATTN_DSV4"


class DeepseekV41ROCMAiterMegaAttnAttention(DeepseekV41ROCMAiterMLAAttention):
    """ROCm mega-attention layer for DeepSeek V4.1.

    Q preparation still applies RoPE and uses the ordinary head-major layout.
    Prefill and decode read ROCm's paged 584-byte cache records directly and
    fuse sparse attention with inverse RoPE and MXFP8 output.
    """

    backend_cls = DeepseekV41ROCMAiterMLAMegAttnBackend

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: QuantizedActivation,
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            output.data.zero_()
            output.scale.zero_()
            return

        assert isinstance(attn_metadata, dict)
        assert q.shape[1] == self.padded_heads
        rocm_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            DeepseekV41ROCMAiterSparseSWAMetadata,
            attn_metadata[self.swa_cache_layer.prefix],
        )

        swa_only = self.compress_ratio == 0
        self_kv_cache = None if swa_only else self._compressed_kv_cache()
        swa_kv_cache = self.swa_cache_layer.kv_cache

        num_decode_tokens = swa_metadata.num_decode_tokens

        if swa_metadata.num_prefills > 0:
            self._forward_prefill_mega(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                attn_metadata=rocm_metadata,
                swa_metadata=swa_metadata,
                output=(
                    output.data[num_decode_tokens:],
                    output.scale[num_decode_tokens:],
                ),
            )
        if swa_metadata.num_decodes > 0:
            self._forward_decode_mega(
                q=q[:num_decode_tokens],
                positions=positions[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=rocm_metadata,
                swa_only=swa_only,
                output=(
                    output.data[:num_decode_tokens],
                    output.scale[:num_decode_tokens],
                ),
            )

    def _forward_decode_mega(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekV41ROCMAiterSparseSWAMetadata,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> int:
        """Read paged KV and fuse attention, inverse RoPE and MXFP8 cast."""
        indices = swa_metadata.decode_swa_ragged_indices
        indptr = swa_metadata.decode_swa_ragged_indptr
        assert indices is not None and indptr is not None
        extra_indices = extra_indptr = None
        if not swa_only:
            assert kv_cache is not None
            extra_indices, extra_indptr, _ = self._decode_topk_ragged(
                swa_metadata,
                attn_metadata,
                swa_metadata.num_decodes,
                swa_metadata.num_decode_tokens,
            )
        torch.ops._rocm_C.fused_norm_rope_attn_rope_cast_decode(
            q,
            self.swa_cache_layer.kv_cache,
            indices,
            indptr,
            None if swa_only else kv_cache,
            extra_indices,
            extra_indptr,
            self.attn_sink[: q.shape[1]],
            self.scale,
            positions,
            self.rotary_emb.cos_sin_cache,
            current_platform.is_fp8_fnuz(),
            output[0],
            output[1],
        )
        return q.shape[0]

    def _forward_prefill_mega(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: DeepseekV41ROCMAiterSparseSWAMetadata,
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Resolve paged KV and fuse dequantization, attention and MXFP8 output."""
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        first = swa_metadata.num_decode_tokens
        tokens = swa_metadata.num_prefill_tokens
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        starts = swa_metadata.query_start_loc
        assert seq_lens is not None and gather_lens is not None and starts is not None
        assert self.topk_indices_buffer is not None
        extra_table = None
        if self.compress_ratio:
            assert attn_metadata is not None and compressed_k_cache is not None
            extra_table = attn_metadata.block_table[
                num_decodes : num_decodes + num_prefills
            ]
        torch.ops._rocm_C.fused_norm_rope_attn_rope_cast_prefill(
            q[:tokens],
            swa_k_cache,
            swa_metadata.block_table[num_decodes : num_decodes + num_prefills],
            compressed_k_cache,
            extra_table,
            self.topk_indices_buffer[first : first + tokens],
            starts[num_decodes : num_decodes + num_prefills + 1],
            seq_lens[:num_prefills],
            gather_lens[:num_prefills],
            self.window_size,
            self.compress_ratio,
            self.attn_sink[: q.shape[1]],
            self.scale,
            positions[:tokens],
            self.rotary_emb.cos_sin_cache,
            current_platform.is_fp8_fnuz(),
            output[0][:tokens],
            output[1][:tokens],
        )
