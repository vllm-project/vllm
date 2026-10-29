# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 block-sparse attention over an NVFP4 K/V cache on ROCm (gfx950).

The cache spec is NVIDIA's, but the bytes inside a page are laid out for the
FlyDSL kernels, so pages are not interchangeable with NVIDIA's.
"""

from dataclasses import replace
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
    MiniMaxM3SparseMetadataBuilder,
    minimax_m3_query_token_positions,
)
from vllm.utils.torch_utils import get_dtype_size, nvfp4_kv_cache_full_dim
from vllm.v1.attention.backend import AttentionLayer, CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout


class MiniMaxM3SparseRocmNvfp4MetadataBuilder(MiniMaxM3SparseMetadataBuilder):
    """Adds each prefill token's request and absolute position."""

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3SparseMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        p = metadata.prefill
        if p is not None and p.query_req_id is None:
            p.query_req_id, p.query_abs_pos = minimax_m3_query_token_positions(
                p.cu_seqlens_q, p.context_lens, metadata.num_prefill_tokens
            )
        return metadata


class MiniMaxM3SparseRocmNvfp4Backend(MiniMaxM3SparseBackend):
    """MiniMax M3 sparse attend over NVFP4 K/V pages (ROCm, gfx950)."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["nvfp4"]

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3SparseRocmNvfp4MetadataBuilder"]:
        return MiniMaxM3SparseRocmNvfp4MetadataBuilder

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """Store each head's K and V as two slots of packed data + block scales."""
        if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_nvfp4:
            return spec
        return replace(
            spec,
            num_head_slots=2 * spec.num_kv_heads,
            state_content_bytes=nvfp4_kv_cache_full_dim(spec.head_size)
            * get_dtype_size(spec.dtype),
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...] | None:
        # The kernels read the page stride from stride(0) and address pages in
        # 64 bits, so block-outer packing works too (opt in with BLHNC).
        return (KVCacheLayout.LBHNC, KVCacheLayout.BLHNC)


def rocm_nvfp4_unsupported_reason(
    num_heads: int, num_kv_heads: int, head_dim: int, topk_blocks: int, block_size: int
) -> str | None:
    """Why the ROCm NVFP4 attend cannot serve this layer, None when it can."""
    from vllm.models.minimax_m3.amd.ops.nvfp4_prefill.host import (
        supports_nvfp4_sparse_prefill,
    )

    if not supports_nvfp4_sparse_prefill(
        num_heads, num_kv_heads, head_dim, topk_blocks, block_size
    ):
        return (
            "needs gfx950, head_dim 128, 16 query heads per KV head and 16 "
            f"selected 128-token blocks (got num_heads={num_heads}, "
            f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, "
            f"topk_blocks={topk_blocks}, block_size={block_size})"
        )
    return None


class MiniMaxM3SparseRocmNvfp4Impl(MiniMaxM3SparseImpl):
    """FlyDSL NVFP4 sparse decode (bf16 compute) and prefill (fp16 compute)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        reason = rocm_nvfp4_unsupported_reason(
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.topk_blocks,
            self.block_size,
        )
        if reason is not None:
            raise ValueError(f"MiniMax M3 NVFP4 KV cache on ROCm {reason}")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.models.minimax_m3.amd.ops.nvfp4_prefill.host import (
            minimax_m3_nvfp4_sparse_prefill,
        )
        from vllm.models.minimax_m3.amd.ops.nvfp4_sparse_decode.host import (
            minimax_m3_nvfp4_sparse_decode_fly,
        )

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output  # profiling run; caches unbound
        md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(md, MiniMaxM3SparseMetadata)

        nd = md.num_decode_tokens
        num_tokens = md.num_actual_tokens
        topk = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk is not None
        q = query[:num_tokens].view(-1, self.num_heads, self.head_size)
        out = output[:num_tokens].view(-1, self.num_heads, self.head_size)
        k_scale = layer._k_scale  # type: ignore[attr-defined]
        v_scale = layer._v_scale  # type: ignore[attr-defined]

        if md.num_decodes > 0:
            d = md.decode
            assert d is not None
            minimax_m3_nvfp4_sparse_decode_fly(
                q[:nd].contiguous(),
                kv_cache,
                topk[:, :nd, :],
                d.block_table,
                d.seq_lens,
                self.scale,
                out[:nd],
                k_scale=k_scale,
                v_scale=v_scale,
                decode_query_len=d.decode_query_len,
            )
        if md.num_prefills > 0:
            p = md.prefill
            assert p is not None
            assert p.query_req_id is not None and p.query_abs_pos is not None
            minimax_m3_nvfp4_sparse_prefill(
                q[nd:],
                kv_cache,
                topk[:, nd:num_tokens, :],
                p.block_table,
                p.query_req_id,
                p.query_abs_pos,
                self.num_kv_heads,
                self.scale,
                out[nd:],
                k_scale=k_scale,
                v_scale=v_scale,
            )
        return output
