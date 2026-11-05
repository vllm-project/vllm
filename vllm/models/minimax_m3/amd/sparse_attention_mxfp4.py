# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 block-sparse attention over an MXFP4 K/V cache on ROCm (gfx950).

Each head's K and V are two slots of one 8704 B page (``mxfp4_pages``). The
top-k block ids the indexer wrote are turned into one page table row per query
token, and ``pa_decode_mxfp4`` attends each row as a length-1 sequence with the
query as e4m3.
"""

from dataclasses import replace
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.amd.indexer_mxfp4 import rocm_mxfp4_aiter_unavailable_reason
from vllm.models.minimax_m3.amd.ops.mxfp4_pages import (
    HEAD_DIM,
    PAGE_BYTES,
    PAGE_TOKENS,
    TOKEN_BYTES,
    mxfp4_page_views,
)
from vllm.models.minimax_m3.amd.ops.sparse_pa import (
    minimax_m3_build_sparse_block_table_decode,
    minimax_m3_build_sparse_block_table_prefill,
)
from vllm.models.minimax_m3.amd.sparse_attention_nvfp4 import (
    MiniMaxM3SparseRocmNvfp4MetadataBuilder,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

# Query heads per KV head one pa_decode_mxfp4 tile serves.
MAX_GROUP_SIZE = 16


class MiniMaxM3SparseRocmMxfp4Backend(MiniMaxM3SparseBackend):
    """MiniMax M3 sparse attend over MXFP4 K/V pages (ROCm, gfx950)."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["mxfp4"]

    @staticmethod
    def get_builder_cls() -> type[MiniMaxM3SparseRocmNvfp4MetadataBuilder]:
        # Adds each prefill token's request and absolute position, which the
        # sparse block-table builder reads.
        return MiniMaxM3SparseRocmNvfp4MetadataBuilder

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """Store each head's K and V as two slots of 128 tokens x 68 B."""
        if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_mxfp4:
            return spec
        return replace(
            spec,
            num_head_slots=2 * spec.num_kv_heads,
            state_content_bytes=TOKEN_BYTES,
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...] | None:
        return (KVCacheLayout.LBHNC,)


def rocm_mxfp4_attention_unsupported_reason(
    num_heads: int, num_kv_heads: int, head_dim: int, block_size: int
) -> str | None:
    """Why the ROCm MXFP4 attend cannot serve this layer, None when it can."""
    if not current_platform.is_rocm():
        return "needs ROCm"
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return "needs gfx950"
    if (
        head_dim != HEAD_DIM
        or block_size != PAGE_TOKENS
        or num_kv_heads != 1
        or num_heads % num_kv_heads
        or num_heads // num_kv_heads > MAX_GROUP_SIZE
    ):
        return (
            f"needs head_dim {HEAD_DIM}, block_size {PAGE_TOKENS}, one KV head "
            f"and at most {MAX_GROUP_SIZE} query heads per rank (got "
            f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
            f"head_dim={head_dim}, block_size={block_size})"
        )
    return rocm_mxfp4_aiter_unavailable_reason()


def mxfp4_sparse_attend(
    q: torch.Tensor,  # [rows, H, 128] e4m3
    kv_cache: torch.Tensor,  # [pages, 2, 128, 68] uint8, one KV head
    sparse_bt: torch.Tensor,  # [rows, topk] int32 page ids
    sparse_ctx: torch.Tensor,  # [rows] int32 tokens to attend
    out: torch.Tensor,  # [rows, H, 128] bf16
    scale: float,
) -> None:
    """Attend each row to its selected pages as a length-1 sequence."""
    from aiter.ops.flydsl.pa_decode_mxfp4 import pa_decode_mxfp4

    pages = kv_cache.view(kv_cache.shape[0], 2, PAGE_BYTES)
    k_data, k_scale = mxfp4_page_views(pages[:, :1])
    v_data, v_scale = mxfp4_page_views(pages[:, 1:])
    pa_decode_mxfp4(
        out,
        q,
        k_data,
        k_scale,
        v_data,
        v_scale,
        sparse_ctx,
        sparse_bt,
        scale,
        max_partitions=1,
        max_context_length=sparse_bt.shape[1] * PAGE_TOKENS,
    )


class MiniMaxM3SparseRocmMxfp4Impl(MiniMaxM3SparseImpl):
    """``pa_decode_mxfp4`` over the pages each query token's top-k selects."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        reason = rocm_mxfp4_attention_unsupported_reason(
            self.num_heads, self.num_kv_heads, self.head_size, self.block_size
        )
        if reason is not None:
            raise ValueError(f"MiniMax M3 MXFP4 KV cache on ROCm {reason}")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output  # profiling run; caches unbound
        md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(md, MiniMaxM3SparseMetadata)

        nd = md.num_decode_tokens
        num_tokens = md.num_actual_tokens
        topk = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk is not None
        assert query.dtype == torch.float8_e4m3fn
        q = query[:num_tokens].view(-1, self.num_heads, self.head_size)
        out = output[:num_tokens].view(-1, self.num_heads, self.head_size)

        if md.num_decodes > 0:
            d = md.decode
            assert d is not None
            sparse_bt, sparse_ctx = minimax_m3_build_sparse_block_table_decode(
                topk[:, :nd, :],
                d.block_table,
                d.seq_lens,
                d.decode_query_len,
                block_page_stride=1,
                pages_per_block=1,
            )
            mxfp4_sparse_attend(
                q[:nd], kv_cache, sparse_bt, sparse_ctx, out[:nd], self.scale
            )
        if md.num_prefills > 0:
            p = md.prefill
            assert p is not None
            assert p.query_req_id is not None and p.query_abs_pos is not None
            sparse_bt, sparse_ctx = minimax_m3_build_sparse_block_table_prefill(
                topk[:, nd:num_tokens, :],
                p.block_table,
                p.query_req_id,
                p.query_abs_pos,
                block_page_stride=1,
                pages_per_block=1,
            )
            mxfp4_sparse_attend(
                q[nd:], kv_cache, sparse_bt, sparse_ctx, out[nd:], self.scale
            )
        return output
