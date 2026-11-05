# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 block-sparse attention over an MXFP4 K/V cache on ROCm.

Each head's K and V are two slots of one 8704 B page (``mxfp4_pages``). The
top-k block ids the indexer wrote are turned into one page table row per query
token (and KV head), and each row is attended as a length-1 sequence.

gfx950 runs AITER's FlyDSL ``pa_decode_mxfp4`` with the query as e4m3. gfx942
has no FP4 converts or MX-scaled MFMA, so it runs the Triton attend in
``ops/mxfp4_attn_gfx942``, which decodes the pages exactly in registers and
keeps the query in the activation dtype; its page table is the page-16 one the
gfx942 MXFP4 indexer's top-k emits, read in place.
"""

import functools
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
    ASM_PAGE_SIZE,
    minimax_m3_build_sparse_block_table_decode,
    minimax_m3_build_sparse_block_table_prefill,
)
from vllm.models.minimax_m3.amd.sparse_attention_msa import (
    MiniMaxM3SparseAiterPADecodeMetadata,
    MiniMaxM3SparseAiterPAPrefillMetadata,
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
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

# Query heads per KV head one pa_decode_mxfp4 tile serves.
MAX_GROUP_SIZE = 16
# Page-16 ids per 128-token block in the table the gfx942 indexer emits.
PAGE16_PER_BLOCK = PAGE_TOKENS // ASM_PAGE_SIZE


class MiniMaxM3SparseRocmMxfp4MetadataBuilder(MiniMaxM3SparseRocmNvfp4MetadataBuilder):
    """Also hands the gfx942 MXFP4 indexer the block table its top-k resolves
    the selection through.

    The indexer emits ``(block * 8 + j) * H + h`` for each selected block, so
    it is given the plain block table (one K/V page set per block), and the
    attend recovers ``block`` as ``entry // (8 * H)``.
    """

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3SparseMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if _mxfp4_attention_arch() != "gfx942":
            return metadata
        block_table = common_attn_metadata.block_table_tensor
        nd = metadata.num_decodes
        if metadata.decode is not None:
            metadata.decode = MiniMaxM3SparseAiterPADecodeMetadata(
                **vars(metadata.decode), page16_block_table=block_table[:nd]
            )
        if metadata.prefill is not None:
            metadata.prefill = MiniMaxM3SparseAiterPAPrefillMetadata(
                **vars(metadata.prefill), page16_block_table=block_table[nd:]
            )
        return metadata


class MiniMaxM3SparseRocmMxfp4Backend(MiniMaxM3SparseBackend):
    """MiniMax M3 sparse attend over MXFP4 K/V pages (ROCm gfx942 / gfx950)."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["mxfp4"]

    @staticmethod
    def get_builder_cls() -> type[MiniMaxM3SparseRocmMxfp4MetadataBuilder]:
        # Adds each prefill token's request and absolute position, which the
        # sparse block-table builder reads, and the gfx942 indexer's table.
        return MiniMaxM3SparseRocmMxfp4MetadataBuilder

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


def _mxfp4_attention_arch() -> str | None:
    from vllm.platforms.rocm import on_gfx942, on_gfx950

    if on_gfx950():
        return "gfx950"
    if on_gfx942():
        return "gfx942"
    return None


def rocm_mxfp4_attention_unsupported_reason(
    num_heads: int, num_kv_heads: int, head_dim: int, block_size: int
) -> str | None:
    """Why the ROCm MXFP4 attend cannot serve this layer, None when it can."""
    if not current_platform.is_rocm():
        return "needs ROCm"
    arch = _mxfp4_attention_arch()
    if arch is None:
        return "needs gfx942 or gfx950"
    # gfx942 reads the per-(token, KV head) table the indexer's top-k emits,
    # so it serves more than one KV head per rank.
    max_kv_heads = 2 if arch == "gfx942" else 1
    if (
        head_dim != HEAD_DIM
        or block_size != PAGE_TOKENS
        or num_kv_heads > max_kv_heads
        or num_heads % num_kv_heads
        or num_heads // num_kv_heads > MAX_GROUP_SIZE
    ):
        return (
            f"needs head_dim {HEAD_DIM}, block_size {PAGE_TOKENS}, at most "
            f"{max_kv_heads} KV heads and {MAX_GROUP_SIZE} query heads per KV "
            f"head per rank on {arch} (got num_heads={num_heads}, "
            f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, "
            f"block_size={block_size})"
        )
    return rocm_mxfp4_aiter_unavailable_reason(arch)


def mxfp4_sparse_attend(
    q: torch.Tensor,  # [rows, H, 128] e4m3
    kv_cache: torch.Tensor,  # [pages, 2, 128, 68] uint8, one KV head
    sparse_bt: torch.Tensor,  # [rows, topk] int32 page ids
    sparse_ctx: torch.Tensor,  # [rows] int32 tokens to attend
    out: torch.Tensor,  # [rows, H, 128] bf16
    scale: float,
) -> None:
    """Attend each row to its selected pages as a length-1 sequence (gfx950)."""
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


@functools.cache
def _aiter_mxfp4_decode():
    """AITER's gluon MXFP4 sparse paged-attention decode, or None when this
    AITER lacks it (the Triton attend is used then)."""
    try:
        from aiter.ops.triton.gluon.pa_decode_mxfp4_gluon import (
            pa_decode_mxfp4_gluon,
        )
    except ImportError:
        return None
    return pa_decode_mxfp4_gluon


class MiniMaxM3SparseRocmMxfp4Impl(MiniMaxM3SparseImpl):
    """MXFP4 sparse attend over the pages each query token's top-k selects."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        reason = rocm_mxfp4_attention_unsupported_reason(
            self.num_heads, self.num_kv_heads, self.head_size, self.block_size
        )
        if reason is not None:
            raise ValueError(f"MiniMax M3 MXFP4 KV cache on ROCm {reason}")
        self.is_gfx942 = _mxfp4_attention_arch() == "gfx942"
        self.prefill_scratch = None
        if self.is_gfx942:
            from vllm.config import get_current_vllm_config
            from vllm.models.minimax_m3.amd.ops.mxfp4_attn_gfx942 import (
                Mxfp4PrefillScratch,
            )

            # One scratch per device, made here so the memory profile charges
            # it: room for two max-length prefill requests per step.
            max_len = get_current_vllm_config().model_config.max_model_len
            self.prefill_scratch = Mxfp4PrefillScratch.get(
                2 * cdiv(max_len, PAGE_TOKENS),
                self.num_kv_heads,
                torch.device("cuda", torch.cuda.current_device()),
            )

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
        if self.is_gfx942:
            return self._forward_gfx942(layer, query, kv_cache, output, md)

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

    def _forward_gfx942(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        md: MiniMaxM3SparseMetadata,
    ) -> torch.Tensor:
        """Attend through the page-16 table the gfx942 indexer emitted.

        Row ``t * H + h`` of the layer's ``sparse_bt_buffer`` holds, for query
        token ``t`` and KV head ``h``, ``(block * 8 + j) * H + h`` for the
        selected blocks, compacted with the causal tail last, and
        ``sparse_ctx_buffer`` the tokens it attends; entry ``8 * i`` names
        block ``entry // (8 * H)``, which is how the attend reads it.
        """
        from vllm.models.minimax_m3.amd.ops.mxfp4_attn_gfx942 import (
            mxfp4_sparse_attend_gfx942,
            mxfp4_sparse_attend_prefill_fp8_gfx942,
        )

        num_tokens = md.num_actual_tokens
        kvh = self.num_kv_heads
        sparse_bt = layer.sparse_bt_buffer  # type: ignore[attr-defined]
        sparse_ctx = layer.sparse_ctx_buffer  # type: ignore[attr-defined]
        assert sparse_bt is not None and sparse_ctx is not None, (
            "the gfx942 MXFP4 attend reads the page table its indexer emits"
        )
        q = query[:num_tokens].view(-1, self.num_heads, self.head_size)
        out = output[:num_tokens].view(-1, self.num_heads, self.head_size)
        common = dict(
            num_kv_heads=kvh,
            max_blocks=self.topk_blocks,
            bt_col_stride=PAGE16_PER_BLOCK,
            page_div=PAGE16_PER_BLOCK * kvh,
        )
        nd = md.num_decode_tokens
        p = md.prefill
        if p is not None and self._transcode_prefill_once(md):
            # Many prefill rows select the same blocks: transcode every block
            # of the prefill requests once to fp8 page-16 pages with per-token
            # power-of-two scales and run AITER's sparse paged attention in
            # fp8, the kernel the fp8 K/V cache uses.
            assert self.prefill_scratch is not None
            if mxfp4_sparse_attend_prefill_fp8_gfx942(
                q[nd:], kv_cache, sparse_bt[nd * kvh : num_tokens * kvh],
                sparse_ctx[nd * kvh : num_tokens * kvh], out[nd:], self.scale,
                num_kv_heads=kvh, block_table=p.block_table, seq_lens=p.seq_lens,
                max_seq_len=p.max_seq_len, scratch=self.prefill_scratch,
            ):  # fmt: skip
                num_tokens = nd
        if num_tokens > 0:
            pa_decode_mxfp4 = _aiter_mxfp4_decode()
            if pa_decode_mxfp4 is not None and q.dtype == torch.bfloat16:
                # AITER's gluon MXFP4 decode: e2m1 decoded to e4m3fnuz in
                # registers and fp8 MFMAs, per-token power-of-two scales.
                pa_decode_mxfp4(
                    out[:num_tokens], q[:num_tokens], kv_cache,
                    sparse_ctx[: num_tokens * kvh], sparse_bt[: num_tokens * kvh],
                    self.scale, kvh,
                )  # fmt: skip
            else:
                mxfp4_sparse_attend_gfx942(
                    q[:num_tokens], kv_cache, sparse_bt[: num_tokens * kvh],
                    sparse_ctx[: num_tokens * kvh], out[:num_tokens], self.scale,
                    **common,
                )  # fmt: skip
        return output

    def _transcode_prefill_once(self, md: MiniMaxM3SparseMetadata) -> bool:
        """Whether the prefill rows select more blocks than their requests
        hold, so that transcoding each block once is the cheaper way round."""
        p = md.prefill
        assert p is not None
        blocks = p.block_table.shape[0] * cdiv(p.max_seq_len, PAGE_TOKENS)
        return md.num_prefill_tokens * self.topk_blocks > 2 * blocks
