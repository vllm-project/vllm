# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 lightning indexer over MXFP4 index-K pages on ROCm (gfx950).

AITER's FlyDSL scorers read the e2m1 / e8m0 pages and write the same
``[heads, rows, blocks]`` fp32 block scores the other indexers do; the top-k
is the one ``indexer_nvfp4`` uses. The scorers do not apply the init / local
forced blocks, so ``_pin_forced_blocks`` writes AITER's sentinels between the
two.
"""

from dataclasses import dataclass, fields, replace
from typing import ClassVar

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.amd.indexer_msa import (
    MSA_INDEX_HEAD_DIM,
    MSA_SCORE_TYPE,
    MSA_SPARSE_BLOCK_SIZE,
    MSA_TOPK_BLOCKS,
    MiniMaxM3IndexerMSABackend,
    MiniMaxM3IndexerMSAMetadata,
    MiniMaxM3IndexerMSAMetadataBuilder,
    _score_width,
)
from vllm.models.minimax_m3.amd.ops.mxfp4_pages import PAGE_BYTES, mxfp4_page_views
from vllm.models.minimax_m3.common.indexer import MiniMaxM3IndexerImpl
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec

# AITER's forced-selection scores (see the NVFP4 scorers).
SENTINEL_INIT = 1e30
SENTINEL_LOCAL = 1e29

AITER_REQUIREMENT = (
    "needs an aiter with the MXFP4 FlyDSL ops (ROCm/aiter#7056, #7057, #7060 and #7061)"
)


def rocm_mxfp4_aiter_unavailable_reason() -> str | None:
    """Why the AITER MXFP4 ops cannot be used here, None when they can."""
    if not rocm_aiter_ops.is_enabled():
        return "needs VLLM_ROCM_USE_AITER=1"
    try:
        from aiter.ops import flydsl
        from aiter.ops.flydsl.pa_decode_mxfp4 import pa_decode_mxfp4  # noqa: F401

        for name in (
            "minimax_m3_index_score_fp4_flydsl",
            "minimax_m3_index_score_prefill_fp4_flydsl",
            "minimax_m3_index_score_work_map",
        ):
            getattr(flydsl, name)
    except (ImportError, AttributeError):
        return AITER_REQUIREMENT
    return None


def rocm_mxfp4_indexer_unsupported_reason(
    *,
    topk_blocks: int,
    sparse_block_size: int,
    num_index_heads: int,
    index_head_dim: int,
    score_type: str,
) -> str | None:
    """Why the ROCm MXFP4 indexer cannot serve this config, None when it can."""
    if not current_platform.is_rocm():
        return "needs ROCm"
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return "needs gfx950"
    if (topk_blocks, sparse_block_size, index_head_dim, score_type) != (
        MSA_TOPK_BLOCKS,
        MSA_SPARSE_BLOCK_SIZE,
        MSA_INDEX_HEAD_DIM,
        MSA_SCORE_TYPE,
    ):
        return (
            f"needs topk_blocks={MSA_TOPK_BLOCKS}, block_size="
            f"{MSA_SPARSE_BLOCK_SIZE}, index_head_dim={MSA_INDEX_HEAD_DIM} and "
            f"score_type={MSA_SCORE_TYPE!r}"
        )
    if num_index_heads != 1:
        return (
            "needs one index head per rank (tensor_parallel_size equal to "
            f"the {num_index_heads} index heads of the checkpoint)"
        )
    return rocm_mxfp4_aiter_unavailable_reason()


@triton.jit
def _pin_forced_blocks_kernel(
    score_ptr,  # [H, rows, width] fp32
    seq_lens_ptr,  # decode: [num_reqs]
    num_valid_ptr,  # prefill: [rows] causal block count of each row
    stride_h,
    stride_row,
    query_len,
    init_blocks,
    local_blocks,
    DECODE: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SENTINEL_INIT: tl.constexpr,
    SENTINEL_LOCAL: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    if DECODE:
        seq_len = tl.load(seq_lens_ptr + row // query_len)
        kv_len = tl.maximum(seq_len - query_len + row % query_len + 1, 0)
        num_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    else:
        num_blocks = tl.load(num_valid_ptr + row)
    score_row = score_ptr + head.to(tl.int64) * stride_h + row.to(tl.int64) * stride_row
    j = tl.arange(0, BLOCK)
    local_start = tl.maximum(num_blocks - local_blocks, 0)
    # A block that is both init and local ends up local; the two stores below
    # touch disjoint blocks.
    value = tl.where(j >= local_start, SENTINEL_LOCAL, SENTINEL_INIT)
    tl.store(score_row + j, value, mask=(j < init_blocks) & (j < num_blocks))
    local_block = local_start + j
    tl.store(
        score_row + local_block,
        SENTINEL_LOCAL,
        mask=(j < local_blocks)
        & (local_block < num_blocks)
        & (local_block >= init_blocks),
    )


def _pin_forced_blocks(
    score: torch.Tensor,  # [H, rows, width] fp32
    *,
    init_blocks: int,
    local_blocks: int,
    seq_lens: torch.Tensor | None = None,
    query_len: int = 1,
    num_valid_pages: torch.Tensor | None = None,
) -> None:
    """Write AITER's init / local sentinels on the causal blocks of every row.

    Decode rows are recovered from ``seq_lens`` and ``query_len``, prefill rows
    carry their block counts in ``num_valid_pages``.
    """
    heads, rows, _ = score.shape
    if rows == 0 or init_blocks + local_blocks == 0:
        return
    _pin_forced_blocks_kernel[(rows, heads)](
        score,
        seq_lens if seq_lens is not None else num_valid_pages,
        num_valid_pages if num_valid_pages is not None else seq_lens,
        score.stride(0),
        score.stride(1),
        query_len,
        init_blocks,
        local_blocks,
        DECODE=seq_lens is not None,
        BLOCK=triton.next_power_of_2(max(init_blocks, local_blocks)),
        BLOCK_SIZE=MSA_SPARSE_BLOCK_SIZE,
        SENTINEL_INIT=SENTINEL_INIT,
        SENTINEL_LOCAL=SENTINEL_LOCAL,
    )


@dataclass
class MiniMaxM3IndexerMxfp4Metadata(MiniMaxM3IndexerMSAMetadata):
    """Adds the decode scorer's work map, built once per step."""

    decode_work_map: torch.Tensor | None = None


class MiniMaxM3IndexerMxfp4MetadataBuilder(MiniMaxM3IndexerMSAMetadataBuilder):
    """The decode scorer's work map lives in one persistent buffer.

    A captured decode graph bakes in the work map's address and its row count
    (the scorer's grid). Both therefore have to survive every replay: the map is
    refilled in place each step, and is always built for the model-length block
    bound, which is also the bound the graph is captured with (the model runner
    captures with ``max_seq_len == max_model_len``).
    """

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.num_index_heads = 1
        self.max_model_len = vllm_config.model_config.max_model_len
        self._max_decode_reqs = vllm_config.scheduler_config.max_num_seqs
        self._work_map_buf: torch.Tensor | None = None

    def _work_map_buffer(self) -> torch.Tensor:
        if self._work_map_buf is None:
            from aiter.ops.flydsl import minimax_m3_index_score_work_map_capacity

            max_block = cdiv(self.max_model_len, self.sparse_block_size)
            # The exact grid is not monotonic in the batch or the query length,
            # so size for the capacity of every decode query length.
            rows = max(
                minimax_m3_index_score_work_map_capacity(
                    self._max_decode_reqs, max_block, query_len, self.num_index_heads
                )
                for query_len in range(1, self.max_decode_query_len + 1)
            )
            self._work_map_buf = torch.empty(
                (rows, 2), dtype=torch.int32, device=self.device
            )
        return self._work_map_buf

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3IndexerMxfp4Metadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        work_map = None
        d = metadata.decode
        if d is not None:
            from aiter.ops.flydsl import minimax_m3_index_score_work_map

            # The scorer's block bound is part of its grid: a replayed graph
            # keeps the one it was captured with, so every step uses it.
            d = replace(d, max_seq_len=self.max_model_len)
            metadata.decode = d
            work_map = minimax_m3_index_score_work_map(
                d.seq_lens,
                cdiv(d.max_seq_len, self.sparse_block_size),
                d.decode_query_len,
                self.num_index_heads,
                out=self._work_map_buffer(),
            )
        return MiniMaxM3IndexerMxfp4Metadata(
            **{f.name: getattr(metadata, f.name) for f in fields(metadata)},
            decode_work_map=work_map,
        )


class MiniMaxM3IndexerMxfp4Backend(MiniMaxM3IndexerMSABackend):
    @staticmethod
    def get_builder_cls() -> type[MiniMaxM3IndexerMxfp4MetadataBuilder]:
        return MiniMaxM3IndexerMxfp4MetadataBuilder


def _new_score(
    heads: int, rows: int, max_seq_len: int, device: torch.device
) -> torch.Tensor:
    """Uninitialized: the top-k reads only each row's causal blocks."""
    return torch.empty(
        (heads, rows, _score_width(max_seq_len, MSA_SPARSE_BLOCK_SIZE)),
        dtype=torch.float32,
        device=device,
    )


def mxfp4_index_topk_decode(
    index_query: torch.Tensor,  # [num_reqs * query_len, H, 128] bf16 or e4m3
    index_pages: torch.Tensor,  # [pages, 8704] uint8
    block_table: torch.Tensor,  # [num_reqs, >= max_blocks] int32
    seq_lens: torch.Tensor,  # [num_reqs] int32
    work_map: torch.Tensor | None,
    topk_out: torch.Tensor,  # [H, num_reqs * query_len, topk] int32
    *,
    max_seq_len: int,
    query_len: int,
    scale: float,
    init_blocks: int,
    local_blocks: int,
) -> None:
    """Top-k block ids of ``query_len`` decode rows per request."""
    from aiter.ops.flydsl import minimax_m3_index_score_fp4_flydsl

    from .ops.index_topk_decode.host import minimax_m3_index_topk_decode_indices

    heads, rows = index_query.shape[1], index_query.shape[0]
    max_block = cdiv(max_seq_len, MSA_SPARSE_BLOCK_SIZE)
    k_data, k_scale = mxfp4_page_views(index_pages)
    score = _new_score(heads, rows, max_seq_len, index_query.device)
    minimax_m3_index_score_fp4_flydsl(
        index_query,
        k_data,
        k_scale,
        block_table,
        seq_lens,
        query_len,
        heads,
        scale,
        max_block,
        out=score[:, :, :max_block],
        work_map=work_map,
    )
    _pin_forced_blocks(
        score,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        seq_lens=seq_lens,
        query_len=query_len,
    )
    minimax_m3_index_topk_decode_indices(
        score, topk_out, seq_lens, max_seq_len=max_seq_len, query_len=query_len
    )


def mxfp4_index_topk_prefill(
    index_query: torch.Tensor,  # [total_q, H, 128] bf16 or e4m3
    index_pages: torch.Tensor,  # [pages, 8704] uint8
    block_table: torch.Tensor,  # [num_reqs, >= max_blocks] int32
    cu_seqlens_q: torch.Tensor,  # [num_reqs + 1] int32
    seq_lens: torch.Tensor,  # [num_reqs] int32
    context_lens: torch.Tensor,  # [num_reqs] int32, tokens before this chunk
    num_valid_pages: torch.Tensor,  # [total_q] int32, causal blocks of each row
    topk_out: torch.Tensor,  # [H, total_q, topk] int32
    *,
    max_query_len: int,
    max_seq_len: int,
    scale: float,
    init_blocks: int,
    local_blocks: int,
) -> None:
    """Top-k block ids of ragged prefill rows."""
    from aiter.ops.flydsl import minimax_m3_index_score_prefill_fp4_flydsl

    from .ops.index_topk_decode.host import minimax_m3_index_topk_prefill_indices

    heads, rows = index_query.shape[1], index_query.shape[0]
    k_data, k_scale = mxfp4_page_views(index_pages)
    score = _new_score(heads, rows, max_seq_len, index_query.device)
    minimax_m3_index_score_prefill_fp4_flydsl(
        index_query,
        k_data,
        k_scale,
        block_table,
        cu_seqlens_q,
        seq_lens,
        context_lens,
        max_query_len,
        max_seq_len,
        scale,
        out=score[:, :, : cdiv(max_seq_len, MSA_SPARSE_BLOCK_SIZE)],
    )
    _pin_forced_blocks(
        score,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        num_valid_pages=num_valid_pages,
    )
    minimax_m3_index_topk_prefill_indices(
        score, topk_out, max_seq_len=max_seq_len, num_valid_pages=num_valid_pages
    )


class MiniMaxM3IndexerRocmMxfp4Impl(MiniMaxM3IndexerImpl):
    """AITER FlyDSL MXFP4 index scores + top-k block ids, prefill and decode."""

    indexer_backend_cls: ClassVar[type[AttentionBackend]] = MiniMaxM3IndexerMxfp4Backend

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.indexer_kv_dtype != "mxfp4":
            raise ValueError(
                f"{type(self).__name__} requires indexer_kv_dtype='mxfp4', got "
                f"{self.indexer_kv_dtype!r}"
            )
        reason = rocm_mxfp4_indexer_unsupported_reason(
            topk_blocks=self.topk_blocks,
            sparse_block_size=self.block_size,
            num_index_heads=self.num_index_heads,
            index_head_dim=self.index_head_dim,
            score_type=self.score_type,
        )
        if reason is not None:
            raise ValueError(f"MiniMax M3 MXFP4 indexer on ROCm {reason}")

    def forward(
        self,
        index_query: torch.Tensor,
        **_aiter_table_args,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None  # profiling run; caches unbound
        md = attn_metadata[self.index_cache.prefix]
        assert isinstance(md, MiniMaxM3IndexerMxfp4Metadata)
        num_tokens = md.num_actual_tokens
        nd = md.num_decode_tokens
        iq = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        pages = self.index_cache.kv_cache
        pages = pages.view(pages.shape[0], PAGE_BYTES)
        buf = self.topk_indices_buffer
        if buf is None:
            buf = torch.empty(
                (self.num_index_heads, num_tokens, self.topk_blocks),
                dtype=torch.int32,
                device=iq.device,
            )

        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None
        if md.num_decodes > 0:
            d = md.decode
            assert d is not None
            decode_topk = buf[:, :nd, :]
            mxfp4_index_topk_decode(
                iq[:nd],
                pages,
                d.block_table,
                d.seq_lens,
                md.decode_work_map,
                decode_topk,
                max_seq_len=d.max_seq_len,
                query_len=d.decode_query_len,
                scale=self.scale,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
            )

        if md.num_prefills > 0:
            p = md.prefill
            assert p is not None and md.prefill_num_valid_pages is not None
            prefill_topk = buf[:, nd:num_tokens, :]
            mxfp4_index_topk_prefill(
                iq[nd:],
                pages,
                p.block_table,
                p.cu_seqlens_q,
                p.seq_lens,
                p.context_lens,
                md.prefill_num_valid_pages,
                prefill_topk,
                max_query_len=p.max_query_len,
                max_seq_len=p.max_seq_len,
                scale=self.scale,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
            )
        return decode_topk, prefill_topk
