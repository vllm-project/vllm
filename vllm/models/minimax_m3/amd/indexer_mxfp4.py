# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 lightning indexer over MXFP4 index-K pages on ROCm.

gfx950 uses AITER's MiniMax block scorer. gfx942 (index cache only; main K/V
keeps its fp8 cache) uses AITER's FP4 MQA scorers from ROCm/aiter#7017 with
signed scores, reading the paged index cache in place: decode scores token
logits and reduces them to 128-token block maxima, prefill packs 16 query rows
per MFMA tile and reduces to page maxima in-kernel. Both then share the
forced init/local pinning and AITER's sparse block top-k, which also emits the
attend's page table.
"""

import inspect
from dataclasses import dataclass, fields, replace
from typing import ClassVar

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.amd.indexer_msa import (
    MAX_SUPPORTED_BLOCKS,
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
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerDecodeMetadata,
    MiniMaxM3IndexerImpl,
)
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
AITER_GFX942_REQUIREMENT = (
    "needs an aiter containing ROCm/aiter#7017 with signed (relu=False), "
    "paged in-place (kv_page_stride) scorers and the row-packed prefill scorer"
)
GFX942_SCORERS = (
    "flydsl_pa_mqa_logits_fp4_gfx942",
    "flydsl_pa_mqa_logits_fp4_gfx942_longctx",
)
GFX942_ROWPACK_ARGS = ("kv_page_stride", "kv_scale_page_stride", "kv_scale_offset")
GFX942_DECODE_ARGS = ("relu", *GFX942_ROWPACK_ARGS)
# Decode bounds above this use AITER's split-context scorer.
GFX942_LONGCTX_THRESHOLD = 65536


def _mxfp4_arch() -> str | None:
    from vllm.platforms.rocm import on_gfx942, on_gfx950

    if on_gfx950():
        return "gfx950"
    if on_gfx942():
        return "gfx942"
    return None


def rocm_mxfp4_aiter_unavailable_reason(arch: str | None = None) -> str | None:
    """Why the AITER MXFP4 ops cannot be used here, None when they can."""
    if not rocm_aiter_ops.is_enabled():
        return "needs VLLM_ROCM_USE_AITER=1"
    arch = arch or _mxfp4_arch()
    try:
        from aiter.ops import flydsl

        if arch == "gfx942":
            from aiter.ops.flydsl.kernels.mqa_logits import (
                pa_mqa_logits_fp4_prefill_gfx942 as prefill,
            )
            from aiter.ops.flydsl.kernels.mqa_logits import (
                pa_mqa_page_max_fp4_rowpack_prefill_gfx942 as rowpack,
            )
            from aiter.ops.msa_attention import pa_sparse_block_topk  # noqa: F401

            required = [
                (getattr(flydsl, name), GFX942_DECODE_ARGS) for name in GFX942_SCORERS
            ]
            required.append(
                (
                    rowpack.flydsl_pa_mqa_page_max_fp4_rowpack_prefill_gfx942,
                    GFX942_ROWPACK_ARGS,
                )
            )
            for fn in (
                flydsl.recommended_gfx942_fp4_longctx_config,
                prefill.compute_varqlen_windows,
                rowpack.rowpack_prefill_inputs_gfx942,
            ):
                required.append((fn, ()))
            for fn, args in required:
                params = inspect.signature(fn).parameters
                if not all(arg in params for arg in args):
                    return AITER_GFX942_REQUIREMENT
        else:
            from aiter.ops.flydsl.pa_decode_mxfp4 import (  # noqa: F401
                pa_decode_mxfp4,
            )

            for name in (
                "minimax_m3_index_score_fp4_flydsl",
                "minimax_m3_index_score_prefill_fp4_flydsl",
                "minimax_m3_index_score_work_map",
            ):
                getattr(flydsl, name)
    except (ImportError, AttributeError):
        return AITER_GFX942_REQUIREMENT if arch == "gfx942" else AITER_REQUIREMENT
    return None


def rocm_mxfp4_indexer_unsupported_reason(
    *,
    topk_blocks: int,
    sparse_block_size: int,
    num_index_heads: int,
    index_head_dim: int,
    score_type: str,
    max_model_len: int,
) -> str | None:
    """Why the ROCm MXFP4 indexer cannot serve this config, None when it can."""
    if not current_platform.is_rocm():
        return "needs ROCm"
    arch = _mxfp4_arch()
    if arch is None:
        return "needs gfx942 or gfx950"
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
    max_blocks = cdiv(max_model_len, sparse_block_size)
    if max_blocks > MAX_SUPPORTED_BLOCKS:
        return (
            f"max_model_len={max_model_len} needs {max_blocks} blocks per row, "
            f"more than the {MAX_SUPPORTED_BLOCKS} pa_sparse_block_topk supports"
        )
    expected_heads = (1, 2) if arch == "gfx942" else (1,)
    if num_index_heads not in expected_heads:
        return (
            f"needs num_index_heads in {expected_heads} on {arch}, "
            f"got num_index_heads={num_index_heads}"
        )
    return rocm_mxfp4_aiter_unavailable_reason(arch)


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
class Gfx942DecodePlan:
    """Per-step inputs shared by every gfx942 index layer, in persistent
    buffers so a captured decode replays from stable addresses."""

    page64_table: torch.Tensor
    scorer_kwargs: dict


@dataclass
class MiniMaxM3IndexerMxfp4Metadata(MiniMaxM3IndexerMSAMetadata):
    """Adds the decode scorer's work map (gfx950) or decode plan (gfx942)."""

    decode_work_map: torch.Tensor | None = None
    gfx942_decode_plan: Gfx942DecodePlan | None = None


class MiniMaxM3IndexerMxfp4MetadataBuilder(MiniMaxM3IndexerMSAMetadataBuilder):
    """The decode work map (gfx950) or plan (gfx942) lives in persistent buffers.

    A captured decode graph bakes in their addresses and the scorer's grid, so
    they are refilled in place each step and always sized for the model-length
    block bound the graph is captured with.
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
        self.is_gfx942 = _mxfp4_arch() == "gfx942"
        self._page64_table: torch.Tensor | None = None
        self._cta_info: dict[tuple[int, int], tuple[dict, torch.Tensor]] = {}

    def _gfx942_decode_plan(
        self, d: MiniMaxM3IndexerDecodeMetadata
    ) -> Gfx942DecodePlan:
        from .ops.mxfp4_gfx942 import (
            KERNEL_PAGES_PER_BLOCK,
            expand_block_table_page64_into,
        )

        batch = d.seq_lens.shape[0]
        width = cdiv(self.max_model_len, self.sparse_block_size)
        if self._page64_table is None:
            self._page64_table = torch.zeros(
                (self._max_decode_reqs, KERNEL_PAGES_PER_BLOCK * width),
                dtype=torch.int32,
                device=self.device,
            )
        table = expand_block_table_page64_into(
            d.block_table[:, :width], self._page64_table
        )
        if d.max_seq_len <= GFX942_LONGCTX_THRESHOLD:
            return Gfx942DecodePlan(table, {})
        from aiter.ops.flydsl import recommended_gfx942_fp4_longctx_config
        from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_common import (
            compute_varctx_schedule,
        )

        key = (batch, d.decode_query_len)
        if key not in self._cta_info:
            cfg = recommended_gfx942_fp4_longctx_config(
                d.max_seq_len, batch_size=batch, next_n=d.decode_query_len
            )
            self._cta_info[key] = (
                cfg,
                torch.empty(
                    (cfg["parallel_unit_num"], 4), dtype=torch.int32, device=self.device
                ),
            )
        cfg, cta_info = self._cta_info[key]
        _, _, total = compute_varctx_schedule(
            d.seq_lens,
            cfg["block_k"],
            cfg["parallel_unit_num"],
            d.max_seq_len,
            next_n=d.decode_query_len,
            cta_info_out=cta_info,
        )
        return Gfx942DecodePlan(table, dict(cfg, cta_info=cta_info, total_ctas=total))

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
        work_map = plan = None
        d = metadata.decode
        if d is not None:
            # Both scorers compile the output width into their captured launch.
            d = replace(d, max_seq_len=self.max_model_len)
            metadata.decode = d
            if self.is_gfx942:
                plan = self._gfx942_decode_plan(d)
            else:
                from aiter.ops.flydsl import minimax_m3_index_score_work_map

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
            gfx942_decode_plan=plan,
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


def mxfp4_gfx942_index_scores_decode(
    index_query: torch.Tensor,
    index_pages: torch.Tensor,
    seq_lens: torch.Tensor,
    plan: Gfx942DecodePlan,
    *,
    max_seq_len: int,
    query_len: int,
    scale: float,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Pinned signed block scores of the decode rows."""
    from aiter.ops.flydsl import (
        flydsl_pa_mqa_logits_fp4_gfx942,
        flydsl_pa_mqa_logits_fp4_gfx942_longctx,
    )

    from .ops.mxfp4_gfx942 import (
        decode_block_scores_gfx942,
        gfx942_mxfp4_cache_strides,
        quantize_q_gfx942,
    )

    long_ctx = bool(plan.scorer_kwargs)
    scorer = (
        flydsl_pa_mqa_logits_fp4_gfx942_longctx
        if long_ctx
        else flydsl_pa_mqa_logits_fp4_gfx942
    )
    batch, (rows, heads) = seq_lens.shape[0], index_query.shape[:2]
    k, k_scale, cache_kw = gfx942_mxfp4_cache_strides(index_pages)
    score = _new_score(heads, rows, max_seq_len, index_query.device)
    for head in range(heads):
        q, q_scale, weights = quantize_q_gfx942(index_query, head)
        long_kwargs = {}
        if long_ctx:
            # A caller-built schedule leaves the -inf fill to the caller; the
            # host-side Q-scale check cannot run inside a captured decode.
            long_kwargs = dict(
                plan.scorer_kwargs,
                validate_q_scale=False,
                out=torch.full(
                    (rows, max_seq_len),
                    float("-inf"),
                    dtype=torch.float32,
                    device=index_query.device,
                ),
            )
        logits = scorer(
            q.view(batch, query_len, *q.shape[1:]),
            q_scale.view(batch, query_len, *q_scale.shape[1:]),
            k,
            k_scale,
            plan.page64_table,
            weights,
            seq_lens,
            max_seq_len,
            weight_scale=scale,
            next_n=query_len,
            kv_block_size=64,
            relu=False,
            **cache_kw,
            **long_kwargs,
        )
        decode_block_scores_gfx942(
            logits,
            seq_lens,
            score[head],
            query_len=query_len,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            sentinel_init=SENTINEL_INIT,
            sentinel_local=SENTINEL_LOCAL,
        )
    return score


def mxfp4_gfx942_index_scores_prefill(
    index_query,
    index_pages,
    block_table,
    cu_seqlens_q,
    seq_lens,
    num_valid_pages,
    *,
    max_seq_len,
    scale,
    init_blocks,
    local_blocks,
) -> torch.Tensor:
    """Signed 128-token block maxima of every prefill row.

    AITER packs 16 rows of a sequence into one MFMA tile and reduces to 64-token
    page maxima in-kernel, so no ``[rows, max_seq_len]`` logits are formed.
    """
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill_gfx942 import (
        compute_varqlen_windows,
    )
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_page_max_fp4_rowpack_prefill_gfx942 import (  # noqa: E501
        flydsl_pa_mqa_page_max_fp4_rowpack_prefill_gfx942,
        rowpack_prefill_inputs_gfx942,
    )
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from .ops.mxfp4_gfx942 import (
        expand_block_table_page64,
        gfx942_mxfp4_cache_strides,
    )

    rows, heads = index_query.shape[:2]
    k, k_scale, cache_kw = gfx942_mxfp4_cache_strides(index_pages)
    table = expand_block_table_page64(block_table)
    windows = compute_varqlen_windows(cu_seqlens_q, seq_lens, rows)
    width = _score_width(max_seq_len, MSA_SPARSE_BLOCK_SIZE)
    score = torch.empty(
        (heads, rows, width), dtype=torch.float32, device=index_query.device
    )
    page_max = torch.empty(
        (rows, 2 * width), dtype=torch.float32, device=index_query.device
    )
    for head in range(heads):
        packed, scales = dynamic_mxfp4_quant(index_query[:, head].contiguous())
        inputs = rowpack_prefill_inputs_gfx942(
            packed, scales.view(torch.uint8).view(rows, -1), cu_seqlens_q, *windows
        )
        page_max.fill_(float("-inf"))
        flydsl_pa_mqa_page_max_fp4_rowpack_prefill_gfx942(
            *inputs[:2],
            k,
            k_scale,
            table,
            *inputs[2:],
            max_seq_len,
            page_max,
            weight_scale=scale,
            **cache_kw,
        )
        score[head] = page_max.view(rows, width, 2).amax(-1)
    _pin_forced_blocks(
        score,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        num_valid_pages=num_valid_pages,
    )
    return score


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
            max_model_len=get_current_vllm_config().model_config.max_model_len,
        )
        if reason is not None:
            raise ValueError(f"MiniMax M3 MXFP4 indexer on ROCm {reason}")
        self.sparse_bt_buffer: torch.Tensor | None = None
        self.sparse_ctx_buffer: torch.Tensor | None = None

    @property
    def pages_per_block(self) -> int:
        from .ops.sparse_pa import ASM_PAGE_SIZE

        return self.block_size // ASM_PAGE_SIZE

    def _table_rows(self, lo: int, hi: int) -> tuple[torch.Tensor, torch.Tensor]:
        rows = (hi - lo) * self.num_kv_heads
        if self.sparse_bt_buffer is None or self.sparse_ctx_buffer is None:
            device = self.index_cache.kv_cache.device
            return (
                torch.empty(
                    (rows, self.topk_blocks * self.pages_per_block),
                    dtype=torch.int32,
                    device=device,
                ),
                torch.empty(rows, dtype=torch.int32, device=device),
            )
        return (
            self.sparse_bt_buffer[lo * self.num_kv_heads : hi * self.num_kv_heads],
            self.sparse_ctx_buffer[lo * self.num_kv_heads : hi * self.num_kv_heads],
        )

    def forward(
        self,
        index_query: torch.Tensor,
        **aiter_table_args,
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
        is_gfx942 = _mxfp4_arch() == "gfx942"
        if is_gfx942:
            from aiter.ops.msa_attention import pa_sparse_block_topk

        if md.num_decodes > 0:
            d = md.decode
            assert d is not None
            decode_topk = buf[:, :nd, :]
            if is_gfx942:
                page16 = aiter_table_args.get("decode_page16_block_table")
                assert page16 is not None
                assert md.gfx942_decode_plan is not None
                score = mxfp4_gfx942_index_scores_decode(
                    iq[:nd],
                    pages,
                    d.seq_lens,
                    md.gfx942_decode_plan,
                    max_seq_len=d.max_seq_len,
                    query_len=d.decode_query_len,
                    scale=self.scale,
                    init_blocks=self.init_blocks,
                    local_blocks=self.local_blocks,
                )
                sparse_bt, sparse_ctx = self._table_rows(0, nd)
                pa_sparse_block_topk(
                    score,
                    decode_topk,
                    page16,
                    d.seq_lens,
                    sparse_bt=sparse_bt,
                    sparse_ctx=sparse_ctx,
                    max_seq_len=d.max_seq_len,
                    block_size=self.block_size,
                    query_len=d.decode_query_len,
                    num_kv_heads=self.num_kv_heads,
                    pages_per_block=self.pages_per_block,
                )
            else:
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
            if is_gfx942:
                page16 = aiter_table_args.get("prefill_page16_block_table")
                assert page16 is not None
                assert md.prefill_row_req_id is not None
                assert md.prefill_kv_lens is not None
                score = mxfp4_gfx942_index_scores_prefill(
                    iq[nd:],
                    pages,
                    p.block_table,
                    p.cu_seqlens_q,
                    p.seq_lens,
                    md.prefill_num_valid_pages,
                    max_seq_len=p.max_seq_len,
                    scale=self.scale,
                    init_blocks=self.init_blocks,
                    local_blocks=self.local_blocks,
                )
                sparse_bt, sparse_ctx = self._table_rows(nd, num_tokens)
                pa_sparse_block_topk(
                    score,
                    prefill_topk,
                    page16,
                    p.seq_lens,
                    sparse_bt=sparse_bt,
                    sparse_ctx=sparse_ctx,
                    max_seq_len=p.max_seq_len,
                    block_size=self.block_size,
                    num_valid_pages=md.prefill_num_valid_pages,
                    row_req_id=md.prefill_row_req_id,
                    kv_lens=md.prefill_kv_lens,
                    num_kv_heads=self.num_kv_heads,
                    pages_per_block=self.pages_per_block,
                )
            else:
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
