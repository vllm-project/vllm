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
    "needs an aiter containing ROCm/aiter#7017 with the paged in-place "
    "(kv_page_stride) row-packed page-max scorer taking a caller schedule "
    "(cta_info)"
)
GFX942_ROWPACK_ARGS = (
    "kv_page_stride",
    "kv_scale_page_stride",
    "kv_scale_offset",
    "cta_info",
    "total_ctas",
)
# Tokens per CTA chunk of the row-packed scorer (64 per wave, 4 waves).
GFX942_DECODE_BLOCK_K = 256


def gfx942_decode_topk_backend(
    rows: int, heads: int, max_seq_len: int, block_size: int
) -> str:
    """The decode block selector, ``triton`` or ``aiter``. Both produce the
    same top-k (same blocks in the same order, -1 padding) and the same sparse
    block table and context lengths; ``triton`` splits each row across
    programs and wins at small batch, ``aiter``'s one-workgroup-per-row radix
    select wins once many long rows fill the GPU. Decided from launch-time
    shapes only, so a captured graph keeps its choice."""
    from .ops.mxfp4_decode_topk import triton_topk_preferred

    return (
        "triton"
        if triton_topk_preferred(rows, heads, max_seq_len, block_size)
        else "aiter"
    )


_BLOCK_OUT_SUPPORT: list[bool] = []


def _aiter_rowpack_has_block_out() -> bool:
    if not _BLOCK_OUT_SUPPORT:
        try:
            from aiter.ops.flydsl.kernels.mqa_logits import (
                pa_mqa_page_max_fp4_rowpack_prefill_gfx942 as rowpack,
            )

            ok = hasattr(
                rowpack, "compile_pa_mqa_block_score_fp4_rowpack_decode_gfx942"
            )
        except ImportError:
            ok = False
        _BLOCK_OUT_SUPPORT.append(ok)
    return _BLOCK_OUT_SUPPORT[0]


_FP8Q_SUPPORT: list[bool] = []


def _aiter_rowpack_has_fp8q() -> bool:
    if not _FP8Q_SUPPORT:
        try:
            from aiter.ops.flydsl.kernels.mqa_logits import (
                pa_mqa_page_max_fp4_rowpack_prefill_gfx942 as rowpack,
            )

            fn = rowpack.compile_pa_mqa_page_max_fp4_rowpack_prefill_gfx942
            ok = "q_fp8" in inspect.signature(fn.__wrapped__).parameters
        except (ImportError, AttributeError):
            ok = False
        _FP8Q_SUPPORT.append(ok)
    return _FP8Q_SUPPORT[0]


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
                (
                    rowpack.flydsl_pa_mqa_page_max_fp4_rowpack_prefill_gfx942,
                    GFX942_ROWPACK_ARGS,
                )
            ]
            for fn in (
                prefill.compute_varqlen_windows,
                prefill.compute_prefill_schedule,
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
    buffers so a captured decode replays from stable addresses.

    Decode runs AITER's row-packed page-max scorer: each request's
    ``H * query_len`` (index head, MTP row) pairs fill one 16-slot MFMA group,
    so a K tile is loaded and e2m1-decoded once per request rather than once
    per row and head, and 64-token page maxima are reduced in-kernel instead of
    round-tripping ``[rows, max_seq_len]`` fp32 logits through HBM.
    """

    page64_table: torch.Tensor
    # Row-packed scorer operands, refilled per layer by the Q pack kernel.
    q_fp4: torch.Tensor  # [max_reqs, 16, 64] uint8
    q_scale: torch.Tensor  # [max_reqs, 1, 4, 16, 4] uint8
    slot_meta: torch.Tensor  # [3 * 16 * max_reqs] int32
    group_to_batch: torch.Tensor  # [max_reqs] int32, arange
    group_starts: torch.Tensor  # [max_reqs] int32, zeros
    page_max: torch.Tensor  # [H * max_rows, 2 * width] fp32
    # The schedule is built once per step (it depends only on seq_lens), not
    # once per layer.
    cta_info: torch.Tensor
    num_ctas: int
    block_k: int
    # FP8-Q decode (aiter scorer with ``q_fp8``): e4m3fnuz Q and per-slot fp32
    # scales replace q_fp4 / q_scale.
    q_fp8: torch.Tensor | None = None  # [max_reqs, 16, 128] uint8
    q_scale_f32: torch.Tensor | None = None  # [max_reqs, 16] fp32


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
        self._gfx942_buffers: dict[str, torch.Tensor] | None = None
        hf = vllm_config.model_config.hf_text_config
        sparse_cfg = getattr(hf, "sparse_attention_config", None) or {}
        tp = vllm_config.parallel_config.tensor_parallel_size
        self._gfx942_index_heads = max(
            1, sparse_cfg.get("sparse_num_index_heads", 1) // tp
        )

    def _gfx942_decode_plan(
        self, d: MiniMaxM3IndexerDecodeMetadata
    ) -> Gfx942DecodePlan:
        from .ops.mxfp4_gfx942 import (
            KERNEL_HEADS as KERNEL_GROUP_ROWS,
        )
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
        from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill_gfx942 import (  # noqa: E501
            CTA_INFO_WIDTH,
            compute_prefill_schedule,
        )

        if self._gfx942_buffers is None:
            reqs = self._max_decode_reqs
            rows = reqs * self.max_decode_query_len
            cus = torch.cuda.get_device_properties(self.device).multi_processor_count
            # The scorer's default occupancy: four CTAs per CU.
            num_ctas = max(4 * cus, reqs)
            self._gfx942_buffers = dict(
                q_fp4=torch.zeros(
                    (reqs, KERNEL_GROUP_ROWS, 64), dtype=torch.uint8, device=self.device
                ),
                q_scale=torch.zeros(
                    (reqs, 1, 4, KERNEL_GROUP_ROWS, 4),
                    dtype=torch.uint8,
                    device=self.device,
                ),
                # FP8-Q decode operands (2 KiB + 64 B per request).
                q_fp8=torch.zeros(
                    (reqs, KERNEL_GROUP_ROWS, 128),
                    dtype=torch.uint8,
                    device=self.device,
                ),
                q_scale_f32=torch.zeros(
                    (reqs, KERNEL_GROUP_ROWS), dtype=torch.float32, device=self.device
                ),
                slot_meta=torch.zeros(
                    3 * KERNEL_GROUP_ROWS * reqs, dtype=torch.int32, device=self.device
                ),
                group_to_batch=torch.arange(
                    reqs, dtype=torch.int32, device=self.device
                ),
                group_starts=torch.zeros(reqs, dtype=torch.int32, device=self.device),
                # Never filled: every page of a row's causal blocks is written by
                # the scorer, and nothing downstream reads past them.
                page_max=torch.empty(
                    (self._gfx942_index_heads * rows, KERNEL_PAGES_PER_BLOCK * width),
                    dtype=torch.float32,
                    device=self.device,
                ),
                cta_info=torch.empty(
                    (num_ctas, CTA_INFO_WIDTH), dtype=torch.int32, device=self.device
                ),
            )
        buf = self._gfx942_buffers
        # One schedule per step, shared by every index layer. The grid is fixed
        # at the buffer's CTA count, so a captured decode replays any batch.
        _, cta_info, num_ctas = compute_prefill_schedule(
            buf["group_to_batch"][:batch],
            buf["group_starts"][:batch],
            d.seq_lens,
            GFX942_DECODE_BLOCK_K,
            buf["cta_info"].shape[0],
            d.max_seq_len,
            cta_info_out=buf["cta_info"],
        )
        return Gfx942DecodePlan(
            page64_table=table,
            q_fp4=buf["q_fp4"],
            q_scale=buf["q_scale"],
            slot_meta=buf["slot_meta"],
            group_to_batch=buf["group_to_batch"],
            group_starts=buf["group_starts"],
            page_max=buf["page_max"],
            cta_info=cta_info,
            num_ctas=num_ctas,
            block_k=GFX942_DECODE_BLOCK_K,
            q_fp8=buf["q_fp8"],
            q_scale_f32=buf["q_scale_f32"],
        )

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


# launcher + the specialization-relevant args -> FlyDSL CallState. The decode
# indexer runs in the attention layer's eager break, so each of the 60 layers
# pays the launcher's host-side signature binding and cache-key build per
# step, which costs more than the kernel itself; a resolved CallState skips it.
#
# The key holds no tensor and no data pointer: a CallState re-reads every
# pointer from the args it is called with (FlyDSL's own fast path does the
# same), so it is keyed on what the compiled specialization depends on only.
# Keying on data_ptr and keeping the bound args alive leaked one entry, and the
# tensors it references, per call once an operand is a per-step activation or
# a fresh output (the fused decode scorer's index Q and score buffer): device
# memory then runs out mid-serve (hipErrorLaunchOutOfResources).
_ROWPACK_CALLS: dict[tuple, object] = {}


def _rowpack_call_key(launcher, args: tuple) -> tuple:
    return (id(launcher),) + tuple(
        (a.dtype, tuple(a.shape), tuple(a.stride()))
        if isinstance(a, torch.Tensor)
        else ("stream" if isinstance(a, torch.cuda.Stream) else a)
        for a in args
    )


def _launch_rowpack_cached(launcher, args: tuple) -> None:
    key = _rowpack_call_key(launcher, args)
    state = _ROWPACK_CALLS.get(key)
    if state is not None:
        state(args)
        return
    launcher(*args)  # compiles on first use and populates the CallState cache
    try:
        bound = launcher._sig.bind(*args)
        bound.apply_defaults()
        if len(bound.arguments) != len(args):
            raise TypeError("launcher has defaulted parameters")
        cache_key = launcher._build_full_cache_key(
            bound.arguments,
            owner_cls=None,
            bound_self=None,
            effective_hints=launcher._effective_compile_hints(),
        )
        _ROWPACK_CALLS[key] = launcher._call_state_cache[cache_key]
    except (AttributeError, KeyError, TypeError):
        # A FlyDSL without these internals: stay on the public launcher.
        _ROWPACK_CALLS[key] = lambda a: launcher(*a)


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
    """Pinned signed block scores of the decode rows.

    The index Q is scored as e4m3fnuz (one fp32 scale per row) against the
    MXFP4 K cache, as gfx950's indexer does: same MFMA, no Q e2m1 decode, and
    no MXFP4 Q rounding (the dominant top-k recall loss). One row-packed
    scorer launch covers every index head and MTP row of the batch (see
    :class:`Gfx942DecodePlan`); it packs Q and the slot metadata in its
    prologue and writes pinned 128-token block scores from its epilogue.

    Capability fallbacks (same results): an aiter whose scorer lacks the
    block-score variant runs pack / page-max scorer / fold as three launches,
    and one whose scorer lacks ``q_fp8`` scores an MXFP4 Q that way.
    """
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_page_max_fp4_rowpack_prefill_gfx942 import (  # noqa: E501
        compile_pa_mqa_page_max_fp4_rowpack_prefill_gfx942,
    )

    from .ops.mxfp4_gfx942 import (
        KERNEL_HEADS,
        decode_page_max_to_block_scores_gfx942,
        gfx942_mxfp4_cache_strides,
        pack_decode_rows_fp8_gfx942,
        pack_decode_rows_gfx942,
    )

    q_fp8 = _aiter_rowpack_has_fp8q() and plan.q_fp8 is not None
    batch, (rows, heads) = seq_lens.shape[0], index_query.shape[:2]
    k, k_scale, cache_kw = gfx942_mxfp4_cache_strides(index_pages)
    if q_fp8 and batch and _aiter_rowpack_has_block_out():
        return _mxfp4_gfx942_index_scores_decode_fused(
            index_query,
            seq_lens,
            plan,
            k,
            k_scale,
            cache_kw,
            max_seq_len=max_seq_len,
            query_len=query_len,
            scale=scale,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )
    if q_fp8:
        assert plan.q_scale_f32 is not None
        q_buf, qs_buf = plan.q_fp8, plan.q_scale_f32
        slot_meta = pack_decode_rows_fp8_gfx942(
            index_query, seq_lens, q_buf, qs_buf, plan.slot_meta, query_len=query_len
        )
        cache_kw = dict(cache_kw, q_fp8=True)
    else:
        q_buf, qs_buf = plan.q_fp4, plan.q_scale
        slot_meta = pack_decode_rows_gfx942(
            index_query, seq_lens, q_buf, qs_buf, plan.slot_meta, query_len=query_len
        )
    page_max = plan.page_max[: heads * rows]
    # Every page of each row's causal window is written by the scheduled
    # CTAs and the fold below reads no further, so no -inf fill is needed.
    launcher, _ = compile_pa_mqa_page_max_fp4_rowpack_prefill_gfx942(
        block_k=plan.block_k,
        kv_block_size=64,
        max_blocks_per_seq=plan.page64_table.shape[1],
        num_warps=plan.block_k // 64,
        head_dim=index_query.shape[-1],
        **cache_kw,
    )
    _launch_rowpack_cached(
        launcher,
        (
            page_max,
            q_buf[:batch],
            qs_buf[:batch],
            k,
            k_scale,
            plan.page64_table,
            slot_meta,
            plan.cta_info,
            page_max.stride(0),
            page_max.shape[1],
            batch * KERNEL_HEADS,
            page_max.numel(),
            float(scale),
            plan.num_ctas,
            torch.cuda.current_stream(),
        ),
    )
    score = _new_score(heads, rows, max_seq_len, index_query.device)
    decode_page_max_to_block_scores_gfx942(
        page_max.view(heads, rows, -1),
        seq_lens,
        score,
        query_len=query_len,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sentinel_init=SENTINEL_INIT,
        sentinel_local=SENTINEL_LOCAL,
    )
    return score


def _mxfp4_gfx942_index_scores_decode_fused(
    index_query: torch.Tensor,
    seq_lens: torch.Tensor,
    plan: Gfx942DecodePlan,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    cache_kw: dict,
    *,
    max_seq_len: int,
    query_len: int,
    scale: float,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """FP8-Q decode scores in one launch: the scorer packs the bf16 Q in its
    prologue and writes pinned 128-token block scores directly (see
    ``compile_pa_mqa_block_score_fp4_rowpack_decode_gfx942``). A Q the
    prologue cannot read (not bf16, or odd strides) is packed by a separate
    launch first."""
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_page_max_fp4_rowpack_prefill_gfx942 import (  # noqa: E501
        compile_pa_mqa_block_score_fp4_rowpack_decode_gfx942,
    )

    from .ops.mxfp4_gfx942 import KERNEL_HEADS, pack_decode_rows_fp8_gfx942

    batch, (rows, heads) = seq_lens.shape[0], index_query.shape[:2]
    assert plan.q_fp8 is not None and plan.q_scale_f32 is not None
    if heads * query_len > KERNEL_HEADS:
        raise ValueError(
            f"{heads} index heads x {query_len} decode rows exceed the "
            f"{KERNEL_HEADS}-slot MFMA group"
        )
    fuse_q = (
        index_query.dtype == torch.bfloat16
        and index_query.stride(2) == 1
        and index_query.stride(0) % 2 == 0
        and index_query.stride(1) % 2 == 0
    )
    if fuse_q:
        q_arg, qs_arg, meta = index_query, plan.q_scale_f32[:batch], plan.slot_meta
        q_strides = (index_query.stride(0), index_query.stride(1))
    else:
        q_arg, qs_arg = plan.q_fp8[:batch], plan.q_scale_f32[:batch]
        meta = pack_decode_rows_fp8_gfx942(
            index_query,
            seq_lens,
            plan.q_fp8,
            plan.q_scale_f32,
            plan.slot_meta,
            query_len=query_len,
        )
        q_strides = (0, 0)
    score = _new_score(heads, rows, max_seq_len, index_query.device)
    out2d = score.view(heads * rows, score.shape[2])
    launcher, _ = compile_pa_mqa_block_score_fp4_rowpack_decode_gfx942(
        block_k=plan.block_k,
        max_blocks_per_seq=plan.page64_table.shape[1],
        head_dim=index_query.shape[-1],
        kv_page_stride=cache_kw["kv_page_stride"],
        kv_scale_page_stride=cache_kw["kv_scale_page_stride"],
        kv_scale_offset=cache_kw["kv_scale_offset"],
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sentinel_init=SENTINEL_INIT,
        sentinel_local=SENTINEL_LOCAL,
        fused_q_heads=heads if fuse_q else 0,
        fused_q_len=query_len if fuse_q else 1,
    )
    _launch_rowpack_cached(
        launcher,
        (
            out2d,
            q_arg,
            qs_arg,
            k,
            k_scale,
            plan.page64_table,
            meta,
            plan.cta_info,
            out2d.stride(0),
            out2d.shape[1],
            batch * KERNEL_HEADS,
            out2d.numel(),
            float(scale),
            seq_lens,
            q_strides[0],
            q_strides[1],
            rows,
            plan.num_ctas,
            torch.cuda.current_stream(),
        ),
    )
    return score


def mxfp4_gfx942_index_scores_prefill_fp8copy(
    index_query: torch.Tensor,  # [rows, H, 128] bf16
    index_pages: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    context_lens: torch.Tensor,
    num_valid_pages: torch.Tensor,
    *,
    max_query_len: int,
    max_seq_len: int,
    scale: float,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Prefill index scores from an exact e4m3fnuz copy of the MXFP4 index K.

    e2m1 * 2^e with the RMS-normed index K's block exponents is exactly
    representable in e4m3fnuz, so the transient fp8 copy holds the same K the
    MXFP4 scorer decodes; the copy feeds the fp8 MFMA
    Triton score kernel the fp8 indexer uses. Q is rounded to e4m3fnuz per
    element there (the fp8 indexer's Q precision; finer than MXFP4 Q).
    The copy is gathered in request-block order into a compact buffer so it
    never needs the cache's full size.
    """
    from .ops.index_topk import minimax_m3_index_score
    from .ops.mxfp4_gfx942 import mxfp4_pages_to_fp8_gfx942

    rows, heads = index_query.shape[:2]
    nblk = block_table.shape[1]
    batch = block_table.shape[0]
    fp8 = current_platform.fp8_dtype()
    need = batch * nblk
    buf = _FP8_COPY.get(index_pages.device)
    if buf is None or buf.shape[0] < need:
        buf = torch.empty((need, 128, 128), dtype=fp8, device=index_pages.device)
        _FP8_COPY[index_pages.device] = buf
    from .ops.mxfp4_gfx942 import MANAGER_BLOCK_BYTES

    mxfp4_pages_to_fp8_gfx942(
        index_pages.reshape(-1, MANAGER_BLOCK_BYTES),
        block_table.reshape(-1),
        buf[:need],
    )
    compact_bt = torch.arange(need, dtype=torch.int32, device=block_table.device).view(
        batch, nblk
    )
    score = minimax_m3_index_score(
        index_query.to(fp8),
        buf[:need],
        compact_bt,
        cu_seqlens_q,
        seq_lens,
        context_lens,
        max_query_len,
        max_seq_len,
        heads,
    )
    # The top-k reads power-of-two-padded rows; copy into that layout.
    out = _new_score(heads, rows, max_seq_len, index_query.device)
    w = min(out.shape[2], score.shape[2])
    torch.mul(score[..., :w], scale, out=out[..., :w])
    score = out
    _pin_forced_blocks(
        score,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        num_valid_pages=num_valid_pages,
    )
    return score


_FP8_COPY: dict[torch.device, torch.Tensor] = {}


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
        cfg = get_current_vllm_config()
        self._workspace_dims = (
            cfg.scheduler_config.max_num_batched_tokens,
            cfg.model_config.max_model_len,
        )
        self.sparse_bt_buffer: torch.Tensor | None = None
        self.sparse_ctx_buffer: torch.Tensor | None = None

    def reserve_workspace(self) -> None:
        """Make the memory-profiling run see the transient score buffers of a
        full prefill chunk at max_model_len.

        The profile run skips ``forward`` (caches unbound), so without this the
        fp32 ``[H, max_num_batched_tokens, blocks]`` score tensors
        a long-context prefill allocates are unaccounted for and the KV cache
        is sized into them (OOM at 512k / 16k-token chunks on TP4).
        """
        if _mxfp4_arch() != "gfx942":
            return
        rows, msl = self._workspace_dims
        heads = self.num_index_heads
        dev = self.index_cache.kv_cache.device
        if dev.type != "cuda":
            dev = torch.device("cuda", torch.cuda.current_device())
        score = torch.empty(
            (heads, rows, _score_width(msl, MSA_SPARSE_BLOCK_SIZE)),
            dtype=torch.float32,
            device=dev,
        )
        # The fp8-copy prefill's unpadded block scores, live alongside the
        # padded copy the top-k reads.
        page_max = torch.empty(
            (heads * rows, 2 * cdiv(msl, MSA_SPARSE_BLOCK_SIZE)),
            dtype=torch.float32,
            device=dev,
        )
        # Touch both so the caching allocator's peak includes them.
        score[0, 0, 0] = 0
        page_max[0, 0] = 0
        if not getattr(MiniMaxM3IndexerRocmMxfp4Impl, "_logged_reserve", False):
            MiniMaxM3IndexerRocmMxfp4Impl._logged_reserve = True
            from vllm.logger import init_logger

            init_logger(__name__).info(
                "MiniMax M3 MXFP4 indexer: reserving %.2f GiB of prefill "
                "score workspace in the memory profile (rows=%d, max_len=%d)",
                (score.numel() + page_max.numel()) * 4 / 2**30,
                rows,
                msl,
            )
        del score, page_max

    @property
    def pages_per_block(self) -> int:
        from .ops.sparse_pa import ASM_PAGE_SIZE

        return self.block_size // ASM_PAGE_SIZE

    def _decode_topk_counter(self, rows: int) -> torch.Tensor | None:
        """Per-row arrival counters of the Triton selector's merge. Zero
        between launches (the merging program resets its row), so a persistent
        buffer keeps a captured graph's address fixed. ``None`` lets the
        selector allocate zeros itself (eager, no shared top-k buffer)."""
        counter = getattr(self, "topk_completion_counter", None)
        if counter is None or counter.shape[1] < rows:
            return None
        return counter

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
            # Profiling run, caches unbound: charge the prefill transients.
            self.reserve_workspace()
            return None, None
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
                topk_args = dict(
                    sparse_bt=sparse_bt,
                    sparse_ctx=sparse_ctx,
                    max_seq_len=d.max_seq_len,
                    block_size=self.block_size,
                    query_len=d.decode_query_len,
                    num_kv_heads=self.num_kv_heads,
                    pages_per_block=self.pages_per_block,
                )
                backend = gfx942_decode_topk_backend(
                    nd, self.num_index_heads, d.max_seq_len, self.block_size
                )
                if backend == "triton":
                    from .ops.mxfp4_decode_topk import mxfp4_decode_topk

                    mxfp4_decode_topk(
                        score,
                        decode_topk,
                        page16,
                        d.seq_lens,
                        counter=self._decode_topk_counter(nd),
                        **topk_args,
                    )
                else:
                    pa_sparse_block_topk(
                        score, decode_topk, page16, d.seq_lens, **topk_args
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
                score = mxfp4_gfx942_index_scores_prefill_fp8copy(
                    iq[nd:],
                    pages,
                    p.block_table,
                    p.cu_seqlens_q,
                    p.seq_lens,
                    p.context_lens,
                    md.prefill_num_valid_pages,
                    max_query_len=p.max_query_len,
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
