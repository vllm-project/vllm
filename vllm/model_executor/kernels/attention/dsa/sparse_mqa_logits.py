# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepGEMM sparse MQA-logits path for the DeepSeek V4.1 two-level indexer.

The candidate-source indexer publishes the top candidate blocks (in units of
``candidate_block_size`` compressed positions). Instead of scoring every KV
position and masking the dense logits down to those blocks, this path scores
only the candidate tokens with DeepGEMM's sparse MQA logits kernels
(``fp8_fp4_(paged_)sparse_mqa_logits``, DeepGEMM >= 2.8, SM100 only), runs
DeepSelect's top-k directly on the bf16 logits they produce, and remaps the
selected columns back to request-local positions.

Only the MXFP4 indexer cache is supported: the kernels require UE8M0-packed
(granularity-32) Q/KV scales, which is exactly the MXFP4 cache layout. The
FP8 indexer cache stores one fp32 scale per quant block instead.

The functions here take caller-owned output buffers (``sparse_indices``,
``end``, ``col_indices``) so the hot path allocates nothing; the indexer
metadata builder owns them.
"""

import importlib.util

import torch

from vllm.model_executor.layers.indexer_topk import (
    deep_select_topk,
    get_deep_select_stride_requirement,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_sparse_mqa_logits,
    fp8_fp4_sparse_mqa_logits,
    get_paged_sparse_mqa_logits_metadata,
    get_sparse_mqa_logits_metadata,
)

# Sparse-block sizes (in tokens) supported by the DeepGEMM kernels.
SPARSE_BLOCK_KV_CHOICES = (16, 8)


def has_deep_select() -> bool:
    """Whether the DeepSelect top-k extension is built and this GPU runs it."""
    return (
        importlib.util.find_spec("vllm._deepselect_C") is not None
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
    )


def check_deep_select_layout(num_sparse_cols: int, topk_tokens: int) -> None:
    """Raise unless DeepSelect accepts the sparse logits layout.

    The sparse kernels return contiguous bf16 ``[rows, num_sparse_cols]``
    logits and the top-k writes int32 ``[rows, topk_tokens]`` indices;
    DeepSelect needs both row strides aligned (1024B / 32B on SM100).
    """
    in_align, out_align = get_deep_select_stride_requirement()
    logits_row_bytes = num_sparse_cols * torch.bfloat16.itemsize
    if num_sparse_cols >= 2**23 or logits_row_bytes % in_align != 0:
        raise ValueError(
            f"DeepSelect needs the sparse logits row ({num_sparse_cols} bf16 "
            f"columns, {logits_row_bytes} B) to be {in_align}B-aligned."
        )
    if topk_tokens > 4096 or (topk_tokens * torch.int32.itemsize) % out_align != 0:
        raise ValueError(
            f"DeepSelect needs topk <= 4096 with a {out_align}B-aligned int32 "
            f"output row, got topk={topk_tokens}."
        )


_INT32_BIG = tl.constexpr(2**31 - 1)


@triton.jit(do_not_specialize_on_alignment=["end_ptr"])
def _expand_candidates_kernel(
    cand_ptr,
    cand_stride,
    ks_ptr,
    ks_stride,
    ke_ptr,
    ke_stride,
    si_ptr,
    si_stride,
    end_ptr,
    end_stride,
    K: tl.constexpr,
    S: tl.constexpr,
    RATIO: tl.constexpr,
    SBK: tl.constexpr,
):
    """One program per row: expand + filter + sort candidate blocks.

    Each row's valid sparse blocks come out sorted ascending (no dedup pass:
    production candidates are unique, and the kernel tolerates repeats),
    padded by repeating the last valid block — DeepGEMM's padding convention.
    ``end`` counts valid sparse-token columns; valid columns form a prefix
    because block ids are sorted.
    """
    row = tl.program_id(0)
    ks = tl.load(ks_ptr + row * ks_stride)
    ke = tl.load(ke_ptr + row * ke_stride)
    cols = tl.arange(0, K)
    cand = tl.load(cand_ptr + row * cand_stride + cols)  # K is pow2, no mask
    if RATIO == 2:
        cand = tl.interleave(cand * RATIO, cand * RATIO + 1)

    # Keep only blocks inside the row's KV range [0, ke - ks); the lower
    # bound is free since candidates are non-negative.
    kv_len = tl.maximum(ke - ks, 0)
    max_block = (kv_len + SBK - 1) // SBK
    valid = (cand >= 0) & (cand < max_block)
    vals = tl.where(valid, cand, _INT32_BIG)
    vals = tl.sort(vals)

    # Anchor at the row's ks rounded down to a sparse-block boundary
    # (identity for the paged path, where ks == 0).
    base = ks // SBK
    si = tl.where(vals != _INT32_BIG, vals, 0) + base
    n_valid = tl.sum(valid.to(tl.int32))
    last_rel = tl.max(tl.where(valid, cand, -1))
    pad_val = tl.where(n_valid > 0, last_rel + base, base)
    tl.store(
        si_ptr + row * si_stride + tl.arange(0, S),
        tl.where(tl.arange(0, S) < n_valid, si, pad_val),
    )

    # Valid sparse-token column count; only the last block can be partial.
    # Absolute start of the last block: (base + last_rel) * SBK + ks % SBK,
    # which is exactly last_rel * SBK + ks.
    last_start = last_rel * SBK + ks
    fill = tl.minimum(tl.maximum(ke - last_start, 1), SBK)
    end = tl.where(n_valid > 0, (n_valid - 1) * SBK + fill, 0)
    tl.store(end_ptr + row * end_stride, end)


@triton.jit
def _sparse_topk_remap_kernel(
    col_ptr,
    col_stride,
    si_ptr,
    si_stride,
    ks_ptr,
    ks_stride,
    out_ptr,
    out_stride,
    k,
    width,
    SBK: tl.constexpr,
    K_POW2: tl.constexpr,
):
    """Remap sparse top-k columns to request-local positions.

    Column ``j * SBK + o`` scores the token at
    ``(si[row, j] - ks // SBK) * SBK + o``; columns outside ``[0, width)``
    (including DeepSelect's sentinel for NaN rows) become -1.
    """
    row = tl.program_id(0)
    ks_base = tl.load(ks_ptr + row * ks_stride) // SBK
    cols = tl.arange(0, K_POW2)
    c = tl.load(col_ptr + row * col_stride + cols, mask=cols < k, other=-1)
    valid = (c >= 0) & (c < width)
    cc = tl.where(valid, c, 0)
    blocks = tl.load(si_ptr + row * si_stride + cc // SBK)
    pos = (blocks - ks_base) * SBK + cc % SBK
    tl.store(out_ptr + row * out_stride + cols, tl.where(valid, pos, -1), mask=cols < k)


def pick_sparse_block_kv(candidate_block_size: int) -> int:
    """Largest sparse-block size (tokens) dividing the candidate block size."""
    for sbk in SPARSE_BLOCK_KV_CHOICES:
        if candidate_block_size % sbk == 0:
            return sbk
    raise ValueError(
        f"candidate_block_size={candidate_block_size} is not a multiple of "
        f"any supported sparse_block_kv {SPARSE_BLOCK_KV_CHOICES}."
    )


def candidate_blocks_to_sparse_indices(
    candidate_blocks: torch.Tensor,
    row_ks: torch.Tensor,
    row_ke: torch.Tensor,
    candidate_block_size: int,
    sparse_block_kv: int,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand request-local candidate blocks into DeepGEMM sparse block ids.

    Args:
        candidate_blocks: [rows, K] int32 request-local candidate block ids
            (-1 padded), in units of ``candidate_block_size`` positions. K
            must be a power of two (the in-kernel sort); rows may be strided.
        row_ks: [rows] int32 per-row K range start; bounds are in the same
            (packed-workspace) coordinates the sparse kernel iterates over.
            Pass zeros for the paged path, whose blocks are context-relative.
        row_ke: [rows] int32 per-row K range end, in the same coordinates as
            ``row_ks``.
        candidate_block_size: Positions per candidate block.
        sparse_block_kv: Positions per sparse block (8 or 16).
        out: Optional ``(sparse_indices, end)`` buffers to write into,
            shaped [rows, K * ratio] and [rows], both int32.

    Returns:
        sparse_indices: [rows, K * ratio] int32 block ids for DeepGEMM,
            sorted ascending per row with the valid prefix first (unique in
            production, where candidates are unique) and the remaining slots
            repeating the last valid block (the kernel's padding convention).
            With the unaligned-ks variant, block ``i`` starts at
            ``i * sparse_block_kv + ks % sparse_block_kv``.
        end: [rows] int32 count of valid sparse-token columns; valid columns
            always form a prefix because the block ids are sorted.

    """
    rows, num_candidates = candidate_blocks.shape
    ratio = candidate_block_size // sparse_block_kv
    assert ratio * sparse_block_kv == candidate_block_size and ratio in (1, 2), (
        candidate_block_size,
        sparse_block_kv,
    )
    assert num_candidates & (num_candidates - 1) == 0, num_candidates
    num_sparse = num_candidates * ratio
    if out is None:
        device = candidate_blocks.device
        out = (
            torch.empty(rows, num_sparse, dtype=torch.int32, device=device),
            torch.empty(rows, dtype=torch.int32, device=device),
        )
    sparse_indices, end = out
    assert sparse_indices.shape == (rows, num_sparse) and end.shape == (rows,)
    assert candidate_blocks.stride(1) == 1 and sparse_indices.stride(1) == 1
    for t in (candidate_blocks, row_ks, row_ke, sparse_indices, end):
        assert t.dtype == torch.int32, t.dtype

    _expand_candidates_kernel[(rows,)](
        candidate_blocks,
        candidate_blocks.stride(0),
        row_ks,
        row_ks.stride(0),
        row_ke,
        row_ke.stride(0),
        sparse_indices,
        sparse_indices.stride(0),
        end,
        end.stride(0),
        K=num_candidates,
        S=num_sparse,
        RATIO=ratio,
        SBK=sparse_block_kv,
        num_warps=16,
    )
    return sparse_indices, end


def sparse_topk_remap(
    logits: torch.Tensor,
    sparse_indices: torch.Tensor,
    end: torch.Tensor,
    row_ks: torch.Tensor,
    sparse_block_kv: int,
    topk_tokens: int,
    topk_indices: torch.Tensor,
    *,
    col_indices: torch.Tensor,
) -> None:
    """Row top-k over sparse logits, remapped to request-local positions.

    Column ``j * sparse_block_kv + o`` of ``logits`` scores the token at
    request-local position
    ``(sparse_indices[row, j] - ks // sparse_block_kv) * sparse_block_kv + o``.

    DeepSelect consumes the bf16 logits as they come out of the sparse
    kernels; the valid columns form a prefix (block ids are sorted), so the
    per-row ``end`` bounds drive it directly. Slots beyond a row's valid
    count come back as -1.

    Args:
        logits: [rows, width] bf16 sparse logits, as produced by the sparse
            kernels.
        sparse_indices: [rows, S] int32 sparse block ids backing ``logits``.
        end: [rows] int32 per-row count of valid logit columns.
        row_ks: [rows] int32 per-row K range start.
        sparse_block_kv: Positions per sparse block (8 or 16).
        topk_tokens: Number of tokens to keep per row.
        topk_indices: [rows, topk_tokens] int32 output buffer.
        col_indices: [rows, topk_tokens] int32 scratch for the sparse-column
            top-k result (row stride 32B-aligned for DeepSelect).

    """
    rows, width = logits.shape
    assert logits.dtype == torch.bfloat16 and width >= topk_tokens
    assert end.dtype == torch.int32 and end.shape == (rows,)
    cols = col_indices[:rows]
    deep_select_topk(logits, topk_tokens, end=end, output_idx=cols)
    _sparse_topk_remap_kernel[(rows,)](
        cols,
        cols.stride(0),
        sparse_indices,
        sparse_indices.stride(0),
        row_ks,
        row_ks.stride(0),
        topk_indices,
        topk_indices.stride(0),
        topk_tokens,
        width,
        SBK=sparse_block_kv,
        K_POW2=triton.next_power_of_2(topk_tokens),
    )


def sparse_mqa_logits_prefill_chunk(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
    sparse_block_kv: int,
    topk_tokens: int,
    topk_indices: torch.Tensor,
    *,
    sparse_indices: torch.Tensor,
    end: torch.Tensor,
    col_indices: torch.Tensor,
    kernel_metadata: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse-MQA logits + top-k for one prefill chunk (packed KV workspace).

    Args:
        q: [rows, H, D] packed Q values (MXFP4 viewed as int8).
        q_scale: [rows, H] int32 packed UE8M0 Q scales.
        k_quant: [total_kv, D] packed K workspace (MXFP4 viewed as int8).
        k_scale: [total_kv] int32 packed UE8M0 K scales.
        weights: [rows, H] bf16 per-head weights; the sparse kernels take
            bf16 and do not fold the Q scale in.
        cu_seqlen_ks: [rows] int32 per-token K start bounds in the packed
            workspace.
        cu_seqlen_ke: [rows] int32 per-token K end bounds in the packed
            workspace.
        candidate_blocks: [rows, K] int32 request-local candidate block ids.
        candidate_block_size: Positions per candidate block.
        sparse_block_kv: Positions per sparse block (8 or 16).
        topk_tokens: Number of tokens to keep per row.
        topk_indices: [rows, topk_tokens] output buffer.
        sparse_indices: Caller-owned scratch, see
            `candidate_blocks_to_sparse_indices`.
        end: Caller-owned scratch, see `candidate_blocks_to_sparse_indices`.
        col_indices: Caller-owned scratch, see `sparse_topk_remap`.
        kernel_metadata: DeepGEMM schedule from a previous call with the same
            candidates and bounds (i.e. another indexer layer in the same
            step). When given, the candidate expansion is skipped and
            ``sparse_indices``/``end`` are assumed to be up to date.

    Returns:
        The DeepGEMM schedule metadata used, for reuse by later layers.

    """
    assert weights.dtype == torch.bfloat16, weights.dtype
    if kernel_metadata is None:
        candidate_blocks_to_sparse_indices(
            candidate_blocks,
            cu_seqlen_ks,
            cu_seqlen_ke,
            candidate_block_size,
            sparse_block_kv,
            out=(sparse_indices, end),
        )
        kernel_metadata = get_sparse_mqa_logits_metadata(
            cu_seqlen_ks,
            cu_seqlen_ke,
            k_quant.shape[0],
            sparse_indices,
            q.dtype,
            sparse_block_kv,
        )
    logits = fp8_fp4_sparse_mqa_logits(
        (q, q_scale),
        (k_quant, k_scale),
        weights,
        kernel_metadata,
        sparse_indices.shape[1],
        sparse_block_kv,
    )
    sparse_topk_remap(
        logits,
        sparse_indices,
        end,
        cu_seqlen_ks,
        sparse_block_kv,
        topk_tokens,
        topk_indices,
        col_indices=col_indices,
    )
    return kernel_metadata


def sparse_mqa_logits_paged_decode(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    row_indices: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
    sparse_block_kv: int,
    topk_tokens: int,
    topk_indices: torch.Tensor,
    *,
    row_ks: torch.Tensor,
    sparse_indices: torch.Tensor,
    end: torch.Tensor,
    col_indices: torch.Tensor,
    kernel_metadata: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse-MQA logits + top-k for paged decode rows.

    Every query is one row (spec-decode rows are flattened by the metadata
    builder), so ``block_table`` is already per row.

    Args:
        q: [rows, 1, H, D] packed Q values (MXFP4 viewed as int8).
        q_scale: [rows, 1, H] int32 packed UE8M0 Q scales.
        kv_cache: 4D [num_pages, page_kv, 1, head_bytes] uint8 fused cache
            view; the page stride must be 512B-aligned.
        weights: [rows, H] bf16 per-head weights.
        context_lens: [rows] int32 per-row (compressed) context lengths.
        block_table: [rows, P] int32 per-row page table, ``stride(-1) == 1``.
        row_indices: [rows] int32 row -> request map (pairing only affects
            scheduling).
        candidate_blocks: [rows, K] int32 candidate block ids.
        candidate_block_size: Positions per candidate block.
        sparse_block_kv: Positions per sparse block (8 or 16).
        topk_tokens: Number of tokens to keep per row.
        topk_indices: [rows, topk_tokens] output buffer.
        row_ks: [rows] int32 zeros (paged blocks are context-relative).
        sparse_indices: Caller-owned scratch.
        end: Caller-owned scratch.
        col_indices: Caller-owned scratch.
        kernel_metadata: See `sparse_mqa_logits_prefill_chunk`.

    Returns:
        The DeepGEMM schedule metadata used, for reuse by later layers.

    """
    rows = q.shape[0]
    page_kv = kv_cache.shape[1]
    assert weights.dtype == torch.bfloat16, weights.dtype
    assert kv_cache.stride(0) % 512 == 0, (
        "paged cache page stride must be 512B-aligned: "
        f"shape={tuple(kv_cache.shape)} stride={kv_cache.stride()}"
    )
    assert page_kv % sparse_block_kv == 0, (page_kv, sparse_block_kv)
    assert block_table.shape[0] == rows and block_table.stride(-1) == 1
    assert context_lens.shape == (rows,) and context_lens.is_contiguous()

    if kernel_metadata is None:
        candidate_blocks_to_sparse_indices(
            candidate_blocks,
            row_ks,
            context_lens,
            candidate_block_size,
            sparse_block_kv,
            out=(sparse_indices, end),
        )
        kernel_metadata = get_paged_sparse_mqa_logits_metadata(
            context_lens,
            block_table,
            row_indices,
            page_kv,
            sparse_indices,
            q.dtype,
            sparse_block_kv,
        )
    logits = fp8_fp4_paged_sparse_mqa_logits(
        (q, q_scale),
        kv_cache,
        weights,
        kernel_metadata,
        sparse_indices.shape[1],
        sparse_block_kv,
    )
    sparse_topk_remap(
        logits,
        sparse_indices,
        end,
        row_ks,
        sparse_block_kv,
        topk_tokens,
        topk_indices,
        col_indices=col_indices,
    )
    return kernel_metadata
