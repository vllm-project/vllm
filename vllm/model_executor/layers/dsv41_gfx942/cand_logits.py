# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The indexer logits of the DSpark candidate positions on gfx942.

Layers 24 to 36 take their top 512 only from the 2048 candidate blocks of 8
positions that layer 20 chose for each decode row. vLLM computes such a
layer's logits for every position of the row with AITER's
deepgemm_fp8_paged_mqa_logits. Then topk942's gatherCandidates copies the
16384 candidate logits of each row into a compact row, and topK512Candidates
takes the top 512 of that row. At 128k context the dense call scores 8 times
more positions than the top-k reads.

_cand_logits_kernel computes only the candidate logits and writes the compact
row directly. Position i of row r holds the logit of column
``8 * cand[r][i // 8] + i % 8`` and that column as its id. It holds -inf and id
-1 when the block entry is -1 or the column is at or past the row's end.
This is what gatherCandidates writes.

Each logit is computed with the code of AITER's gfx942 kernel
(_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle in
aiter/ops/triton/gluon/pa_mqa_logits.py). It uses the same MFMA layout. It
multiplies the queries of all heads with the column's key starting from zero,
multiplies by the key's scale, applies ReLU, multiplies by the head weights,
and adds the heads up with the same reduction. The sum over the heads runs
inside one wave, along the heads of one column, so its order does not depend
on which position a column holds. Only the address of each column's key
differs from the dense kernel. So every logit is bit-identical to the dense
kernel's logit of that position (tests/kernels/test_dsv41_gfx942_cand_logits.py).
"""

import torch
from aiter.ops.triton.gluon import pa_mqa_logits as _aiter_logits
from aiter.ops.triton.gluon.pa_decode_gluon import get_cdna_version

from vllm.triton_utils import tl

# vLLM has no wrapper for Gluon, so the kernel uses the Gluon modules that
# AITER's dense kernel was built with. Gluon's constexpr is Triton's
# tl.constexpr, so the kernel's annotations use tl.constexpr.
gluon = _aiter_logits.gluon
gl = _aiter_logits.gl

# The same reduction function and the same MFMA layout choice as the dense
# kernel. Both are taken from AITER's module so that the two kernels cannot
# drift apart.
_sum_combine = _aiter_logits._sum_combine
_Use_2d_instr_shape_mfma_layout = _aiter_logits._Use_2d_instr_shape_mfma_layout

# The columns of one MFMA stage. The dense kernel runs ChunkK = 256 in two
# stages of 128 columns.
COLUMNS = 128
# Each program scores this many stages of one row. 16384 candidates are 128
# stages, so a 6 row step runs 6 * 128 programs at 1 stage a program. Each
# stage waits on three dependent loads (candidate block, page, keys), and many
# programs in flight hide that latency best. At 6 rows and 128k context, 1
# stage a program takes 6.8 us and 8 stages take 14.0 us.
STAGES_PER_PROGRAM = 1


@gluon.jit
def _cand_logits_kernel(
    Q_buffer,
    stride_q_row,
    stride_q_heads,
    KV_buffer,
    stride_k_seq,
    scale_buffer,
    stride_scale_seq,
    seq_lens,
    next_n,
    block_table,
    stride_bt_row,
    cand,
    stride_cand_row,
    weights,
    stride_w_row,
    out_logits,
    out_ids,
    row_len,
    splits_per_row,
    SeqLensIs2D: tl.constexpr,
    StagesPerProgram: tl.constexpr,
    ChunkQ: tl.constexpr,
    HiddenDim: tl.constexpr,
    KVBlockSize: tl.constexpr,
    CandBlock: tl.constexpr,
    CDNA_VERSION: tl.constexpr,
):
    # The layouts of the dense gfx942 kernel with ChunkK = 256.
    NumWarps: tl.constexpr = 4
    ThreadsPerWarp: tl.constexpr = 64
    ChunkKPerStage: tl.constexpr = 128
    MFMAPerWarp: tl.constexpr = ChunkKPerStage // 16 // NumWarps
    ValQMPerThread: tl.constexpr = ChunkQ // (
        NumWarps * ThreadsPerWarp // (HiddenDim // 16)
    )
    layout_q: tl.constexpr = gl.BlockedLayout(
        size_per_thread=[ValQMPerThread, 16],
        threads_per_warp=[ThreadsPerWarp // (HiddenDim // 16), HiddenDim // 16],
        warps_per_cta=[NumWarps, 1],
        order=[1, 0],
    )
    if _Use_2d_instr_shape_mfma_layout:
        mfma_layout: tl.constexpr = gl.amd.AMDMFMALayout(
            version=CDNA_VERSION,
            instr_shape=[16, 16],
            transposed=False,
            warps_per_cta=[1, NumWarps],
            tiles_per_warp=[1, MFMAPerWarp],
        )
    else:
        mfma_layout: tl.constexpr = gl.amd.AMDMFMALayout(  # type: ignore[no-redef]
            version=CDNA_VERSION,
            instr_shape=[16, 16, 32],
            transposed=False,
            warps_per_cta=[1, NumWarps],
            tiles_per_warp=[1, MFMAPerWarp],
        )
    mfma_layout_a: tl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    mfma_layout_b: tl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    layout_scale: tl.constexpr = gl.SliceLayout(1, mfma_layout)
    layout_col_b: tl.constexpr = gl.SliceLayout(0, mfma_layout_b)
    layout_col: tl.constexpr = gl.SliceLayout(0, mfma_layout)
    # The cache stores each page of KVBlockSize keys shuffled in groups of 16
    # keys, the layout the dense kernel's offset_k_fixed reads.
    ShuffleRows: tl.constexpr = 16

    pid = tl.program_id(0)
    row = pid // splits_per_row
    split = pid % splits_per_row
    # The row's end as topk942's rowEndOf computes it.
    if SeqLensIs2D:
        row_end = gl.load(seq_lens + row)
    else:
        row_end = gl.load(seq_lens + row // next_n) - next_n + row % next_n + 1
    row_end = gl.maximum(row_end, 0)

    q = gl.amd.cdna3.buffer_load(
        ptr=Q_buffer,
        offsets=row * stride_q_row
        + (gl.arange(0, ChunkQ, layout=gl.SliceLayout(1, layout_q)) * stride_q_heads)[
            :, None
        ]
        + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_q))[None, :],
    )
    mfma_q = gl.convert_layout(q, mfma_layout_a)
    scale_weight = gl.amd.cdna3.buffer_load(
        ptr=weights,
        offsets=row * stride_w_row + gl.arange(0, ChunkQ, layout=layout_scale),
    )
    dims = gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b))
    dim_offsets = dims % 16 + dims // 16 * (ShuffleRows * 16)
    zero = gl.zeros((ChunkQ, ChunkKPerStage), dtype=tl.float32, layout=mfma_layout)

    for stage in range(StagesPerProgram):
        first = (split * StagesPerProgram + stage) * ChunkKPerStage
        cols = first + gl.arange(0, ChunkKPerStage, layout=layout_col_b)
        in_row = cols < row_len
        block = gl.load(
            cand + row * stride_cand_row + cols // CandBlock, mask=in_row, other=-1
        )
        position = block * CandBlock + cols % CandBlock
        live = (block >= 0) & (position < row_end)
        # A column that is not live reads the first key of the row's first
        # page, so that every load stays inside the cache. Its logit is
        # replaced with -inf below.
        position = tl.where(live, position, 0)
        # The dense kernel reads row r's keys through the page table row of
        # its request, r // next_n.
        page = gl.load(
            block_table + (row // next_n) * stride_bt_row + position // KVBlockSize
        )
        page = page.to(gl.int64)
        key = position % KVBlockSize
        col_offsets = key % ShuffleRows * 16 + key // ShuffleRows * (
            ShuffleRows * HiddenDim
        )
        k = gl.load(
            KV_buffer
            + dim_offsets[:, None]
            + col_offsets[None, :]
            + page[None, :] * stride_k_seq
        )
        k_scale = gl.load(scale_buffer + page * stride_scale_seq + key)

        mfma_k = gl.convert_layout(k, mfma_layout_b)
        o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)
        k_scale = gl.convert_layout(k_scale, layout_col)
        o = o * k_scale[None, :]
        o = gl.maximum(o, 0.0)
        o = o * scale_weight[:, None]
        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)

        cols_out = gl.convert_layout(cols, layout_col)
        live_out = gl.convert_layout(live, layout_col)
        position_out = gl.convert_layout(position, layout_col)
        store_mask = cols_out < row_len
        gl.store(
            out_logits + row * row_len + cols_out,
            tl.where(live_out, logits, float("-inf")),
            mask=store_mask,
        )
        gl.store(
            out_ids + row * row_len + cols_out,
            tl.where(live_out, position_out, -1),
            mask=store_mask,
        )


def split_cache(kv_cache: torch.Tensor):
    """The FP8 keys and the FP32 key scales of an indexer cache of shape
    [pages, page size, 1, dim + 4], split the way AITER's
    deepgemm_fp8_paged_mqa_logits splits it: each page holds page size x dim
    FP8 values and then page size FP32 scales."""
    from aiter import dtypes

    pages, page_size, _, index_dim = kv_cache.shape
    dim = index_dim - 4
    flat = kv_cache.view(-1, page_size * index_dim)
    keys = flat[..., : page_size * dim].view(dtypes.fp8)
    scales = flat[..., page_size * dim :].view(torch.float32)
    return keys, scales


def candidate_logits(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    next_n: int,
    block_table: torch.Tensor,
    candidates: torch.Tensor,
    cand_block: int,
    out_logits: torch.Tensor,
    out_ids: torch.Tensor,
    stages_per_program: int = 0,
) -> None:
    """Writes the compact candidate rows described in the module docstring.
    stages_per_program overrides STAGES_PER_PROGRAM when it is not 0.

    q_fp8 is [rows, heads, dim] FP8, with row r the draft position r % next_n
    of request r // next_n. kv_cache is the paged indexer cache, weights
    [rows, heads] FP32, block_table one row of pages for each request,
    seq_lens the lengths as the dense call gets them, candidates
    [rows, blocks] int32 with -1 padding, and out_logits and out_ids
    contiguous [rows, blocks * cand_block]."""
    rows, heads, dim = q_fp8.shape
    page_size = kv_cache.shape[1]
    row_len = candidates.shape[1] * cand_block
    assert out_logits.shape == (rows, row_len) and out_logits.is_contiguous()
    assert out_ids.shape == (rows, row_len) and out_ids.is_contiguous()
    # The kernel reads keys in the cache's groups of 16, so a page must hold
    # whole groups. Each column finds its own page, so a candidate block may
    # cross a page boundary.
    assert page_size % 16 == 0, f"page size {page_size} is not a multiple of 16"
    keys, scales = split_cache(kv_cache)
    per_program = stages_per_program or STAGES_PER_PROGRAM
    stages = triton_cdiv(row_len, COLUMNS)
    splits = triton_cdiv(stages, per_program)
    _cand_logits_kernel[(rows * splits,)](
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        keys,
        keys.stride(0),
        scales,
        scales.stride(0),
        seq_lens,
        next_n,
        block_table,
        block_table.stride(0),
        candidates,
        candidates.stride(0),
        weights,
        weights.stride(0),
        out_logits,
        out_ids,
        row_len,
        splits,
        SeqLensIs2D=seq_lens.dim() == 2,
        StagesPerProgram=per_program,
        ChunkQ=heads,
        HiddenDim=dim,
        KVBlockSize=page_size,
        CandBlock=cand_block,
        CDNA_VERSION=get_cdna_version(),
        num_warps=4,
    )


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b
