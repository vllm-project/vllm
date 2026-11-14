# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal extend attention over a paged fp8 KV cache with fp8 Q.

Each program takes BLOCK_M // GQA query tokens of one request and the GQA query heads
of one KV head (rows token-major), so each K/V tile serves all GQA heads. Context and
new tokens share one loop; only tiles past the first query's position are masked
(bottom-right aligned). P is stored as 2^P_SHIFT * exp2(.) in fp8, and the row sums
come from an fp8 MFMA of P with a ones block rather than a per-element reduction.
When the grid is below two programs per CU, the context is split over NSPLIT programs
whose partial outputs are merged by a second kernel.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units

_P_SHIFT = 8.0
_L_COLS = 32
_BLOCK_M = 256
_BLOCK_N = 128


@triton.jit
def _tile(
    acc,
    m_i,
    l_acc,
    ones,
    q,
    start,
    n,
    d,
    bt,
    kvh,
    seq_len,
    q_pos,
    qk_scale,
    K,
    V,
    k_s0,
    k_s1,
    k_s2,
    v_s0,
    v_s1,
    v_s2,
    PAGE: tl.constexpr,
    P_SHIFT: tl.constexpr,
    MASK: tl.constexpr,
):
    kk = start + n
    if MASK:
        kv_ok = kk < seq_len
        page = tl.load(bt + kk // PAGE, mask=kv_ok, other=0).to(tl.int64)
    else:
        page = tl.load(bt + kk // PAGE).to(tl.int64)
    off = kk % PAGE
    kt = tl.load(
        K + page[None, :] * k_s0 + off[None, :] * k_s1 + kvh * k_s2 + d[:, None]
    )
    v_ptr = V + page[:, None] * v_s0 + off[:, None] * v_s1 + kvh * v_s2 + d[None, :]
    v = tl.load(v_ptr, mask=kv_ok[:, None], other=0.0) if MASK else tl.load(v_ptr)
    s = tl.dot(q, kt)
    if MASK:
        s = tl.where((kk[None, :] <= q_pos[:, None]) & kv_ok[None, :], s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, 1) * qk_scale)
    p = tl.math.exp2(s * qk_scale - (m_new - P_SHIFT)[:, None]).to(v.dtype)
    alpha = tl.math.exp2(m_i - m_new)
    l_acc = tl.dot(p, ones, l_acc * alpha[:, None])
    acc = tl.dot(p, v, acc * alpha[:, None])
    return acc, m_new, l_acc


@triton.jit
def _fp8_paged_extend_kernel(
    Out,
    Q,
    K,
    V,
    BT,
    SeqLens,
    CuQ,
    QDesc,
    KDesc,
    VDesc,
    sm_scale,
    bt_stride,
    q_s0,
    q_s1,
    o_s0,
    o_s1,
    k_s0,
    k_s1,
    k_s2,
    v_s0,
    v_s1,
    v_s2,
    PartO,
    PartL,
    num_kv,
    num_tokens,
    GQA: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
    P_SHIFT: tl.constexpr,
    L_COLS: tl.constexpr,
    NSPLIT: tl.constexpr,
):
    qb = tl.program_id(0)
    seq = tl.program_id(1)
    kvh = tl.program_id(2) % num_kv
    sp = tl.program_id(2) // num_kv
    q_start = tl.load(CuQ + seq)
    q_len = tl.load(CuQ + seq + 1) - q_start
    t0 = qb * BLOCK_T
    if t0 >= q_len:
        return
    seq_len = tl.load(SeqLens + seq)
    ctx = seq_len - q_len

    BM: tl.constexpr = BLOCK_T * GQA
    rows = tl.arange(0, BM)
    tok = t0 + rows // GQA
    head = kvh * GQA + rows % GQA
    d = tl.arange(0, D)
    row_ok = tok < q_len
    q = tl.load(
        Q
        + (q_start + tok).to(tl.int64)[:, None] * q_s0
        + head[:, None] * q_s1
        + d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )
    qk_scale = sm_scale * 1.4426950408889634 * tl.load(QDesc) * tl.load(KDesc)

    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_acc = tl.zeros([BM, L_COLS], tl.float32)
    ones = tl.full([BLOCK_N, L_COLS], 1.0, tl.float32).to(K.dtype.element_ty)
    acc = tl.zeros([BM, D], tl.float32)
    bt = BT + seq.to(tl.int64) * bt_stride
    n = tl.arange(0, BLOCK_N)
    q_pos = ctx + tok
    # Tiles below full_end are visible to every row and are split; the masked tail
    # goes to the last split.
    full_end = (ctx + t0 + 1) // BLOCK_N * BLOCK_N
    end = ctx + tl.minimum(t0 + BLOCK_T, q_len)
    n_full = full_end // BLOCK_N
    per = tl.cdiv(n_full, NSPLIT)
    lo = tl.minimum(sp * per, n_full) * BLOCK_N
    hi = tl.minimum(sp * per + per, n_full) * BLOCK_N
    tail = tl.where(sp == NSPLIT - 1, full_end, end)
    for start in range(lo, hi, BLOCK_N):
        acc, m_i, l_acc = _tile(
            acc,
            m_i,
            l_acc,
            ones,
            q,
            start,
            n,
            d,
            bt,
            kvh,
            seq_len,
            q_pos,
            qk_scale,
            K,
            V,
            k_s0,
            k_s1,
            k_s2,
            v_s0,
            v_s1,
            v_s2,
            PAGE,
            P_SHIFT,
            False,
        )
    for start in range(tail, end, BLOCK_N):
        acc, m_i, l_acc = _tile(
            acc,
            m_i,
            l_acc,
            ones,
            q,
            start,
            n,
            d,
            bt,
            kvh,
            seq_len,
            q_pos,
            qk_scale,
            K,
            V,
            k_s0,
            k_s1,
            k_s2,
            v_s0,
            v_s1,
            v_s2,
            PAGE,
            P_SHIFT,
            True,
        )

    l_i = tl.sum(l_acc, 1) * (1.0 / L_COLS)
    if NSPLIT == 1:
        o = acc * (tl.load(VDesc) / l_i)[:, None]
        tl.store(
            Out
            + (q_start + tok).to(tl.int64)[:, None] * o_s0
            + head[:, None] * o_s1
            + d[None, :],
            o.to(Out.dtype.element_ty),
            mask=row_ok[:, None],
        )
    else:
        live = l_i > 0
        o = acc / tl.where(live, l_i, 1.0)[:, None]
        lse = tl.where(
            live, m_i + tl.math.log2(tl.where(live, l_i, 1.0)), float("-inf")
        )
        H = num_kv * GQA
        prow = (sp * num_tokens + q_start + tok).to(tl.int64) * H + head
        tl.store(PartO + prow[:, None] * D + d[None, :], o, mask=row_ok[:, None])
        tl.store(PartL + prow, lse, mask=row_ok)


@triton.jit
def _merge_splits_kernel(
    Out,
    PartO,
    PartL,
    VDesc,
    o_s0,
    o_s1,
    num_tokens,
    H: tl.constexpr,
    D: tl.constexpr,
    NSPLIT: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, H)
    d = tl.arange(0, D)
    m = tl.full([H], float("-inf"), tl.float32)
    for sp in tl.static_range(NSPLIT):
        m = tl.maximum(m, tl.load(PartL + (sp * num_tokens + t) * H + h))
    w_sum = tl.zeros([H], tl.float32)
    o = tl.zeros([H, D], tl.float32)
    for sp in tl.static_range(NSPLIT):
        row = (sp * num_tokens + t) * H + h
        w = tl.math.exp2(tl.load(PartL + row) - m)
        w_sum += w
        o += w[:, None] * tl.load(PartO + row[:, None] * D + d[None, :])
    o = o * (tl.load(VDesc) / w_sum)[:, None]
    tl.store(
        Out + t * o_s0 + h[:, None] * o_s1 + d[None, :], o.to(Out.dtype.element_ty)
    )


def fp8_paged_extend_supported(
    num_heads: int, num_kv_heads: int, head_size: int
) -> bool:
    gqa = num_heads // num_kv_heads
    return (
        num_heads % num_kv_heads == 0
        and num_heads & (num_heads - 1) == 0
        and gqa & (gqa - 1) == 0
        and gqa <= _BLOCK_M
        and head_size == 64
    )


def _num_splits(programs: int, max_seq_len: int) -> int:
    # One 8-warp program fills a CU at waves_per_eu=2: split only below two programs
    # per CU, aiming at four, with at least 16 tiles per split.
    num_cus = num_compute_units()
    if programs >= 2 * num_cus:
        return 1
    n = triton.next_power_of_2(triton.cdiv(4 * num_cus, programs))
    return max(1, min(n, 8, max_seq_len // (16 * _BLOCK_N)))


def fp8_paged_extend_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Causal attention of each request's new tokens over its whole KV sequence.

    Args:
        q: [num_tokens, num_heads, 64] fp8, last dim contiguous.
        k_cache: [num_blocks, block_size, num_kv_heads, 64] fp8 view, last dim
            contiguous; already holds the new tokens.
        v_cache: Same layout as k_cache.
        out: [num_tokens, num_heads, 64], written in place.
        cu_seqlens_q: [num_reqs + 1] int32 query offsets.
        seq_lens: [num_reqs] int32 KV lengths including the new tokens.
        block_table: [num_reqs, max_blocks] int32.
        max_query_len: Longest query in the batch.
        max_seq_len: Longest KV length in the batch.
        q_descale: Per-tensor fp32 scale of q, one element.
        k_descale: Per-tensor fp32 scale of k_cache, one element.
        v_descale: Per-tensor fp32 scale of v_cache, one element.
        softmax_scale: Scale applied to q.k.

    Returns:
        out.

    """
    num_tokens, num_heads, head_size = q.shape
    num_kv_heads = k_cache.shape[2]
    gqa = num_heads // num_kv_heads
    block_t = _BLOCK_M // gqa
    q_blocks = triton.cdiv(max_query_len, block_t)
    nsplit = _num_splits(q_blocks * seq_lens.numel() * num_kv_heads, max_seq_len)
    if nsplit > 1:
        part_o = torch.empty(
            nsplit,
            num_tokens,
            num_heads,
            head_size,
            device=q.device,
            dtype=torch.float32,
        )
        part_l = torch.empty(
            nsplit, num_tokens, num_heads, device=q.device, dtype=torch.float32
        )
    else:
        part_o = part_l = out
    grid = (q_blocks, seq_lens.numel(), num_kv_heads * nsplit)
    _fp8_paged_extend_kernel[grid](
        out,
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        cu_seqlens_q,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        block_table.stride(0),
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        part_o,
        part_l,
        num_kv_heads,
        num_tokens,
        GQA=gqa,
        D=head_size,
        PAGE=k_cache.shape[1],
        BLOCK_T=block_t,
        BLOCK_N=_BLOCK_N,
        P_SHIFT=_P_SHIFT,
        L_COLS=_L_COLS,
        NSPLIT=nsplit,
        num_warps=8,
        num_stages=3,
        waves_per_eu=2,
        matrix_instr_nonkdim=32,
    )
    if nsplit > 1:
        _merge_splits_kernel[(num_tokens,)](
            out,
            part_o,
            part_l,
            v_descale,
            out.stride(0),
            out.stride(1),
            num_tokens,
            H=num_heads,
            D=head_size,
            NSPLIT=nsplit,
            num_warps=4,
        )
    return out
