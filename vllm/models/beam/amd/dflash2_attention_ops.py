# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-KV sliding-window attention for the Beam DFlash2 drafter on ROCm.

The drafter attends a few query tokens per request to a long paged context
under a (non-)causal sliding window. One program per (request, query block,
KV head, segment) walks a slice of the *visible* key range, with all query
heads that share the KV head packed into the MMA rows; a second kernel merges
the per-segment partials with log-sum-exp weights.

KV cache: logical ``[num_blocks, num_kv_heads, block_size, 2 * head_size]``
with K in ``[..., :head_size]`` and V in ``[..., head_size:]``; any stride
order whose innermost dimension is contiguous is accepted. The cache is the
query dtype or per-tensor-scaled fp8.
"""

import torch

from vllm.triton_utils import tl, triton

_MAX_BLOCK_M = 128
_MAX_SEGMENTS = 64
_TILE_N = 64


@triton.jit
def _dflash2_split_kv_partial(
    Q,
    KV,
    BLOCK_TABLE,
    QUERY_START_LOC,
    SEQ_LENS,
    PART_O,
    PART_M,
    PART_L,
    K_SCALE,
    V_SCALE,
    scale_log2,
    stride_qt,
    stride_qh,
    stride_kvb,
    stride_kvh,
    stride_kvn,
    stride_bt,
    num_q_blocks,
    num_heads,
    WINDOW: tl.constexpr,
    CAUSAL: tl.constexpr,
    QUERIES_PER_KV: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_PAD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    FP8_KV: tl.constexpr,
):
    seq = tl.program_id(0) // num_q_blocks
    q_block = tl.program_id(0) % num_q_blocks
    kv_head = tl.program_id(1)
    segment = tl.program_id(2)

    q_start = tl.load(QUERY_START_LOC + seq)
    q_len = tl.load(QUERY_START_LOC + seq + 1) - q_start
    first_tok = q_block * BLOCK_Q
    if first_tok >= q_len:
        return
    seq_len = tl.load(SEQ_LENS + seq)
    ctx_len = seq_len - q_len

    rows = tl.arange(0, BLOCK_M)
    tok = first_tok + rows // QUERIES_PER_KV
    head = kv_head * QUERIES_PER_KV + rows % QUERIES_PER_KV
    row_ok = (rows < BLOCK_Q * QUERIES_PER_KV) & (tok < q_len)
    d = tl.arange(0, HEAD_PAD)
    d_ok = d < HEAD_SIZE

    q_tok = (q_start + tok).to(tl.int64)
    q = tl.load(
        Q + q_tok[:, None] * stride_qt + head[:, None] * stride_qh + d[None, :],
        mask=row_ok[:, None] & d_ok[None, :],
        other=0.0,
    )
    q_pos = ctx_len + tok

    last_tok = tl.minimum(first_tok + BLOCK_Q, q_len) - 1
    lo = 0
    if WINDOW > 0:
        lo = tl.maximum(ctx_len + first_tok - WINDOW + 1, 0)
    hi = seq_len
    if CAUSAL:
        hi = ctx_len + last_tok + 1
    elif WINDOW > 0:
        hi = tl.minimum(seq_len, ctx_len + last_tok + WINDOW)
    num_tiles = tl.cdiv(tl.maximum(hi - lo, 0), TILE_N)
    tiles_per_segment = tl.cdiv(num_tiles, NUM_SEGMENTS)
    tile_begin = segment * tiles_per_segment
    tile_end = tl.minimum(tile_begin + tiles_per_segment, num_tiles)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_PAD], dtype=tl.float32)
    offs_n = tl.arange(0, TILE_N)
    bt_row = BLOCK_TABLE + seq.to(tl.int64) * stride_bt
    # fp8 K/V is exact in the query dtype: fold k_scale into the logits and
    # v_scale into the partial output.
    qk_scale = scale_log2
    if FP8_KV:
        qk_scale = scale_log2 * tl.load(K_SCALE)

    for t in range(tile_begin, tile_end):
        pos = lo + t * TILE_N + offs_n
        kv_ok = pos < hi
        block = tl.load(bt_row + pos // PAGE_SIZE, mask=kv_ok, other=0).to(tl.int64)
        base = (
            block * stride_kvb
            + kv_head * stride_kvh
            + (pos % PAGE_SIZE).to(tl.int64) * stride_kvn
        )
        kv_mask = kv_ok[:, None] & d_ok[None, :]
        k = tl.load(KV + base[:, None] + d[None, :], mask=kv_mask, other=0.0)
        v = tl.load(
            KV + base[:, None] + HEAD_SIZE + d[None, :], mask=kv_mask, other=0.0
        )
        if FP8_KV:
            k = k.to(q.dtype)
            v = v.to(q.dtype)

        s = tl.dot(q, tl.trans(k)) * qk_scale
        dist = q_pos[:, None] - pos[None, :]
        ok = row_ok[:, None] & kv_ok[None, :]
        if CAUSAL:
            ok = ok & (dist >= 0)
        if WINDOW > 0:
            ok = ok & (dist < WINDOW)
            if not CAUSAL:
                ok = ok & (dist > -WINDOW)
        s = tl.where(ok, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(s - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    if FP8_KV:
        acc = acc * tl.load(V_SCALE)
    ml_off = (q_tok * num_heads + head) * NUM_SEGMENTS + segment
    tl.store(
        PART_O + ml_off[:, None] * HEAD_PAD + d[None, :],
        acc,
        mask=row_ok[:, None],
    )
    tl.store(PART_M + ml_off, m_i, mask=row_ok)
    tl.store(PART_L + ml_off, l_i, mask=row_ok)


@triton.jit
def _dflash2_split_kv_merge(
    OUT,
    PART_O,
    PART_M,
    PART_L,
    QUERY_START_LOC,
    num_seqs,
    stride_ot,
    stride_oh,
    num_heads,
    NUM_SEGMENTS: tl.constexpr,
    SEGMENTS_PAD: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_PAD: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    if tok >= tl.load(QUERY_START_LOC + num_seqs):
        return

    seg = tl.arange(0, SEGMENTS_PAD)
    seg_ok = seg < NUM_SEGMENTS
    d = tl.arange(0, HEAD_PAD)
    base = (tok.to(tl.int64) * num_heads + head) * NUM_SEGMENTS
    m_s = tl.load(PART_M + base + seg, mask=seg_ok, other=float("-inf"))
    l_s = tl.load(PART_L + base + seg, mask=seg_ok, other=0.0)
    live = l_s > 0.0
    m_max = tl.max(tl.where(live, m_s, float("-inf")), 0)
    m_max = tl.where(m_max == float("-inf"), 0.0, m_max)
    w = tl.where(live, tl.exp2(m_s - m_max), 0.0)
    l_sum = tl.sum(l_s * w, 0)
    o_s = tl.load(
        PART_O + (base + seg)[:, None] * HEAD_PAD + d[None, :],
        mask=live[:, None],
        other=0.0,
    )
    o = tl.sum(o_s * w[:, None], 0)
    o = tl.where(l_sum > 0.0, o / l_sum, 0.0)
    tl.store(
        OUT + tok.to(tl.int64) * stride_ot + head * stride_oh + d,
        o.to(OUT.dtype.element_ty),
        mask=d < HEAD_SIZE,
    )


def split_kv_workspace(
    max_rows: int, num_heads: int, head_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 partial (O, m, l) buffers for up to ``max_rows`` (token,
    segment) pairs of ``num_heads`` heads."""
    head_pad = triton.next_power_of_2(head_size)
    part_o = torch.empty(
        max_rows * num_heads * head_pad, dtype=torch.float32, device=device
    )
    part_m = torch.empty(max_rows * num_heads, dtype=torch.float32, device=device)
    part_l = torch.empty_like(part_m)
    return part_o, part_m, part_l


def choose_num_segments(
    num_seqs: int,
    num_q_blocks: int,
    num_kv_heads: int,
    num_tokens: int,
    max_rows: int,
    target_programs: int,
) -> int:
    """Smallest power-of-two segment count that fills ``target_programs``
    partial programs, capped by the workspace."""
    base = max(1, num_seqs * num_q_blocks * num_kv_heads)
    segments = 1
    while segments < _MAX_SEGMENTS and base * segments * 2 <= target_programs:
        segments *= 2
    while segments > 1 and num_tokens * segments > max_rows:
        segments //= 2
    return segments


def dflash2_split_kv_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_seqs: int,
    max_query_len: int,
    scale: float,
    sliding_window: int,
    causal: bool,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    target_programs: int,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """``query``/``output``: ``[num_tokens, num_heads, head_size]``.

    An fp8 ``kv_cache`` holds ``K / k_scale`` and ``V / v_scale``; the scales
    are one-element fp32 tensors.
    """
    num_tokens, num_heads, head_size = query.shape
    num_blocks, num_kv_heads, page_size, kv_dim = kv_cache.shape
    assert kv_dim == 2 * head_size and kv_cache.stride(-1) == 1
    fp8_kv = kv_cache.dtype != query.dtype
    if fp8_kv:
        assert kv_cache.dtype.itemsize == 1 and kv_cache.dtype.is_floating_point
        assert k_scale is not None and v_scale is not None
    assert query.stride(-1) == 1 and output.stride(-1) == 1
    assert num_heads % num_kv_heads == 0
    queries_per_kv = num_heads // num_kv_heads
    assert queries_per_kv <= _MAX_BLOCK_M
    if num_tokens == 0 or num_seqs == 0:
        return

    part_o, part_m, part_l = workspace
    head_pad = triton.next_power_of_2(head_size)
    max_rows = part_m.numel() // num_heads
    assert num_tokens <= max_rows, "DFlash2 split-KV workspace too small"

    block_m = min(
        _MAX_BLOCK_M, max(16, triton.next_power_of_2(max_query_len * queries_per_kv))
    )
    block_q = block_m // queries_per_kv
    num_q_blocks = triton.cdiv(max_query_len, block_q)
    num_segments = choose_num_segments(
        num_seqs, num_q_blocks, num_kv_heads, num_tokens, max_rows, target_programs
    )

    _dflash2_split_kv_partial[(num_seqs * num_q_blocks, num_kv_heads, num_segments)](
        query,
        kv_cache,
        block_table,
        query_start_loc,
        seq_lens,
        part_o,
        part_m,
        part_l,
        k_scale if fp8_kv else part_l,
        v_scale if fp8_kv else part_l,
        scale * 1.4426950408889634,
        query.stride(0),
        query.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        num_q_blocks,
        num_heads,
        WINDOW=sliding_window if sliding_window is not None else 0,
        CAUSAL=causal,
        QUERIES_PER_KV=queries_per_kv,
        HEAD_SIZE=head_size,
        HEAD_PAD=head_pad,
        PAGE_SIZE=page_size,
        BLOCK_M=block_m,
        BLOCK_Q=block_q,
        TILE_N=_TILE_N,
        NUM_SEGMENTS=num_segments,
        FP8_KV=fp8_kv,
        num_warps=4,
        num_stages=1,
    )
    _dflash2_split_kv_merge[(num_tokens, num_heads)](
        output,
        part_o,
        part_m,
        part_l,
        query_start_loc,
        num_seqs,
        output.stride(0),
        output.stride(1),
        num_heads,
        NUM_SEGMENTS=num_segments,
        SEGMENTS_PAD=max(2, triton.next_power_of_2(num_segments)),
        HEAD_SIZE=head_size,
        HEAD_PAD=head_pad,
        num_warps=1,
    )
