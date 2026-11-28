# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""4-bit (e2m1) KV cache attention for the large-head global layers of Gemma4 on ROCm gfx950.

Cache format (per token and KV head): K and V rows of ``head_size // 2`` bytes each. Byte ``j`` holds
channel ``j`` in its low nibble and channel ``j + head_size // 2`` in its high nibble (the two neox
halves of the rotated key). A nibble is an OCP e2m1 code (sign, 2-bit exponent, 1-bit mantissa) of
``x / scale`` with one static scale per tensor (``k_scale`` for K, ``v_scale`` for V): K is the output of
a norm with a learned weight and V of a weightless RMSNorm, so their per-channel magnitudes are tightly
bounded and no per-block scales are stored (exactly 0.5 B per element).

Decode reads the cache with the scaled MFMA (`tl.dot_scaled`, e4m3 x e2m1): S = Q'K^T takes the packed K
bytes as the K-packed operand; P V takes the packed V bytes as the K-packed operand of two dots whose
left operand holds P in the even / odd slots of a zero-interleaved K axis, one per nibble. Prefill-style
batches dequantise the blocks they reference into an fp8 scratch pool and use the regular kernels.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _kv4_decode_3d(
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,
    kc_ptr,
    vc_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    q_descale_ptr,
    qk_scale_in,
    out_factor,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    sk0: tl.int64,
    sk1: tl.constexpr,
    sk2: tl.int64,
    sv0: tl.int64,
    sv2: tl.int64,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    HALF: tl.constexpr = HEAD_SIZE // 2
    TILES_PER_BLOCK: tl.constexpr = BLOCK_SIZE // TILE_SIZE
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)
    qk_scale = qk_scale_in * 1.4426950408889634 * tl.load(q_descale_ptr)
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    tiles_per_segment = tl.cdiv(seq_len, NUM_SEGMENTS * TILE_SIZE)
    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_SIZE)
    offs_j = tl.arange(0, HALF)
    offs_t = tl.arange(0, TILE_SIZE)
    qmask = offs_m < num_queries_per_kv
    qh = kv_head_idx * num_queries_per_kv + offs_m
    # K axis of the packed dot is (byte j, nibble b) = channel j + HALF * b
    ch = offs_k // 2 + HALF * (offs_k % 2)
    Q = tl.load(
        query_ptr + seq_idx * query_stride_0 + qh[:, None] * query_stride_1 + ch[None, :],
        mask=qmask[:, None],
        other=0.0,
    )
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc0 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_M, HALF], dtype=tl.float32)
    num_tiles = tl.cdiv(seq_len, TILE_SIZE)
    bt_off = seq_idx * block_table_stride
    zeros8 = tl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.float32).to(tl.float8e4nv)
    seg_lo = segm_idx * tiles_per_segment
    seg_hi = min((segm_idx + 1) * tiles_per_segment, num_tiles)
    n_full = seq_len // TILE_SIZE
    k_t = offs_t[:, None] * sk1 + offs_j[None, :]
    for j in range(seg_lo, min(seg_hi, n_full)):
        pb = tl.load(block_tables_ptr + bt_off + j // TILES_PER_BLOCK).to(tl.int64)
        t0 = (j % TILES_PER_BLOCK) * TILE_SIZE * sk1
        kb = kc_ptr + pb * sk0 + kv_head_idx * sk2 + t0
        vb = vc_ptr + pb * sv0 + kv_head_idx * sv2 + t0
        Kb = tl.trans(tl.load(kb + k_t, cache_modifier=".cg"))
        S = qk_scale * tl.dot_scaled(Q, None, "e4m3", Kb, None, "e2m1")
        S = tl.where(qmask[:, None], S, float("-inf"))
        m_j = tl.maximum(M, tl.max(S, axis=1))
        P = tl.math.exp2(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.math.exp2(M - m_j)
        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        Vb = tl.load(vb + k_t, cache_modifier=".cg")
        P8 = P.to(tl.float8e4nv)
        A0 = tl.reshape(tl.join(P8, zeros8), [BLOCK_M, 2 * TILE_SIZE])
        A1 = tl.reshape(tl.join(zeros8, P8), [BLOCK_M, 2 * TILE_SIZE])
        acc0 = tl.dot_scaled(A0, None, "e4m3", Vb, None, "e2m1", acc=acc0)
        acc1 = tl.dot_scaled(A1, None, "e4m3", Vb, None, "e2m1", acc=acc1)
    if seg_hi > n_full and seg_lo <= n_full:
        seq_offset = n_full * TILE_SIZE + offs_t
        tile_mask = seq_offset < seq_len
        pb = tl.load(block_tables_ptr + bt_off + n_full // TILES_PER_BLOCK).to(tl.int64)
        t0 = (n_full % TILES_PER_BLOCK) * TILE_SIZE * sk1
        kb = kc_ptr + pb * sk0 + kv_head_idx * sk2 + t0
        vb = vc_ptr + pb * sv0 + kv_head_idx * sv2 + t0
        Kb = tl.trans(tl.load(kb + k_t, mask=tile_mask[:, None], other=0, cache_modifier=".cg"))
        S = qk_scale * tl.dot_scaled(Q, None, "e4m3", Kb, None, "e2m1")
        S = tl.where(qmask[:, None] & tile_mask[None, :], S, float("-inf"))
        m_j = tl.maximum(M, tl.max(S, axis=1))
        P = tl.math.exp2(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.math.exp2(M - m_j)
        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        Vb = tl.load(vb + k_t, mask=tile_mask[:, None], other=0, cache_modifier=".cg")
        P8 = P.to(tl.float8e4nv)
        A0 = tl.reshape(tl.join(P8, zeros8), [BLOCK_M, 2 * TILE_SIZE])
        A1 = tl.reshape(tl.join(zeros8, P8), [BLOCK_M, 2 * TILE_SIZE])
        acc0 = tl.dot_scaled(A0, None, "e4m3", Vb, None, "e2m1", acc=acc0)
        acc1 = tl.dot_scaled(A1, None, "e4m3", Vb, None, "e2m1", acc=acc1)
    acc0 = acc0 * out_factor
    acc1 = acc1 * out_factor
    base = (
        seq_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS * HEAD_SIZE)
        + qh[:, None] * (NUM_SEGMENTS * HEAD_SIZE)
        + segm_idx * HEAD_SIZE
    )
    tl.store(segm_output_ptr + base + offs_j[None, :], acc0, mask=qmask[:, None])
    tl.store(segm_output_ptr + base + HALF + offs_j[None, :], acc1, mask=qmask[:, None])
    so = seq_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS) + qh * NUM_SEGMENTS + segm_idx
    tl.store(segm_max_ptr + so, M, mask=qmask)
    tl.store(segm_expsum_ptr + so, L, mask=qmask)


_SCALE_TENSORS: dict = {}


def scale_tensor(value: float, device) -> torch.Tensor:
    """Device scalar for a static scale; create it before CUDA graph capture (first use outside capture)."""
    device = torch.device(device)
    key = (value, device.type, device.index if device.index is not None else torch.cuda.current_device())
    t = _SCALE_TENSORS.get(key)
    if t is None:
        t = torch.tensor(value, dtype=torch.float32, device=device)
        _SCALE_TENSORS[key] = t
    return t


TILE = 64


def _pick_num_segments(num_seqs: int, num_kv_heads: int) -> int:
    n = -(-2048 // max(1, num_seqs * num_kv_heads))
    n = 1 << max(0, (n - 1).bit_length())
    return min(max(n, 4), 64)


def kv4_decode_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float,
    q_descale: torch.Tensor,
    k_scale: float,
    v_scale: float,
    num_segments: int | None = None,
) -> None:
    """Decode attention (one query token per sequence) over the 4-bit cache.

    q: [S, Hq, D] fp8; key_cache/value_cache: [B, bs, Hkv, D // 2] uint8 views, bs == 64;
    out: [S, Hq, D] (bf16/fp16).
    """
    from aiter.ops.triton._triton_kernels.attention.unified_attention import reduce_segments

    ns, nq, hd = q.shape
    nkv = key_cache.shape[2]
    bs = key_cache.shape[1]
    assert bs % TILE == 0 and hd == 512, (bs, hd)
    assert key_cache.stride(3) == 1 and value_cache.stride(3) == 1
    assert key_cache.stride(1) == value_cache.stride(1)
    nseg = num_segments or _pick_num_segments(ns, nkv)
    segm_output = torch.empty(ns, nq, nseg, hd, dtype=torch.float32, device=q.device)
    segm_max = torch.empty(ns, nq, nseg, dtype=torch.float32, device=q.device)
    segm_expsum = torch.empty(ns, nq, nseg, dtype=torch.float32, device=q.device)
    _kv4_decode_3d[(ns, nkv, nseg)](
        segm_output,
        segm_max,
        segm_expsum,
        q,
        key_cache,
        value_cache,
        block_table,
        seqused_k,
        q_descale,
        softmax_scale * k_scale,
        v_scale,
        block_table.stride(0),
        q.stride(0),
        q.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(2),
        nq,
        nq // nkv,
        TILE,
        hd,
        16,
        nseg,
        bs,
        num_warps=2,
        waves_per_eu=2,
        num_stages=1,
    )
    reduce_segments[(ns, nq)](
        output_ptr=out,
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        seq_lens_ptr=seqused_k,
        num_seqs=ns,
        num_query_heads=nq,
        out_scale_ptr=None,
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        block_table_stride=block_table.stride(0),
        HEAD_SIZE=hd,
        HEAD_SIZE_PADDED=hd,
        V_HEAD_SIZE=hd,
        V_HEAD_SIZE_PADDED=hd,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=None,
        NUM_SEGMENTS_PER_SEQ=nseg,
        TILE_SIZE=TILE,
        enable_fp_fusion=True,
        num_warps=4,
        waves_per_eu=2,
        num_stages=1,
    )


@triton.jit
def _kv4_to_fp8(x):
    # e2m1 nibble (int32, 0..15) -> e4m3 byte: value bits (c << 2) + 48, 0.5 (code 1) -> 48, 0 -> 0
    c = x & 7
    mag = tl.where(c < 2, c * 48, (c << 2) + 48)
    return mag | ((x & 8) << 4)


@triton.jit
def _kv4_dequant_blocks(
    src_ptr,
    dst_ptr,
    bt_ptr,
    seqlen_ptr,
    bt_stride: tl.int64,
    s0: tl.int64,
    s1: tl.int64,
    s2: tl.constexpr,
    d0: tl.int64,
    d1: tl.int64,
    d2: tl.constexpr,
    MAXB: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HALF: tl.constexpr,
):
    # scratch blocks hold 64 tokens; the source block holds BLOCK_SIZE (a multiple of 64)
    SB: tl.constexpr = 64
    seq = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    if b * SB >= tl.load(seqlen_ptr + seq):
        return
    pb = tl.load(bt_ptr + seq * bt_stride + (b * SB) // BLOCK_SIZE).to(tl.int64)
    offs_t = tl.arange(0, SB)
    offs_j = tl.arange(0, HALF)
    tok = (b * SB) % BLOCK_SIZE + offs_t
    src = src_ptr + pb * s0 + h * s1 + tok[:, None] * s2 + offs_j[None, :]
    dst = dst_ptr + (seq * MAXB + b).to(tl.int64) * d0 + h * d1 + offs_t[:, None] * d2 + offs_j[None, :]
    for part in range(2):  # K then V
        x = tl.load(src + part * HALF).to(tl.int32)
        lo = _kv4_to_fp8(x & 15).to(tl.uint8)
        hi = _kv4_to_fp8(x >> 4).to(tl.uint8)
        tl.store(dst + part * 2 * HALF, lo)
        tl.store(dst + part * 2 * HALF + HALF, hi)


def kv4_dequant_to_fp8(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dequantise the blocks of the given sequences into an fp8 scratch pool of 64-token blocks.

    kv_cache: the layer's [B, H, bs, D] uint8 cache (D bytes = K row + V row of D / 2 each, D = head_size).
    Returns (key, value, block_table) in the regular fp8 layout [N, 64, H, head_size] holding the raw
    e2m1 values (the kernels' k/v descales apply the static scales).
    """
    nb_total, nkv, bs, c = kv_cache.shape
    ns = seqused_k.shape[0]
    half = c // 2
    hd = c
    sb = 64
    max_blocks = -(-max_seq_len // sb)
    fp8 = current_platform.fp8_dtype()
    scratch = torch.empty((ns * max_blocks, nkv, sb, 2 * hd), dtype=torch.uint8, device=kv_cache.device)
    _kv4_dequant_blocks[(ns, max_blocks, nkv)](
        kv_cache,
        scratch,
        block_table,
        seqused_k,
        block_table.stride(0),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        scratch.stride(0),
        scratch.stride(1),
        scratch.stride(2),
        max_blocks,
        bs,
        half,
        num_warps=4,
    )
    scratch = scratch.view(fp8)
    k, v = scratch.transpose(1, 2).split(hd, dim=-1)
    return k, v, _scratch_block_table(ns, max_blocks, kv_cache.device)


_SCRATCH_BT: dict = {}


def _scratch_block_table(ns: int, max_blocks: int, device) -> torch.Tensor:
    key = (ns, max_blocks, device)
    t = _SCRATCH_BT.get(key)
    if t is None:
        t = torch.arange(ns * max_blocks, dtype=torch.int32, device=device).view(ns, max_blocks)
        _SCRATCH_BT[key] = t
    return t
