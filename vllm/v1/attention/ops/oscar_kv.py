# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OSCAR KV cache for the TurboQuant attention backend.

Presets ``turboquant_k3v3_oscar``, ``turboquant_k3v2_oscar`` and
``turboquant_k2v2_oscar``. Per head and token, K and V are each
Hadamard-rotated, clipped at a per-row |x| quantile and quantized to 2- or
3-bit asymmetric uniform codes, with a fp16 scale/min that also folds in a
row-norm correction. For head_dim 256 a slot is K3V3 200 B, K3V2 168 B,
K2V2 136 B (turboquant_4bit_nc: 262 B).

Vector record (rotated space), head_dim D, chunk width W = D / 4:
  [0:W]          low 2-bit plane   byte j, bits 2c..2c+1 = dim c*W + j
  [W:W+W/2]      high 1-bit plane  byte m, bit 2c+e      = dim c*W + 2m + e  (3-bit)
  [SO:SO+2]      scale fp16,  [SO+2:SO+4] min fp16       SO = W (2-bit), W + W/2 (3-bit)
Decode never dequantizes these tokens: codes (0..7) are exact in fp16, so
  score = scale * (q . code) + min * sum(q),   sum p*v = (p*scale) . code + sum(p*min)
run as four K=W tensor-core dots per tile.

Recent window: the last ``recent`` tokens of every request are attended at
high precision (int8 per token-head with a fp32 absmax scale). The quantized
copy is always written; tokens landing on a page inside some request's recent
window are also written to a staging row. Rows are assigned per *physical
page* by ``OscarWindow.update`` (metadata builder, GPU-only, no host sync), so
prefix-shared pages share staging and the pages of a finished turn stay
usable by the next turn until their row is recycled. A staged token is used
only if the page still owns the row AND the staged checksum equals the
checksum of the quantized K record (its scale/min word mixed with a hash of
its low code plane), so block reuse, KV offload restore or row recycling can
never surface stale values: they fall back to the quantized copy.
"""

import math
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton

try:
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover
    libdevice = None

DECODE_BLOCK_N = 64
# Upper bound on splits of the quantized range (workspace sizing). A launch
# uses splits_for(B, Hk), which only depends on the batch, so every captured
# CUDA graph keeps a constant grid.
DECODE_SPLITS = 48
DECODE_TARGET_PROGRAMS = 96
# Per-row |x| clip quantile by bit width (sweep on Ternary-Bonsai-2-27B K/V).
CLIP_K = {2: 0.96, 3: 0.98}
CLIP_V = {2: 0.92, 3: 0.96}
# Staging rows are sized for this many concurrent windows, and capped so all
# layers' staging takes at most this fraction of the free GPU memory seen when
# it is first allocated (the profiling run, so it is accounted for before the
# KV cache is sized). Pages beyond it fall back to their codes (accuracy only).
WINDOW_REQS = 8
WINDOW_MEMORY_FRACTION = 0.1
SUPPORTED_HEAD_DIMS = (128, 256)


def splits_for(rows: int, kv_heads: int) -> int:
    return max(1, min(DECODE_SPLITS, -(-DECODE_TARGET_PROGRAMS // (rows * kv_heads))))


def _scale_offset(bits: int, head_dim: int) -> int:
    w = head_dim // 4
    return w + w // 2 if bits == 3 else w


@dataclass(frozen=True)
class OscarFormat:
    kbits: int
    vbits: int
    recent: int
    head_dim: int = 256
    # The first ``sink`` tokens are attended from the window copy as well (attention
    # sinks; large effect on e.g. Qwen3, small on Ternary-Bonsai). One page.
    sink: int = 16

    @property
    def krec(self) -> int:
        return _scale_offset(self.kbits, self.head_dim) + 4

    @property
    def slot(self) -> int:
        return self.krec + _scale_offset(self.vbits, self.head_dim) + 4

    @property
    def window_tiles(self) -> int:
        """Decode programs over the window: recent tiles, plus one for the sink."""
        return self.recent // DECODE_BLOCK_N + (1 if self.sink else 0)

    def _need(self, clip: float) -> int:  # rank (from the top) of the clip threshold
        d = self.head_dim
        return d - min(d - 1, int(clip * d))

    @property
    def need_k(self) -> int:
        return self._need(CLIP_K[self.kbits])

    @property
    def need_v(self) -> int:
        return self._need(CLIP_V[self.vbits])

    @classmethod
    def of(cls, tq_config) -> "OscarFormat":
        return cls(
            tq_config.key_quant_bits,
            tq_config.value_quant_bits,
            tq_config.oscar_recent,
            tq_config.head_dim,
            tq_config.oscar_sink,
        )


# ---------------------------------------------------------------- helpers
@triton.jit
def _bfly(x, R: tl.constexpr, N: tl.constexpr, HS: tl.constexpr):
    y = tl.permute(tl.reshape(x, (R, N // (2 * HS), 2, HS)), (0, 1, 3, 2))
    a, c = tl.split(y)
    return tl.reshape(tl.permute(tl.join(a + c, a - c), (0, 1, 3, 2)), (R, N))


@triton.jit
def _fwht(x, R: tl.constexpr, D: tl.constexpr):
    """X (R, D) fp32 -> x @ H (Sylvester order, orthonormal; H = H^T = H^-1)."""
    x = _bfly(x, R, D, 1)
    x = _bfly(x, R, D, 2)
    x = _bfly(x, R, D, 4)
    x = _bfly(x, R, D, 8)
    x = _bfly(x, R, D, 16)
    x = _bfly(x, R, D, 32)
    x = _bfly(x, R, D, 64)
    if D == 256:
        x = _bfly(x, R, D, 128)
        x = x * 0.0625
    else:
        x = x * 0.08838834764831845  # 1 / sqrt(128)
    return x


@triton.jit
def _planes(base, offs_j, mask, BITS: tl.constexpr, W: tl.constexpr):
    """Records at base (BN,) -> low plane, high plane (byte j//2), scale, min."""
    SO: tl.constexpr = W + W // 2 if BITS == 3 else W
    lo = tl.load(base[:, None] + offs_j[None, :], mask=mask[:, None], other=0).to(
        tl.int32
    )
    if BITS == 3:
        hi = tl.load(
            base[:, None] + W + (offs_j // 2)[None, :], mask=mask[:, None], other=0
        ).to(tl.int32)
    else:
        hi = lo
    sc = tl.load(
        (base + SO).to(tl.pointer_type(tl.float16), bitcast=True), mask=mask, other=0
    )
    mn = tl.load(
        (base + SO + 2).to(tl.pointer_type(tl.float16), bitcast=True),
        mask=mask,
        other=0,
    )
    return lo, hi, sc.to(tl.float32), mn.to(tl.float32)


@triton.jit
def _code(lo, hi, offs_j, C: tl.constexpr, BITS: tl.constexpr):
    c = (lo >> (2 * C)) & 3
    if BITS == 3:
        c = c | (((hi >> (2 * C + offs_j % 2)[None, :]) & 1) << 2)
    return c.to(tl.float16)


@triton.jit
def _code_hash(lo, W: tl.constexpr):
    """int32 hash of a (R, W) low code plane (byte values), mod 2^32. The scale/min
    word alone is too clustered across rows to tell different content apart."""
    coef = (2 * tl.arange(0, W) + 1) * -1640531535  # odd multiples of 0x9E3779B1
    return tl.sum(lo * coef[None, :], 1)


@triton.jit
def _staged(
    base,
    blk,
    off,
    valid,
    PAGE_MAP,
    ROW_OWNER,
    CHK,
    s_chk_r,
    s_chk_h,
    h,
    KBITS: tl.constexpr,
    W: tl.constexpr,
):
    """Staging row of each token and whether its staged copy is trustworthy."""
    SO: tl.constexpr = W + W // 2 if KBITS == 3 else W
    row = tl.load(PAGE_MAP + blk, mask=valid, other=-1)
    has = valid & (row >= 0)
    owner = tl.load(ROW_OWNER + row, mask=has, other=-1)
    chk = tl.load(
        CHK + row.to(tl.int64) * s_chk_r + h * s_chk_h + off, mask=has, other=0
    )
    word = tl.load(
        (base + SO).to(tl.pointer_type(tl.int32), bitcast=True), mask=has, other=1
    )
    lo = tl.load(
        base[:, None] + tl.arange(0, W)[None, :], mask=has[:, None], other=0
    ).to(tl.int32)
    return row.to(tl.int64), has & (owner == blk) & (chk == (word ^ _code_hash(lo, W)))


# ---------------------------------------------------------------- store
@triton.jit
def _quant_vec(
    src, base, valid, NEED: tl.constexpr, BITS: tl.constexpr, D: tl.constexpr
):
    """Rotate, clip, quantize and pack one (1, D) row.

    Returns the rotated row (fp32) and the record checksum."""
    W: tl.constexpr = D // 4
    LV: tl.constexpr = 2**BITS - 1
    SO: tl.constexpr = W + W // 2 if BITS == 3 else W
    offs_d = tl.arange(0, D)
    x = tl.load(src + offs_d[None, :], mask=valid, other=0.0).to(tl.float32)
    xr = _fwht(x, 1, D)
    # Divisions use div_rn (IEEE round-to-nearest): the default approximate division
    # can flip a code at a .5 boundary, which makes results differ from a reference.
    # clip threshold = NEED-th largest |x| (ties counted, like a sort-based quantile)
    a = tl.abs(xr)
    cum = tl.zeros([1], tl.int32)
    thr = tl.zeros([1], tl.float32)
    done = tl.zeros([1], tl.int1)
    for _ in tl.static_range(NEED):
        m = tl.max(a, 1)
        eq = a == m[:, None]
        cnt = tl.sum(eq.to(tl.int32), 1)
        hit = (done == 0) & (cum + cnt >= NEED)
        thr = tl.where(hit, m, thr)
        done = done | hit
        cum += cnt
        a = tl.where(eq, -1.0, a)
    xc = tl.minimum(tl.maximum(xr, -thr[:, None]), thr[:, None])
    mn = tl.min(xc, 1)
    sc = libdevice.div_rn(tl.maximum(tl.max(xc, 1) - mn, 1e-8), LV * 1.0)
    q = tl.minimum(
        tl.maximum(
            libdevice.rint(libdevice.div_rn(xc - mn[:, None], sc[:, None])), 0.0
        ),
        LV * 1.0,
    )
    deq = q * sc[:, None] + mn[:, None]
    f = libdevice.div_rn(
        tl.sqrt(tl.sum(xr * xr, 1)), tl.maximum(tl.sqrt(tl.sum(deq * deq, 1)), 1e-8)
    )
    qi = q.to(tl.int32)
    lo = tl.sum(
        (tl.reshape(qi, (1, 4, W)) & 3) << (2 * tl.arange(0, 4))[None, :, None], 1
    )
    tl.store(base + tl.arange(0, W)[None, :], lo.to(tl.uint8), mask=valid)
    if BITS == 3:
        hsh = (
            2 * tl.arange(0, 4)[None, :, None, None]
            + tl.arange(0, 2)[None, None, None, :]
        )
        hi = tl.sum(tl.sum(((tl.reshape(qi, (1, 4, W // 2, 2)) >> 2) & 1) << hsh, 3), 1)
        tl.store(base + W + tl.arange(0, W // 2)[None, :], hi.to(tl.uint8), mask=valid)
    sc16 = (sc * f).to(tl.float16)
    mn16 = (mn * f).to(tl.float16)
    tl.store(
        (base + SO).to(tl.pointer_type(tl.float16), bitcast=True), sc16, mask=valid
    )
    tl.store(
        (base + SO + 2).to(tl.pointer_type(tl.float16), bitcast=True), mn16, mask=valid
    )
    word = (sc16.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF) | (
        mn16.to(tl.int16, bitcast=True).to(tl.int32) << 16
    )
    return xr, word ^ _code_hash(lo, W)


@triton.jit
def _oscar_store_kernel(
    K,
    V,
    SLOTMAP,
    KV,
    PAGE_MAP,
    ROW_OWNER,
    STK,
    STV,
    SSC,
    CHK,
    s_kn,
    s_kh,
    s_vn,
    s_vh,
    s_kvb,
    s_kvh,
    s_st_r,
    s_st_h,
    s_chk_r,
    s_chk_h,
    PAGE: tl.constexpr,
    NEED_K: tl.constexpr,
    NEED_V: tl.constexpr,
    KBITS: tl.constexpr,
    VBITS: tl.constexpr,
    KREC: tl.constexpr,
    S_POS: tl.constexpr,
    D: tl.constexpr,
):
    """Grid (N, Hk), one warp per token-head."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    slot = tl.load(SLOTMAP + t).to(tl.int64)
    valid = slot >= 0
    blk = slot // PAGE
    off = slot % PAGE
    base = KV + blk * s_kvb + h * s_kvh + off * S_POS + tl.zeros([1], tl.int64)
    kr, kw = _quant_vec(K + t * s_kn + h * s_kh, base, valid, NEED_K, KBITS, D)
    vr, _ = _quant_vec(V + t * s_vn + h * s_vh, base + KREC, valid, NEED_V, VBITS, D)
    row = tl.load(PAGE_MAP + blk, mask=valid, other=-1)
    stage = valid & (row >= 0)
    stage = stage & (tl.load(ROW_OWNER + row, mask=stage, other=-1) == blk)
    row = row.to(tl.int64)
    st = row * s_st_r + h * s_st_h + off * D + tl.arange(0, D)[None, :]
    ks = libdevice.div_rn(tl.maximum(tl.max(tl.abs(kr), 1), 1e-8), 127.0)
    vs = libdevice.div_rn(tl.maximum(tl.max(tl.abs(vr), 1), 1e-8), 127.0)
    tl.store(
        STK + st,
        libdevice.rint(libdevice.div_rn(kr, ks[:, None])).to(tl.int8),
        mask=stage,
    )
    tl.store(
        STV + st,
        libdevice.rint(libdevice.div_rn(vr, vs[:, None])).to(tl.int8),
        mask=stage,
    )
    sc = row * s_chk_r + h * s_chk_h + off + tl.zeros([1], tl.int64)
    tl.store(SSC + 2 * sc, ks, mask=stage)
    tl.store(SSC + 2 * sc + 1, vs, mask=stage)
    tl.store(CHK + sc, kw, mask=stage)


def cache_strides(kv_cache: torch.Tensor, slot: int) -> tuple[int, int, int]:
    """(page, head, position) byte strides of a uint8 cache that is either
    position-major (num_pages, page, Hk, slot) or head-major
    (num_pages, Hk, 1, page * slot)."""
    if kv_cache.shape[-1] == slot:
        return kv_cache.stride(0), kv_cache.stride(2), kv_cache.stride(1)
    return kv_cache.stride(0), kv_cache.stride(1), slot


def cache_page_size(kv_cache: torch.Tensor, slot: int) -> int:
    """Tokens per page of either cache layout (see cache_strides)."""
    return (
        kv_cache.shape[1] if kv_cache.shape[-1] == slot else kv_cache.shape[-1] // slot
    )


def oscar_store(key, value, kv_cache, slot_mapping, win, stage):
    """key/value (N, Hk, D) unrotated."""
    N, Hk = key.shape[0], key.shape[1]
    f = win.fmt
    stk, stv, ssc, chk = stage
    s_blk, s_head, s_pos = cache_strides(kv_cache, f.slot)
    _oscar_store_kernel[(N, Hk)](
        key,
        value,
        slot_mapping,
        kv_cache,
        win.page_map,
        win.row_owner,
        stk,
        stv,
        ssc,
        chk,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        s_blk,
        s_head,
        stk.stride(0),
        stk.stride(1),
        chk.stride(0),
        chk.stride(1),
        PAGE=win.page,
        NEED_K=f.need_k,
        NEED_V=f.need_v,
        KBITS=f.kbits,
        VBITS=f.vbits,
        KREC=f.krec,
        S_POS=s_pos,
        D=f.head_dim,
        num_warps=1,
    )


# ---------------------------------------------------------------- decode
@triton.jit
def _rot_q_kernel(Q, QR, s_qb, s_qh, HQ: tl.constexpr, D: tl.constexpr):
    i = tl.program_id(0)
    offs = tl.arange(0, D)[None, :]
    x = tl.load(Q + (i // HQ) * s_qb + (i % HQ) * s_qh + offs).to(tl.float32)
    tl.store(QR + i * D + offs, _fwht(x, 1, D).to(tl.float16))


@triton.jit
def _softmax_step(s, valid, m_i, l_i):
    s = tl.where(valid[None, :], s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    # a fully masked tile (short context) keeps m = -inf: subtract 0, not inf - inf
    m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp(m_i - m_use)
    p = tl.exp(s - m_use[:, None])
    return p, alpha, m_new, l_i * alpha + tl.sum(p, 1)


@triton.jit
def _wchunk(
    st_ptr,
    ss,
    ok,
    lo,
    hi,
    sc,
    mn,
    offs_j,
    C: tl.constexpr,
    BITS: tl.constexpr,
    W: tl.constexpr,
):
    """Window chunk C (BN, W) fp16: the staged copy, else the quantized value."""
    x = (
        tl.load(st_ptr + C * W, mask=ok[:, None], other=0).to(tl.float32) * ss[:, None]
    ).to(tl.float16)
    deq = (
        _code(lo, hi, offs_j, C, BITS).to(tl.float32) * sc[:, None] + mn[:, None]
    ).to(tl.float16)
    return tl.where(ok[:, None], x, deq)


@triton.jit
def _oscar_decode_stage1(
    QR,
    KV,
    BT,
    SEQ,
    PAGE_MAP,
    ROW_OWNER,
    STK,
    STV,
    SSC,
    CHK,
    MID,
    s_kvb,
    s_kvh,
    s_bt,
    s_st_r,
    s_st_h,
    s_chk_r,
    s_chk_h,
    s_mb,
    s_mh,
    s_ms,
    sm_scale,
    HQ: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    RECENT: tl.constexpr,
    SINK: tl.constexpr,
    KBITS: tl.constexpr,
    VBITS: tl.constexpr,
    KREC: tl.constexpr,
    S_POS: tl.constexpr,
    D: tl.constexpr,
):
    """Grid (B, Hk, NUM_SPLITS + window tiles). Splits cover the quantized range
    [min(SINK, r0), r0) with r0 = L - RECENT; the remaining programs cover the recent
    window one tile each, then (SINK > 0) one tile for the sink [0, min(SINK, r0))."""
    W: tl.constexpr = D // 4
    b = tl.program_id(0)
    hk = tl.program_id(1)
    sp = tl.program_id(2)
    L = tl.load(SEQ + b)
    offs_g = tl.arange(0, GP)
    offs_j = tl.arange(0, W)
    gmask = offs_g < G
    qp = QR + (b * HQ + hk * G + offs_g)[:, None] * D + offs_j[None, :]
    q0 = tl.load(qp, mask=gmask[:, None], other=0.0)
    q1 = tl.load(qp + W, mask=gmask[:, None], other=0.0)
    q2 = tl.load(qp + 2 * W, mask=gmask[:, None], other=0.0)
    q3 = tl.load(qp + 3 * W, mask=gmask[:, None], other=0.0)
    qsum = tl.sum(
        q0.to(tl.float32) + q1.to(tl.float32) + q2.to(tl.float32) + q3.to(tl.float32), 1
    )
    m_i = tl.full([GP], float("-inf"), tl.float32)
    l_i = tl.zeros([GP], tl.float32)
    a0 = tl.zeros([GP, W], tl.float32)
    a1 = tl.zeros([GP, W], tl.float32)
    a2 = tl.zeros([GP, W], tl.float32)
    a3 = tl.zeros([GP, W], tl.float32)
    am = tl.zeros([GP], tl.float32)
    kv_head = KV + hk * s_kvh
    r0 = tl.maximum(L - RECENT, 0)
    s0 = tl.minimum(SINK, r0)
    if sp < NUM_SPLITS:
        per = tl.cdiv(tl.cdiv(r0 - s0, NUM_SPLITS), BLOCK_N) * BLOCK_N
        start = s0 + sp * per
        end = tl.minimum(start + per, r0)
        for t0 in range(start, end, BLOCK_N):
            offs_n = t0 + tl.arange(0, BLOCK_N)
            nmask = offs_n < end
            blk = tl.load(BT + b * s_bt + offs_n // PAGE, mask=nmask, other=0).to(
                tl.int64
            )
            base = kv_head + blk * s_kvb + (offs_n % PAGE) * S_POS
            lo, hi, sc, mn = _planes(base, offs_j, nmask, KBITS, W)
            dq = tl.dot(q0, tl.trans(_code(lo, hi, offs_j, 0, KBITS)))
            dq = tl.dot(q1, tl.trans(_code(lo, hi, offs_j, 1, KBITS)), dq)
            dq = tl.dot(q2, tl.trans(_code(lo, hi, offs_j, 2, KBITS)), dq)
            dq = tl.dot(q3, tl.trans(_code(lo, hi, offs_j, 3, KBITS)), dq)
            s = (dq * sc[None, :] + qsum[:, None] * mn[None, :]) * sm_scale
            p, alpha, m_i, l_i = _softmax_step(s, nmask, m_i, l_i)
            lo, hi, sc, mn = _planes(base + KREC, offs_j, nmask, VBITS, W)
            ps = (p * sc[None, :]).to(tl.float16)
            a0 = tl.dot(ps, _code(lo, hi, offs_j, 0, VBITS), a0 * alpha[:, None])
            a1 = tl.dot(ps, _code(lo, hi, offs_j, 1, VBITS), a1 * alpha[:, None])
            a2 = tl.dot(ps, _code(lo, hi, offs_j, 2, VBITS), a2 * alpha[:, None])
            a3 = tl.dot(ps, _code(lo, hi, offs_j, 3, VBITS), a3 * alpha[:, None])
            am = am * alpha + tl.sum(p * mn[None, :], 1)
    else:
        tile = sp - NUM_SPLITS
        is_sink = tile >= RECENT // BLOCK_N
        pos = tl.where(is_sink, 0, r0 + tile * BLOCK_N) + tl.arange(0, BLOCK_N)
        wmask = pos < tl.where(is_sink, s0, L)
        blk = tl.load(BT + b * s_bt + pos // PAGE, mask=wmask, other=0).to(tl.int64)
        off = pos % PAGE
        base = kv_head + blk * s_kvb + off * S_POS
        row, ok = _staged(
            base,
            blk,
            off,
            wmask,
            PAGE_MAP,
            ROW_OWNER,
            CHK,
            s_chk_r,
            s_chk_h,
            hk,
            KBITS,
            W,
        )
        fb = wmask & (ok == 0)
        st = (row * s_st_r + hk * s_st_h + off * D)[:, None] + offs_j[None, :]
        sci = row * s_chk_r + hk * s_chk_h + off
        kss = tl.load(SSC + 2 * sci, mask=ok, other=0.0)
        vss = tl.load(SSC + 2 * sci + 1, mask=ok, other=0.0)
        lo, hi, sc, mn = _planes(base, offs_j, fb, KBITS, W)
        s = tl.dot(
            q0,
            tl.trans(_wchunk(STK + st, kss, ok, lo, hi, sc, mn, offs_j, 0, KBITS, W)),
        )
        s = tl.dot(
            q1,
            tl.trans(_wchunk(STK + st, kss, ok, lo, hi, sc, mn, offs_j, 1, KBITS, W)),
            s,
        )
        s = tl.dot(
            q2,
            tl.trans(_wchunk(STK + st, kss, ok, lo, hi, sc, mn, offs_j, 2, KBITS, W)),
            s,
        )
        s = tl.dot(
            q3,
            tl.trans(_wchunk(STK + st, kss, ok, lo, hi, sc, mn, offs_j, 3, KBITS, W)),
            s,
        )
        p, alpha, m_i, l_i = _softmax_step(s * sm_scale, wmask, m_i, l_i)
        lo, hi, sc, mn = _planes(base + KREC, offs_j, fb, VBITS, W)
        pp = p.to(tl.float16)
        a0 = tl.dot(pp, _wchunk(STV + st, vss, ok, lo, hi, sc, mn, offs_j, 0, VBITS, W))
        a1 = tl.dot(pp, _wchunk(STV + st, vss, ok, lo, hi, sc, mn, offs_j, 1, VBITS, W))
        a2 = tl.dot(pp, _wchunk(STV + st, vss, ok, lo, hi, sc, mn, offs_j, 2, VBITS, W))
        a3 = tl.dot(pp, _wchunk(STV + st, vss, ok, lo, hi, sc, mn, offs_j, 3, VBITS, W))
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    lse = tl.where(l_i > 0, m_i + tl.log(l_safe), float("-inf"))
    am = am[:, None]
    l_safe = l_safe[:, None]
    # Short contexts leave most splits/tiles empty: write only their lse (-inf).
    op = (
        MID + b * s_mb + (hk * G + offs_g)[:, None] * s_mh + sp * s_ms + offs_j[None, :]
    )
    om = gmask[:, None] & (l_i > 0)[:, None]
    tl.store(op, (a0 + am) / l_safe, mask=om)
    tl.store(op + W, (a1 + am) / l_safe, mask=om)
    tl.store(op + 2 * W, (a2 + am) / l_safe, mask=om)
    tl.store(op + 3 * W, (a3 + am) / l_safe, mask=om)
    tl.store(MID + b * s_mb + (hk * G + offs_g) * s_mh + sp * s_ms + D, lse, mask=gmask)


@triton.jit
def _oscar_decode_stage2(
    MID,
    OUT,
    s_mb,
    s_mh,
    s_ms,
    s_ob,
    s_oh,
    NS: tl.constexpr,
    NSP: tl.constexpr,
    D: tl.constexpr,
):
    """Merge split partials by LSE and rotate back. grid (B, Hq)."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, NSP)
    smask = offs_s < NS
    row = MID + b * s_mb + h * s_mh + offs_s * s_ms
    lse = tl.load(row + D, mask=smask, other=float("-inf"))
    w = tl.exp(lse - tl.max(lse, 0))
    w = w / tl.sum(w, 0)
    offs_d = tl.arange(0, D)
    o = tl.load(
        row[:, None] + offs_d[None, :],
        mask=(smask & (lse > float("-inf")))[:, None],
        other=0.0,
    )
    o = _fwht(tl.reshape(tl.sum(o * w[:, None], 0), (1, D)), 1, D)
    tl.store(OUT + b * s_ob + h * s_oh + offs_d[None, :], o.to(OUT.dtype.element_ty))


def oscar_decode(
    query, kv_cache, block_table, seq_lens, win, stage, scale, mid, qr, out
):
    """Query (B, Hq, D) unrotated; mid/qr/out from decode_workspace()."""
    B, Hq = query.shape[0], query.shape[1]
    stk, stv, ssc, chk = stage
    Hk = stk.shape[1]
    f = win.fmt
    D = f.head_dim
    s_blk, s_head, s_pos = cache_strides(kv_cache, f.slot)
    ns_q = splits_for(B, Hk)
    _rot_q_kernel[(B * Hq,)](
        query, qr, query.stride(0), query.stride(1), HQ=Hq, D=D, num_warps=1
    )
    _oscar_decode_stage1[(B, Hk, ns_q + f.window_tiles)](
        qr,
        kv_cache,
        block_table,
        seq_lens,
        win.page_map,
        win.row_owner,
        stk,
        stv,
        ssc,
        chk,
        mid,
        s_blk,
        s_head,
        block_table.stride(0),
        stk.stride(0),
        stk.stride(1),
        chk.stride(0),
        chk.stride(1),
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        scale,
        HQ=Hq,
        G=Hq // Hk,
        GP=max(16, triton.next_power_of_2(Hq // Hk)),
        PAGE=win.page,
        BLOCK_N=DECODE_BLOCK_N,
        NUM_SPLITS=ns_q,
        RECENT=f.recent,
        SINK=f.sink,
        KBITS=f.kbits,
        VBITS=f.vbits,
        KREC=f.krec,
        S_POS=s_pos,
        D=D,
        num_warps=4,
        num_stages=1,
    )
    ns = ns_q + f.window_tiles
    _oscar_decode_stage2[(B, Hq)](
        mid,
        out,
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        out.stride(0),
        out.stride(1),
        NS=ns,
        NSP=triton.next_power_of_2(ns),
        D=D,
        num_warps=4,
    )
    return out


# ------------------------------------------- dequant (continuation prefill)
@triton.jit
def _deq_rec(vb, offs, BITS: tl.constexpr, D: tl.constexpr):
    """One vector record -> (1, D) fp32, rotated space."""
    W: tl.constexpr = D // 4
    SO: tl.constexpr = W + W // 2 if BITS == 3 else W
    lo = tl.load(vb + (offs % W)).to(tl.int32)
    code = (lo >> (2 * (offs // W))) & 3
    if BITS == 3:
        hi = tl.load(vb + W + (offs % W) // 2).to(tl.int32)
        code = code | (((hi >> (2 * (offs // W) + offs % 2)) & 1) << 2)
    sc = tl.load((vb + SO).to(tl.pointer_type(tl.float16), bitcast=True)).to(tl.float32)
    mn = tl.load((vb + SO + 2).to(tl.pointer_type(tl.float16), bitcast=True)).to(
        tl.float32
    )
    return (code.to(tl.float32) * sc + mn)[None, :]


@triton.jit
def _oscar_dequant_kernel(
    KV,
    BT,
    PAGE_MAP,
    ROW_OWNER,
    STK,
    STV,
    SSC,
    CHK,
    KO,
    VO,
    s_kvb,
    s_kvh,
    s_st_r,
    s_st_h,
    s_chk_r,
    s_chk_h,
    s_oh,
    s_ot,
    PAGE: tl.constexpr,
    KBITS: tl.constexpr,
    VBITS: tl.constexpr,
    KREC: tl.constexpr,
    S_POS: tl.constexpr,
    D: tl.constexpr,
):
    """Grid (num_tokens, Hk): unrotated fp16 K/V of cached positions.

    Uses the staged copy where it is valid."""
    W: tl.constexpr = D // 4
    t = tl.program_id(0)
    h = tl.program_id(1)
    blk = tl.load(BT + t // PAGE).to(tl.int64)
    off = t % PAGE
    base = KV + blk * s_kvb + h * s_kvh + off * S_POS
    one = tl.full([1], 1, tl.int1)
    z = tl.zeros([1], tl.int64)
    row, ok = _staged(
        base + z,
        blk + z,
        off + z,
        one,
        PAGE_MAP,
        ROW_OWNER,
        CHK,
        s_chk_r,
        s_chk_h,
        h,
        KBITS,
        W,
    )
    offs = tl.arange(0, D)
    st = (row * s_st_r + h * s_st_h + off * D)[:, None] + offs[None, :]
    sci = row * s_chk_r + h * s_chk_h + off
    kss = tl.load(SSC + 2 * sci, mask=ok, other=0.0)
    vss = tl.load(SSC + 2 * sci + 1, mask=ok, other=0.0)
    xk = _deq_rec(base, offs, KBITS, D)
    xk = tl.where(
        ok[:, None],
        tl.load(STK + st, mask=ok[:, None], other=0).to(tl.float32) * kss[:, None],
        xk,
    )
    tl.store(KO + h * s_oh + t * s_ot + offs[None, :], _fwht(xk, 1, D).to(tl.float16))
    xv = _deq_rec(base + KREC, offs, VBITS, D)
    xv = tl.where(
        ok[:, None],
        tl.load(STV + st, mask=ok[:, None], other=0).to(tl.float32) * vss[:, None],
        xv,
    )
    tl.store(VO + h * s_oh + t * s_ot + offs[None, :], _fwht(xv, 1, D).to(tl.float16))


def oscar_dequant(kv_cache, block_table_row, num_tokens, win, stage, k_out, v_out):
    """k_out/v_out (Hk, >=num_tokens, D) fp16 views."""
    stk, stv, ssc, chk = stage
    Hk = stk.shape[1]
    f = win.fmt
    s_blk, s_head, s_pos = cache_strides(kv_cache, f.slot)
    _oscar_dequant_kernel[(num_tokens, Hk)](
        kv_cache,
        block_table_row,
        win.page_map,
        win.row_owner,
        stk,
        stv,
        ssc,
        chk,
        k_out,
        v_out,
        s_blk,
        s_head,
        stk.stride(0),
        stk.stride(1),
        chk.stride(0),
        chk.stride(1),
        k_out.stride(0),
        k_out.stride(1),
        PAGE=win.page,
        KBITS=f.kbits,
        VBITS=f.vbits,
        KREC=f.krec,
        S_POS=s_pos,
        D=f.head_dim,
        num_warps=1,
    )


# ---------------------------------------------------- recent-window page table
def pages_per_window(fmt: OscarFormat, page: int) -> int:
    return fmt.recent // page + 1 + -(-fmt.sink // page)


def window_rows(
    fmt: OscarFormat,
    page: int,
    max_reqs: int,
    num_layers: int,
    num_kv_heads: int,
    free_bytes: int,
) -> int:
    """Staging rows for one KV cache group (see WINDOW_REQS)."""
    ppw = pages_per_window(fmt, page)
    # int8 K and V, fp32 K/V scales, int32 checksum per token-head
    row_bytes = num_kv_heads * page * (2 * fmt.head_dim + 12)
    budget = int(WINDOW_MEMORY_FRACTION * free_bytes) // (num_layers * row_bytes)
    return max(ppw, min((min(max_reqs, WINDOW_REQS) + 2) * ppw, budget))


class OscarWindow:
    """Physical page -> staging row table, shared by one KV cache group's layers."""

    def __init__(self, fmt: OscarFormat, num_pages: int, page: int, rows: int, device):
        self.fmt = fmt
        self.page = page
        self.sink_pages = -(-fmt.sink // page)
        self.pages_per_req = pages_per_window(fmt, page)
        self.rows = rows
        # a trailing trash entry each: masked scatters need no boolean indexing
        # (which would sync with the host)
        self.page_map = torch.full(
            (num_pages + 1,), -1, dtype=torch.int32, device=device
        )
        self.row_owner = torch.full(
            (self.rows + 1,), -1, dtype=torch.int32, device=device
        )
        self.last_used = torch.zeros(self.rows + 1, dtype=torch.int64, device=device)
        self.step = torch.zeros(1, dtype=torch.int64, device=device)
        self._ar = torch.arange(self.pages_per_req - self.sink_pages, device=device)
        self._ar_sink = torch.arange(self.sink_pages, device=device)

    def alloc_stage(self, num_kv_heads: int, device):
        """One layer's (K int8, V int8, K/V scales fp32, checksum int32)."""
        shape = (self.rows, num_kv_heads, self.page, self.fmt.head_dim)
        return (
            torch.zeros(shape, dtype=torch.int8, device=device),
            torch.zeros(shape, dtype=torch.int8, device=device),
            torch.zeros((*shape[:3], 2), dtype=torch.float32, device=device),
            torch.zeros(shape[:3], dtype=torch.int32, device=device),
        )

    def update(self, block_table: torch.Tensor, seq_lens: torch.Tensor) -> None:
        """Map every page holding one of the first ``sink`` or last ``recent``
        tokens of each request to a staging row. seq_lens are post-step lengths,
        so this step's new tokens are covered too."""
        n_pages = self.page_map.shape[0] - 1
        trash_p, trash_r = n_pages, self.rows
        L = seq_lens.long()
        idx = ((L - self.fmt.recent).clamp(min=0) // self.page)[:, None] + self._ar[
            None, :
        ]
        idx = torch.cat([idx, self._ar_sink[None, :].expand(L.shape[0], -1)], 1)
        valid = (idx * self.page < L[:, None]) & (idx < block_table.shape[1])
        blk = torch.gather(
            block_table, 1, idx.clamp(max=block_table.shape[1] - 1)
        ).long()
        valid &= (blk >= 0) & (blk < n_pages)
        blk = torch.where(valid, blk, trash_p).flatten()
        valid = valid.flatten()
        row = self.page_map[blk].long()
        hit = valid & (row >= 0) & (self.row_owner[row.clamp(min=0)].long() == blk)
        self.step += 1
        self.last_used.index_put_(
            (torch.where(hit, row, trash_r),), self.step.expand(blk.shape[0])
        )
        # one new row per distinct missing page
        miss = valid & ~hit
        key = torch.where(miss, blk, n_pages + 1)
        skey, order = torch.sort(key)
        first_sorted = torch.ones_like(skey, dtype=torch.bool)
        first_sorted[1:] = skey[1:] != skey[:-1]
        first = torch.zeros_like(miss)
        first[order] = first_sorted
        first &= miss
        in_use = torch.zeros(self.rows + 1, dtype=torch.int64, device=blk.device)
        in_use.index_put_((torch.where(hit, row, trash_r),), torch.ones_like(row))
        # free rows first, least recently used first
        cand = torch.argsort(
            in_use[: self.rows] * (1 << 62) + self.last_used[: self.rows]
        )
        k = torch.cumsum(first.long(), 0) - 1
        # more missing pages than free rows (> WINDOW_REQS windows): rest unstaged
        first &= k < (in_use[: self.rows] == 0).sum()
        new_row = torch.where(first, cand[k.clamp(min=0, max=self.rows - 1)], trash_r)
        self.page_map.index_put_((torch.where(first, blk, trash_p),), new_row.int())
        self.row_owner.index_put_((new_row,), torch.where(first, blk, -1).int())
        self.last_used.index_put_((new_row,), self.step.expand(blk.shape[0]))


class WindowHolder:
    """One KV cache group's window table: registered by the builder, filled by
    the first KV write (which knows the page count)."""

    win: OscarWindow | None = None
    num_layers: int = 1


_WINDOWS: dict[str, WindowHolder] = {}


def get_window(layer_name: str) -> WindowHolder:
    return _WINDOWS[layer_name]


def register_window(layer_names, holder: WindowHolder) -> None:
    holder.num_layers = len(layer_names)
    for n in layer_names:
        _WINDOWS[n] = holder


def decode_workspace(
    fmt: OscarFormat, tokens: int, num_heads: int, num_kv_heads: int, out_dtype
):
    """Shapes for WorkspaceManager.get_simultaneous for a decode of ``tokens`` rows:
    split partials (o | lse), rotated q, output."""
    d = fmt.head_dim
    splits = splits_for(tokens, num_kv_heads) + fmt.window_tiles
    return (
        ((tokens, num_heads, splits, d + 1), torch.float32),
        ((tokens, num_heads, d), torch.float16),
        ((tokens, num_heads, d), out_dtype),
    )


def max_decode_workspace(
    fmt: OscarFormat, max_tokens: int, num_heads: int, num_kv_heads: int, out_dtype
):
    """decode_workspace() of the batch size that needs the most bytes (splits shrink
    as the batch grows, so it is not always the largest batch)."""

    def nbytes(specs):
        return sum(
            math.prod(shape) * torch.empty((), dtype=dt).element_size()
            for shape, dt in specs
        )

    return max(
        (
            decode_workspace(fmt, b, num_heads, num_kv_heads, out_dtype)
            for b in range(1, max_tokens + 1)
        ),
        key=nbytes,
    )
