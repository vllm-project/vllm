# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M1c ``mla_attn``: Kimi-K3's MLA decode attention at TP8 + DCP1, the bf16
query (M1a ``qbf16``) against the fp8 latent cache, split over the KV: one pass
a split, every split's partial (bf16 o, f32 LSE) for a combine (M1b's).

One launch, ``BLOCKS`` CTAs x ``THREADS`` threads. A CTA a (head group, split):
16 query rows (``16 / qlen`` heads x qlen positions) against a chunk of the
request's KV. The split's CTAs (every head group) share an XCD, so a chunk
streams from HBM once into that XCD's L2.

    block of 128 tokens (wave w its tile of 16, the waves in step)
        K tile -> registers (a lane 144 B of a token: the MFMA's A, the K order
        permuted alike in Q's B), S^T = K Q^T (18 bf16 MFMAs); V -> LDS
        the block's row max over the waves (LDS) -> P = exp2(S - m) bf16 ->
        LDS; the running O, l rescaled (online softmax)
        O^T += V^T P^T: wave w its 64 latents, V^T from LDS (a 4x4 byte
        transpose a dword), P^T as the softmax left it (a lane its row)
    end: l summed over the waves -> o = O / l (bf16), LSE = m + log l

A split with no token of the request (or none before its causal bound) leaves
LSE -inf and o 0. Rows are causal: position i of the request's qlen sees its
KV up to ``seq_len - qlen + i``.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import LOG2E
from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import (
    ENTRY,
    KVL,
    NH,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ops import (
    _hw_f32,
    bf16_pair,
    butterfly,
    fp8x8_bf16_pk,
    global_load,
    hw_exp2,
    hw_rcp,
    ld_i32,
    memrealtime,
    mfma_bf16,
    rsrc,
    traced,
    uniform,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    key_tuple,
    source_digest,
)

XCDS = 8
TILE = 16  # tokens a wave's tile
BLK = TILE * WAVES  # tokens a block
CHUNK = 64  # bytes of a token a K load reads (lane group g: CHUNK p + 16 g)
KLOADS = ENTRY // CHUNK  # 9: a row's 4 lanes 64 contiguous bytes a load (one
#   128 B line, not four)
KSTEPS = ENTRY // 32  # 18 MFMA steps of QK
VROW = KVL // 4 + 4  # an LDS V row (dwords): 528 B, the lane groups' reads
#   (rows 4 g ..) 16 banks apart
VCOLS = KVL // WAVES  # 64 latents a wave
LN2 = 0.6931471805599453
NEG = -1.0e30  # the running max before any token (finite: no inf - inf)

assert ENTRY % CHUNK == 0 and KVL % CHUNK == 0 and KVL // 4 % 16 == 0
assert VCOLS == 64 and THREADS == WAVES * 64

_STREAM = fx.Stream(None)
_SOURCES = source_digest(".")


def splits(qlen: int) -> int:
    """KV splits of the launch (every request's together): the head groups'
    CTAs an XCD hold a whole number of splits."""
    hg = NH // (TILE // qlen)
    return XCDS * (BLOCKS // XCDS // hg)


def geometry(qlen: int):
    """(heads a CTA, head groups, splits an XCD)."""
    ht = TILE // qlen
    hg = NH // ht
    return ht, hg, BLOCKS // XCDS // hg


def ctas(qlen: int) -> int:
    """The working CTAs (0 .. ctas): a done flag each when fused (``cta``'s
    ``done``), at (split, head group) ``split * hg + group``."""
    _, hg, spx = geometry(qlen)
    return XCDS * spx * hg


@dataclass(frozen=True)
class MlaAttnBuild:
    tokens: int  # S = requests x qlen
    qlen: int  # 4 or 8
    stamps: bool = False  # debug: phase timestamps (memrealtime) over part_o
    dbg: int = 0  # debug: 1 no K loads in the loop (block 0's K throughout)


def perm(a, b, sel):
    """v_perm_b32: byte k of the result is byte sel_k of (a: 7..4, b: 3..0)."""
    return fx.Int32(
        llvm.call_intrinsic(
            T.i32,
            "llvm.amdgcn.perm",
            [fx.Int32(a).ir_value(), fx.Int32(b).ir_value(), fx.Int32(sel).ir_value()],
            [],
            [],
        )
    )


def transpose4(d):
    """Four dwords (row j: bytes of columns 0..3) -> four (column i: bytes of
    rows 0..3)."""
    lo01 = perm(d[1], d[0], 0x05010400)
    hi01 = perm(d[1], d[0], 0x07030602)
    lo23 = perm(d[3], d[2], 0x05010400)
    hi23 = perm(d[3], d[2], 0x07030602)
    return [
        perm(lo23, lo01, 0x05040100),
        perm(lo23, lo01, 0x07060302),
        perm(hi23, hi01, 0x05040100),
        perm(hi23, hi01, 0x07060302),
    ]


@traced
def stamp(c, slot):
    # a trace-time if around a device one (not one condition)
    if const_expr(c["stamps"]):  # noqa: SIM102
        if (c["wave"] == 0) & (c["lane"] == 0):
            bo.buffer_store(
                fx.Int32(memrealtime() & 0x7FFFFFFF),
                rsrc(c["args"]["part_o"]),
                c["bid"] * 64 + fx.min(slot, 63),
            )


def hw_log2(x):
    return _hw_f32("llvm.amdgcn.log.f32", x)


def idx_load(c, b):
    """The page index of this lane's token (``lane % 16``) of block b's tile of
    this wave, clamped into the chunk (a block past the last loads its rows)."""
    tok = c["c0"] + (b * WAVES + c["wave"]) * TILE + c["lane"] % TILE
    return ld_i32(c["args"]["pages"], c["p0"] + fx.min(tok, c["last"]))


def k_loads(c, idx):
    """A K tile's lane share of token row ``idx``: 16 B at 64 p + 16 g, p < 9."""
    a = c["args"]
    base = a["kv"] + fx.Int64(idx) * ENTRY + fx.Int64((c["lane"] // TILE) * 16)
    return [fx.Vector(global_load(base + CHUNK * p, 4)) for p in range(KLOADS)]


@traced
def stage_block(c, b, kw, st):
    """One block (``kw`` this wave's K tile) into the running (m, l, O)."""
    lane, wave, g = c["lane"], c["wave"], c["lane"] // TILE
    row = lane % TILE
    m_run, l_run, o = st
    # S^T: token (4 g + j) x row (lane % 16)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    words = [kw[p][e] for p in range(KLOADS) for e in range(4)]
    for s in range_constexpr(KSTEPS):
        ka = fp8x8_bf16_pk(words[2 * s : 2 * s + 2])
        acc = mfma_bf16(ka, c["q"][s], acc)
    # V (bytes < 512: loads p < 8) -> this wave's LDS tile, row lane % 16
    vrow = c["vt"] + (wave * TILE + row) * VROW + 4 * g
    for p in range_constexpr(KVL // CHUNK):
        fx.ptr_store(kw[p], vrow + CHUNK // 4 * p)
    # scores (log2 units), causal + chunk mask
    t0 = c["c0"] + (b * WAVES + wave) * TILE + 4 * g
    sc = []
    for j in range_constexpr(4):
        ok = (t0 + j <= c["last"]) & (t0 + j <= c["bound"])
        sc.append(ok.select(acc[j] * c["scale"], fx.Float32(float("-inf"))))
    mx = fx.max(fx.max(sc[0], sc[1]), fx.max(sc[2], sc[3]))
    mx = butterfly(mx, (32, 16), fx.max)
    if lane < TILE:
        fx.ptr_store(mx, c["xm"] + wave * TILE + row)
    gpu.barrier()
    stamp(c, 2 + 3 * b + 0)
    mb = fx.ptr_load(c["xm"] + row)
    for w in range_constexpr(1, WAVES):
        mb = fx.max(mb, fx.ptr_load(c["xm"] + w * TILE + row))
    m_new = fx.max(m_run, mb)
    alpha = hw_exp2(m_run - m_new)
    p = [hw_exp2(sc[j] - m_new) for j in range(4)]
    ps = butterfly((p[0] + p[1]) + (p[2] + p[3]), (32, 16))
    l_run = l_run * alpha + ps
    pw = [bf16_pair(p[0], p[1]), bf16_pair(p[2], p[3])]
    fx.ptr_store(
        fx.Vector.from_elements(pw, fx.Float32).bitcast(fx.Int32),
        c["xp"] + (wave * 64 + lane) * 2,
    )
    gpu.barrier()
    stamp(c, 2 + 3 * b + 1)
    # O^T += V^T P^T, two tiles an MFMA (k: tile 2 a's tokens 4 g .., then
    # tile 2 a + 1's): latents 64 wave + 4 (lane % 16) + i of N-tile i
    o = [
        fx.Vector.from_elements([o[i][j] * alpha for j in range(4)], fx.Float32)
        for i in range(4)
    ]
    for a2 in range_constexpr(WAVES // 2):
        pb = [
            fx.Vector(
                fx.ptr_load(
                    c["xp"] + ((2 * a2 + u) * 64 + lane) * 2,
                    result_type=fx.Vector.make_type(2, fx.Int32),
                )
            )
            for u in range(2)
        ]
        pv = fx.Vector.from_elements(
            [pb[0][0], pb[0][1], pb[1][0], pb[1][1]], fx.Int32
        ).bitcast(fx.BFloat16)
        tr = []
        for u in range_constexpr(2):
            vb = c["vt"] + ((2 * a2 + u) * TILE + 4 * g) * VROW + wave * 16 + row
            tr.append(transpose4([fx.ptr_load(vb + j * VROW) for j in range(4)]))
        for i in range_constexpr(4):
            va = fp8x8_bf16_pk([tr[0][i], tr[1][i]])
            o[i] = mfma_bf16(va, pv, o[i])
    gpu.barrier()
    stamp(c, 2 + 3 * b + 2)
    return m_new, l_run, o


@traced
def run(c):
    lane, wave, g = c["lane"], c["wave"], c["lane"] // TILE
    row = lane % TILE
    m_run = fx.Float32(NEG)
    l_run = fx.Float32(0.0)
    o = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(4)]
    # K two blocks ahead, the indices three (a deeper ring in registers spills:
    # 256 VGPRs at 8 waves)
    stamp(c, 1)
    b = fx.Int32(0)
    k0 = k_loads(c, idx_load(c, 0))
    k1 = k_loads(c, idx_load(c, 1))
    i2 = idx_load(c, 2)
    if const_expr(c["dbg"] == 1):
        while b < c["nblk"]:
            m_run, l_run, o = stage_block(c, b, k0, (m_run, l_run, o))
            b = b + 1
    else:
        while b < c["nblk"]:
            k2 = k_loads(c, i2)
            i2 = idx_load(c, b + 3)
            m_run, l_run, o = stage_block(c, b, k0, (m_run, l_run, o))
            k0 = k1
            k1 = k2
            b = b + 1
    # l over the waves (each its tiles' sum, at the common running max)
    if lane < TILE:
        fx.ptr_store(l_run, c["xm"] + wave * TILE + row)
    gpu.barrier()
    lt = fx.ptr_load(c["xm"] + row)
    for w in range_constexpr(1, WAVES):
        lt = lt + fx.ptr_load(c["xm"] + w * TILE + row)
    a = c["args"]
    empty = lt == fx.Float32(0.0)
    inv = empty.select(fx.Float32(0.0), hw_rcp(lt) * c["vscale"])
    # lane: latents 64 wave + 16 g + (4 j + i) of its row
    words = [
        bf16_pair(o[i][j] * inv, o[i + 1][j] * inv) for j in range(4) for i in (0, 2)
    ]
    at = ((c["split"] * c["S"] + c["t"]) * NH + c["h"]) * KVL + wave * VCOLS + 16 * g
    for half in range_constexpr(2):
        v = fx.Vector.from_elements(words[4 * half : 4 * half + 4], fx.Float32)
        if const_expr(not c["stamps"]):
            bo.buffer_store(
                v, rsrc(a["part_o"]), at // 2 + 4 * half, cache_modifier=c["cm"]
            )
    stamp(c, 63)
    if (wave == 0) & (lane < TILE):
        lse = m_run + hw_log2(lt)
        lse = empty.select(fx.Float32(float("-inf")), lse * fx.Float32(LN2))
        bo.buffer_store(
            lse,
            rsrc(a["part_lse"]),
            (c["split"] * c["S"] + c["t"]) * NH + c["h"],
            cache_modifier=c["cm"],
        )


def check(s: int, q: int):
    assert q in (4, 8) and s % q == 0
    assert splits(q) % (s // q) == 0, (splits(q), s // q)


def smem():
    """The LDS of a CTA (``cta``'s vt / xm / xp)."""

    @fx.struct
    class Smem:
        vt: fx.Array[fx.Int32, WAVES * TILE * VROW, 16]
        xm: fx.Array[fx.Float32, WAVES * TILE, 16]
        xp: fx.Array[fx.Int32, WAVES * 64 * 2, 16]

    return Smem


@traced
def cta(c):
    """CTA ``c["bid"]``'s (head group, split): its partial stored (cache
    modifier ``c["cm"]``), then ``c["done"](flag)`` when given (``ctas``).
    ``c["args"]``: qb, kv, pages, indptr, part_o, part_lse, scale, vscale."""
    s, q, a = c["S"], c["Q"], c["args"]
    ht, hg, spx = geometry(q)
    nreq = s // q
    per_req = splits(q) // nreq
    bid, tid = c["bid"], c["tid"]
    lane = tid % 64
    wave = tid // 64
    g = lane // TILE
    row = lane % TILE
    x = bid % XCDS
    u = bid // XCDS
    if u < spx * hg:
        gs = x * spx + u // hg  # global split
        req = gs % nreq
        split = gs // nreq
        hd = (u % hg) * ht + row // q
        pos = row % q
        t = req * q + pos
        p0 = uniform(ld_i32(a["indptr"], req))
        L = uniform(ld_i32(a["indptr"], req + 1)) - p0
        ct = ((L + per_req - 1) // per_req + TILE - 1) // TILE * TILE
        c0 = split * ct
        c1 = fx.min(L, c0 + ct)
        ntile = (fx.max(c1 - c0, 0) + TILE - 1) // TILE
        # Q^T's B operand: row lane % 16, step st its 8 latents at
        # 64 (st // 2) + 16 g + 8 (st % 2) (K's order, bf16)
        qo = ((t * NH + hd) * ENTRY + 16 * g) // 2
        qv = [
            fx.Vector(
                bo.buffer_load(
                    rsrc(a["qb"]),
                    qo + (CHUNK * (st // 2) + 8 * (st % 2)) // 2,
                    vec_width=4,
                    dtype=T.i32,
                )
            ).bitcast(fx.BFloat16)
            for st in range(KSTEPS)
        ]
        # a new dict (the branch would carry a mutated ``c`` out as values)
        d = dict(
            c,
            lane=lane,
            wave=wave,
            q=qv,
            t=t,
            h=hd,
            split=req * per_req + split,
            p0=p0,
            c0=c0,
            last=fx.max(c1 - 1, 0),
            bound=L - q + pos,
            nblk=(ntile + WAVES - 1) // WAVES,
            scale=fx.Int32(a["scale"]).bitcast(fx.Float32) * fx.Float32(LOG2E),
            vscale=fx.Int32(a["vscale"]).bitcast(fx.Float32),
        )
        stamp(d, 0)
        run(d)
        if const_expr("done" in d):
            d["done"]((req * per_req + split) * hg + u % hg)


_BUILDS: dict = {}


def build(key: MlaAttnBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s, q = key.tokens, key.qlen
    check(s, q)
    Smem = smem()

    keyed = key_tuple(key, _SOURCES)
    name = f"k3_mono_mla_attn_s{s}_q{q}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def mla_attn(
        qb: Int64,
        kv: Int64,
        pages: Int64,
        indptr: Int64,
        part_o: Int64,
        part_lse: Int64,
        scale: Int32,
        vscale: Int32,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        cta(
            {
                "S": s,
                "Q": q,
                "stamps": key.stamps,
                "dbg": key.dbg,
                "bid": bid,
                "tid": tid,
                "cm": 0,
                "vt": lds.vt.ptr,
                "xm": lds.xm.ptr,
                "xp": lds.xp.ptr,
                "args": {
                    "qb": qb,
                    "kv": kv,
                    "pages": pages,
                    "indptr": indptr,
                    "part_o": part_o,
                    "part_lse": part_lse,
                    "scale": scale,
                    "vscale": vscale,
                },
            }
        )

    @flyc.jit
    def launch(
        qb: Int64,
        kv: Int64,
        pages: Int64,
        indptr: Int64,
        part_o: Int64,
        part_lse: Int64,
        scale: Int32,
        vscale: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        mla_attn(qb, kv, pages, indptr, part_o, part_lse, scale, vscale).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    _BUILDS[key] = launch
    return launch


def partial_shapes(key: MlaAttnBuild):
    n = splits(key.qlen)
    return (n, key.tokens, NH, KVL), (n, key.tokens, NH)


def f32_bits(x: float) -> int:
    return int(torch.tensor(float(x), dtype=torch.float32).view(torch.int32))


def mla_attn(
    key: MlaAttnBuild, *, q, kv, pages, indptr, part_o, part_lse, scale, kv_scale=1.0
):
    """Host launcher. ``q`` [S, 12, 576] bf16; ``kv`` [tokens, 576] fp8 rows
    (dense); ``pages`` int32 flat token indices, ``indptr`` [requests + 1]
    int32 (a request's KV ``pages[indptr[r]:indptr[r + 1]]``); ``part_o`` /
    ``part_lse`` of ``partial_shapes(key)``, bf16 / f32. ``scale`` the softmax
    scale, ``kv_scale`` the cache's dequant (folded into QK and o)."""
    so, sl = partial_shapes(key)
    assert part_o.shape == so and part_o.dtype == torch.bfloat16
    assert part_lse.shape == sl and part_lse.dtype == torch.float32
    assert q.shape == (key.tokens, NH, ENTRY), q.shape
    a = host_args(
        q=q, kv=kv, pages=pages, indptr=indptr, scale=scale, kv_scale=kv_scale
    )
    build(key)(
        *a[:4],
        part_o.data_ptr(),
        part_lse.data_ptr(),
        *a[4:],
        stream=torch.cuda.current_stream(),
    )


def host_args(*, q, kv, pages, indptr, scale, kv_scale=1.0):
    """(qb, kv, pages, indptr, scale, vscale) of the kernel (``mla_attn``'s
    inputs but the partials; K2's fused M1c's)."""
    assert (
        q.dtype == torch.bfloat16 and q.is_contiguous() and q.shape[1:] == (NH, ENTRY)
    )
    assert kv.element_size() == 1 and kv.stride(-1) == 1 and kv.size(-1) == ENTRY
    assert kv.stride(0) == ENTRY
    assert pages.dtype == torch.int32 and indptr.dtype == torch.int32
    return [
        q.data_ptr(),
        kv.data_ptr(),
        pages.data_ptr(),
        indptr.data_ptr(),
        f32_bits(scale * kv_scale),
        f32_bits(kv_scale),
    ]
