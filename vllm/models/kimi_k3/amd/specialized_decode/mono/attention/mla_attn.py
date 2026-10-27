# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M1c ``mla_attn``: Kimi-K3's MLA decode attention at TP8 + DCP1, the bf16
query (M1a ``qbf16``) against the fp8 latent cache, split over the KV: one pass
a split, every split's partial (bf16 o, f32 LSE) for a combine (M1b's).

One launch, ``BLOCKS`` CTAs x ``THREADS`` threads. A CTA a (head group, split):
16 query rows (``16 / qlen`` heads x qlen positions) against a chunk of the
request's KV. The split's CTAs (every head group) share an XCD, so a chunk
streams from HBM once into that XCD's L2.

    block of 128 tokens (wave w its tile of 16, the waves in step), its K
    rows DMA'd into one of two LDS slots a block ahead (global_load_lds, the
    chunks spread over the block before: ``DMA``; this wave's chunks waited
    for one at a time, so the top barrier only frees the slot)
        K tile LDS -> registers (a lane 32 B a line of a token: the MFMA's A, the K
        order permuted alike in Q's B), S^T = K Q^T: the fp8 K as is against
        Q's exact fp8 hi + lo (``q_split``), 9 16x16x128 MFMAs a row tile
        the block's row max over the waves (LDS) -> P = exp2(S - m) 2^PLOG,
        fp8 hi + lo -> LDS; the running O, l rescaled (online softmax)
        O^T += V^T P^T: wave w its 64 latents, the fp8 V^T straight from the
        slot (``ds_read_b64_tr_b8``) against P's hi and lo, 4 16x16x128 MFMAs
        a row tile and part
    end: l summed over the waves -> o = O / l (bf16), LSE = m + log l

A split with no token of the request (or none before its causal bound) leaves
LSE -inf and o 0. Rows are causal: position i of the request's qlen sees its
KV up to ``seq_len - qlen + i``.
"""

from dataclasses import dataclass, replace

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import LOG2E
from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import (
    ENTRY,
    KVL,
    NH,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ops import (
    _hw_f32,
    as_ir_value,
    bf16_pair,
    butterfly,
    hw_exp2,
    hw_rcp,
    ld_i32,
    memrealtime,
    permlane_swap,
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
CHUNK = 64  # bytes a token a K chunk (a DMA: 64 lanes x 16 B a tile)
KLOADS = ENTRY // CHUNK  # 9 DMAs a tile: 8 of 8 tokens' whole 128 B lines (a
#   token's 8 lanes, a quarter the addresses of 4 B a lane), the tail's 16 x 64
QSTEPS = KLOADS // 2  # 4 fp8 MFMA steps of QK a Q part (a lane 32 B: K
#   line s, chunks 2 s, 2 s + 1), then the tail (chunk 8 twice against its
#   hi | lo)
TILEW = TILE * CHUNK // 4  # 256 dwords: a tile's chunk p, one DMA
SLOT = KLOADS * WAVES * TILEW  # a block's K (dwords): [chunk p][tile][TILEW]
LINE = 2 * CHUNK  # 128 B: chunks 2 s, 2 s + 1 hold line s (bytes 128 s ..) of
#   the tile's tokens 0..7, 8..15 (a DMA 8 whole lines, ``lines_at``); chunk 8
#   (the tail, 64 B a token) its 16 tokens (``kpos``)
VCOLS = KVL // WAVES  # 64 latents a wave
LN2 = 0.6931471805599453
NEG = -1.0e30  # the running max before any token (finite: no inf - inf)

assert ENTRY % CHUNK == 0 and KVL % CHUNK == 0 and KVL // 4 % 16 == 0
assert VCOLS == 64 and THREADS == WAVES * 64 and KVL // CHUNK == WAVES

# Block b + 1's K chunks go out spread over block b (one at a time): issued
# together they hold the waves at the issue for the whole block's DMA (~1 us,
# the CU's load path), and issued late their landing shows at the next wait.
# Hook -> chunks: "q<s>" after QK step s (``QSTEPS``: the tail), "p" after
# the P barrier, "v<i>" before latent tile i's V^T reads, "m<k>" before PV's
# MFMAs of (row tile k // 2, P part k % 2); a schedule by row tiles (a
# spread over QK's first steps, P and PV within ~2%; bunched 5-20% slower).
# Among PV's MFMAs is the ~2% faster, but in the fused K2 builds at 2 row
# tiles it leaves a Q part (QK's first B) spilled: a scratch reload a block,
# whose vmcnt(0) waits out every DMA in flight.
# At 1 row tile (the block's K the longer wait) two each after QK's first
# steps: 2-3% faster than one.
DMA = {
    1: {"q0": 2, "q1": 2, "q2": 1, "q3": 1, "p": 1, "m0": 2},
    2: {"q0": 1, "q1": 1, "q2": 1, "q3": 1, "p": 2, "v0": 1, "v1": 1, "v2": 1},
}
assert all(sum(d.values()) == KLOADS for d in DMA.values())

# P to the PV MFMA as two fp8 terms (hi + lo, of p 2^PLOG: p <= 1, hi's
# rounding error in lo's 3 more bits, ~bf16 overall), against the slot's raw
# fp8 V: fp8 MFMAs, no V conversion
PLOG = 8  # P 2^PLOG up (p <= 1: <= 256 of E4M3's 448), lo normal to p 2^-10

# Warp specialization (the row tiles in ``WSPEC``): a SIMD's two waves on the
# two halves of a block, software-pipelined. Waves 0..QKW-1 QK and softmax of
# block b (wave w tiles 2 w, 2 w + 1, both row tiles), waves QKW.. PV of block
# b - 1 (wave QKW + v latents 128 v .., V^T in registers, read a block ahead):
# one wave's softmax VALU under the other's MFMAs. A step two barriers: B1 the
# max exchange (the PV waves amid their MFMAs, P(b - 1) read), B2 P(b) and
# alpha(b) written, V^T(b) read and block b + 1 landed. P in ``xp`` (read by
# B1(b + 1), written after it). Every wave DMAs its tile (as ``k_dma``) of
# block b + 2 into block b's slot, chunks ..WS_K1 after B1(b) (its V^T read
# before it, its QK done), the rest early in step b + 1 (the DMAs over the
# step, not a burst a step), then the indices of block b + 3
WSPEC = (2,)
QKW = WAVES // 2
WS_B1 = 4  # PV's latent tiles (of 8) before B1
WS_K1 = 3  # a tile's chunks DMA'd after B1 two blocks ahead (the rest one)


_STREAM = fx.Stream(None)
_SOURCES = source_digest(".")


RTILES = {4: 1, 8: 2}  # row tiles a CTA by qlen (3 head groups either way)


def splits(qlen: int, rt: int = 0) -> int:
    """KV splits of the launch (every request's together): the head groups'
    CTAs an XCD hold a whole number of splits."""
    return XCDS * geometry(qlen, rt)[2]


def geometry(qlen: int, rt: int = 0):
    """(heads a CTA, head groups, splits an XCD); a CTA ``rt`` row tiles of 16
    (``RTILES`` by default)."""
    ht = TILE * (rt or RTILES[qlen]) // qlen
    hg = NH // ht
    return ht, hg, BLOCKS // XCDS // hg


def ctas(qlen: int, rt: int = 0) -> int:
    """The working CTAs (0 .. ctas): a done flag each when fused (``cta``'s
    ``done``), at (split, head group) ``split * hg + group``."""
    _, hg, spx = geometry(qlen, rt)
    return XCDS * spx * hg


@dataclass(frozen=True)
class MlaAttnBuild:
    tokens: int  # S = requests x qlen
    qlen: int  # 4 or 8
    stamps: bool = False  # debug: phase timestamps (memrealtime) over split 0
    #   of part_o: 2 + 4 b + (wait done, QK + max barrier, P barrier, PV end)
    dbg: int = 0  # debug: 1 no K DMA in the loop (stale slots), 2 no part_o stores
    rt: int = 0  # row tiles a CTA (0: ``RTILES``)
    wide: bool = False  # a KV pool over 4 GiB (``wide_kv``): 64-bit DMA
    #   addresses (else a 32-bit offset off the pool's SGPR base)
    ws: int = -1  # warp specialization (-1: by ``WSPEC``)


def wspec(rt: int, ws: int = -1) -> bool:
    return rt in WSPEC if ws < 0 else bool(ws)


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


def mfma_f8(a, b, c):
    """16x16x128 E4M3 x E4M3 MFMA, unscaled (scale operands 0: LLVM's
    v_mfma_f32_16x16x128_f8f6f4): ``a`` / ``b`` 8 dwords (QK and PV)."""
    zero = as_ir_value(fx.Int32(0))
    return fx.Vector(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.vec(4, T.f32),
            [as_ir_value(a), as_ir_value(b), as_ir_value(c), 0, 0, 0, zero, 0, zero],
        )
    )


def fp8_unpack4(w):
    """One dword of four E4M3 -> four f32 (exact)."""
    out = []
    for hi in (False, True):
        v = fx.Vector(rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), as_ir_value(w), hi))
        out += [v[0], v[1]]
    return out


def fp8_pack4(a, b, c, d):
    """Four f32 (in range) -> one dword of four E4M3, RNE (``ops``' but the
    first pair over a poison dword: no zeroing move)."""
    lo = fx.Int32(
        rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(llvm.mlir_poison(T.i32)), False)
    )
    return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, c, d, lo, True))


def fp8_split4(x):
    """Four f32 (in range) -> (hi, lo) dwords of four E4M3: x ~ hi + lo (the
    differences as pairs, as the unpack gives them: no shuffling moves)."""
    hi = fp8_pack4(*x)
    lo = []
    for k in range(2):
        hf = fx.Vector(rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), as_ir_value(hi), k == 1))
        d = fx.Vector.from_elements([x[2 * k], x[2 * k + 1]], fx.Float32) - hf
        lo += [d[0], d[1]]
    return hi, fp8_pack4(*lo)


def xp_at(r, h, k, lane):
    """The P dword of (row tile r, hi / lo h, tiles 4 k ..) of ``lane``."""
    return (((r * 2 + h) * 2 + k) * 64 + lane) * 4


@traced
def q_split(c, rows, at):
    """The CTA's bf16 Q rows (slot 1: [row][ENTRY]) -> E4M3 hi, lo of q 2^k
    (a power of two a row: its |max| in [128, 256)): hi + lo is q exactly,
    bf16's 8 bits E4M3's 4 + 4 (lo q - hi, a multiple of hi's ulp / 16; but
    below 2^-9 of 2^k: 2^-17 of the row's |max|). At dword ``at``: [hi |
    lo][row][ENTRY] bytes, the tail chunk's also at ``qt``; a row's 2^-k at
    ``xm`` + row. A row 16 threads, 4 latents a unit (``tid % 16`` + 16 i)."""
    tid = c["tid"]
    rr = tid // 16
    if rr < rows:
        src = c["ks"] + SLOT + rr * (ENTRY // 2)
        us = [tid % 16 + 16 * i for i in range(ENTRY // 4 // 16)]
        qs = [
            fx.Vector(
                fx.ptr_load(src + 2 * u, result_type=fx.Vector.make_type(2, fx.Int32))
            ).bitcast(fx.BFloat16)
            for u in us
        ]
        x = [fx.Float32(q[e]) for q in qs for e in range(4)]
        am = fx.Float32(1.0e-30)
        for k in range_constexpr(len(x)):
            am = fx.max(am, fx.max(x[k], fx.Float32(0.0) - x[k]))
        am = butterfly(am, (8, 4, 2, 1), fx.max)
        e = (am.bitcast(fx.Int32) >> 23) & 0xFF
        e = fx.max(fx.min(e, fx.Int32(253)), fx.Int32(8))
        qi = ((261 - e) << 23).bitcast(fx.Float32)
        if tid % 16 == 0:
            fx.ptr_store(((e - 7) << 23).bitcast(fx.Float32), c["xm"] + rr)
        for i in range_constexpr(len(us)):
            u = us[i]
            y = [x[4 * i + j] * qi for j in range(4)]
            hi = fp8_pack4(*y)
            lo = fp8_pack4(
                *[
                    y[j] - fx.Float32(rocdl.cvt_f32_fp8(T.f32, as_ir_value(hi), j))
                    for j in range(4)
                ]
            )
            fx.ptr_store(hi, c["ks"] + at + rr * (ENTRY // 4) + u)
            fx.ptr_store(lo, c["ks"] + at + (rows + rr) * (ENTRY // 4) + u)
            if const_expr(i == len(us) - 1):
                # the tail chunk's hi | lo kept apart (``qt``: read a block)
                tb = c["qt"] + rr * (2 * CHUNK // 4) + u - 2 * QSTEPS * CHUNK // 4
                fx.ptr_store(hi, tb)
                fx.ptr_store(lo, tb + CHUNK // 4)


def q_parts(c, r, at, rows):
    """Row tile r's B operands from ``q_split``'s: QK step s's hi, lo
    (s < ``QSTEPS``: line s), 8 dwords each (a lane 32 B at 32 g of the line,
    K's order); the tail's stay in LDS (``qt``)."""
    row = r * TILE + c["lane"] % TILE
    g = c["lane"] // TILE

    def ld(part, m, k):
        return fx.Vector(
            fx.ptr_load(
                c["ks"]
                + at
                + (part * rows + row) * (ENTRY // 4)
                + (
                    32 * m + 8 * g + 4 * k
                    if const_expr(c["lines"])
                    else 16 * (2 * m + k) + 4 * g
                ),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )

    def cat(a, b):
        return fx.Vector.from_elements(
            [a[i] for i in range(4)] + [b[i] for i in range(4)], fx.Int32
        )  # noqa: E501

    parts = []
    for m in range_constexpr(QSTEPS):
        for h in range_constexpr(2):
            parts.append(cat(ld(h, m, 0), ld(h, m, 1)))
    return parts


def kpos(t, cc):
    """The 16 B position (of 64) in a tile's tail chunk of token t's 16 B cc:
    a lane group's K read (tokens 0..15, one cc) over the 64 banks."""
    return 16 * cc + 4 * ((t % 4 + cc) % 4) + t // 4


def sig(t, tp):
    """The 16 B column swizzle (XOR) of token t (of 16) of a tile of parity
    tp in a line chunk: a K read (tokens 0..15, one column) and a V^T read
    (tokens 4 g + j, tiles 2 n, 2 n + 1) both over the 64 banks."""
    return ((t >> 1) & 3) | ((((t >> 3) ^ tp) & 1) << 2)


# tokens t, t + 8 of a lane (``dma_offs``): columns 64 B apart
assert all(sig(t + 8, tp) == sig(t, tp) ^ 4 for t in range(8) for tp in (0, 1))


def lines_at(t, tp, col):
    """Token t's 16 B ``col`` of line s: its byte in the slot (from line s's
    chunk 2 s, a tile 1 KB: 8 tokens x 128 B), the DMA lane ``8 (t % 8) +
    (col ^ sig)`` of chunk 2 s + t // 8."""
    return (t // 8) * WAVES * TILEW * 4 + (t % 8) * LINE + 16 * (col ^ sig(t, tp))


def dma_lane(lane):
    """(token, 16 B cc) tail DMA lane ``lane`` carries (it lands at position
    lane)."""
    cc = lane // 16
    return 4 * (lane % 4) + ((lane // 4) % 4 + 4 - cc) % 4, cc


def vtr_lane(lane, wave, i, lines=True):
    """This lane's byte in a block's (line wave // 2) of the 8 B it gives a
    V^T transpose read of latent tile i (``tr_read``; tiles 2 n .. at + 2 n
    TILEW): lane k of a 16 its half k % 2 of the 16 B 4 (wave % 2) + i of
    token 4 g + k // 2 % 4 of tile k // 8 -> lane m: latent 64 wave + 16 i +
    m, tokens 4 g .. of tiles 2 n, 2 n + 1."""
    g, k = lane // 16, lane % 16
    t = 4 * g + (k // 2) % 4
    if not lines:
        # the 64 B chunk layout: latent tile i at chunk wave
        return (k // 8) * TILEW * 4 + 16 * kpos(t, i) + 8 * (k % 2)
    col = 4 * (wave % 2) + i
    return (k // 8) * TILEW * 4 + lines_at(t, k // 8, col) + 8 * (k % 2)


def tr_read(addr):
    """ds_read_b64_tr_b8 at LDS byte ``addr``: a lane m of 16 byte m % 8 of
    the 8 B of lanes m // 8 + 2 e (byte e), as two dwords."""
    p3 = llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), as_ir_value(fx.Int32(addr)))
    return fx.Vector(rocdl.ds_read_tr8_b64(T.vec(2, T.i32), p3).result)


def idx_load(c, b):
    """The page indices of this lane's DMA tokens of block b's tile of this
    wave (the tail's, lines' tokens lane // 8, + 8), clamped into the chunk
    (a block past the last loads its rows)."""
    t0 = c["c0"] + (b * WAVES + c["wave"]) * TILE
    if const_expr(not c["lines"]):
        return (ld_i32(c["args"]["pages"], c["p0"] + fx.min(t0 + c["dt"], c["last"])),)
    return tuple(
        ld_i32(c["args"]["pages"], c["p0"] + fx.min(t0 + t, c["last"]))
        for t in (c["dt"], c["lane"] // 8, c["lane"] // 8 + 8)
    )


def dma_offs(c, idx):
    """``k_dma``'s sources of ``idx_load``'s indices: the tail chunk's (16 B
    dcc), line chunks' tokens lane // 8 and + 8 (their 16 B column
    ``sig``-swizzled); bytes off the pool (``wide``: addresses)."""
    cols = (
        (c["dcc"] * 16, c["dcol"], c["dcol"] ^ 64)
        if const_expr(c["lines"])
        else (c["dcc"] * 16,)
    )
    if const_expr(c["wide"]):
        return tuple(
            c["args"]["kv"] + fx.Int64(i) * ENTRY + fx.Int64(col)
            for i, col in zip(idx, cols)
        )
    return tuple(fx.Int32(i) * ENTRY + col for i, col in zip(idx, cols))


def k_dma(c, b, offs, ps=None, tile=None):
    """Block b's K tile of this wave (or ``tile``; ``dma_offs``) -> slot
    b % 2: a DMA a chunk (64 lanes x 16 B: line s's 8 tokens, the tail's 16).
    Inline asm: LLVM's own wait for an LDS DMA would hold this block's slot
    reads for the next block's DMA too; ``vmwait`` / ``wait_dma`` are the only
    waits."""
    ps = range(KLOADS) if ps is None else ps
    tile = c["wave"] if tile is None else tile
    dst = c["ks_b"] + ((b % 2) * SLOT + tile * TILEW) * 4

    def at(p):
        # (offset, immediate): chunk 2 s + h line s of tokens 8 h ..
        if const_expr(p == KLOADS - 1 or not c["lines"]):
            return offs[0], CHUNK * p
        return offs[1 + p % 2], LINE * (p // 2)

    if const_expr(c["wide"]):
        for p in ps:
            o, imm = at(p)
            lds_dma(o + imm, dst + p * WAVES * TILEW * 4)
    else:
        # a 32-bit offset off the pool's SGPR base, the line in the
        # immediate (half the address VGPRs a DMA reads; the immediate
        # offsets the LDS side too, M0 takes it back)
        for p in ps:
            o, imm = at(p)
            llvm.inline_asm(
                None,
                [
                    as_ir_value(o),
                    as_ir_value(fx.Int64(c["args"]["kv"])),
                    as_ir_value(uniform(dst + p * WAVES * TILEW * 4 - imm)),
                ],
                "s_mov_b32 m0, $2\n\ts_nop 0\n\t"
                f"global_load_lds_dwordx4 $0, $1 offset:{imm}",
                "v,s,s,~{m0},~{memory}",
                has_side_effects=True,
            )


def lds_dma(src, dst):
    """A global_load_lds_dwordx4: lane i's 16 B at ``src`` -> LDS byte
    ``dst`` (uniform) + 16 i. (Every LDS DMA here an ``s_nop`` between its M0
    write and the load: right after the write the load could read a stale M0,
    the warp-specialized waves' K / index DMAs then landing in another slot:
    nondeterministic outputs.)"""
    llvm.inline_asm(
        None,
        [as_ir_value(fx.Int64(src)), as_ir_value(uniform(dst))],
        "s_mov_b32 m0, $1\n\ts_nop 0\n\tglobal_load_lds_dwordx4 $0, off",
        "v,s,~{m0},~{memory}",
        has_side_effects=True,
    )


@traced
def idx_dma(c, b, t0=None, n=1):
    """Block b's page indices of this wave's tile (or tiles t0 .. t0 + n) ->
    its LDS index slot b % 2 (lanes 0 .. 16 n, token lane's): a DMA, so it
    lands in order with the K DMAs (a load would make LLVM wait on every DMA
    before it to read it)."""
    t0 = c["wave"] if t0 is None else t0
    if c["lane"] < n * TILE:
        tok = c["c0"] + (b * WAVES + t0) * TILE + c["lane"]
        src = c["args"]["pages"] + fx.Int64(c["p0"] + fx.min(tok, c["last"])) * 4
        dst = c["xi_b"] + ((b % 2) * WAVES + t0) * TILE * 4
        llvm.inline_asm(
            None,
            [as_ir_value(fx.Int64(src)), as_ir_value(uniform(dst))],
            "s_mov_b32 m0, $1\n\ts_nop 0\n\tglobal_load_lds_dword $0, off",
            "v,s,~{m0},~{memory}",
            has_side_effects=True,
        )


def idx_lds(c, b):
    """This lane's DMA tokens' page indices of block b (``idx_dma``'s; as
    ``idx_load``)."""
    xb = c["xi"] + ((b % 2) * WAVES + c["wave"]) * TILE
    if const_expr(not c["lines"]):
        return (fx.ptr_load(xb + c["dt"]),)
    return (
        fx.ptr_load(xb + c["dt"]),
        fx.ptr_load(xb + c["lane"] // 8),
        fx.ptr_load(xb + c["lane"] // 8 + 8),
    )


def vmwait(n):
    """s_waitcnt vmcnt(n): at most n of this wave's loads / DMAs in flight
    (they land in order)."""
    llvm.inline_asm(
        None, [], f"s_waitcnt vmcnt({n})", "~{memory}", has_side_effects=True
    )


def wait_dma():
    """This wave's DMAs (and loads) landed; the CTA's after the barrier."""
    llvm.inline_asm(None, [], "s_waitcnt vmcnt(0)", "", has_side_effects=True)
    gpu.barrier()


def xm_row(c, r, row):
    """The waves' values of row ``row`` of row tile r in xm ([r][row][wave]:
    two b128 reads)."""
    v = [
        fx.Vector(
            fx.ptr_load(
                c["xm"] + (r * TILE + row) * WAVES + 4 * k,
                result_type=fx.Vector.make_type(4, fx.Float32),
            )
        )
        for k in range(2)
    ]
    return [v[w // 4][w % 4] for w in range(WAVES)]


def v_reads(c, b, dma):
    """V^T of the slot's block b by transpose reads, the A of a 16x16x128 fp8
    MFMA a latent tile i: a lane's 32 B tile w's tokens 4 g .. at 4 w (P's
    order); ``dma`` the hook ``v<i>``'s K chunks before tile i's reads."""
    vw = (c["wave"] // 2) * 2 if const_expr(c["lines"]) else c["wave"]
    vt = c["ks_b"] + ((b % 2) * SLOT + vw * WAVES * TILEW) * 4
    va = []
    for i in range_constexpr(4):
        dma(f"v{i}")
        tr = [tr_read(vt + c["vtr"][i] + n * 2 * TILEW * 4) for n in range(4)]
        va.append(
            fx.Vector.from_elements(
                [tr[n][e] for n in range(4) for e in range(2)], fx.Int32
            )
        )
    return va


@traced
def stage_block(c, b, st, dma, kwait, masked):
    """One block (its K in slot b % 2) into the running (m, l, O) of each of
    the CTA's row tiles (K and V^T read once for all); ``dma(hook)``
    issues the hook's of block b + 1's K chunks (``DMA``), ``kwait(p)`` waits
    for this wave's chunk p of block b; ``masked``: a block with a token past
    the chunk or a row's causal bound (else every score kept)."""
    lane, wave = c["lane"], c["wave"]
    row = lane % TILE
    rt = c["rt"]
    m_run, l_run, o = st
    kt = c["ks"] + (b % 2) * SLOT + wave * TILEW

    def kld(at):
        return fx.Vector(
            fx.ptr_load(kt + at, result_type=fx.Vector.make_type(4, fx.Int32))
        )

    def kstep(s):
        """Step s's K words: line s's 16 B columns 2 g, 2 g + 1 (chunks 2 s,
        2 s + 1; the tail: chunk 8's g), once landed."""
        if const_expr(not c["lines"]):
            ats = [
                p * WAVES * TILEW + c["qpos"][2]
                for p in ([2 * s, 2 * s + 1] if s < QSTEPS else [2 * s])
            ]
        else:
            ats = (
                [2 * s * WAVES * TILEW + c["qpos"][k] for k in range(2)]
                if s < QSTEPS
                else [2 * s * WAVES * TILEW + c["qpos"][2]]
            )
        kwait(2 * s + 1 if s < QSTEPS else 2 * s)
        return [w for at in ats for w in (lambda v: [v[e] for e in range(4)])(kld(at))]

    # S^T: token (4 g + j) x row (lane % 16), a tile r of rows each; fp8 K
    # straight into the MFMA against Q's hi and lo (``q_parts``), a step's K
    # read during the step before (``sched_barrier``: no hoisting past it)
    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(rt)]
    nxt = kstep(0)
    for s in range_constexpr(QSTEPS + 1):
        words = nxt
        if const_expr(s < QSTEPS):
            nxt = kstep(s + 1)
            ka = fx.Vector.from_elements(words, fx.Int32)
            for r in range_constexpr(rt):
                for h in range_constexpr(2):
                    acc[r] = mfma_f8(ka, c["q"][r][2 * s + h], acc[r])
        else:
            # the tail: chunk 8 twice against its hi | lo (from ``qt``: in
            # VGPRs they'd spill, the reload a stall a block)
            ka = fx.Vector.from_elements(words + words, fx.Int32)
            for r in range_constexpr(rt):
                qb = c["qt"] + (r * TILE + row) * (2 * CHUNK // 4) + 4 * (lane // 16)
                qtl = [
                    fx.Vector(
                        fx.ptr_load(
                            qb + h * CHUNK // 4,
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                    )
                    for h in range(2)
                ]
                qtv = fx.Vector.from_elements(
                    [qtl[h][e] for h in range(2) for e in range(4)], fx.Int32
                )
                acc[r] = mfma_f8(ka, qtv, acc[r])
        dma(f"q{s}")
        rocdl.sched_barrier(0)
    # scores (log2 units: acc qscale, qscale > 0, so the max on acc's),
    # causal + chunk mask
    sc = [[acc[r][j] for j in range(4)] for r in range(rt)]
    if const_expr(masked):
        t0 = c["c0"] + (b * WAVES + wave) * TILE + 4 * (lane // TILE)
        ok = [(t0 + j <= c["last"]) & (t0 + j <= c["bound"]) for j in range(4)]
        ninf = fx.Float32(float("-inf"))
        sc = [[ok[j].select(sc[r][j], ninf) for j in range(4)] for r in range(rt)]
    mx = [
        fx.max(fx.max(sc[r][0], sc[r][1]), fx.max(sc[r][2], sc[r][3]))
        for r in range(rt)
    ]
    if const_expr(rt == 2):
        # the two row tiles' halves in one swap: lanes 32 r .. row tile r's
        x, y = permlane_swap(32, mx[0].bitcast(fx.Int32), mx[1].bitcast(fx.Int32))
        m2 = butterfly(
            fx.max(x.bitcast(fx.Float32), y.bitcast(fx.Float32)), (16,), fx.max
        )
        if lane % 32 < TILE:
            fx.ptr_store(m2, c["xm"] + ((lane // 32) * TILE + row) * WAVES + wave)
    else:
        for r in range_constexpr(rt):
            m1 = butterfly(mx[r], (32, 16), fx.max)
            if lane < TILE:
                fx.ptr_store(m1, c["xm"] + (r * TILE + row) * WAVES + wave)
    gpu.barrier()
    stamp(c, 2 + 4 * b + 1)
    dma("p")
    mb = []
    for r in range_constexpr(rt):
        xw = xm_row(c, r, row)
        mb.append(
            fx.max(
                fx.max(fx.max(xw[0], xw[1]), fx.max(xw[2], xw[3])),
                fx.max(fx.max(xw[4], xw[5]), fx.max(xw[6], xw[7])),
            )
            * c["qscale"][r]
        )
    m_new, alpha = [], []
    for r in range_constexpr(rt):
        m_new.append(fx.max(m_run[r], mb[r]))
        alpha.append(hw_exp2(m_run[r] - m_new[r]))
        # p 2^PLOG (l, O alike: the normalization undoes it)
        nmp = fx.Float32(PLOG) - m_new[r]
        p = [hw_exp2(fmath.fma(sc[r][j], c["qscale"][r], nmp)) for j in range(4)]
        # l a lane's (its 4 tokens'), summed over the lanes at the end
        # (pairs, as the split's: no shuffling moves)
        pp = [
            fx.Vector.from_elements(p[2 * k : 2 * k + 2], fx.Float32) for k in range(2)
        ]
        ps = pp[0] + pp[1]
        l_run[r] = l_run[r] * alpha[r] + (ps[0] + ps[1])
        # [r][hi, lo][tile // 4][lane][tile % 4]: a lane's 8 tiles two b128
        pw = fp8_split4(p)
        for h in range_constexpr(2):
            fx.ptr_store(pw[h], c["xp"] + xp_at(r, h, wave // 4, lane) + wave % 4)
    gpu.barrier()
    stamp(c, 2 + 4 * b + 2)
    # O^T += V^T P^T, the block an MFMA a (latent tile i, P part) (k: tile
    # w's tokens 4 g .. at 4 w): latents 64 wave + 16 i + 4 g + j, chunk p =
    # wave of the slot; o[4 r + i] row tile r's (P's hi and lo into it alike)
    o = [
        fx.Vector.from_elements(
            [o[4 * r + i][j] * alpha[r] for j in range(4)], fx.Float32
        )
        for r in range(rt)
        for i in range(4)
    ]
    va = v_reads(c, b, dma)
    for r in range_constexpr(rt):
        for h in range_constexpr(2):
            pb = [
                fx.Vector(
                    fx.ptr_load(
                        c["xp"] + xp_at(r, h, k, lane),
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                )
                for k in range(2)
            ]
            pv = fx.Vector.from_elements(
                [pb[w // 4][w % 4] for w in range(WAVES)], fx.Int32
            )
            dma(f"m{2 * r + h}")
            for i in range_constexpr(4):
                o[4 * r + i] = mfma_f8(va[i], pv, o[4 * r + i])
    # the PV MFMAs before the next block's index wait (in order, it would
    # hold them for this block's DMAs)
    rocdl.sched_barrier(0)
    stamp(c, 2 + 4 * b + 3)
    return m_new, l_run, o


def block(c, b, st, masked):
    """Block b (``stage_block``) behind the top barrier, its DMA hooks."""
    rt = c["rt"]
    # the slot block b + 1's DMA takes was the last block's (its PV
    # over): the waves in step; block b's own K waits a chunk
    gpu.barrier()
    stamp(c, 2 + 4 * b)
    idx_dma(c, b + 2)
    # block b + 1's indices landed (newer: block b's 9 chunks, b + 2's
    # indices)
    vmwait(KLOADS + 1)
    i1 = dma_offs(c, idx_lds(c, b + 1))
    # in flight (oldest first): block b's K chunks, block b + 2's
    # indices, then block b + 1's chunks as sent
    sent = [0]

    def kwait(p):
        # newer than chunk p: chunks p + 1 .., the indices, the sent
        vmwait(KLOADS - p + sent[0])

    def dma(hook):
        n = DMA[min(rt, 2)].get(hook, 0)
        if const_expr(c["dbg"] != 1 and n > 0):
            # pinned where the schedule puts it
            rocdl.sched_barrier(0)
            k_dma(c, b + 1, i1, range(sent[0], sent[0] + n))
            rocdl.sched_barrier(0)
        sent[0] += n

    return stage_block(c, b, st, dma, kwait, masked)


@traced
def run(c):
    """The blocks (block 0's DMA and block 1's indices issued) and the
    partial's stores."""
    lane, wave, g = c["lane"], c["wave"], c["lane"] // TILE
    row = lane % TILE
    rt = c["rt"]
    m_run = [fx.Float32(NEG) for _ in range(rt)]
    l_run = [fx.Float32(0.0) for _ in range(rt)]
    o = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(4 * rt)]
    # K a block ahead (slot b % 2), the indices two (``xi``, DMA'd too)
    stamp(c, 1)
    # the blocks every score of is kept (their last token at most the
    # chunk's and position 0's bound) unmasked, then the rest
    b = fx.Int32(0)
    while b < c["nfull"]:
        m_run, l_run, o = block(c, b, (m_run, l_run, o), False)
        b = b + 1
    while b < c["nblk"]:
        m_run, l_run, o = block(c, b, (m_run, l_run, o), True)
        b = b + 1
    # l over the lanes, then the waves (each its tiles' sum, at the common
    # running max)
    l_run = [butterfly(l_run[r], (32, 16)) for r in range(rt)]
    if lane < TILE:
        for r in range_constexpr(rt):
            fx.ptr_store(l_run[r], c["xm"] + (r * TILE + row) * WAVES + wave)
    gpu.barrier()
    a = c["args"]
    for r in range_constexpr(rt):
        xw = xm_row(c, r, row)
        lt = xw[0]
        for w in range_constexpr(1, WAVES):
            lt = lt + xw[w]
        empty = lt == fx.Float32(0.0)
        inv = empty.select(fx.Float32(0.0), hw_rcp(lt) * c["vscale"])
        # lane: latents 64 wave + 16 i + 4 g + j of its row; tiles 2 n,
        # 2 n + 1 traded over the 16-lane rows (``permlane_swap``): an even
        # g its 8 latents 4 g .. of tile 2 n, an odd g 4 (g - 1) .. of 2 n + 1
        th = (c["split"] * c["S"] + c["t"]) * NH + c["h"][r]
        pk = [
            [
                bf16_pair(o[4 * r + i][j] * inv, o[4 * r + i][j + 1] * inv).bitcast(
                    fx.Int32
                )
                for j in (0, 2)
            ]
            for i in range(4)
        ]
        for n in range_constexpr(2):
            sw = [permlane_swap(16, pk[2 * n][e], pk[2 * n + 1][e]) for e in range(2)]
            lat = wave * VCOLS + 16 * (2 * n + g % 2) + 4 * (g - g % 2)
            v = fx.Vector.from_elements(
                [sw[0][0], sw[1][0], sw[0][1], sw[1][1]], fx.Int32
            )
            # stamps: split 0's rows hold them; the rest still store (PV live)
            live = (c["split"] > 0) if const_expr(c["stamps"]) else True
            if const_expr(c["dbg"] != 2):  # noqa: SIM102
                if live:
                    bo.buffer_store(
                        v,
                        rsrc(a["part_o"]),
                        (th * KVL + lat) // 2,
                        cache_modifier=c["cm"],
                    )
        if (wave == 0) & (lane < TILE):
            lse = m_run[r] + (hw_log2(lt) - fx.Float32(PLOG))
            lse = empty.select(fx.Float32(float("-inf")), lse * fx.Float32(LN2))
            bo.buffer_store(lse, rsrc(a["part_lse"]), th, cache_modifier=c["cm"])
    stamp(c, 63)


def chunk_dma(c, src, imm, dst):
    """One chunk DMA (``k_dma``'s): ``src`` + ``imm`` (an offset off the pool,
    or ``wide`` an address) -> LDS byte ``dst`` (uniform; lane i its 16 B at
    + 16 i)."""
    if const_expr(c["wide"]):
        lds_dma(src + imm, dst)
    else:
        llvm.inline_asm(
            None,
            [
                as_ir_value(src),
                as_ir_value(fx.Int64(c["args"]["kv"])),
                as_ir_value(uniform(dst - imm)),
            ],
            "s_mov_b32 m0, $2\n\ts_nop 0\n\t"
            f"global_load_lds_dwordx4 $0, $1 offset:{imm}",
            "v,s,s,~{m0},~{memory}",
            has_side_effects=True,
        )


def ws_offs(c, b, t):
    """This lane's DMA source of tile t of block b (``dma_offs``, its index
    from the index slot)."""
    return dma_offs(
        c, (fx.ptr_load(c["xi"] + ((b % 2) * WAVES + t) * TILE + c["dt"]),)
    )[0]


def ws_dmas(c, bb, ps, hooks=("all",)):
    """This wave's DMA hook of block bb's tile's chunks ``ps`` (``k_dma``'s),
    spread over ``hooks`` (the first ones one more)."""
    if const_expr(c["dbg"] == 1):
        return lambda hook: None
    offs = (ws_offs(c, bb, c["wave"]),)
    ps = list(ps)
    nh = len(hooks)
    cut = [0] + [len(ps) - (len(ps) * (nh - i)) // nh for i in range(1, nh + 1)]
    at = {h: ps[cut[i] : cut[i + 1]] for i, h in enumerate(hooks)}

    def dma(hook):
        if const_expr(len(at.get(hook, [])) > 0):
            rocdl.sched_barrier(0)
            k_dma(c, bb, offs, at[hook])
            rocdl.sched_barrier(0)

    return dma


def ws_vtile(c, b, i):
    """V^T latent tile i (of 8: 128 v + 16 i ..) of the slot's block b
    (``v_reads``'): 8 dwords."""
    vt = c["vb"] + ((b % 2) * SLOT + (i // 4) * WAVES * TILEW) * 4 + c["vtr"][i % 4]
    tr = [tr_read(vt + n * 2 * TILEW * 4) for n in range(4)]
    return fx.Vector.from_elements(
        [tr[n][e] for n in range(4) for e in range(2)], fx.Int32
    )


@traced
def ws_qk_block(c, b, st, masked):
    """QK wave step b: S of block b's tiles 2 w, 2 w + 1 (``stage_block``'s
    QK), the max over the QK waves (B1), block b + 2's tile DMA'd, P(b) -> xp,
    alpha -> xm (B2: block b + 1's landed)."""
    lane, wave = c["lane"], c["wave"]
    row = lane % TILE
    g = lane // TILE
    rt = c["rt"]
    m_run, l_run = st
    # block b + 1's chunks WS_K1.. over the QK steps, then block b + 3's
    # indices (its slot's read by then)
    early = ws_dmas(c, b + 1, range(WS_K1, KLOADS), [f"k{s}" for s in range(QSTEPS)])
    slot = c["ks"] + (b % 2) * SLOT
    kt = slot + 2 * wave * TILEW + c["qpos"][2]

    def kstep(s):
        ps = [2 * s, 2 * s + 1] if s < QSTEPS else [2 * s]
        return [
            [
                v[e]
                for p in ps
                for v in [
                    fx.Vector(
                        fx.ptr_load(
                            kt + p * WAVES * TILEW + j * TILEW,
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                    )
                ]
                for e in range(4)
            ]
            for j in range(2)
        ]

    acc = [[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(rt)] for _ in range(2)]
    nxt = kstep(0)
    for s in range_constexpr(QSTEPS + 1):
        words = nxt
        if const_expr(s < QSTEPS):
            nxt = kstep(s + 1)
            early(f"k{s}")
            if const_expr(s == QSTEPS - 1):
                idx_dma(c, b + 3)
            for j in range_constexpr(2):
                ka = fx.Vector.from_elements(words[j], fx.Int32)
                for r in range_constexpr(rt):
                    for h in range_constexpr(2):
                        acc[j][r] = mfma_f8(ka, c["q"][r][2 * s + h], acc[j][r])
        else:
            for r in range_constexpr(rt):
                qb = c["qt"] + (r * TILE + row) * (2 * CHUNK // 4) + 4 * g
                qtl = [
                    fx.Vector(
                        fx.ptr_load(
                            qb + h * CHUNK // 4,
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                    )
                    for h in range_constexpr(2)
                ]
                qtv = fx.Vector.from_elements(
                    [qtl[h][e] for h in range(2) for e in range(4)], fx.Int32
                )
                for j in range_constexpr(2):
                    ka = fx.Vector.from_elements(words[j] + words[j], fx.Int32)
                    acc[j][r] = mfma_f8(ka, qtv, acc[j][r])
        rocdl.sched_barrier(0)
    sc = [[[acc[j][r][e] for e in range(4)] for r in range(rt)] for j in range(2)]
    if const_expr(masked):
        ninf = fx.Float32(float("-inf"))
        for j in range_constexpr(2):
            t0 = c["c0"] + (b * WAVES + 2 * wave + j) * TILE + 4 * g
            ok = [(t0 + e <= c["last"]) & (t0 + e <= c["bound"]) for e in range(4)]
            sc[j] = [
                [ok[e].select(sc[j][r][e], ninf) for e in range(4)] for r in range(rt)
            ]
    mx = []
    for r in range_constexpr(rt):
        m4 = [fx.max(sc[0][r][e], sc[1][r][e]) for e in range(4)]
        mx.append(fx.max(fx.max(m4[0], m4[1]), fx.max(m4[2], m4[3])))
    x, y = permlane_swap(32, mx[0].bitcast(fx.Int32), mx[1].bitcast(fx.Int32))
    m2 = butterfly(fx.max(x.bitcast(fx.Float32), y.bitcast(fx.Float32)), (16,), fx.max)
    if lane % 32 < TILE:
        fx.ptr_store(m2, c["xm"] + ((lane // 32) * TILE + row) * WAVES + wave)
    gpu.barrier()
    ws_dmas(c, b + 2, range(WS_K1))("all")
    m_new, alpha = [], []
    for r in range_constexpr(rt):
        xw = fx.Vector(
            fx.ptr_load(
                c["xm"] + (r * TILE + row) * WAVES,
                result_type=fx.Vector.make_type(4, fx.Float32),
            )
        )
        mb = fx.max(fx.max(xw[0], xw[1]), fx.max(xw[2], xw[3])) * c["qscale"][r]
        m_new.append(fx.max(m_run[r], mb))
        alpha.append(hw_exp2(m_run[r] - m_new[r]))
        nmp = fx.Float32(PLOG) - m_new[r]
        lsum = None
        for j in range_constexpr(2):
            p = [hw_exp2(fmath.fma(sc[j][r][e], c["qscale"][r], nmp)) for e in range(4)]
            pp = [
                fx.Vector.from_elements(p[2 * k : 2 * k + 2], fx.Float32)
                for k in range(2)
            ]
            lsum = pp[0] + pp[1] if lsum is None else lsum + pp[0] + pp[1]
            pw = fp8_split4(p)
            t = 2 * wave + j
            for h in range_constexpr(2):
                fx.ptr_store(pw[h], c["xp"] + xp_at(r, h, t // 4, lane) + t % 4)
        assert lsum is not None
        l_run[r] = l_run[r] * alpha[r] + (lsum[0] + lsum[1])
    if (wave == 0) & (lane < TILE):
        for r in range_constexpr(rt):
            fx.ptr_store(alpha[r], c["xm"] + (r * TILE + row) * WAVES + QKW)
    vmwait(0 if const_expr(c["dbg"] == 1) else WS_K1)
    gpu.barrier()
    return m_new, l_run


@traced
def ws_qk(c):
    """The QK waves' steps (``ws_qk_block``), the last block's PV's barriers,
    then the l over the waves and the LSE."""
    lane, wave = c["lane"], c["wave"]
    row = lane % TILE
    rt = c["rt"]
    m_run = [fx.Float32(NEG) for _ in range(rt)]
    l_run = [fx.Float32(0.0) for _ in range(rt)]
    # block 1's tile's chunks ..WS_K1 (the rest step 0's)
    ws_dmas(c, 1, range(WS_K1))("all")
    b = fx.Int32(0)
    while b < c["nfull"]:
        m_run, l_run = ws_qk_block(c, b, (m_run, l_run), False)
        b = b + 1
    while b < c["nblk"]:
        m_run, l_run = ws_qk_block(c, b, (m_run, l_run), True)
        b = b + 1
    vmwait(0)
    gpu.barrier()
    gpu.barrier()
    l_run = [butterfly(l_run[r], (32, 16)) for r in range(rt)]
    if lane < TILE:
        for r in range_constexpr(rt):
            fx.ptr_store(l_run[r], c["xm"] + (r * TILE + row) * WAVES + wave)
    gpu.barrier()
    if (wave == 0) & (lane < TILE):
        for r in range_constexpr(rt):
            lt = ws_l(c, r, row)
            empty = lt == fx.Float32(0.0)
            th = (c["split"] * c["S"] + c["t"]) * NH + c["h"][r]
            lse = m_run[r] + (hw_log2(lt) - fx.Float32(PLOG))
            lse = empty.select(fx.Float32(float("-inf")), lse * fx.Float32(LN2))
            bo.buffer_store(
                lse, rsrc(c["args"]["part_lse"]), th, cache_modifier=c["cm"]
            )


def ws_l(c, r, row):
    """Row ``row`` of row tile r's l: the QK waves' (xm) summed."""
    xw = fx.Vector(
        fx.ptr_load(
            c["xm"] + (r * TILE + row) * WAVES,
            result_type=fx.Vector.make_type(4, fx.Float32),
        )
    )
    return (xw[0] + xw[1]) + (xw[2] + xw[3])


@traced
def ws_pv_block(c, b, st):
    """PV wave step b: O += V^T P^T of block b - 1 (its V^T ``va`` read in
    step b - 1, P from xp, alpha from xm), B1 after ``WS_B1``
    latent tiles, V^T(b) read before it (block b's slot then free); after B1
    block b + 2's chunks DMA'd (block b + 1's landed by B2)."""
    lane = c["lane"]
    row = lane % TILE
    rt = c["rt"]
    o, va = st
    early = ws_dmas(c, b + 1, range(WS_K1, KLOADS), [f"m{i}" for i in range(WS_B1)])
    xp = c["xp"]
    al = [fx.ptr_load(c["xm"] + (r * TILE + row) * WAVES + QKW) for r in range(rt)]
    pv = []
    for r in range_constexpr(rt):
        for h in range_constexpr(2):
            pb = [
                fx.Vector(
                    fx.ptr_load(
                        xp + xp_at(r, h, k, lane),
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                )
                for k in range(2)
            ]
            pv.append(
                fx.Vector.from_elements(
                    [pb[w // 4][w % 4] for w in range(WAVES)], fx.Int32
                )
            )
    o = [
        fx.Vector.from_elements([o[8 * r + i][j] * al[r] for j in range(4)], fx.Float32)
        for r in range(rt)
        for i in range(8)
    ]
    nva = []
    dma = None
    for i in range_constexpr(8):
        if const_expr(i == WS_B1):
            nva += [ws_vtile(c, b, k) for k in range(WS_B1, 8)]
            idx_dma(c, b + 3)
            gpu.barrier()
            dma = ws_dmas(c, b + 2, range(WS_K1), [f"m{k}" for k in range(WS_B1, 8)])
        for r in range_constexpr(rt):
            for h in range_constexpr(2):
                o[8 * r + i] = mfma_f8(va[i], pv[2 * r + h], o[8 * r + i])
        if const_expr(i < WS_B1):
            nva.append(ws_vtile(c, b, i))
            early(f"m{i}")
        else:
            assert dma is not None
            dma(f"m{i}")
    if const_expr(WS_B1 >= 8):
        idx_dma(c, b + 3)
        gpu.barrier()
        ws_dmas(c, b + 2, range(WS_K1))("all")
    vmwait(0 if const_expr(c["dbg"] == 1) else WS_K1)
    gpu.barrier()
    return o, nva


@traced
def ws_pv(c):
    """The PV waves' steps: step 0 V^T(0) and block 1's DMA, then
    ``ws_pv_block`` a block (one behind the QK waves), then o = O / l."""
    lane, wave, g = c["lane"], c["wave"], c["lane"] // TILE
    row = lane % TILE
    rt = c["rt"]
    o = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(8 * rt)]
    # step 0: block 1 (landed by B2), block 3's indices (after block 1's
    # read theirs: the slot's), V^T(0), block 2 after B1
    ws_dmas(c, 1, range(KLOADS))("all")
    idx_dma(c, 3)
    va = [ws_vtile(c, 0, i) for i in range(8)]
    gpu.barrier()
    ws_dmas(c, 2, range(WS_K1))("all")
    vmwait(0 if const_expr(c["dbg"] == 1) else WS_K1)
    gpu.barrier()
    b = fx.Int32(1)
    while b < c["nblk"] + 1:
        o, va = ws_pv_block(c, b, (o, va))
        b = b + 1
    # the DMAs past the last block (clamped) land before the CTA's LDS goes
    vmwait(0)
    gpu.barrier()
    a = c["args"]
    v = wave - QKW
    for r in range_constexpr(rt):
        lt = ws_l(c, r, row)
        empty = lt == fx.Float32(0.0)
        inv = empty.select(fx.Float32(0.0), hw_rcp(lt) * c["vscale"])
        th = (c["split"] * c["S"] + c["t"]) * NH + c["h"][r]
        for hf in range_constexpr(2):
            pk = [
                [
                    bf16_pair(
                        o[8 * r + 4 * hf + i][j] * inv,
                        o[8 * r + 4 * hf + i][j + 1] * inv,
                    ).bitcast(fx.Int32)
                    for j in (0, 2)
                ]
                for i in range_constexpr(4)
            ]
            for n in range_constexpr(2):
                sw = [
                    permlane_swap(16, pk[2 * n][e], pk[2 * n + 1][e]) for e in range(2)
                ]
                lat = (2 * v + hf) * VCOLS + 16 * (2 * n + g % 2) + 4 * (g - g % 2)
                vv = fx.Vector.from_elements(
                    [sw[0][0], sw[1][0], sw[0][1], sw[1][1]], fx.Int32
                )
                if const_expr(c["dbg"] != 2):
                    bo.buffer_store(
                        vv,
                        rsrc(a["part_o"]),
                        (th * KVL + lat) // 2,
                        cache_modifier=c["cm"],
                    )


@traced
def ws_run(c):
    """Warp-specialized blocks (``WSPEC``): the QK waves', the PV waves'."""
    wv = uniform(c["wave"])
    if wv < QKW:
        ws_qk(c)
    if wv >= QKW:
        ws_pv(c)


def check(s: int, q: int, rt: int = 0):
    assert q in (4, 8) and s % q == 0
    # slot 1 holds the bf16 Q rows and their fp8 hi | lo (``q_split``)
    assert (rt or RTILES[q]) * TILE * ENTRY <= SLOT
    assert NH % geometry(q, rt)[0] == 0
    # warp-specialized: 2 row tiles
    assert not wspec(rt or RTILES[q]) or (rt or RTILES[q]) == 2
    assert splits(q, rt) % (s // q) == 0, (splits(q, rt), s // q)


def smem(qlen: int, rt: int = 0, ws: int = -1):
    """The LDS of a CTA (``cta``'s ks / xm / xp / xi; xp P's hi | lo, then
    the tail Q's tail hi | lo, ``qt``)."""
    rt = rt or RTILES[qlen]
    xpn = rt * WAVES * 64 * 2

    @fx.struct
    class Smem:
        ks: fx.Array[fx.Int32, 2 * SLOT, 16]
        xm: fx.Array[fx.Float32, rt * WAVES * TILE, 16]
        xp: fx.Array[fx.Int32, xpn + rt * TILE * 2 * CHUNK // 4, 16]
        xi: fx.Array[fx.Int32, 2 * WAVES * TILE, 16]

    return Smem


@traced
def cta(c):
    """CTA ``c["bid"]``'s (head group, split): its partial stored (cache
    modifier ``c["cm"]``), then ``c["done"](flag)`` when given (``ctas``).
    ``c["args"]``: qb, kv, pages, indptr, part_o, part_lse, scale, vscale."""
    s, q, a = c["S"], c["Q"], c["args"]
    rt = c.get("rt") or RTILES[q]
    ws = wspec(rt, c.get("ws", -1))
    ht, hg, spx = geometry(q, rt)
    nreq = s // q
    per_req = splits(q, rt) // nreq
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
        hd = [(u % hg) * ht + r * (TILE // q) + row // q for r in range(rt)]
        pos = row % q
        t = req * q + pos
        p0 = uniform(ld_i32(a["indptr"], req))
        L = uniform(ld_i32(a["indptr"], req + 1)) - p0
        # an empty request's p0 may be pages' end: its (unused) block-0 loads
        # read pages[0] instead
        p0 = (L > 0).select(p0, fx.Int32(0))
        ct = ((L + per_req - 1) // per_req + TILE - 1) // TILE * TILE
        c0 = split * ct
        c1 = fx.min(L, c0 + ct)
        ntile = (fx.max(c1 - c0, 0) + TILE - 1) // TILE
        dt, dcc = dma_lane(lane)
        # a new dict (the branch would carry a mutated ``c`` out as values)
        d = dict(
            c,
            rt=rt,
            # the line layout (a DMA 8 tokens' whole lines) at 1 row tile;
            # at 2 its extra VGPRs spill in the fused K2 builds (a reload a
            # block, whose vmcnt(0) waits out every DMA): 64 B chunks
            lines=rt == 1,
            lane=lane,
            wave=wave,
            dt=dt,
            dcc=dcc,
            dcol=16 * ((lane % 8) ^ sig(lane // 8, wave % 2)) if rt == 1 else 0,
            ks_b=fx.Int32(fx.ptrtoint(c["ks"])),
            xi_b=fx.Int32(fx.ptrtoint(c["xi"])),
            qt=c["xp"] + rt * WAVES * 64 * 2,
            p0=p0,
            c0=c0,
            last=fx.max(c1 - 1, 0),
        )
        # block 0's DMA before the Q loads (one in-order vmcnt: the indices'
        # wait would hold for them too)
        k_dma(d, 0, dma_offs(d, idx_load(d, 0)))
        idx_dma(d, 1)
        if const_expr(ws):
            idx_dma(d, 2)
        # the CTA's Q rows (16 r + row: [r][row][ENTRY] bf16) -> slot 1
        # once, a DMA a 1 KB (a wave every WAVES-th; the last repeated):
        # the waves' own loads of it would outweigh a K block 8 x
        rowb = 2 * ENTRY
        nq = rt * TILE * rowb // 1024
        for i in range_constexpr((nq + WAVES - 1) // WAVES):
            k = fx.min(wave + WAVES * i, nq - 1) * 64 + lane
            lr = k // (rowb // 16)
            qh = (u % hg) * ht + (lr // TILE) * (TILE // q) + (lr % TILE) // q
            qt = req * q + (lr % TILE) % q
            lds_dma(
                a["qb"] + fx.Int64((qt * NH + qh) * rowb + k % (rowb // 16) * 16),
                d["ks_b"] + (SLOT * 4 + (k - lane) * 16),
            )
        wait_dma()
        # Q -> fp8 hi / lo (slot 1 after the bf16 rows), then Q^T's B operands
        rows = rt * TILE
        at = SLOT + rows * ENTRY // 2
        q_split(d, rows, at)
        gpu.barrier()
        qv = [q_parts(d, r, at, rows) for r in range(rt)]
        scale = fx.Int32(a["scale"]).bitcast(fx.Float32) * fx.Float32(LOG2E)
        qsc = [scale * fx.ptr_load(c["xm"] + r * TILE + row) for r in range(rt)]
        # every wave's reads before block 1's DMA into slot 1
        gpu.barrier()
        d.update(
            qpos=(
                [lines_at(row, wave % 2, 2 * g + k) // 4 for k in range(2)]
                if rt == 1
                else [None, None]
            )
            + [4 * kpos(row, g)],
            vtr=[vtr_lane(lane, wave, i, rt == 1) for i in range(4)],
            vb=d["ks_b"] + 2 * (wave - QKW) * WAVES * TILEW * 4,
            q=qv,
            qscale=qsc,
            t=t,
            h=hd,
            split=req * per_req + split,
            bound=L - q + pos,
            nblk=(ntile + WAVES - 1) // WAVES,
            nfull=fx.max(fx.min(c1, L - q + 1) - c0, 0) // BLK,
            vscale=fx.Int32(a["vscale"]).bitcast(fx.Float32),
        )
        stamp(d, 0)
        if const_expr(ws):
            ws_run(d)
        else:
            run(d)
        if const_expr("done" in d):
            d["done"]((req * per_req + split) * hg + u % hg)


_BUILDS: dict = {}


def build(key: MlaAttnBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s, q = key.tokens, key.qlen
    check(s, q, key.rt)
    Smem = smem(q, key.rt, key.ws)

    keyed = key_tuple(key, _SOURCES)
    name = f"k3_mono_mla_attn_s{s}_q{q}_r{key.rt or RTILES[q]}"
    name += "_w" if key.wide else ""
    name += "_ws" if wspec(key.rt or RTILES[q], key.ws) else ""

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
                "rt": key.rt,
                "wide": key.wide,
                "ws": key.ws,
                "bid": bid,
                "tid": tid,
                "cm": 0,
                "ks": lds.ks.ptr,
                "xm": lds.xm.ptr,
                "xp": lds.xp.ptr,
                "xi": lds.xi.ptr,
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
    n = splits(key.qlen, key.rt)
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
    build(replace(key, wide=key.wide or wide_kv(kv)))(
        *a[:4],
        part_o.data_ptr(),
        part_lse.data_ptr(),
        *a[4:],
        stream=torch.cuda.current_stream(),
    )


def wide_kv(kv) -> bool:
    """The build's ``wide`` for KV pool ``kv``: a DMA's 32-bit offset off the
    pool's base (a token row's end at most) won't reach past 4 GiB."""
    return kv.numel() > 2**32


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
