# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501, SIM102
"""CuTe DSL cluster sparse decode (FP8 e4m3 KV, 16 selected pages of 128 tokens, MXFP8 output).

fp16-P online-softmax cluster decode. Every floating-point operation is spelled as an explicit PTX instruction
(`fma.rn.f32`, `mul.f32` / `add.f32` / `sub.f32`, `ex2.approx.f32`, `rcp.rn.f32`, `div.rn.f32`), so the
outputs do not depend on compiler contraction.

One cluster of CL CTAs (CL = 16 or 8) per (token row, KV head): CTA `rank` runs the selected pages
rank, rank + CL, ... (PPC = 16 / CL pages) with an online softmax in base 2 (S = q.k in fp32 from exact
fp8 -> fp16 operands on `mma.sync` m16n8k16, P rounded to fp16 before P @ V), then pushes its normalized partial of
Q head `row` and that head's log-sum-exp to CTA `row / HPC` over DSMEM (`st.async` completing an mbarrier);
CTA r merges HPC = HQ / CL heads (at least 1) by their LSE, writes the bf16 output and its MXFP8 quantization
(E4M3 values, UE8M0 scales in the 128x4 swizzled layout the o_proj GEMV reads plus a row-major copy).

HQ = Q heads per KV head: 16 (TP4) or 8 (TP8). The MMA keeps 16 rows; at 8 heads rows 8..15 of q are zero and their
partials are dropped, and at CL 16 only CTAs 0..7 merge (one head each); the others finish after their push.

PDL: the CTA arrives on the cluster barrier, then waits on the grid dependency (`griddepcontrol.wait`) and at once
lets its dependents launch. `prewait` (PRE builds): bit 0 reads the lens before the wait (they are written at the
step start), bit 1 also q (written before the previous kernel's own grid wait returned); only the page tables are
the previous kernel's output. PRE = False reads every input after the wait.

Kernel arguments are raw device addresses (int64) and byte strides, so one compiled kernel serves every tensor
of its (CL, HQ, PRE, min blocks, PDL) build.
"""

from __future__ import annotations

from typing import Any

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass import dsl_user_op

TOPK = 16
DIM = 128
PAGE = 128
THREADS = 128
PITCH = DIM + 8  # halves per fp16 row (272 B): conflict-free ldmatrix
KPITCH = DIM + 16  # bytes per E4M3 K row (144 B): conflict-free 2-byte B-fragment loads
MMA_ROWS = 16  # Q rows of the MMA tile

# Shared memory (bytes; the CUDA struct's layout): qs fp16 [16][PITCH], vs fp16
# [128][PITCH], ps fp16 [16][PITCH], ks e4m3 [128][KPITCH], recv f32 [16][128], recv_lse
# f32 [16], red f32 [4][16], rowm f32 [16], rowl f32 [16], rbar u64.
QS = 0
VS = QS + MMA_ROWS * PITCH * 2
PS = VS + PAGE * PITCH * 2
KS = PS + MMA_ROWS * PITCH * 2
RECV = KS + PAGE * KPITCH
R_LSE = RECV + TOPK * DIM * 4
RED = R_LSE + TOPK * 4
ROWM = RED + 4 * MMA_ROWS * 4
ROWL = ROWM + MMA_ROWS * 4
RBAR = ROWL + MMA_ROWS * 4
SMEM_BYTES = RBAR + 8
assert (QS, VS, PS, KS, RECV, R_LSE, RED, ROWL, RBAR) == (
    0,
    4352,
    39168,
    43520,
    61952,
    70144,
    70208,
    70528,
    70592,
)

NEG_INF = float("-inf")


SMEM_PER_SM = 228 * 1024  # SM100 family
RESERVED_SMEM_PER_CTA = 1024


def smem_alloc(cap: int) -> int:
    """Shared memory a CTA allocates: SMEM_BYTES, or (cap > 0) padded so that at most `cap` CTAs fit per SM (the
    residency that balances two waves of clusters better)."""
    if cap <= 0:
        return SMEM_BYTES
    need = SMEM_PER_SM // (cap + 1) - RESERVED_SMEM_PER_CTA + 1
    return max(SMEM_BYTES, (need + 1023) // 1024 * 1024)


def heads_per_cta(hq: int, cl: int) -> int:
    return max(1, hq // cl)


def recv_bytes(hq: int, cl: int) -> int:
    """Mbarrier transaction bytes a merging CTA expects: HPC * CL partials of 128 fp32 plus their LSE."""
    return heads_per_cta(hq, cl) * cl * (DIM + 1) * 4


# ------------------------------------------------------------------------------------------------ inline PTX
def _asm(ret, args, text, cons, side=True, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm

    return llvm.inline_asm(ret, [a.ir_value(loc=loc, ip=ip) for a in args], text, cons, has_side_effects=side,
                           is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)  # fmt: skip


def _i32t():
    from cutlass._mlir.extras import types as T

    return T.i32()


def _f32t():
    from cutlass._mlir.extras import types as T

    return T.f32()


def _unpack(out, ty, n, wrap, loc, ip):
    from cutlass._mlir.dialects import llvm

    return [wrap(llvm.extractvalue(ty, out, [i], loc=loc, ip=ip)) for i in range(n)]


def _struct(ty, n):
    from cutlass._mlir.dialects import llvm

    return llvm.StructType.get_literal([ty] * n)


F32, I32, I64 = cutlass.Float32, cutlass.Int32, cutlass.Int64


@dsl_user_op
def _f2(op, a, b, *, loc=None, ip=None):
    """Binary fp32 op as an explicit PTX instruction (`mul.f32`, `add.f32`, `sub.f32`, `max.f32`, `div.rn.f32`)."""
    return F32(
        _asm(
            _f32t(),
            [F32(a), F32(b)],
            f"{op} $0, $1, $2;",
            "=f,f,f",
            side=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f1(op, a, *, loc=None, ip=None):
    """Unary fp32 op (`ex2.approx.f32`, `lg2.approx.f32`, `rcp.rn.f32`, `rcp.approx.ftz.f32`, `abs.f32`)."""
    return F32(
        _asm(_f32t(), [F32(a)], f"{op} $0, $1;", "=f,f", side=False, loc=loc, ip=ip)
    )


@dsl_user_op
def _fma(a, b, c, *, loc=None, ip=None):
    return F32(_asm(_f32t(), [F32(a), F32(b), F32(c)], "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", side=False,
                    loc=loc, ip=ip))  # fmt: skip


def _mul(a, b):
    return _f2("mul.f32", a, b)


def _add(a, b):
    return _f2("add.f32", a, b)


def _sub(a, b):
    return _f2("sub.f32", a, b)


def _fmax(a, b):
    return _f2("max.f32", a, b)


def _ex2(a):
    return _f1("ex2.approx.f32", a)


@dsl_user_op
def _bits(x, *, loc=None, ip=None):
    return I32(
        _asm(_i32t(), [F32(x)], "mov.b32 $0, $1;", "=r,f", side=False, loc=loc, ip=ip)
    )


@dsl_user_op
def _float(w, *, loc=None, ip=None):
    return F32(
        _asm(_f32t(), [I32(w)], "mov.b32 $0, $1;", "=f,r", side=False, loc=loc, ip=ip)
    )


@dsl_user_op
def _shfl_bfly(x, off: int, *, loc=None, ip=None):
    return F32(
        _asm(
            _f32t(),
            [F32(x)],
            f"shfl.sync.bfly.b32 $0, $1, {off}, 31, -1;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cvt_e4m3x2(w, hi: bool, *, loc=None, ip=None):
    """The fp16x2 of the E4M3 pair in the low (or high) 16 bits of `w` (exact)."""
    sel = "{l, h}" if not hi else "{h, l}"  # mov.b32 {a, b}: a = the low half
    return I32(_asm(_i32t(), [I32(w)],
                    "{\n\t.reg .b16 l, h;\n\tmov.b32 " + sel + ", $1;\n\tcvt.rn.f16x2.e4m3x2 $0, l;\n\t}", "=r,r",
                    side=False, loc=loc, ip=ip))  # fmt: skip


def _cvt16(words):
    """16 E4M3 bytes (4 words) -> 16 fp16 (8 words), the CUDA `cvt16` order."""
    out = []
    for w in words:
        out.append(_cvt_e4m3x2(w, False))
        out.append(_cvt_e4m3x2(w, True))
    return out


@dsl_user_op
def _f16x2(lo, hi, *, loc=None, ip=None):
    """`__floats2half2_rn(lo, hi)` as a word."""
    return I32(_asm(_i32t(), [F32(hi), F32(lo)], "cvt.rn.f16x2.f32 $0, $1, $2;", "=r,f,f", side=False,
                    loc=loc, ip=ip))  # fmt: skip


@dsl_user_op
def _bf16_hi(x, *, loc=None, ip=None):
    """bf16(x) (RN) in the high half of a word (= the fp32 bits of the rounded value)."""
    return I32(_asm(_i32t(), [F32(x)],
                    "{\n\t.reg .b16 h, z;\n\tcvt.rn.bf16.f32 h, $1;\n\tmov.b16 z, 0;\n\tmov.b32 $0, {z, h};\n\t}",
                    "=r,f", side=False, loc=loc, ip=ip))  # fmt: skip


@dsl_user_op
def _e4m3_lo(x, *, loc=None, ip=None):
    """E4M3 (RN, satfinite) of x in the low byte (`cvt.rn.satfinite.e4m3x2.f32 q2, 0, x`)."""
    return I32(_asm(_i32t(), [F32(x)],
                    "{\n\t.reg .b16 q;\n\tcvt.rn.satfinite.e4m3x2.f32 q, 0f00000000, $1;\n\tcvt.u32.u16 $0, q;\n\t}",
                    "=r,f", side=False, loc=loc, ip=ip))  # fmt: skip


# shared memory (32-bit shared-window addresses)
@dsl_user_op
def _sts_v4(addr, w, *, loc=None, ip=None):
    _asm(None, [I32(addr), *(I32(x) for x in w)], "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r",
         loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def _sts_b32(addr, w, *, loc=None, ip=None):
    _asm(None, [I32(addr), I32(w)], "st.shared.b32 [$0], $1;", "r,r", loc=loc, ip=ip)


@dsl_user_op
def _sts_f32(addr, x, *, loc=None, ip=None):
    _asm(None, [I32(addr), F32(x)], "st.shared.f32 [$0], $1;", "r,f", loc=loc, ip=ip)


@dsl_user_op
def _lds_f32(addr, *, loc=None, ip=None):
    return F32(
        _asm(_f32t(), [I32(addr)], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip)
    )


@dsl_user_op
def _lds_cvt(addr, *, loc=None, ip=None):
    """The fp16x2 of the E4M3 pair at a shared address (2-byte load)."""
    return I32(_asm(_i32t(), [I32(addr)],
                    "{\n\t.reg .b16 h;\n\tld.shared.b16 h, [$1];\n\tcvt.rn.f16x2.e4m3x2 $0, h;\n\t}", "=r,r",
                    loc=loc, ip=ip))  # fmt: skip


@dsl_user_op
def _ldsm_x4(addr, trans: bool, *, loc=None, ip=None):
    t = ".trans" if trans else ""
    out = _asm(_struct(_i32t(), 4), [I32(addr)],
               f"ldmatrix.sync.aligned.m8n8.x4{t}.shared.b16 {{$0, $1, $2, $3}}, [$4];", "=r,=r,=r,=r,r",
               loc=loc, ip=ip)  # fmt: skip
    return _unpack(out, _i32t(), 4, I32, loc, ip)


@dsl_user_op
def _mma(c, a, b0, b1, *, loc=None, ip=None):
    """mma.sync m16n8k16 row.col f32 += f16 x f16."""
    out = _asm(_struct(_f32t(), 4), [*(I32(x) for x in a), I32(b0), I32(b1), *(F32(x) for x in c)],
               "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
               "{$10, $11, $12, $13};", "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", side=False, loc=loc, ip=ip)  # fmt: skip
    return _unpack(out, _f32t(), 4, F32, loc, ip)


@dsl_user_op
def _bar_sync(*, loc=None, ip=None):
    _asm(None, [], "bar.sync 0;", "~{memory}", loc=loc, ip=ip)


# cluster / mbarrier / PDL
@dsl_user_op
def _cluster_rank(*, loc=None, ip=None):
    return I32(
        _asm(
            _i32t(),
            [],
            "mov.u32 $0, %cluster_ctarank;",
            "=r",
            side=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mbar_init_expect(bar, nbytes, *, loc=None, ip=None):
    _asm(None, [I32(bar), I32(nbytes)],
         "mbarrier.init.shared::cta.b64 [$0], 1;\n\tfence.mbarrier_init.release.cluster;\n\t"
         "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;", "r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def _cluster_arrive_relaxed(*, loc=None, ip=None):
    _asm(
        None, [], "barrier.cluster.arrive.relaxed.aligned;", "~{memory}", loc=loc, ip=ip
    )


@dsl_user_op
def _cluster_wait(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.wait.aligned;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _grid_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _launch_dependents(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


@dsl_user_op
def _mapa(addr, rank, *, loc=None, ip=None):
    return I32(_asm(_i32t(), [I32(addr), I32(rank)], "mapa.shared::cluster.u32 $0, $1, $2;", "=r,r,r", side=False,
                    loc=loc, ip=ip))  # fmt: skip


@dsl_user_op
def _st_async_v2(addr, a, b, bar, *, loc=None, ip=None):
    _asm(None, [I32(addr), F32(a), F32(b), I32(bar)],
         "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [$0], {$1, $2}, [$3];", "r,f,f,r,~{memory}",
         loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def _st_async_f32(addr, a, bar, *, loc=None, ip=None):
    _asm(None, [I32(addr), F32(a), I32(bar)],
         "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [$0], $1, [$2];", "r,f,r,~{memory}",
         loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def _try_wait_cluster(bar, *, loc=None, ip=None):
    return I32(_asm(_i32t(), [I32(bar)],
                    "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], 0;\n\t"
                    "selp.u32 $0, 1, 0, p;\n\t}", "=r,r,~{memory}", loc=loc, ip=ip))  # fmt: skip


# global memory (64-bit addresses)
@dsl_user_op
def _ld_b32_if(addr, pred, other, *, loc=None, ip=None):
    """`pred ? *(int *)addr : other` (no load when not pred)."""
    return I32(_asm(_i32t(), [I64(addr), I32(pred), I32(other)],
                    "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b32 $0, $3;\n\t@p ld.global.b32 $0, [$1];\n\t}",
                    "=&r,l,r,r", loc=loc, ip=ip))  # fmt: skip


@dsl_user_op
def _ld_v4_if(addr, pred, nc: bool, *, loc=None, ip=None):
    """16 bytes at addr (`ld.global[.nc].v4`) when pred, else zeros."""
    sp = ".nc" if nc else ""
    out = _asm(_struct(_i32t(), 4), [I64(addr), I32(pred)],
               "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $5, 0;\n\tmov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\tmov.b32 $2, 0;\n\t"
               f"mov.b32 $3, 0;\n\t@p ld.global{sp}.v4.b32 {{$0, $1, $2, $3}}, [$4];\n\t}}", "=&r,=&r,=&r,=&r,l,r",
               loc=loc, ip=ip)  # fmt: skip
    return _unpack(out, _i32t(), 4, I32, loc, ip)


@dsl_user_op
def _ld_v4(addr, nc: bool, *, loc=None, ip=None):
    sp = ".nc" if nc else ""
    out = _asm(_struct(_i32t(), 4), [I64(addr)], f"ld.global{sp}.v4.b32 {{$0, $1, $2, $3}}, [$4];",
               "=r,=r,=r,=r,l", loc=loc, ip=ip)  # fmt: skip
    return _unpack(out, _i32t(), 4, I32, loc, ip)


@dsl_user_op
def _st_b16_hi(addr, w, *, loc=None, ip=None):
    """Store the high 16 bits of w."""
    _asm(None, [I64(addr), I32(w)], "{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, $1;\n\tst.global.b16 [$0], h;\n\t}",
         "l,r", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def _st_b8(addr, w, *, loc=None, ip=None):
    _asm(None, [I64(addr), I32(w)], "{\n\t.reg .b16 b;\n\tcvt.u16.u32 b, $1;\n\tst.global.b8 [$0], b;\n\t}", "l,r",
         loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def _red_release_add1(addr, *, loc=None, ip=None):
    _asm(
        None,
        [I64(addr)],
        "red.release.gpu.global.add.u64 [$0], 1;",
        "l,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_f32_if(addr, x, pred, *, loc=None, ip=None):
    _asm(None, [I32(addr), F32(x), I32(pred)],
         "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p st.shared.f32 [$0], $1;\n\t}", "r,f,r", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def _st_async_f32_if(addr, a, bar, pred, *, loc=None, ip=None):
    _asm(None, [I32(addr), F32(a), I32(bar), I32(pred)],
         "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $3, 0;\n\t"
         "@p st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [$0], $1, [$2];\n\t}", "r,f,r,r,~{memory}",
         loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def _st_b8_if(addr, w, pred, *, loc=None, ip=None):
    _asm(None, [I64(addr), I32(w), I32(pred)],
         "{\n\t.reg .pred p;\n\t.reg .b16 b;\n\tsetp.ne.b32 p, $2, 0;\n\tcvt.u16.u32 b, $1;\n\t"
         "@p st.global.b8 [$0], b;\n\t}", "l,r,r", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def _red_release_add1_if(addr, pred, *, loc=None, ip=None):
    _asm(None, [I64(addr), I32(pred)],
         "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $1, 0;\n\t@p red.release.gpu.global.add.u64 [$0], 1;\n\t}",
         "l,r,~{memory}", loc=loc, ip=ip)  # fmt: skip


# ------------------------------------------------------------------------------------------------
# device pieces Plain trace-time helpers (no staged control flow inside; every branch is
# a predicate), called from the kernel.
class _Ctx:
    """Per-thread values the pieces share (a namespace, filled by the kernel)."""

    base: Any
    CL: Any
    done_cnt: Any
    g: Any
    HPC: Any
    HQ: Any
    k_base: Any
    k_page: Any
    k_tok: Any
    kvh: Any
    kvr: Any
    l4: Any
    lane: Any
    ln: Any
    nsel: Any
    ntr: Any
    num_sf_k_tiles: Any
    out: Any
    out_tok: Any
    PPC: Any
    q_addr: Any
    q_dst: Any
    q8: Any
    qh: Any
    qk_scale_log2: Any
    rank: Any
    rbar: Any
    sf: Any
    sfp: Any
    st: Any
    tb: Any
    tch: Any
    tid: Any
    tok: Any
    trow: Any
    v_base: Any
    v_page: Any
    v_scale: Any
    v_tok: Any
    warp: Any


def _load_q(c):
    if c.HQ == MMA_ROWS:
        w = _ld_v4(c.q_addr, False)
    else:  # the MMA's padding rows (h >= HQ) are zero
        w = _ld_v4_if(c.q_addr, I32(c.qh < I32(c.HQ)), False)
    h16 = _cvt16(w)
    _sts_v4(c.q_dst, h16[0:4])
    _sts_v4(c.q_dst + I32(16), h16[4:8])


def _fetch(c, slot):
    """Register prefetch of one selected page into c.kvr (K words 0..31, V words 32..63; thread covers token rows
    i * 16 + tid / 8, bytes (tid % 8) * 16) and its token count into c.ntr (0 when the slot is past the selection)."""
    page = _ld_b32_if(c.tb + I64(slot) * I64(4), I32(slot < c.nsel), I32(-1))
    last = c.ln - (c.nsel - I32(1)) * I32(PAGE)
    nt = cutlass.select_(slot == c.nsel - I32(1), last, I32(PAGE))
    nt = cutlass.select_(page < I32(0), I32(0), nt)
    pg = I64(cutlass.select_(page > I32(0), page, I32(0)))
    kb = c.k_base + pg * c.k_page + I64(c.tch)
    vb = c.v_base + pg * c.v_page + I64(c.tch)
    for i in range(8):
        row = c.trow + I32(i * 16)
        valid = I32(row < nt)
        kw = _ld_v4_if(kb + I64(row) * c.k_tok, valid, True)
        vw = _ld_v4_if(vb + I64(row) * c.v_tok, valid, True)
        for e in range(4):
            c.kvr[4 * i + e] = kw[e]
            c.kvr[32 + 4 * i + e] = vw[e]
    c.ntr[0] = nt


def _row_reduce(c, op, dst):
    """dst[h] = op over the 4 warps' `red[w][h]` (threads h < 16; the others read row 0 and store nothing)."""
    lt = I32(c.tid < I32(MMA_ROWS))
    red_t = (
        c.base
        + I32(RED)
        + cutlass.select_(c.tid < I32(MMA_ROWS), c.tid, I32(0)) * I32(4)
    )
    r0 = _lds_f32(red_t)
    r1 = _lds_f32(red_t + I32(64))
    r2 = _lds_f32(red_t + I32(128))
    r3 = _lds_f32(red_t + I32(192))
    _sts_f32_if(c.base + I32(dst) + c.tid * I32(4), op(op(r0, r1), op(r2, r3)), lt)


def _page(c, j, slot):
    """One selected page: stage K / V, S = q k^T, online softmax update, O += P V (state in c.st)."""
    base, lane, warp, g, l4 = c.base, c.lane, c.warp, c.g, c.l4
    for i in range(8):
        row = c.trow + I32(i * 16)
        _sts_v4(
            base + I32(KS) + row * I32(KPITCH) + c.tch,
            [c.kvr[4 * i + e] for e in range(4)],
        )
        vh = _cvt16([c.kvr[32 + 4 * i + e] for e in range(4)])
        vd = base + I32(VS) + (row * I32(PITCH) + c.tch) * I32(2)
        _sts_v4(vd, vh[0:4])
        _sts_v4(vd + I32(16), vh[4:8])
    cur = c.ntr[0]
    _bar_sync()
    if j + 1 < c.PPC:
        _fetch(c, slot + I32(c.CL))  # in flight during this page's compute

    # S = q k^T: warp w covers tokens 32 w .. 32 w + 31 (4 n8 tiles).
    sacc = [[F32(0.0)] * 4 for _ in range(4)]
    a_row = (
        base
        + I32(QS)
        + ((lane & I32(15)) * I32(PITCH) + (lane >> I32(4)) * I32(8)) * I32(2)
    )
    k_row = (
        base + I32(KS) + (warp * I32(32) + (lane >> I32(2))) * I32(KPITCH) + l4 * I32(2)
    )
    for ks in range(DIM // 16):
        a = _ldsm_x4(a_row + I32(ks * 32), False)
        for jj in range(4):
            kr = k_row + I32(jj * 8 * KPITCH + ks * 16)
            sacc[jj] = _mma(sacc[jj], a, _lds_cvt(kr), _lds_cvt(kr + I32(8)))
    mloc = [F32(NEG_INF), F32(NEG_INF)]
    for jj in range(4):
        for i in range(4):
            t = warp * I32(32) + I32(jj * 8 + (i & 1)) + I32(2) * l4
            sv = cutlass.select_(
                t < cur, _fma(sacc[jj][i], c.qk_scale_log2, F32(0.0)), F32(NEG_INF)
            )
            sacc[jj][i] = sv
            mloc[i >> 1] = _fmax(mloc[i >> 1], sv)
    for r in range(2):
        mloc[r] = _fmax(mloc[r], _shfl_bfly(mloc[r], 1))
        mloc[r] = _fmax(mloc[r], _shfl_bfly(mloc[r], 2))
    red_w = base + I32(RED) + (warp * I32(MMA_ROWS) + g) * I32(4)
    lead = I32(l4 == I32(0))
    _sts_f32_if(red_w, mloc[0], lead)
    _sts_f32_if(red_w + I32(32), mloc[1], lead)
    _bar_sync()
    _row_reduce(c, _fmax, ROWM)
    _bar_sync()
    m_new = [None, None]
    fac = [None, None]
    for r in range(2):
        m_old = c.st[16 + r]
        m_new[r] = _fmax(m_old, _lds_f32(base + I32(ROWM) + (g + I32(8 * r)) * I32(4)))
        fac[r] = cutlass.select_(
            m_old == F32(NEG_INF), F32(0.0), _ex2(_sub(m_old, m_new[r]))
        )
        c.st[16 + r] = m_new[r]
    lloc = [F32(0.0), F32(0.0)]
    for jj in range(4):
        for i in (0, 2):
            r = i >> 1
            p0 = _ex2(_sub(sacc[jj][i], m_new[r]))
            p1 = _ex2(_sub(sacc[jj][i + 1], m_new[r]))
            lloc[r] = _add(lloc[r], _add(p0, p1))
            t = warp * I32(32) + I32(jj * 8) + I32(2) * l4
            _sts_b32(
                base + I32(PS) + ((g + I32(8 * r)) * I32(PITCH) + t) * I32(2),
                _f16x2(p0, p1),
            )
    for r in range(2):
        lloc[r] = _add(lloc[r], _shfl_bfly(lloc[r], 1))
        lloc[r] = _add(lloc[r], _shfl_bfly(lloc[r], 2))
    _bar_sync()
    _sts_f32_if(red_w, lloc[0], lead)
    _sts_f32_if(red_w + I32(32), lloc[1], lead)
    acc = [[c.st[4 * jj + i] for i in range(4)] for jj in range(4)]
    for jj in range(4):
        acc[jj][0] = _mul(fac[0], acc[jj][0])
        acc[jj][1] = _mul(fac[0], acc[jj][1])
        acc[jj][2] = _mul(fac[1], acc[jj][2])
        acc[jj][3] = _mul(fac[1], acc[jj][3])
    # O += P V: warp w covers dims 32 w .. 32 w + 31.
    p_row = (
        base
        + I32(PS)
        + ((lane & I32(15)) * I32(PITCH) + (lane >> I32(4)) * I32(8)) * I32(2)
    )
    v_t = (lane & I32(7)) + ((lane >> I32(3)) & I32(1)) * I32(8)
    v_d = warp * I32(32) + (lane >> I32(4)) * I32(8)
    for kk in range(PAGE // 16):
        a = _ldsm_x4(p_row + I32(kk * 32), False)
        for jp in range(2):
            b = _ldsm_x4(
                base
                + I32(VS)
                + ((v_t + I32(kk * 16)) * I32(PITCH) + v_d + I32(jp * 16)) * I32(2),
                True,
            )
            acc[2 * jp] = _mma(acc[2 * jp], a, b[0], b[1])
            acc[2 * jp + 1] = _mma(acc[2 * jp + 1], a, b[2], b[3])
    for jj in range(4):
        for i in range(4):
            c.st[4 * jj + i] = acc[jj][i]
    _bar_sync()
    _row_reduce(c, _add, ROWL)
    _bar_sync()
    for r in range(2):
        c.st[18 + r] = _fma(
            c.st[18 + r], fac[r], _lds_f32(base + I32(ROWL) + (g + I32(8 * r)) * I32(4))
        )
    _bar_sync()  # ks / vs / ps / rowl free for the next page


def _push(c, r):
    """Normalize the partial of Q row g + 8 r and push it with its LSE (log2) to CTA row / HPC, slot
    (row % HPC, rank). Rows >= HQ (the MMA's padding at 8 heads) push nothing."""
    l_r = c.st[18 + r]
    has = l_r > F32(0.0)
    scale = cutlass.select_(has, _f1("rcp.rn.f32", l_r), F32(0.0))
    lse = cutlass.select_(
        has, _add(c.st[16 + r], _f1("lg2.approx.f32", l_r)), F32(NEG_INF)
    )
    scale = _mul(scale, c.v_scale)
    row = c.g + I32(8 * r)
    dst_cta = row // I32(c.HPC)
    slot = (row % I32(c.HPC)) * I32(c.CL) + c.rank
    dst = _mapa(c.base + I32(RECV) + slot * I32(DIM * 4), dst_cta)
    rb = _mapa(c.rbar, dst_cta)
    for jj in range(4):
        d = c.warp * I32(32) + I32(jj * 8) + I32(2) * c.l4
        _st_async_v2(
            dst + d * I32(4),
            _mul(scale, c.st[4 * jj + 2 * r]),
            _mul(scale, c.st[4 * jj + 2 * r + 1]),
            rb,
        )
    lead = I32((c.warp == I32(0)) & (c.l4 == I32(0)))
    _st_async_f32_if(_mapa(c.base + I32(R_LSE) + slot * I32(4), dst_cta), lse, rb, lead)


def _merge(c):
    """Merge the HPC heads of this CTA by their LSE; bf16 output + MXFP8 (values, swizzled and row-major scales)."""
    base, tid, lane, warp, tok = c.base, c.tid, c.lane, c.warp, c.tok
    for hi in range(c.HPC):
        h = c.kvh * I32(c.HQ) + c.rank * I32(c.HPC) + I32(hi)
        lse_c = [_lds_f32(base + I32(R_LSE + (hi * c.CL + j) * 4)) for j in range(c.CL)]
        lse_max = F32(NEG_INF)
        for j in range(c.CL):
            lse_max = _fmax(lse_max, lse_c[j])
        lse_max = cutlass.select_(lse_max == F32(NEG_INF), F32(0.0), lse_max)
        denom = F32(0.0)
        for j in range(c.CL):
            lse_c[j] = _ex2(_sub(lse_c[j], lse_max))
            denom = _add(denom, lse_c[j])
        denom = cutlass.select_(denom > F32(0.0), denom, F32(1.0))
        o = F32(0.0)
        for j in range(c.CL):
            rv = _lds_f32(base + I32(RECV + (hi * c.CL + j) * DIM * 4) + tid * I32(4))
            o = _fma(rv, _f2("div.rn.f32", lse_c[j], denom), o)
        ob = _bf16_hi(o)
        col = h * I32(DIM) + tid
        _st_b16_hi(c.out + I64(tok) * c.out_tok + I64(col) * I64(2), ob)

        x = _float(ob)
        amax = _f1("abs.f32", x)
        for off in (16, 8, 4, 2, 1):
            amax = _fmax(amax, _shfl_bfly(amax, off))
        sfv = _mul(amax, _f1("rcp.approx.ftz.f32", F32(448.0)))
        sb = _bits(sfv)
        e = ((sb >> I32(23)) & I32(0xFF)) + I32((sb & I32(0x7FFFFF)) != I32(0))
        e = cutlass.select_(e < I32(254), e, I32(254))
        qs = cutlass.select_(
            amax != F32(0.0), _float((I32(254) - e) << I32(23)), F32(0.0)
        )
        xq = _mul(x, qs)
        _st_b8(
            c.q8 + I64(tok) * I64(c.num_sf_k_tiles * I32(128)) + I64(col), _e4m3_lo(xq)
        )
        k_sf = h * I32(4) + warp
        sf_off = (
            I64(tok // I32(128)) * I64(c.num_sf_k_tiles) * I64(512)
            + I64((k_sf >> I32(2)) * I32(512))
            + I64(
                (tok & I32(31)) * I32(16)
                + ((tok & I32(127)) >> I32(5)) * I32(4)
                + (k_sf & I32(3))
            )
        )
        l0 = lane == I32(0)
        _st_b8_if(c.sf + sf_off, e, I32(l0 & (c.sf != I64(0))))
        _st_b8_if(
            c.sfp + I64(tok) * I64(c.num_sf_k_tiles * I32(4)) + I64(k_sf), e, I32(l0)
        )


def _finish(c, sync: bool):
    if sync:
        _bar_sync()
    _red_release_add1_if(c.done_cnt, I32((c.tid == I32(0)) & (c.done_cnt != I64(0))))
    _launch_dependents()


# ------------------------------------------------------------------------------------------------ kernel
@cute.kernel
def sparse_decode_kernel(
    q: I64, q_tok: I64, q_head: I64,  # type: ignore[valid-type]  # e4m3 [T, Hq, 128]; byte strides
    k: I64, v: I64, k_page: I64, k_head: I64, k_tok: I64, v_page: I64, v_head: I64, v_tok: I64,  # type: ignore[valid-type]  # e4m3 pages
    tables: I64, tb_head: I64, tb_tok: I64,  # type: ignore[valid-type]  # int32 [Hkv, T, >= 16]; byte strides
    lens: I64, ln_head: I64,  # type: ignore[valid-type]  # int32 [Hkv, T]; byte stride of the head dim
    qk_scale_log2: F32, v_scale: F32,  # type: ignore[valid-type]
    out: I64, out_tok: I64,  # type: ignore[valid-type]  # bf16 [T, Hq * 128]; byte stride
    q8: I64, sf: I64, sfp: I64,  # type: ignore[valid-type]  # MXFP8 values / 128x4-swizzled scales (0 = none) / row-major scales
    done_cnt: I64,  # type: ignore[valid-type]  # u64 completion counter (0 = none)
    num_sf_k_tiles: I32, num_kv_heads: I32, prewait: I32,  # type: ignore[valid-type]
    CL: cutlass.Constexpr[int], HQ: cutlass.Constexpr[int], PRE: cutlass.Constexpr[bool],
    SMEM: cutlass.Constexpr[int],
):  # fmt: skip
    c = _Ctx()
    c.CL, c.HQ, c.PPC = CL, HQ, TOPK // CL
    c.HPC = heads_per_cta(HQ, CL)
    NRECV = HQ // c.HPC
    c.qk_scale_log2, c.v_scale = qk_scale_log2, v_scale
    c.k_page, c.v_page, c.k_tok, c.v_tok = k_page, v_page, k_tok, v_tok
    c.out, c.out_tok, c.q8, c.sf, c.sfp, c.done_cnt, c.num_sf_k_tiles = (
        out,
        out_tok,
        q8,
        sf,
        sfp,
        done_cnt,
        num_sf_k_tiles,
    )
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    c.tid = I32(tid)
    bid = I32(bid)
    c.lane = c.tid & I32(31)
    c.warp = c.tid >> I32(5)
    c.g = c.lane >> I32(2)
    c.l4 = c.lane & I32(3)
    cid = bid // I32(CL)
    c.tok = cid // num_kv_heads
    c.kvh = cid - c.tok * num_kv_heads
    c.rank = _cluster_rank()

    sm = cutlass.Array(
        cutlass.Uint8, SMEM, space=cutlass.AddressSpace.smem, alignment=1024
    )
    c.base = I32(sm.data_ptr().toint())
    c.rbar = c.base + I32(RBAR)

    if cutlass.const_expr(NRECV == CL):
        nbytes = I32(recv_bytes(HQ, CL))
    else:
        nbytes = cutlass.select_(c.rank < I32(NRECV), I32(recv_bytes(HQ, CL)), I32(0))
    if c.tid == I32(0):
        _mbar_init_expect(c.rbar, nbytes)
    _cluster_arrive_relaxed()

    # q -> fp16 rows of the MMA tile: thread covers head tid >> 3, bytes (tid & 7) * 16 .. + 15.
    c.qh = c.tid >> I32(3)
    qc = (c.tid & I32(7)) * I32(16)
    c.q_addr = q + I64(c.tok) * q_tok + I64(c.kvh * I32(HQ) + c.qh) * q_head + I64(qc)
    c.q_dst = c.base + I32(QS) + (c.qh * I32(PITCH) + qc) * I32(2)

    len_addr = lens + I64(c.kvh) * ln_head + I64(c.tok) * I64(4)
    if cutlass.const_expr(PRE):
        pre_len = I32((prewait & I32(1)) != I32(0))
        len_early = _ld_b32_if(len_addr, pre_len, I32(0))
        if (prewait & I32(2)) != I32(0):
            _load_q(c)
        _grid_wait()
        _launch_dependents()
        if (prewait & I32(2)) == I32(0):
            _load_q(c)
        ln = _ld_b32_if(len_addr, I32(1) - pre_len, len_early)
    else:
        _grid_wait()
        _launch_dependents()
        _load_q(c)
        ln = _ld_b32_if(len_addr, I32(1), I32(0))
    c.ln = ln
    nsel = cutlass.select_(ln > I32(0), ln, I32(0))
    nsel = (nsel + I32(PAGE - 1)) >> I32(7)
    c.nsel = cutlass.select_(nsel < I32(TOPK), nsel, I32(TOPK))
    c.tb = tables + I64(c.kvh) * tb_head + I64(c.tok) * tb_tok
    c.k_base = k + I64(c.kvh) * k_head
    c.v_base = v + I64(c.kvh) * v_head
    c.trow = c.tid >> I32(3)
    c.tch = (c.tid & I32(7)) * I32(16)

    # Register prefetch of one page and its token count, and the softmax state (acc
    # [4][4], m_row [2], l_row [2]), in rmem across the page loop.
    c.kvr = cute.make_rmem_tensor((64,), I32)
    c.ntr = cute.make_rmem_tensor((1,), I32)
    c.st = cute.make_rmem_tensor((20,), F32)
    for i in cutlass.range_constexpr(16):
        c.st[i] = F32(0.0)
    c.st[16] = F32(NEG_INF)
    c.st[17] = F32(NEG_INF)
    c.st[18] = F32(0.0)
    c.st[19] = F32(0.0)

    _fetch(c, c.rank)
    for j in cutlass.range_constexpr(TOPK // CL):
        slot = c.rank + I32(j * CL)
        if slot < c.nsel:  # slots increase in j; uniform across the CTA
            _page(c, j, slot)

    _cluster_wait()
    for r in cutlass.range_constexpr(2):
        if cutlass.const_expr(
            HQ == MMA_ROWS or 8 * r < HQ
        ):  # rows >= HQ are the MMA's padding (8 heads: r = 1)
            _push(c, r)

    if cutlass.const_expr(NRECV == CL):
        while _try_wait_cluster(c.rbar) == I32(0):
            pass
        _merge(c)
        _finish(c, True)
    else:
        if c.rank < I32(NRECV):
            while _try_wait_cluster(c.rbar) == I32(0):
                pass
            _merge(c)
            _finish(c, True)
        else:  # no head to merge (8 heads, CL 16)
            _finish(c, False)


@cute.jit
def sparse_decode_launch(
    q: I64, q_tok: I64, q_head: I64,  # type: ignore[valid-type]
    k: I64, v: I64, k_page: I64, k_head: I64, k_tok: I64, v_page: I64, v_head: I64, v_tok: I64,  # type: ignore[valid-type]
    tables: I64, tb_head: I64, tb_tok: I64, lens: I64, ln_head: I64,  # type: ignore[valid-type]
    qk_scale_log2: F32, v_scale: F32,  # type: ignore[valid-type]
    out: I64, out_tok: I64, q8: I64, sf: I64, sfp: I64, done_cnt: I64,  # type: ignore[valid-type]
    num_sf_k_tiles: I32, num_kv_heads: I32, prewait: I32, grid: I32,  # type: ignore[valid-type]
    CL: cutlass.Constexpr[int], HQ: cutlass.Constexpr[int], PRE: cutlass.Constexpr[bool],
    MINB: cutlass.Constexpr[int], use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream, CAP: cutlass.Constexpr[int] = 0,
):  # fmt: skip
    sparse_decode_kernel(
        q, q_tok, q_head, k, v, k_page, k_head, k_tok, v_page, v_head, v_tok, tables, tb_head, tb_tok, lens, ln_head,
        qk_scale_log2, v_scale, out, out_tok, q8, sf, sfp, done_cnt, num_sf_k_tiles, num_kv_heads, prewait,
        CL, HQ, PRE, smem_alloc(CAP),
    ).launch(
        grid=(grid, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(CL, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
        min_blocks_per_mp=MINB,
        max_number_threads=[THREADS, 1, 1],
    )  # fmt: skip
