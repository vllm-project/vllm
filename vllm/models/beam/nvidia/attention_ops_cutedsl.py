# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exact BF16 output gate in CuTe DSL."""

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.utils.torch_utils import current_stream

_LDQ = "ld.global.nc.L1::no_allocate"


def _asm(res_ty, operands, text, constraints, loc=None, ip=None, side_effects=False):
    return llvm.inline_asm(
        res_ty,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_v4(addr: Int64, pf="", *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        _LDQ + pf + ".v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def make_policy(kind: str, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [],
            f"createpolicy.fractional.L2::{kind}.b64 $0, 1.0;",
            "=l",
            loc,
            ip,
        )
    )


@dsl_user_op
def st_v4_hint(addr: Int64, words, pol: Int64, *, loc=None, ip=None):
    ops = [Int64(addr).ir_value(loc=loc, ip=ip)] + [
        Uint32(w).ir_value(loc=loc, ip=ip) for w in words
    ]
    ops.append(Int64(pol).ir_value(loc=loc, ip=ip))
    _asm(
        None,
        ops,
        "st.global.L2::cache_hint.v4.b32 [$0], {$1, $2, $3, $4}, $5;",
        "l,r,r,r,r,l",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def ld_v2(addr: Int64, pf=".L2::128B", *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        _LDQ + pf + ".v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc,
        ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(2)
    ]


@dsl_user_op
def ld_b32(addr: Int64, *, loc=None, ip=None):
    return [
        Uint32(
            _asm(
                T.i32(),
                [Int64(addr).ir_value(loc=loc, ip=ip)],
                _LDQ + ".L2::128B.b32 $0, [$1];",
                "=r,l",
                loc,
                ip,
            )
        )
    ]


@dsl_user_op
def ld_b16(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 h;\nld.global.nc.b16 h, [$1];\ncvt.u32.u16 $0, h;\n}",
            "=r,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def ld_b16_l2_128(addr: Int64, pf=".L2::128B", *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 h;\n" + _LDQ + pf + ".b16 h, [$1];\ncvt.u32.u16 $0, h;\n}",
            "=r,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def ld_bias_b16x2(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 h;\nld.global.nc.b16 h, [$1];\nmov.b32 $0, {h, h};\n}",
            "=r,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def st_vec(addr: Int64, words, *, loc=None, ip=None):
    n = len(words)
    ops = [Int64(addr).ir_value(loc=loc, ip=ip)] + [
        Uint32(w).ir_value(loc=loc, ip=ip) for w in words
    ]
    if n == 4:
        text = "st.global.v4.b32 [$0], {$1, $2, $3, $4};"
    elif n == 2:
        text = "st.global.v2.b32 [$0], {$1, $2};"
    else:
        text = "st.global.b32 [$0], $1;"
    _asm(None, ops, text, "l" + ",r" * n, loc, ip, side_effects=True)


@dsl_user_op
def st_b16(addr: Int64, word, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(word).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 l, h;\nmov.b32 {l, h}, $1;\nst.global.b16 [$0], l;\n}",
        "l,r",
        loc,
        ip,
        side_effects=True,
    )


def _gate_ptx(has_bias, has_temp, has_floor):
    s = [
        "{",
        ".reg .b32 w, lo, hi;",
        ".reg .b64 p, kl, k1;",
        ".reg .f32 a0, a1;",
        "mov.b64 kl, 0xBFB8AA3BBFB8AA3B;",
        "mov.b64 k1, 0x3F8000003F800000;",
    ]
    w = "$1"
    if has_bias:
        s.append("add.rn.bf16x2 w, $1, $3;")
        w = "w"

    def unpack(src):
        s.append(f"shl.b32 lo, {src}, 16;")
        s.append(f"and.b32 hi, {src}, 0xffff0000;")

    if has_temp:
        unpack(w)
        s += [
            "mov.b32 a0, lo;",
            "mov.b32 a1, hi;",
            "div.full.f32 a0, a0, $4;",
            "div.full.f32 a1, a1, $4;",
            "cvt.rn.bf16x2.f32 w, a1, a0;",
        ]
        w = "w"
    unpack(w)
    s += [
        "mov.b64 p, {lo, hi};",
        "mul.rn.f32x2 p, p, kl;",
        "mov.b64 {a0, a1}, p;",
        "ex2.approx.ftz.f32 a0, a0;",
        "ex2.approx.ftz.f32 a1, a1;",
        "mov.b64 p, {a0, a1};",
        "add.rn.f32x2 p, p, k1;",
        "mov.b64 {a0, a1}, p;",
        "rcp.approx.ftz.f32 a0, a0;",
        "rcp.approx.ftz.f32 a1, a1;",
        "cvt.rn.bf16x2.f32 w, a1, a0;",
    ]
    if has_floor:
        s += [".reg .b64 kc, kf;", "mov.b64 kc, {$5, $5};", "mov.b64 kf, {$6, $6};"]
        unpack("w")
        s += [
            "mov.b64 p, {lo, hi};",
            "mul.rn.f32x2 p, p, kc;",
            "mov.b64 {a0, a1}, p;",
            "cvt.rn.bf16x2.f32 w, a1, a0;",
        ]
        unpack("w")
        s += [
            "mov.b64 p, {lo, hi};",
            "add.rn.f32x2 p, p, kf;",
            "mov.b64 {a0, a1}, p;",
            "cvt.rn.bf16x2.f32 w, a1, a0;",
        ]
    s += ["mul.rn.bf16x2 $0, w, $2;", "}"]
    return "\n".join(s)


@dsl_user_op
def gate_pair(
    g, x, b, temp, c1, flo, has_bias, has_temp, has_floor, *, loc=None, ip=None
) -> Uint32:
    ops = [
        Uint32(g).ir_value(loc=loc, ip=ip),
        Uint32(x).ir_value(loc=loc, ip=ip),
        Uint32(b).ir_value(loc=loc, ip=ip),
        Float32(temp).ir_value(loc=loc, ip=ip),
        Float32(c1).ir_value(loc=loc, ip=ip),
        Float32(flo).ir_value(loc=loc, ip=ip),
    ]
    return Uint32(
        _asm(
            T.i32(),
            ops,
            _gate_ptx(has_bias, has_temp, has_floor),
            "=r,r,r,r,f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def gate_pair_default_bias(g, x, b, *, loc=None, ip=None) -> Uint32:
    ops = [
        Uint32(g).ir_value(loc=loc, ip=ip),
        Uint32(x).ir_value(loc=loc, ip=ip),
        Uint32(b).ir_value(loc=loc, ip=ip),
    ]
    return Uint32(
        _asm(T.i32(), ops, _gate_ptx(True, False, False), "=r,r,r,r", loc, ip)
    )


@dsl_user_op
def gate_pair_default_nobias(g, x, *, loc=None, ip=None) -> Uint32:
    ops = [Uint32(g).ir_value(loc=loc, ip=ip), Uint32(x).ir_value(loc=loc, ip=ip)]
    return Uint32(_asm(T.i32(), ops, _gate_ptx(False, False, False), "=r,r,r", loc, ip))


def gate_pair_selected(g, x, b, temp, c1, flo, has_bias, has_temp, has_floor):
    if const_expr(not has_temp and not has_floor):
        if const_expr(has_bias):
            return gate_pair_default_bias(g, x, b)
        return gate_pair_default_nobias(g, x)
    return gate_pair(g, x, b, temp, c1, flo, has_bias, has_temp, has_floor)


def _gate_vec_exp_interleaved_default_ptx(has_bias, npairs):
    """Two BF16x2 pairs with independent MUFU operations issued together."""
    assert npairs == 2
    s = [
        "{",
        ".reg .b32 w0, w1, lo, hi;",
        ".reg .b64 p0, p1, kl, k1;",
        ".reg .f32 a0, a1, a2, a3;",
        "mov.b64 kl, 0xBFB8AA3BBFB8AA3B;",
        "mov.b64 k1, 0x3F8000003F800000;",
    ]
    g0, g1 = "$2", "$3"
    if has_bias:
        s += ["add.rn.bf16x2 w0, $2, $6;", "add.rn.bf16x2 w1, $3, $6;"]
        g0, g1 = "w0", "w1"
    s += [
        f"shl.b32 lo, {g0}, 16;",
        f"and.b32 hi, {g0}, 0xffff0000;",
        "mov.b64 p0, {lo, hi};",
        "mul.rn.f32x2 p0, p0, kl;",
        "mov.b64 {a0, a1}, p0;",
        f"shl.b32 lo, {g1}, 16;",
        f"and.b32 hi, {g1}, 0xffff0000;",
        "mov.b64 p1, {lo, hi};",
        "mul.rn.f32x2 p1, p1, kl;",
        "mov.b64 {a2, a3}, p1;",
        "ex2.approx.ftz.f32 a0, a0;",
        "ex2.approx.ftz.f32 a1, a1;",
        "ex2.approx.ftz.f32 a2, a2;",
        "ex2.approx.ftz.f32 a3, a3;",
        "mov.b64 p0, {a0, a1};",
        "mov.b64 p1, {a2, a3};",
        "add.rn.f32x2 p0, p0, k1;",
        "add.rn.f32x2 p1, p1, k1;",
        "mov.b64 {a0, a1}, p0;",
        "mov.b64 {a2, a3}, p1;",
        "rcp.approx.ftz.f32 a0, a0;",
        "rcp.approx.ftz.f32 a1, a1;",
        "rcp.approx.ftz.f32 a2, a2;",
        "rcp.approx.ftz.f32 a3, a3;",
        "cvt.rn.bf16x2.f32 w0, a1, a0;",
        "cvt.rn.bf16x2.f32 w1, a3, a2;",
        "mul.rn.bf16x2 $0, w0, $4;",
        "mul.rn.bf16x2 $1, w1, $5;",
        "}",
    ]
    return "\n".join(s)


@dsl_user_op
def gate_vec_exp_interleaved_default(gs, xs, b, *, loc=None, ip=None):
    n = len(gs)
    ops = [Uint32(v).ir_value(loc=loc, ip=ip) for v in gs + xs]
    if b is not None:
        ops.append(Uint32(b).ir_value(loc=loc, ip=ip))
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * n),
        ops,
        _gate_vec_exp_interleaved_default_ptx(b is not None, n),
        ",".join(["=r"] * n + ["r"] * len(ops)),
        loc,
        ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(n)
    ]


def _gate_vec_default_ptx(has_bias, npairs):
    s = [
        "{",
        ".reg .b32 w, lo, hi;",
        ".reg .b64 p, kl, k1;",
        ".reg .f32 a0, a1;",
        "mov.b64 kl, 0xBFB8AA3BBFB8AA3B;",
        "mov.b64 k1, 0x3F8000003F800000;",
    ]
    for j in range(npairs):
        g = f"${npairs + j}"
        x = f"${2 * npairs + j}"
        if has_bias:
            s.append(f"add.rn.bf16x2 w, {g}, ${3 * npairs};")
            g = "w"
        s += [
            f"shl.b32 lo, {g}, 16;",
            f"and.b32 hi, {g}, 0xffff0000;",
            "mov.b64 p, {lo, hi};",
            "mul.rn.f32x2 p, p, kl;",
            "mov.b64 {a0, a1}, p;",
            "ex2.approx.ftz.f32 a0, a0;",
            "ex2.approx.ftz.f32 a1, a1;",
            "mov.b64 p, {a0, a1};",
            "add.rn.f32x2 p, p, k1;",
            "mov.b64 {a0, a1}, p;",
            "rcp.approx.ftz.f32 a0, a0;",
            "rcp.approx.ftz.f32 a1, a1;",
            "cvt.rn.bf16x2.f32 w, a1, a0;",
            f"mul.rn.bf16x2 ${j}, w, {x};",
        ]
    s.append("}")
    return "\n".join(s)


@dsl_user_op
def gate_vec_default(gs, xs, b, *, loc=None, ip=None):
    """Default-math (floor=0, temperature=1) gate for 2 or 4 packed BF16x2 pairs."""
    n = len(gs)
    ops = [Uint32(v).ir_value(loc=loc, ip=ip) for v in gs + xs]
    if b is not None:
        ops.append(Uint32(b).ir_value(loc=loc, ip=ip))
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * n),
        ops,
        _gate_vec_default_ptx(b is not None, n),
        ",".join(["=r"] * n + ["r"] * len(ops)),
        loc,
        ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(n)
    ]


def _gate_ptx_scalar(has_bias, has_temp, has_floor):
    s = [
        "{",
        ".reg .b16 g, b, x, w;",
        ".reg .f32 a, r;",
        "cvt.u16.u32 g, $1;",
        "cvt.u16.u32 x, $2;",
    ]
    w = "g"
    if has_bias:
        s += ["cvt.u16.u32 b, $3;", "add.rn.bf16 w, g, b;"]
        w = "w"
    if has_temp:
        s += [
            f"cvt.f32.bf16 a, {w};",
            "div.full.f32 a, a, $4;",
            "cvt.rn.bf16.f32 w, a;",
        ]
        w = "w"
    s += [
        f"cvt.f32.bf16 a, {w};",
        "mul.rn.f32 a, a, 0fBFB8AA3B;",
        "ex2.approx.ftz.f32 a, a;",
        "add.rn.f32 a, a, 0f3F800000;",
        "rcp.approx.ftz.f32 r, a;",
        "cvt.rn.bf16.f32 w, r;",
    ]
    if has_floor:
        s += [
            "cvt.f32.bf16 a, w;",
            "mul.rn.f32 a, a, $5;",
            "cvt.rn.bf16.f32 w, a;",
            "cvt.f32.bf16 a, w;",
            "add.rn.f32 a, a, $6;",
            "cvt.rn.bf16.f32 w, a;",
        ]
    s += ["mul.rn.bf16 w, w, x;", "mov.b32 $0, {w, w};", "}"]
    return "\n".join(s)


@dsl_user_op
def gate_one(
    g, x, b, temp, c1, flo, has_bias, has_temp, has_floor, *, loc=None, ip=None
) -> Uint32:
    ops = [
        Uint32(g).ir_value(loc=loc, ip=ip),
        Uint32(x).ir_value(loc=loc, ip=ip),
        Uint32(b).ir_value(loc=loc, ip=ip),
        Float32(temp).ir_value(loc=loc, ip=ip),
        Float32(c1).ir_value(loc=loc, ip=ip),
        Float32(flo).ir_value(loc=loc, ip=ip),
    ]
    return Uint32(
        _asm(
            T.i32(),
            ops,
            _gate_ptx_scalar(has_bias, has_temp, has_floor),
            "=r,r,r,r,f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def gate_one_default_bias(g, x, b, *, loc=None, ip=None) -> Uint32:
    ops = [
        Uint32(g).ir_value(loc=loc, ip=ip),
        Uint32(x).ir_value(loc=loc, ip=ip),
        Uint32(b).ir_value(loc=loc, ip=ip),
    ]
    return Uint32(
        _asm(T.i32(), ops, _gate_ptx_scalar(True, False, False), "=r,r,r,r", loc, ip)
    )


@dsl_user_op
def gate_one_default_nobias(g, x, *, loc=None, ip=None) -> Uint32:
    ops = [Uint32(g).ir_value(loc=loc, ip=ip), Uint32(x).ir_value(loc=loc, ip=ip)]
    return Uint32(
        _asm(T.i32(), ops, _gate_ptx_scalar(False, False, False), "=r,r,r", loc, ip)
    )


def gate_one_selected(g, x, b, temp, c1, flo, has_bias, has_temp, has_floor):
    if const_expr(not has_temp and not has_floor):
        if const_expr(has_bias):
            return gate_one_default_bias(g, x, b)
        return gate_one_default_nobias(g, x)
    return gate_one(g, x, b, temp, c1, flo, has_bias, has_temp, has_floor)


def _ld_words(addr, VEC, PF256):
    if const_expr(VEC == 8):
        return ld_v4(addr, ".L2::256B" if PF256 else "")
    elif const_expr(VEC == 4):
        return ld_v2(addr, ".L2::256B" if PF256 else ".L2::128B")
    elif const_expr(VEC == 2):
        return ld_b32(addr)
    return [ld_b16_l2_128(addr, ".L2::256B" if PF256 else ".L2::128B")]


@cute.kernel
def _gate2d_kernel(
    x_addr: Int64,
    g_addr: Int64,
    b_addr: Int64,
    o_addr: Int64,
    n_rows: Int32,
    temp: Float32,
    c1: Float32,
    flo: Float32,
    W: cutlass.Constexpr,
    D: cutlass.Constexpr,
    G_STRIDE: cutlass.Constexpr,
    VEC: cutlass.Constexpr,
    E: cutlass.Constexpr,
    R: cutlass.Constexpr,
    BLOCK: cutlass.Constexpr,
    BIAS_MODE: cutlass.Constexpr,
    HAS_TEMP: cutlass.Constexpr,
    HAS_FLOOR: cutlass.Constexpr,
    ROW_EXACT: cutlass.Constexpr,
    ONE_ROW: cutlass.Constexpr,
    ROWS_ON_X: cutlass.Constexpr,
    PF256: cutlass.Constexpr,
    INTERLEAVED8: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    if const_expr(not ROWS_ON_X):
        bx, by = by, bx
    cta_cols = BLOCK * VEC * E
    col_exact = W % cta_cols == 0
    nwords = (VEC + 1) // 2
    has_bias = BIAS_MODE != 0
    row0 = Uint32(0) if const_expr(ONE_ROW) else Uint32(bx) * Uint32(R)
    colbase = Uint32(by) * Uint32(cta_cols) + Uint32(tidx) * Uint32(VEC)

    cols = []
    colcs = []
    bws = []
    for e in cutlass.range_constexpr(E):
        col = colbase + Uint32(e * BLOCK * VEC)
        colc = col if const_expr(col_exact) else cutlass.min(col, Uint32(W - VEC))
        cols.append(col)
        colcs.append(colc)
        if const_expr(BIAS_MODE == 1):
            bw = [ld_bias_b16x2(b_addr + Int64(colc // Uint32(D)) * 2)] * nwords
        elif const_expr(BIAS_MODE == 2):
            bw = []
            for j in cutlass.range_constexpr(nwords):
                lo = ld_b16(b_addr + Int64((colc + Uint32(2 * j)) // Uint32(D)) * 2)
                if const_expr(VEC > 1):
                    hi = ld_b16(
                        b_addr + Int64((colc + Uint32(2 * j + 1)) // Uint32(D)) * 2
                    )
                    bw.append(lo | (hi << 16))
                else:
                    bw.append(lo)
        else:
            bw = [Uint32(0)] * nwords
        bws.append(bw)

    rows = []
    xs = []
    gs = []
    for r in cutlass.range_constexpr(R):
        row = Uint32(0) if const_expr(ONE_ROW) else row0 + Uint32(r)
        rowc = (
            row
            if const_expr(ONE_ROW or ROW_EXACT)
            else cutlass.min(row, Uint32(n_rows) - Uint32(1))
        )
        rows.append(row)
        if const_expr(ONE_ROW):
            x_row = x_addr
            g_row = g_addr
        else:
            x_row = x_addr + Int64(rowc) * (W * 2)
            g_row = g_addr + Int64(rowc) * (G_STRIDE * 2)
        xr = []
        gr = []
        for e in cutlass.range_constexpr(E):
            xr.append(_ld_words(x_row + Int64(colcs[e]) * 2, VEC, PF256))
            gr.append(_ld_words(g_row + Int64(colcs[e]) * 2, VEC, PF256))
        xs.append(xr)
        gs.append(gr)

    outs = []
    if const_expr(VEC == 1):
        flat_g = []
        flat_x = []
        flat_b = []
        for r in cutlass.range_constexpr(R):
            for e in cutlass.range_constexpr(E):
                flat_g.append(gs[r][e][0])
                flat_x.append(xs[r][e][0])
                flat_b.append(bws[e][0])
        n = R * E
        flat_o = []
        for i in cutlass.range_constexpr(0, n, 2):
            if const_expr(i + 1 < n):
                rr = gate_pair_selected(
                    flat_g[i] | (flat_g[i + 1] << 16),
                    flat_x[i] | (flat_x[i + 1] << 16),
                    (flat_b[i] & Uint32(0xFFFF)) | (flat_b[i + 1] << 16),
                    temp,
                    c1,
                    flo,
                    has_bias,
                    HAS_TEMP,
                    HAS_FLOOR,
                )
                flat_o.append(rr)
                flat_o.append(rr >> 16)
            else:
                flat_o.append(
                    gate_one_selected(
                        flat_g[i],
                        flat_x[i],
                        flat_b[i],
                        temp,
                        c1,
                        flo,
                        has_bias,
                        HAS_TEMP,
                        HAS_FLOOR,
                    )
                )
        for r in cutlass.range_constexpr(R):
            outs.append([[flat_o[r * E + e]] for e in range(E)])
    else:
        for r in cutlass.range_constexpr(R):
            orow = []
            for e in cutlass.range_constexpr(E):
                if const_expr(
                    VEC >= 4 and not HAS_TEMP and not HAS_FLOOR and BIAS_MODE == 1
                ):
                    if const_expr(VEC == 4):
                        ow = gate_vec_exp_interleaved_default(
                            gs[r][e], xs[r][e], bws[e][0]
                        )
                    elif const_expr(INTERLEAVED8):
                        ow = gate_vec_exp_interleaved_default(
                            gs[r][e][0:2], xs[r][e][0:2], bws[e][0]
                        ) + gate_vec_exp_interleaved_default(
                            gs[r][e][2:4], xs[r][e][2:4], bws[e][0]
                        )
                    else:
                        ow = gate_vec_default(gs[r][e], xs[r][e], bws[e][0])
                elif const_expr(
                    VEC >= 4 and not HAS_TEMP and not HAS_FLOOR and BIAS_MODE == 0
                ):
                    if const_expr(VEC == 4):
                        ow = gate_vec_exp_interleaved_default(gs[r][e], xs[r][e], None)
                    elif const_expr(INTERLEAVED8):
                        ow = gate_vec_exp_interleaved_default(
                            gs[r][e][0:2], xs[r][e][0:2], None
                        ) + gate_vec_exp_interleaved_default(
                            gs[r][e][2:4], xs[r][e][2:4], None
                        )
                    else:
                        ow = gate_vec_default(gs[r][e], xs[r][e], None)
                else:
                    ow = []
                    for j in cutlass.range_constexpr(nwords):
                        ow.append(
                            gate_pair_selected(
                                gs[r][e][j],
                                xs[r][e][j],
                                bws[e][j],
                                temp,
                                c1,
                                flo,
                                has_bias,
                                HAS_TEMP,
                                HAS_FLOOR,
                            )
                        )
                orow.append(ow)
            outs.append(orow)

    st_pol = make_policy("evict_last") if const_expr(VEC == 8) else None
    for r in cutlass.range_constexpr(R):
        o_row = o_addr if const_expr(ONE_ROW) else o_addr + Int64(rows[r]) * (W * 2)
        for e in cutlass.range_constexpr(E):
            addr = o_row + Int64(cols[e]) * 2
            if const_expr(ROW_EXACT and col_exact):
                if const_expr(VEC == 1):
                    st_b16(addr, outs[r][e][0])
                elif const_expr(st_pol is not None):
                    st_v4_hint(addr, outs[r][e], st_pol)
                else:
                    st_vec(addr, outs[r][e])
            else:
                if const_expr(ROW_EXACT):
                    ok = cols[e] < Uint32(W)
                elif const_expr(col_exact):
                    ok = rows[r] < Uint32(n_rows)
                else:
                    ok = (rows[r] < Uint32(n_rows)) & (cols[e] < Uint32(W))
                if ok:
                    if const_expr(VEC == 1):
                        st_b16(addr, outs[r][e][0])
                    elif const_expr(st_pol is not None):
                        st_v4_hint(addr, outs[r][e], st_pol)
                    else:
                        st_vec(addr, outs[r][e])


@cute.jit
def _gate2d_launch(
    x_addr: Int64,
    g_addr: Int64,
    b_addr: Int64,
    o_addr: Int64,
    n_rows: Int32,
    temp: Float32,
    c1: Float32,
    flo: Float32,
    stream: cuda.CUstream,
    W: cutlass.Constexpr,
    D: cutlass.Constexpr,
    G_STRIDE: cutlass.Constexpr,
    VEC: cutlass.Constexpr,
    E: cutlass.Constexpr,
    R: cutlass.Constexpr,
    BLOCK: cutlass.Constexpr,
    BIAS_MODE: cutlass.Constexpr,
    HAS_TEMP: cutlass.Constexpr,
    HAS_FLOOR: cutlass.Constexpr,
    ROW_EXACT: cutlass.Constexpr,
    ONE_ROW: cutlass.Constexpr,
    ROWS_ON_X: cutlass.Constexpr,
    CLUSTER_LAUNCH: cutlass.Constexpr,
    PF256: cutlass.Constexpr,
    INTERLEAVED8: cutlass.Constexpr,
):
    ncb = (W + BLOCK * VEC * E - 1) // (BLOCK * VEC * E)
    nrb = (n_rows + (R - 1)) // R
    grid = [nrb, ncb, 1] if const_expr(ROWS_ON_X) else [ncb, nrb, 1]
    kernel = _gate2d_kernel(
        x_addr,
        g_addr,
        b_addr,
        o_addr,
        n_rows,
        temp,
        c1,
        flo,
        W,
        D,
        G_STRIDE,
        VEC,
        E,
        R,
        BLOCK,
        BIAS_MODE,
        HAS_TEMP,
        HAS_FLOOR,
        ROW_EXACT,
        ONE_ROW,
        ROWS_ON_X,
        PF256,
        INTERLEAVED8,
    )
    # Small grids: launching as (trivial 1x1x1) clusters measured ~0.1 us lower end-to-
    # end
    # latency on B300, and adding the cooperative (co-resident) attribute on top of that
    # a
    # further 1-3 timer ticks; large grids lose CTA launch throughput this way.
    if const_expr(CLUSTER_LAUNCH == 2):
        kernel.launch(
            grid=grid,
            block=[BLOCK, 1, 1],
            cluster=[1, 1, 1],
            cooperative=True,
            stream=stream,
            preferred_smem_carveout=0,
            min_blocks_per_mp=1,
        )
    elif const_expr(CLUSTER_LAUNCH == 1):
        kernel.launch(
            grid=grid,
            block=[BLOCK, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            preferred_smem_carveout=0,
        )
    else:
        kernel.launch(
            grid=grid, block=[BLOCK, 1, 1], stream=stream, preferred_smem_carveout=0
        )


_cache: dict[tuple, Callable] = {}
_no_coop = set()
_CLUSTER_MAX_CTAS = 768
# Cooperative 1x1x1-cluster launches were measured to be refused above ~1-1.5K CTAs on
# B300
# even when plain occupancy would allow more, so cap them here (and fall back if
# refused).
_COOP_MAX_CTAS = 1024
_sm_count: list[int] = []


def _num_sms():
    if not _sm_count:
        _sm_count.append(
            torch.cuda.get_device_properties(
                torch.accelerator.current_device_index()
            ).multi_processor_count
        )
    return _sm_count[0]


def _max_vec(W, gs, ptrs):
    for vec in (8, 4, 2):
        if W % vec == 0 and gs % vec == 0 and all(p % (2 * vec) == 0 for p in ptrs):
            return vec
    return 1


def _coop_fits(ncta, block, regs):
    """Conservative co-residency bound (no shared memory; regs rounded up per warp)."""
    per_warp = -(-regs * 32 // 256) * 256
    by_regs = 65536 // (per_warp * -(-block // 32))
    per_sm = min(32, 2048 // block, by_regs)
    return ncta <= min(_COOP_MAX_CTAS, _num_sms() * per_sm)


def _choose(T_, W, maxvec, contiguous_gate):
    """Return launch parameters and an optional cooperative register bound."""
    n = T_ * W
    if n <= 512 * 1024:
        # Decode/small-prefill sizes (any floor/temperature). Scalar lanes in 96-ish
        # CTAs
        # up to 64K elements, then 8-byte packets with ~1.5-3K warps in flight.
        pf256 = False
        if n <= 16 * 1024:
            vec, e, block_cap = 1, 1, 128
        elif n <= 32 * 1024:
            # A 256-thread scalar CTA spans 512 B: request 256 B per L2 miss.
            vec, e, block_cap, pf256 = 1, 1, 256, True
        elif n <= 64 * 1024:
            # 512-thread scalar CTAs each cover 1 KB of a row: fetch 256 B per L2 miss.
            vec, e, block_cap, pf256 = 1, 1, 512, True
        elif n <= 128 * 1024:
            vec, e, block_cap = 4, 1, 256
        elif n <= 256 * 1024:
            vec, e, block_cap = 4, 1, 512
        else:
            vec, e, block_cap = 4, 1, 384
        vec = min(vec, maxvec)
        if vec != 1:
            pf256 = False
        need = -(-W // (vec * e))
        return vec, e, 1, min(block_cap, max(32, -(-need // 32) * 32)), pf256, None
    if n <= 32 * 1024 * 1024:
        vec, e, r = 8, 1, 1
    else:
        vec, e, r = 8, 1, 1 if contiguous_gate else 2
    vec = min(vec, maxvec)
    if vec == 8 and n > 1024 * 1024:
        # Two rows per 192-thread CTA keep the T=512 primary grid co-resident.
        ncb = -(-W // (192 * 8))
        if ncb * 192 * 8 - W < 192 * 8 // 2 and _coop_fits(ncb * -(-T_ // 2), 192, 32):
            return 8, 1, 2, 192, False, 32
    need = -(-W // (vec * e))
    block_cap = 256 if vec == 8 and 512 * 1024 <= n <= 1024 * 1024 else 128
    if (
        vec == 8
        and n > 1024 * 1024
        and -(-T_ // r) * -(-W // (256 * vec * e)) <= _CLUSTER_MAX_CTAS
    ):
        block_cap = 256  # keeps ~1-1.5M-element grids within the cluster-launch tier
    block = min(block_cap, max(32, -(-need // 32) * 32))
    return vec, e, r, block, False, None


def run(attn_output, gate, gate_bias, num_heads, head_dim, floor, temperature, output):
    T_ = attn_output.shape[0]
    H = int(num_heads)
    D = int(head_dim)
    W = H * D
    if T_ == 0 or W == 0:
        return
    gs = gate.stride(0) if T_ > 1 else max(gate.stride(0), W)
    has_bias = gate_bias is not None
    floor = float(floor)
    temperature = float(temperature)
    has_temp = temperature != 1.0
    c1 = 1.0 - floor
    has_floor = c1 != 1.0 or floor != 0.0
    ptrs = [attn_output.data_ptr(), gate.data_ptr(), output.data_ptr()]
    b_ptr = gate_bias.data_ptr() if has_bias else 0

    vec, e, r, block, pf256, coop_regs = _choose(T_, W, _max_vec(W, gs, ptrs), gs == W)
    bias_mode = 0 if not has_bias else (1 if D % vec == 0 else 2)
    nrb = -(-T_ // r)
    compile_stride = W if T_ == 1 else gs
    ncta = nrb * -(-W // (block * vec * e))
    launch_mode = 0
    if coop_regs is not None:
        launch_mode = 2
    elif ncta <= _CLUSTER_MAX_CTAS:
        # Cooperative launch needs every CTA resident at once; only claim it at <= half
        # the
        # SM's thread capacity so it cannot fail for any register footprint used here.
        launch_mode = 2 if ncta <= _num_sms() * max(1, 1024 // block) else 1
    # T=512's two-row cooperative tile exposes the final packets' math at the one-wave
    # tail.
    # Interleave independent exponential sigmoid operations for these packets.
    interleaved8 = (
        vec == 8
        and r == 2
        and launch_mode == 2
        and not has_temp
        and not has_floor
        and bias_mode != 2
    )
    key = (
        W,
        D,
        compile_stride,
        vec,
        e,
        r,
        block,
        bias_mode,
        has_temp,
        has_floor,
        T_ % r == 0,
        T_ == 1,
        nrb > 65535,
        launch_mode,
        pf256,
        interleaved8,
    )
    if key in _no_coop:
        key = key[:13] + (1, pf256, interleaved8)
    args = (
        Int64(ptrs[0]),
        Int64(ptrs[1]),
        Int64(b_ptr),
        Int64(ptrs[2]),
        Int32(T_),
        Float32(temperature),
        Float32(c1),
        Float32(floor),
        cuda.CUstream(current_stream().cuda_stream),
    )
    fn = _cache.get(key)
    if fn is None:
        fn = cute.compile(_gate2d_launch, *args, *key)
        _cache[key] = fn
    if launch_mode == 2 and coop_regs is not None:
        try:
            fn(*args)
        except (
            Exception
        ) as exc:  # cooperative grid refused by the driver: launch clustered
            if "COOPERATIVE_LAUNCH_TOO_LARGE" not in str(exc):
                raise
            _no_coop.add(key)
            key = key[:13] + (1, pf256, interleaved8)
            fn = _cache.get(key)
            if fn is None:
                fn = cute.compile(_gate2d_launch, *args, *key)
                _cache[key] = fn
            fn(*args)
        return
    fn(*args)
