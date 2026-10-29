# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers shared by the MiniMax-M3 FlyDSL kernels (gfx950)."""

import functools

import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, llvm
from flydsl.expr import rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

# Cache-policy bits of buffer loads and stores: sc1 is coherent across the XCDs'
# L2s, sc0 | sc1 across the system, nt streams.
SC0, NT, SC1 = 0x1, 0x2, 0x10
LOG2E = 1.4426950409  # softmax runs in exp2
# The first 16 dwords of kernel arguments arrive in SGPRs, so a kernel's first
# loads do not wait for s_load.
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


def run_compiled(exe, *args, waves_per_eu: int | None = None):
    """Compiles and runs ``exe`` on the first call, then dispatches the cache."""
    import flydsl.compiler as flyc
    from flydsl.compiler.kernel_function import CompilationContext

    cf = getattr(exe, "_cf", None)
    if cf is None:
        hints = {"waves_per_eu": waves_per_eu} if waves_per_eu else {}
        with CompilationContext.compile_hints(hints):
            exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def cu_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.cache
def unit_scale(device: torch.device) -> torch.Tensor:
    return torch.ones(1, dtype=torch.float32, device=device)


def global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def as_f32(x):
    return fx.Int32(x).bitcast(fx.Float32)


def permlane_swap(op, d_a, d_b):
    """v_permlane{16,32}_swap; fx has no wrapper."""
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = op(pair_ty, fx.as_ir_value(d_a), fx.as_ir_value(d_b), False, False)
    return (
        fx.Int32(llvm.extractvalue(T.i32, res, [0])),
        fx.Int32(llvm.extractvalue(T.i32, res, [1])),
    )


def maxf_nn(a, b):
    """v_max_f32 with nnan; without it LLVM canonicalizes both inputs first."""
    fm = ir.Attribute.parse("#arith.fastmath<nnan>")
    return fx.Float32(
        arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b), fastmath=fm).result
    )


def xlane4(x, op):
    """``op`` over lanes L, L^16, L^32, L^48, broadcast to all four."""
    a, b = permlane_swap(rocdl.permlane32_swap, bits(x), bits(x))
    m = op(as_f32(a), as_f32(b))
    a, b = permlane_swap(rocdl.permlane16_swap, bits(m), bits(m))
    return op(as_f32(a), as_f32(b))


def perm(hi, lo, sel):
    """v_perm_b32: byte i of the result is byte sel_i of (hi:lo)."""
    return fx.Int32(rocdl.perm_b32(fx.Int32(hi), fx.Int32(lo), fx.Int32(sel)))


def exp2(x):
    """v_exp_f32 without denormal range reduction."""
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(x).ir_value()))


def pk_bf16(a, b):
    """Two fp32 -> one dword of two bf16 (RNE), ``a`` low."""
    return fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(
            fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))
        )
    )


def pk_f16(a, b):
    """Two fp32 -> one dword of two fp16 (RNE), ``a`` low."""
    return (
        Vec.from_elements([fx.Float32(a), fx.Float32(b)], fx.Float32)
        .to(fx.Float16)
        .bitcast(fx.Int32)[0]
    )


def cvt_fp4x2_f16(word, sel):
    """The two e2m1 codes of byte ``sel`` of ``word`` as two fp16 (exact)."""
    return Vec(
        fx.rocdl.cvt_scalef32_pk_f16_fp4(
            Vec.make_type(2, fx.Float16),
            fx.as_ir_value(fx.Int32(word)),
            fx.as_ir_value(fx.Float32(1.0)),
            sel,
        )
    )


def cvt_fp8x2_f16(word, hi):
    """E4M3 bytes 0, 1 (``hi``: 2, 3) of ``word`` as two fp16 (exact)."""
    return Vec(
        fx.rocdl.cvt_scalef32_pk_f16_fp8(
            Vec.make_type(2, fx.Float16),
            fx.as_ir_value(fx.Int32(word)),
            fx.as_ir_value(fx.Float32(1.0)),
            hi,
        )
    )


def mfma_16x16x32(a4, b4, acc, fp16: bool):
    """v_mfma_f32_16x16x32_{bf16,f16} on 4-dword operands."""
    dtype = fx.Float16 if fp16 else fx.BFloat16
    op = fx.rocdl.mfma_f32_16x16x32_f16 if fp16 else fx.rocdl.mfma_f32_16x16x32_bf16
    a = Vec(a4).bitcast(dtype)
    b = Vec(b4).bitcast(dtype)
    return Vec(op(Vec.make_type(4, fx.Float32), [a, b, acc, 0, 0, 0]))


def mfma_fp8_16x16x128(a8, b8, acc4):
    """v_mfma_f32_16x16x128_f8f6f4 on e4m3 operands; zero scales select the
    unscaled instruction."""
    return Vec(
        fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            Vec.make_type(4, fx.Float32), [a8, b8, acc4, 0, 0, 0, 0, 0, 0]
        )
    )


def pack8(h0, h1):
    """Two 4 x i32 halves -> the 8 x i32 operand of the fp8 16x16x128 MFMA."""
    a, b = Vec(h0), Vec(h1)
    return Vec.from_elements(
        [a[k] for k in range(4)] + [b[k] for k in range(4)], fx.Int32
    )


# Byte selectors of the even and odd dims in key order.
PERM_EVEN, PERM_ODD = 0x06040200, 0x07050301


def key_order4(d):
    """4 dwords of e4m3 (16 dims) -> even dims, then odd dims per 8 (key order)."""
    return Vec.from_elements(
        [
            perm(d[1], d[0], PERM_EVEN),
            perm(d[1], d[0], PERM_ODD),
            perm(d[3], d[2], PERM_EVEN),
            perm(d[3], d[2], PERM_ODD),
        ],
        fx.Int32,
    )


def dequant8(c, t_lo, t_hi):
    """8 e2m1 codes (low nibble the even dim) and the group's e4m3 table ->
    the e4m3 keys of the even dims, then of the odd dims."""
    ml = perm(t_hi, t_lo, c & fx.Int32(0x07070707))
    mh = perm(t_hi, t_lo, (c >> 4) & fx.Int32(0x07070707))
    sign = fx.Int32(0x80808080 - (1 << 32))
    return ml ^ ((c << 4) & sign), mh ^ (c & sign)
