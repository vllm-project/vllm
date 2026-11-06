# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Arithmetic and memory helpers shared by the KDA CuTe kernels."""

from functools import partial

import cutlass.cute as cute
from cutlass import Float32, Int32

from .primitives import (
    asm,
    load_words,
)


# The stock composed decode's Triton math (causal_conv1d_update, fused_recurrent_kda,
# rms_norm_gated): tl.exp is ex2.approx of x * log2(e), tl.sqrt is sqrt.approx.ftz and
# an FP32 ``/`` is div.full.f32. For a positive divisor b that is a * rcp.approx(b)
# bit for bit, with a and b first scaled by 1/4 when b > 2^126
# (test_div_full_is_reciprocal_product); so it is computed as that, without
# div.full's slow-path branch, and the reciprocal of a shared divisor once. The
# flush-to-zero forms of ex2 and rcp (no subnormal fixups) give the same bits here:
# every exp below is either added to 1 or at least e^-5 (or A = exp(A_log), which
# only scales a sigmoid's argument), and every reciprocal is of a value in [1e-3,
# 2^126].
@asm(Float32, pure=True)
def triton_exp(value):
    code = "{ .reg .f32 t; mul.f32 t,$1,0f3FB8AA3B; ex2.approx.ftz.f32 $0,t; }"
    return code, "=f,f", (value,)


@asm(Float32, pure=True)
def div_full(numerator, denominator):
    """The quotient by a positive denominator as div.full.f32 computes it."""
    code = "{ .reg .f32 a,b,r; .reg .pred big; mov.f32 a,$1; mov.f32 b,$2; "
    code += "setp.gt.f32 big,b,0f7E800000; @big mul.f32 a,a,0f3E800000; "
    code += "@big mul.f32 b,b,0f3E800000; rcp.approx.ftz.f32 r,b; mul.f32 $0,a,r; }"
    return code, "=f,f,f", (numerator, denominator)


@asm(Float32, pure=True)
def reciprocal(value):
    """1 / x as div.full.f32 computes it for 2^-126 <= x <= 2^126."""
    return "rcp.approx.ftz.f32 $0,$1;", "=f,f", (value,)


@asm(Float32, pure=True)
def mul_rn(a, b):
    return "mul.f32 $0,$1,$2;", "=f,f,f", (a, b)


@asm(Float32, pure=True)
def sqrt_approx(value):
    return "sqrt.approx.ftz.f32 $0,$1;", "=f,f", (value,)


@cute.jit
def triton_sigmoid(value):
    """tl.sigmoid: 1 / (1 + exp(-x))."""
    return div_full(Float32(1), Float32(1) + triton_exp(-value))


load_weight_bf16x4 = partial(load_words, count=2, cache_policy="nc", word_type="u32")


load_activation_bf16x4 = partial(load_words, count=2, word_type="u32", mutable=True)


load_shared_f32x4 = partial(
    load_words, count=4, shared=True, mutable=True, result_type=Float32, word_type="f32"
)


def load_shared_u32x2(pointer):
    return load_words(pointer, count=2, shared=True, mutable=True, word_type="u32")


@asm()
def sync_threads(barrier_id, count):
    """bar.sync with a runtime barrier id."""
    return "bar.sync $0, $1;", "r,r,~{memory}", (Int32(barrier_id), Int32(count))


@asm()
def fence_async_shared():
    """Order this thread's shared-memory writes before later bulk copies."""
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


@asm()
def store_bulk(dst, src, size):
    """Shared -> global bulk copy (one bulk group)."""
    code = "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2; "
    code += "cp.async.bulk.commit_group;"
    return code, "l,r,r,~{memory}", (dst, src, Int32(size))


@asm()
def wait_bulk_reads():
    """The issued bulk stores have read their shared-memory sources."""
    return "cp.async.bulk.wait_group.read 0;", "~{memory}", ()


@asm()
def set_max_registers(count, *, increase):
    op = "inc" if increase else "dec"
    return f"setmaxnreg.{op}.sync.aligned.u32 {count};", "", ()
