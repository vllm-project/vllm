# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device primitives of the mono MoE launch: LDS views, wave arithmetic, the
real-time clock and device-scope stores."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import global_typed_ptr, lds_typed_ptr
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T

from vllm.models.kimi_k3.amd.mono.common.plan import ROUTE_MARKS


def lds_i32(base, off_words):
    return lds_typed_ptr(base, T.i32, byte_offset=fx.Int32(off_words * 4))


def lds_f32(base, off_words):
    return lds_typed_ptr(base, T.f32, byte_offset=fx.Int32(off_words * 4))


def now():
    """The 100 MHz s_memrealtime clock."""
    return fx.Int64(
        _llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], [])
    )


@comm_ops.traced
def ticket_mark(tid, arg_trace, t, k):
    """Trace: clock k of ticket t (0 start, 1 dependencies met, 2 end)."""
    if tid == fx.Int32(0):
        trace = global_typed_ptr(arg_trace, T.i64, align=8)
        trace[fx.Int32(ROUTE_MARKS) + t * fx.Int32(4) + fx.Int32(k)] = now()


@comm_ops.traced
def route_mark(tid, arg_trace, k):
    """Trace: clock k of the routing phase (the sort's workgroup)."""
    if tid == fx.Int32(0):
        trace = global_typed_ptr(arg_trace, T.i64, align=8)
        trace[fx.Int32(k)] = now()


def rcp(x):
    return fx.Float32(
        _llvm.call_intrinsic(
            T.f32, "llvm.amdgcn.rcp.f32", [fx.Float32(x).ir_value()], [], []
        )
    )


def popc(x):
    return fx.Int32(fmath.ctpop(fx.Int32(x).ir_value()))


def popc64(x):
    return fx.Int64(fmath.ctpop(fx.Int64(x).ir_value())).to(fx.Int32)


def lds_or(addr_i64, val):
    """Workgroup-scope LDS fetch-and-or at a raw LDS byte address."""
    _llvm.AtomicRMWOp(
        _llvm.AtomicBinOp._or,
        comm_ops._to_ptr_lds(addr_i64),
        fx.Int32(val).ir_value(),
        _llvm.AtomicOrdering.monotonic,
        syncscope="workgroup",
        alignment=4,
    )


def st_wt(base_i64, index, val, nbytes):
    """Device-scope store: writes through the XCD's L2, so no L2 writeback is needed."""
    _llvm.StoreOp(
        val.ir_value(),
        comm_ops._ptr_plus(base_i64, index, nbytes),
        alignment=nbytes,
        ordering=_llvm.AtomicOrdering.monotonic,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    )


def st_plain(base_i64, index, val, nbytes):
    """Plain store: stays in the XCD's L2 until a release writes it back."""
    _llvm.StoreOp(
        val.ir_value(), comm_ops._ptr_plus(base_i64, index, nbytes), alignment=nbytes
    )


def l1_invalidate():
    """Drop this CU's L1 (the lines another XCD wrote through to memory)."""
    _llvm.InlineAsmOp(None, [], "buffer_inv sc0", "", has_side_effects=True)


def order_key(x):
    """i32 whose signed order is the order of the (non-NaN) f32 ``x``."""
    b = fx.Float32(x).bitcast(fx.Int32)
    return (b < fx.Int32(0)).select(b ^ fx.Int32(0x7FFFFFFF), b)


def key_value(k):
    return ((k < fx.Int32(0)).select(k ^ fx.Int32(0x7FFFFFFF), k)).bitcast(fx.Float32)


def wave_kth_key(key, k):
    """Largest t with at least k lanes of the wave holding key >= t, by bisection
    on ballots."""

    def n_ge(t):
        return fx.Int64(fmath.ctpop(fx.Int64(rocdl.ballot(T.i64, key >= t)).ir_value()))

    t = (n_ge(fx.Int32(0)) >= fx.Int64(k)).select(fx.Int32(0), fx.Int32(-(1 << 31)))
    for b in range_constexpr(30, -1, -1):
        c = t | fx.Int32(1 << b)
        t = (n_ge(c) >= fx.Int64(k)).select(c, t)
    return t
