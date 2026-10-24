# SPDX-License-Identifier: Apache-2.0
"""Shared CuTe device operations."""

from __future__ import annotations

from cutlass import Boolean, Int32, Uint32, Uint64, cute
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import dsl_user_op

from . import _tcgen05

_MAKE_RMEM = getattr(cute, "make_rmem_tensor", None) or cute.make_fragment


def make_rmem(shape, dtype):
    return _MAKE_RMEM(shape, dtype)


@dsl_user_op
def alloc_tmem(taddr: cute.Pointer, ncols: int, *, loc=None, ip=None) -> None:
    nvvm.tcgen05_alloc(
        taddr.to_llvm_ptr(loc=loc, ip=ip),
        Uint32(ncols).ir_value(loc=loc, ip=ip),
        group=_tcgen05.NVVM_CTA_GROUP_MAP[1],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def dealloc_tmem(tmem_addr: Int32, ncols: int, *, loc=None, ip=None) -> None:
    ptr_ty = llvm.PointerType.get(cute.AddressSpace.tmem.value)
    tmem_ptr = llvm.inttoptr(
        ptr_ty, Int32(tmem_addr).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    nvvm.tcgen05_dealloc(
        tmem_ptr,
        Int32(ncols).ir_value(loc=loc, ip=ip),
        group=_tcgen05.NVVM_CTA_GROUP_MAP[1],
        loc=loc,
        ip=ip,
    )


def make_fp16_idesc(mma_m: int, mma_n: int) -> Uint32:
    return Uint32(1 << 4 | mma_n >> 3 << 17 | mma_m >> 4 << 24)


def make_tf32_idesc(mma_m: int, mma_n: int) -> Uint32:
    return Uint32(1 << 4 | 2 << 7 | 2 << 10 | mma_n >> 3 << 17 | mma_m >> 4 << 24)


def make_sdesc_32B_swizzle(base16) -> Uint64:
    half_mode = Uint64(3 << 61)
    lo = Uint64(1 << 16 | 16 << 32 | 1 << 46)
    return base16 + half_mode + half_mode | lo


@dsl_user_op
def mma_tf32(
    d_tmem, a_desc, b_desc, idesc, enable_input_d, *, loc=None, ip=None
) -> None:
    with cute.arch.elect_one():
        nvvm.tcgen05_mma(
            nvvm.Tcgen05MMAKind.TF32,
            _tcgen05.NVVM_CTA_GROUP_MAP[1],
            _tcgen05._make_tmem_llvm_ptr(d_tmem, loc=loc, ip=ip),
            Uint64(a_desc).ir_value(loc=loc, ip=ip),
            Uint64(b_desc).ir_value(loc=loc, ip=ip),
            Int32(idesc).ir_value(loc=loc, ip=ip),
            Boolean(enable_input_d).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
