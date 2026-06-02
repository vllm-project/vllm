# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# this module is named _tcgen05 to avoid name collision with cute.nvgpu.tcgen05

import cutlass
from cutlass import Boolean, Float32, Int32, Uint32, Uint64, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.cutlass_dsl import dsl_user_op

NVVM_CTA_GROUP_MAP = [
    None,
    nvvm.Tcgen05GroupKind.CTA_1,
    nvvm.Tcgen05GroupKind.CTA_2,
]
LDST_MAP = {
    "32x32b": (nvvm.Tcgen05LdStShape.SHAPE_32X32B, 1),
    "16x128b": (nvvm.Tcgen05LdStShape.SHAPE_16X128B, 2),
    "16x256b": (nvvm.Tcgen05LdStShape.SHAPE_16X256B, 4),
}


def _make_tmem_llvm_ptr(addr, *, loc=None, ip=None):
    ptr_ty = llvm.PointerType.get(cute.AddressSpace.tmem.value)
    val = Int32(addr).ir_value(loc=loc, ip=ip)
    return llvm.inttoptr(ptr_ty, val, loc=loc, ip=ip)


@dsl_user_op
def alloc(
    taddr: cute.Pointer,
    cta_group: int = 1,
    *,
    loc=None,
    ip=None,
) -> None:
    nvvm.tcgen05_alloc(
        taddr.to_llvm_ptr(loc=loc, ip=ip),
        Uint32(512).ir_value(loc=loc, ip=ip),
        group=NVVM_CTA_GROUP_MAP[cta_group],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def dealloc(cta_group: int = 1, *, loc=None, ip=None) -> None:
    nvvm.tcgen05_dealloc(
        _make_tmem_llvm_ptr(0, loc=loc, ip=ip),
        Int32(512).ir_value(loc=loc, ip=ip),
        group=NVVM_CTA_GROUP_MAP[cta_group],
        loc=loc,
        ip=ip,
    )


def make_bf16_idesc(
    MMA_M: int,
    MMA_N: int,
    *,
    negate_A: bool = False,
    negate_B: bool = False,
    transpose_A: bool = False,
    transpose_B: bool = False,
):
    idesc = Uint32(
        (1 << 4) | (1 << 7) | (1 << 10) | ((MMA_N >> 3) << 17) | ((MMA_M >> 4) << 24)
    )
    idesc |= Uint32(negate_A) << 13
    idesc |= Uint32(negate_B) << 14
    idesc |= Uint32(transpose_A) << 15
    idesc |= Uint32(transpose_B) << 16
    return idesc


def make_sdesc_128B_swizzle(LBO: int):
    SBO = 8 * 128
    return Uint64((LBO >> 4 << 16) | (SBO >> 4 << 32) | (1 << 46) | (2 << 61))


@dsl_user_op
def mma_f16(
    d_tmem,
    a_desc,
    b_desc,
    idesc,
    enable_input_d,
    cta_group: int = 1,
    *,
    loc=None,
    ip=None,
) -> None:
    nvvm.tcgen05_mma(
        nvvm.Tcgen05MMAKind.F16,
        NVVM_CTA_GROUP_MAP[cta_group],
        _make_tmem_llvm_ptr(d_tmem, loc=loc, ip=ip),
        Uint64(a_desc).ir_value(loc=loc, ip=ip),
        Uint64(b_desc).ir_value(loc=loc, ip=ip),
        Int32(idesc).ir_value(loc=loc, ip=ip),
        Boolean(enable_input_d).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_ts_f16(
    d_tmem,
    a_tmem,
    b_desc,
    idesc,
    enable_input_d,
    cta_group: int = 1,
    *,
    loc=None,
    ip=None,
) -> None:
    nvvm.tcgen05_mma(
        nvvm.Tcgen05MMAKind.F16,
        NVVM_CTA_GROUP_MAP[cta_group],
        _make_tmem_llvm_ptr(d_tmem, loc=loc, ip=ip),
        _make_tmem_llvm_ptr(a_tmem, loc=loc, ip=ip),
        Uint64(b_desc).ir_value(loc=loc, ip=ip),
        Int32(idesc).ir_value(loc=loc, ip=ip),
        Boolean(enable_input_d).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def commit(mbar, cta_mask=None, cta_group: int = 1, *, loc=None, ip=None):
    mbar_llvm = mbar.to_llvm_ptr(loc=loc, ip=ip)
    group = NVVM_CTA_GROUP_MAP[cta_group]
    if cutlass.const_expr(cta_mask is not None):
        nvvm.tcgen05_commit_arrive(
            mbar_llvm,
            multicast_mask=cta_mask.ir_value(loc=loc, ip=ip),
            group=group,
            loc=loc,
            ip=ip,
        )
    else:
        nvvm.tcgen05_commit_arrive(mbar_llvm, group=group, loc=loc, ip=ip)


@dsl_user_op
def ld(row, col, shape: str, num: int, *, loc=None, ip=None):
    nvvm_shape, regs_per_num = LDST_MAP[shape]
    num_regs = regs_per_num * num
    tmem = (Int32(row) << Int32(16)) | Int32(col)
    tmem_ptr = _make_tmem_llvm_ptr(tmem, loc=loc, ip=ip)

    if num_regs == 1:
        reg = nvvm.tcgen05_ld(Int32.mlir_type, nvvm_shape, tmem_ptr, loc=loc, ip=ip)
        reg_f32 = llvm.bitcast(Float32.mlir_type, reg, loc=loc, ip=ip)
        return Float32(reg_f32)

    else:
        vec_i32_ty = ir.VectorType.get([num_regs], Int32.mlir_type, loc=loc)
        vec_f32_ty = ir.VectorType.get([num_regs], Float32.mlir_type, loc=loc)
        regs = nvvm.tcgen05_ld(vec_i32_ty, nvvm_shape, tmem_ptr, loc=loc, ip=ip)
        regs_f32 = llvm.bitcast(vec_f32_ty, regs, loc=loc, ip=ip)
        return cute.TensorSSA(regs_f32, (num_regs,), Float32)


@dsl_user_op
def st(row, col, shape: str, num: int, vals, *, loc=None, ip=None) -> None:
    # if input is TensorSSA, convert to Tensor so we can bitcast
    if isinstance(vals, cute.TensorSSA):
        vals_ = cute.make_rmem_tensor_like(vals)
        vals_.store(vals)
        vals = vals_

    # bitcast to Int32
    vals = cute.recast_tensor(vals, Int32)

    nvvm_shape, regs_per_num = LDST_MAP[shape]
    num_regs = regs_per_num * num
    tmem = (Int32(row) << Int32(16)) | Int32(col)
    tmem_ptr = _make_tmem_llvm_ptr(tmem, loc=loc, ip=ip)

    if num_regs == 1:
        nvvm.tcgen05_st(
            nvvm_shape,
            tmem_ptr,
            vals[0].ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    else:
        vec_i32_ty = ir.VectorType.get([num_regs], Int32.mlir_type, loc=loc)
        val_vec = vector.from_elements(
            vec_i32_ty,
            [vals[i].ir_value(loc=loc, ip=ip) for i in range(num_regs)],
            loc=loc,
            ip=ip,
        )
        nvvm.tcgen05_st(nvvm_shape, tmem_ptr, val_vec, loc=loc, ip=ip)


@dsl_user_op
def fence_after_thread_sync(*, loc=None, ip=None):
    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def fence_before_thread_sync(*, loc=None, ip=None):
    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def wait_ld(*, loc=None, ip=None):
    nvvm.tcgen05_wait(nvvm.Tcgen05WaitKind.LOAD, loc=loc, ip=ip)


@dsl_user_op
def wait_st(*, loc=None, ip=None):
    nvvm.tcgen05_wait(nvvm.Tcgen05WaitKind.STORE, loc=loc, ip=ip)
