# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from cutlass import (
    BFloat16,
    Float8E4M3FN,
    Float16,
    Float32,
    Int32,
    Int64,
    Uint32,
    cute,
)
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op

_TORCH_TO_CUTE_DTYPE = {
    torch.bfloat16: BFloat16,
    torch.float8_e4m3fn: Float8E4M3FN,
}

_CUTE_TO_PTX_DTYPE = {
    BFloat16: "bf16",
    Float16: "f16",
    Float8E4M3FN: "e4m3",
    Float32: "f32",
}

# https://github.com/NVIDIA/cutlass/blob/v4.3.2/include/cute/arch/copy_sm90_desc.hpp#L193-L197
EVICT_NORMAL = Int64(0x1000000000000000)
EVICT_FIRST = Int64(0x12F0000000000000)
EVICT_LAST = Int64(0x14F0000000000000)


@dsl_user_op
def recast_val(x, dtype, *, loc=None, ip=None):
    return dtype(llvm.bitcast(dtype.mlir_type, x.ir_value(loc=loc, ip=ip)))


def simple_tma_copy(atom, src, dst, mbar=None, cache_policy=None):
    """A simple helper that wraps group_modes() and tma_partition()
    NOTE: this should be called WITHOUT cute.elect_one()
    """
    if isinstance(atom.op, cpasync.CopyBulkTensorTileG2SOp):
        gmem = src
        smem = dst
    elif isinstance(atom.op, cpasync.CopyBulkTensorTileS2GOp):
        smem = src
        gmem = dst
    else:
        raise ValueError

    s_part, g_part = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(smem, 0),
        cute.group_modes(gmem, 0),
    )

    if isinstance(atom.op, cpasync.CopyBulkTensorTileG2SOp):
        cute.copy(atom, g_part, s_part, tma_bar_ptr=mbar, cache_policy=cache_policy)
    elif isinstance(atom.op, cpasync.CopyBulkTensorTileS2GOp):
        cute.copy(atom, s_part, g_part, cache_policy=cache_policy)
    else:
        raise ValueError


# can't find the equivalent in nvvm
@dsl_user_op
def fence_before_tma_store(*, loc=None, ip=None):
    llvm.inline_asm(
        T.i32(),
        [],
        "mov.u32 $0, 0;\n\t"
        "fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_sync(a, b, c: cute.Tensor, *, loc=None, ip=None):
    a_ty = _CUTE_TO_PTX_DTYPE[a.element_type]
    b_ty = _CUTE_TO_PTX_DTYPE[b.element_type]
    c_ty = _CUTE_TO_PTX_DTYPE[c.element_type]
    mlir_ty = c.element_type.mlir_type
    K = 256 // a.element_type.width  # 32B

    # Recast expects tensor-backed fragments; materialize SSA fragments here so
    # callsites can pass converted FP8 fragments directly.
    if isinstance(a, cute.TensorSSA):
        a_ = cute.make_rmem_tensor_like(a)
        a_.store(a, loc=loc, ip=ip)
        a = a_
    if isinstance(b, cute.TensorSSA):
        b_ = cute.make_rmem_tensor_like(b)
        b_.store(b, loc=loc, ip=ip)
        b = b_

    a = cute.recast_tensor(a, Int32, loc=loc, ip=ip)
    b = cute.recast_tensor(b, Int32, loc=loc, ip=ip)
    out = llvm.inline_asm(
        llvm.StructType.get_literal([mlir_ty] * 4),
        [a[i].ir_value(loc=loc, ip=ip) for i in range(4)]
        + [b[i].ir_value(loc=loc, ip=ip) for i in range(2)]
        + [c[i].ir_value(loc=loc, ip=ip) for i in range(4)],
        f"mma.sync.aligned.m16n8k{K}.row.col.{c_ty}.{a_ty}.{b_ty}.{c_ty} "
        "{$0, $1, $2, $3}, "
        "{$4, $5, $6, $7}, "
        "{$8, $9}, "
        "{$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([4], mlir_ty, loc=loc),
        [llvm.extractvalue(mlir_ty, out, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 4, c.element_type)


def _bf16x2_unary(asm: str, a: Uint32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip)],
        f"{asm}.bf16x2 $0, $1;",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(out)


def _bf16x2_binary(asm: str, a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        f"{asm}.bf16x2 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(out)


@dsl_user_op
def _bf16x2_abs(a: Uint32, *, loc=None, ip=None) -> Uint32:
    return _bf16x2_unary("abs", a, loc=loc, ip=ip)


@dsl_user_op
def _bf16x2_neg(a: Uint32, *, loc=None, ip=None) -> Uint32:
    return _bf16x2_unary("neg", a, loc=loc, ip=ip)


@dsl_user_op
def _bf16x2_max(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return _bf16x2_binary("max", a, b, loc=loc, ip=ip)


@dsl_user_op
def _bf16x2_mul(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return _bf16x2_binary("mul.rn", a, b, loc=loc, ip=ip)


@dsl_user_op
def _bf16x2_sub(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return _bf16x2_binary("sub.rn", a, b, loc=loc, ip=ip)
