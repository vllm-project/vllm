# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from FlashInfer's flashinfer/comm/mnnvl_cutedsl/cute_dsl_primitives.py
# and runtime.py (flashinfer-ai/flashinfer@139af6f6).

"""CuTe DSL primitives shared by the fused MoE-tail kernels: PTX loads and
stores, NVLS multicast stores, Lamport mailbox helpers, BF16 packing, block
reductions and launch helpers. This module defines no kernel."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint16, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

WARP_SIZE = 32
VEC_BF16 = 8
QUAD_BF16 = 4
PACKED_BYTES = 16
NEGATIVE_ZERO_BF16_BITS = 0x8000
NEGATIVE_ZERO_BF16_PAIR = 0x80008000
NEGATIVE_ZERO_F32_BITS = 0x80000000

# Lamport mailboxes rotate through three generations; the stage-state words
# hold the next generation to publish into and the one being consumed.
LAMPORT_GENERATIONS = 3
NEXT_STAGE = 0
ACTIVE_STAGE = 1


def _asm(result, operands, text, constraints, *, side_effects, loc, ip):
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# Global memory
# ---------------------------------------------------------------------------


@dsl_user_op
def load_global_u32x4(
    pointer: cute.Pointer,
    *,
    volatile: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
):
    """Load one 128-bit fragment as four u32 registers.

    The volatile form is the Lamport polling load. Marking the asm
    side-effecting prevents loop-invariant motion and common-subexpression
    elimination across polling iterations.
    """
    address = pointer.toint(loc=loc, ip=ip)
    opcode = "ld.volatile.global.v4.u32" if volatile else "ld.global.v4.u32"
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        f"{opcode} {{$0, $1, $2, $3}}, [$4];",
        "=r,=r,=r,=r,l",
        side_effects=volatile,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [
            llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip)
            for index in range(4)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def load_global_u32x2(pointer: cute.Pointer, *, loc=None, ip=None):
    address = pointer.toint(loc=loc, ip=ip)
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [address.ir_value(loc=loc, ip=ip)],
        "ld.global.v2.u32 {$0, $1}, [$2];",
        "=r,=r,l",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([2], T.i32(), loc=loc),
        [
            llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip)
            for index in range(2)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 2, Uint32)


@dsl_user_op
def load_global_bf16_as_f32(address: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [address.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 bits;\n\tld.global.b16 bits, [$1];\n\t"
            "cvt.f32.bf16 $0, bits;\n\t}",
            "=f,l",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_volatile_u32(pointer: cute.Pointer, *, loc=None, ip=None) -> Uint32:
    address = pointer.toint(loc=loc, ip=ip)
    return Uint32(
        _asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.u32 $0, [$1];",
            "=r,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_global_u32x4(
    address: Int64,
    packed,
    *,
    volatile: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
) -> None:
    """Store four packed words to an ordinary or NVLS multicast global VA."""
    words = [packed[index].ir_value(loc=loc, ip=ip) for index in range(4)]
    opcode = "st.volatile.global.v4.u32" if volatile else "st.global.v4.u32"
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        f"{opcode} [$0], {{$1, $2, $3, $4}};",
        "l,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_global_u32x2(address: Int64, packed, *, loc=None, ip=None) -> None:
    words = [packed[index].ir_value(loc=loc, ip=ip) for index in range(2)]
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        "st.global.v2.u32 [$0], {$1, $2};",
        "l,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_global_u32(address: Int64, word: Uint32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), word.ir_value(loc=loc, ip=ip)],
        "st.global.u32 [$0], $1;",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_async_release_gpu_add_u32(
    pointer: cute.Pointer, value: Uint32, *, loc=None, ip=None
) -> None:
    """The SM100 arrival primitive FlashInfer's LamportFlags use."""
    address = pointer.toint(loc=loc, ip=ip)
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "red.async.release.global.gpu.add.u32 [$0], $1;",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# NVLS multicast stores
# ---------------------------------------------------------------------------


@dsl_user_op
def stmc_bf16x2(address: Int64, packed: Uint32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), packed.ir_value(loc=loc, ip=ip)],
        "multimem.st.relaxed.sys.global.bf16x2 [$0], $1;",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stmc_bf16x4(address: Int64, packed, *, loc=None, ip=None) -> None:
    words = [packed[index].ir_value(loc=loc, ip=ip) for index in range(2)]
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        "multimem.st.relaxed.sys.global.v2.bf16x2 [$0], {$1, $2};",
        "l,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stmc_bf16x8(address: Int64, packed, *, loc=None, ip=None) -> None:
    """Publish eight BF16 values (four packed words) through NVLS multicast."""
    words = [packed[index].ir_value(loc=loc, ip=ip) for index in range(4)]
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        "multimem.st.relaxed.sys.global.v4.bf16x2 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# Lamport sentinels
#
# A BF16 mailbox marks an empty slot with BF16 negative zero in every lane; an
# FP32 mailbox marks it with FP32 negative zero in every word. Senders replace
# real negative zeros with positive zeros so they never read as empty.
# ---------------------------------------------------------------------------


@dsl_user_op
def store_lamport_sentinel_u32x4(address: Int64, *, loc=None, ip=None) -> None:
    """Reset one BF16 Lamport fragment to BF16 negative zeros."""
    sentinel = Uint32(NEGATIVE_ZERO_BF16_PAIR).ir_value(loc=loc, ip=ip)
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), sentinel, sentinel, sentinel, sentinel],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_lamport_sentinel_128(pointer: cute.Pointer, *, loc=None, ip=None) -> None:
    """Reset one FP32 Lamport fragment to four FP32 negative-zero bit patterns."""
    address = pointer.toint(loc=loc, ip=ip)
    sentinel = Uint32(NEGATIVE_ZERO_F32_BITS).ir_value(loc=loc, ip=ip)
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), sentinel, sentinel, sentinel, sentinel],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@cute.jit
def sanitize_negative_zero_u32(word: Uint32) -> Uint32:
    low = Uint16(word & Uint32(0xFFFF))
    high = Uint16(word >> Uint32(16))
    if low == Uint16(NEGATIVE_ZERO_BF16_BITS):
        word = word & Uint32(0xFFFF0000)
    if high == Uint16(NEGATIVE_ZERO_BF16_BITS):
        word = word & Uint32(0x0000FFFF)
    return word


@cute.jit
def sanitize_negative_zero_u32x2(packed):
    sanitized = cute.make_rmem_tensor(cute.make_layout((2,)), Uint32)
    for index in cutlass.range_constexpr(2):
        sanitized[index] = sanitize_negative_zero_u32(packed[index])
    return sanitized.load()


@cute.jit
def sanitize_negative_zero_u32x4(packed):
    """Turn real BF16 -0 into +0 so it cannot equal the empty sentinel."""
    sanitized = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
    for index in cutlass.range_constexpr(4):
        sanitized[index] = sanitize_negative_zero_u32(packed[index])
    return sanitized.load()


@cute.jit
def fragment_has_negative_zero(packed):
    """Whether a BF16 fragment still holds the BF16 sentinel in any lane."""
    dirty = False
    for index in cutlass.range_constexpr(4):
        word = packed[index]
        dirty = (
            dirty
            | (Uint16(word & Uint32(0xFFFF)) == Uint16(NEGATIVE_ZERO_BF16_BITS))
            | (Uint16(word >> Uint32(16)) == Uint16(NEGATIVE_ZERO_BF16_BITS))
        )
    return dirty


@cute.jit
def fragment_is_dirty(packed):
    """Whether an FP32 fragment still holds the FP32 sentinel in any word."""
    dirty = packed[0] == Uint32(NEGATIVE_ZERO_F32_BITS)
    for index in cutlass.range_constexpr(1, 4):
        dirty = dirty | (packed[index] == Uint32(NEGATIVE_ZERO_F32_BITS))
    return dirty


# ---------------------------------------------------------------------------
# Shared and distributed shared memory
# ---------------------------------------------------------------------------


@dsl_user_op
def map_shared_to_peer(
    smem_pointer: cute.Pointer, peer_rank: Int32, *, loc=None, ip=None
) -> Int32:
    """Map a local shared-memory slot to the same slot in a peer CTA."""
    address = smem_pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    return Int32(
        _asm(
            T.i32(),
            [address, peer_rank.ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_shared_cluster_f32(
    remote_address: Int32, value: Float32, *, loc=None, ip=None
) -> None:
    _asm(
        None,
        [remote_address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.shared::cluster.f32 [$0], $1;",
        "r,f",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_f32x2(pointer: cute.Pointer, *, loc=None, ip=None):
    """Load two aligned FP32 partials from local shared memory."""
    address = Int32(pointer.toint(loc=loc, ip=ip))
    loaded = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [address.ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), loaded, [index], loc=loc, ip=ip))
        for index in range(2)
    )


@dsl_user_op
def load_shared_f32x4(pointer: cute.Pointer, *, loc=None, ip=None):
    """Load four aligned FP32 partials from local shared memory."""
    address = Int32(pointer.toint(loc=loc, ip=ip))
    loaded = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), loaded, [index], loc=loc, ip=ip))
        for index in range(4)
    )


# ---------------------------------------------------------------------------
# BF16 packing and arithmetic
# ---------------------------------------------------------------------------


@dsl_user_op
def packed_u32x4_to_bf16x8(packed, *, loc=None, ip=None):
    values = llvm.bitcast(
        ir.VectorType.get([VEC_BF16], BFloat16.mlir_type, loc=loc),
        packed.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(values, VEC_BF16, BFloat16)


@dsl_user_op
def packed_u32x2_to_bf16x4(packed, *, loc=None, ip=None):
    values = llvm.bitcast(
        ir.VectorType.get([QUAD_BF16], BFloat16.mlir_type, loc=loc),
        packed.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(values, QUAD_BF16, BFloat16)


@dsl_user_op
def bf16x8_to_packed_u32x4(values, *, loc=None, ip=None):
    packed = llvm.bitcast(
        ir.VectorType.get([4], T.i32(), loc=loc),
        values.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def bf16x4_to_packed_u32x2(values, *, loc=None, ip=None):
    packed = llvm.bitcast(
        ir.VectorType.get([2], T.i32(), loc=loc),
        values.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 2, Uint32)


@dsl_user_op
def bf16x2_to_u32(values, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.bitcast(T.i32(), values.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


@dsl_user_op
def f32_to_bf16_bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [value.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 bits;\n\tcvt.rn.bf16.f32 bits, $1;\n\t"
            "cvt.u32.u16 $0, bits;\n\t}",
            "=r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fma_f32_bf16(
    a: BFloat16, b: BFloat16, acc: Float32, *, loc=None, ip=None
) -> Float32:
    a_bits = llvm.bitcast(T.i16(), a.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b_bits = llvm.bitcast(T.i16(), b.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    return Float32(
        _asm(
            T.f32(),
            [a_bits, b_bits, acc.ir_value(loc=loc, ip=ip)],
            "fma.rn.f32.bf16 $0, $1, $2, $3;",
            "=f,h,h,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------
# Warp and block reductions
# ---------------------------------------------------------------------------


@dsl_user_op
def shuffle_sync_idx_u32(
    value: Uint32, source_lane: Int32, *, loc=None, ip=None
) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [value.ir_value(loc=loc, ip=ip), source_lane.ir_value(loc=loc, ip=ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            # Preserve full-warp execution across later divergent consumers.
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def group_leader_block_sum(
    value: Float32,
    warp_sums: cute.Tensor,
    warps: cutlass.Constexpr[int],
    leader_stride: cutlass.Constexpr[int],
) -> Float32:
    """Block sum of the values held by every ``leader_stride``-th lane."""
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    for offset in cutlass.range_constexpr(1, WARP_SIZE):
        if cutlass.const_expr(offset >= leader_stride and (offset & (offset - 1)) == 0):
            value = value + cute.arch.shuffle_sync_bfly(
                value,
                offset=offset,
                mask=-1,
                mask_and_clamp=31,
            )
    if lane == 0:
        cute.arch.store((warp_sums + warp).llvm_ptr, value)
    cute.arch.barrier()

    result = Float32(0.0)
    if warp == 0:
        if lane < Int32(warps):
            result = cute.arch.load(
                (warp_sums + lane).llvm_ptr,
                Float32,
            )
        result = cute.arch.warp_reduction_sum(result)
        if lane == 0:
            cute.arch.store(warp_sums.llvm_ptr, result)
    cute.arch.barrier()
    return cute.arch.load(warp_sums.llvm_ptr, Float32)


@cute.jit
def warp_sum_specialized(
    value: Float32,
    warp_idx: Int32,
    lane: Int32,
    warps: cutlass.Constexpr[int],
    last_warp_lanes: cutlass.Constexpr[int],
    last_warp_mask: cutlass.Constexpr[int],
) -> Float32:
    """Warp sum supporting a compile-time partial final warp."""
    if warp_idx == Int32(warps - 1) and cutlass.const_expr(last_warp_lanes < 32):
        for offset in cutlass.range_constexpr(16, 0, -1):
            # range_constexpr does not provide powers-of-two stepping.
            if cutlass.const_expr(offset in (16, 8, 4, 2, 1)):
                other = cute.arch.shuffle_sync_bfly(
                    value,
                    offset=offset,
                    mask=last_warp_mask,
                    mask_and_clamp=31,
                )
                if (lane ^ Int32(offset)) < Int32(last_warp_lanes):
                    value = value + other
    else:
        for offset in cutlass.range_constexpr(16, 0, -1):
            if cutlass.const_expr(offset in (16, 8, 4, 2, 1)):
                value = value + cute.arch.shuffle_sync_bfly(
                    value,
                    offset=offset,
                    mask=-1,
                    mask_and_clamp=31,
                )
    return value


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------


class CUDAGraphCompatibleWrapper:
    """DLPack view that does not synchronize with the producer stream."""

    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def __dlpack__(self, stream=None):
        # stream=-1 skips producer sync; CuTe launches on the current stream,
        # which is the capturing one under CUDA graphs.
        return self.tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


def to_cute(tensor: torch.Tensor, assumed_align: int = 16) -> cute.Tensor:
    return from_dlpack(
        CUDAGraphCompatibleWrapper(tensor.detach()), assumed_align=assumed_align
    )


def to_cute_dynamic(
    tensor: torch.Tensor, alignment: int, *, divisibility: int
) -> cute.Tensor:
    """A 1-D view whose length is a runtime shape divisible by ``divisibility``."""
    return to_cute(tensor, alignment).mark_compact_shape_dynamic(
        mode=0, divisibility=divisibility
    )


def to_cute_dynamic_m(
    tensor: torch.Tensor, *, mode: int, assumed_align: int = 16
) -> cute.Tensor:
    """Expose exactly one compact tensor mode as a runtime shape.

    Model dimensions remain part of the compiled tensor type. Only the token
    mode is symbolic, so changing M within an op's capacity reuses the same
    compiled kernel.
    """
    return to_cute(tensor, assumed_align).mark_compact_shape_dynamic(
        mode=mode, stride_order=tensor.dim_order()
    )


def make_fake_dynamic_compact_tensor(dtype, *, alignment: int, divisibility: int):
    return make_fake_compact_tensor(
        dtype, (cute.sym_int32(divisibility=divisibility),), assumed_align=alignment
    )


def current_cu_stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)
