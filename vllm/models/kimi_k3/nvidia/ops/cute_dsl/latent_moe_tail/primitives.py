# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared CuTe DSL primitives; this module does not define a CUDA kernel."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint16, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

VEC_BF16 = 8
PACKED_BYTES = 16
NUM_LAMPORT_BUFFERS = 3
NEG_ZERO_F32_BITS = 0x80000000
NEG_ZERO_BF16_BITS = 0x8000


class CUDAGraphCompatibleWrapper:
    """DLPack view that does not synchronize with the producer stream."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __dlpack__(self, stream=None):
        return self.tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


def to_cute(tensor: torch.Tensor, assumed_align: int = 16) -> cute.Tensor:
    return from_dlpack(
        CUDAGraphCompatibleWrapper(tensor.detach()), assumed_align=assumed_align
    )


def to_cute_dynamic_m(
    tensor: torch.Tensor,
    *,
    mode: int,
    assumed_align: int = 16,
) -> cute.Tensor:
    """Expose exactly one compact tensor mode as a runtime shape.

    Model dimensions remain part of the compiled tensor type. Only the token
    mode is symbolic, so changing M within an Op's capacity reuses the same
    compiled kernel.
    """

    return to_cute(tensor, assumed_align).mark_compact_shape_dynamic(
        mode=mode,
        stride_order=tensor.dim_order(),
    )


@dsl_user_op
def fma_f32_bf16(
    a: BFloat16,
    b: BFloat16,
    acc: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    a_bits = llvm.bitcast(T.i16(), a.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b_bits = llvm.bitcast(T.i16(), b.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.f32(),
        [a_bits, b_bits, acc.ir_value(loc=loc, ip=ip)],
        "fma.rn.f32.bf16 $0, $1, $2, $3;",
        "=f,h,h,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Float32(result)


@dsl_user_op
def load_global_u32x4(
    pointer: cute.Pointer,
    *,
    volatile: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
):
    """Load one 128-bit fragment as four u32 registers.

    The volatile form is the Lamport polling load.  Marking the asm
    side-effecting prevents loop-invariant motion and common-subexpression
    elimination across polling iterations.
    """

    address = pointer.toint(loc=loc, ip=ip)
    opcode = "ld.volatile.global.v4.u32" if volatile else "ld.global.v4.u32"
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        f"{opcode} {{$0, $1, $2, $3}}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=volatile,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


def _make_top16_bf16_finalize_asm() -> str:
    lines = [
        "{",
        ".reg .pred _valid<16>;",
        ".reg .s32 _index<16>;",
        ".reg .b32 _weight_words<8>;",
        ".reg .b16 _weights<16>;",
        ".reg .b32 _data<64>;",
        ".reg .b16 _lo, _hi;",
        ".reg .f32 _acc<8>;",
        ".reg .u64 _address<16>;",
        "ld.global.v4.u32 {_index0, _index1, _index2, _index3}, [$5];",
        "ld.global.v4.u32 {_index4, _index5, _index6, _index7}, [$5+16];",
        "ld.global.v4.u32 {_index8, _index9, _index10, _index11}, [$5+32];",
        "ld.global.v4.u32 {_index12, _index13, _index14, _index15}, [$5+48];",
        "ld.global.v4.u32 "
        "{_weight_words0, _weight_words1, _weight_words2, "
        "_weight_words3}, [$6];",
        "ld.global.v4.u32 "
        "{_weight_words4, _weight_words5, _weight_words6, "
        "_weight_words7}, [$6+16];",
    ]
    for pair in range(8):
        lines.append(
            f"mov.b32 {{_weights{2 * pair}, _weights{2 * pair + 1}}}, "
            f"_weight_words{pair};"
        )
    for element in range(8):
        lines.append(f"mov.f32 _acc{element}, 0f00000000;")
    for route in range(16):
        data = 4 * route
        lines.extend(
            [
                f"setp.ge.s32 _valid{route}, _index{route}, 0;",
                f"@_valid{route} mad.wide.s32 _address{route}, "
                f"_index{route}, 7168, $4;",
                f"@_valid{route} ld.global.v4.u32 "
                f"{{_data{data}, _data{data + 1}, _data{data + 2}, "
                f"_data{data + 3}}}, [_address{route}];",
            ]
        )
    for route in range(16):
        for pair in range(4):
            data = 4 * route + pair
            element = 2 * pair
            lines.extend(
                [
                    f"@_valid{route} mov.b32 {{_lo, _hi}}, _data{data};",
                    f"@_valid{route} fma.rn.f32.bf16 _acc{element}, _lo, "
                    f"_weights{route}, _acc{element};",
                    f"@_valid{route} fma.rn.f32.bf16 _acc{element + 1}, _hi, "
                    f"_weights{route}, _acc{element + 1};",
                ]
            )
    for pair in range(4):
        lines.append(f"cvt.rn.bf16x2.f32 ${pair}, _acc{2 * pair + 1}, _acc{2 * pair};")
    lines.append("}")
    return "\n".join(lines)


_TOP16_BF16_FINALIZE_ASM = _make_top16_bf16_finalize_asm()


@dsl_user_op
def finalize_top16_bf16(
    gemm2_vector: cute.Pointer,
    route_indices: cute.Pointer,
    route_weights: cute.Pointer,
    *,
    loc=None,
    ip=None,
):
    """Finalize one Kimi K3 top-16 vector with wide metadata loads."""

    addresses = [
        pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
        for pointer in (gemm2_vector, route_indices, route_weights)
    ]
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        addresses,
        _TOP16_BF16_FINALIZE_ASM,
        "=r,=r,=r,=r,l,l,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


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

    words = [packed[i].ir_value(loc=loc, ip=ip) for i in range(4)]
    opcode = "st.volatile.global.v4.u32" if volatile else "st.global.v4.u32"
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        f"{opcode} [$0], {{$1, $2, $3, $4}};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_global_u32x2(
    address: Int64,
    packed,
    *,
    loc=None,
    ip=None,
) -> None:
    words = [packed[i].ir_value(loc=loc, ip=ip) for i in range(2)]
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        "st.global.v2.u32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_global_u32(
    address: Int64,
    word: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            word.ir_value(loc=loc, ip=ip),
        ],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stmc_bf16x8(address: Int64, packed, *, loc=None, ip=None) -> None:
    """Publish eight BF16 values through an NVLS multicast mapping."""

    words = [packed[i].ir_value(loc=loc, ip=ip) for i in range(4)]
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        "multimem.st.relaxed.sys.global.v4.bf16x2 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_lamport_sentinel_128(pointer: cute.Pointer, *, loc=None, ip=None) -> None:
    """Reset one Lamport fragment to four FP32 negative-zero bit patterns."""

    address = pointer.toint(loc=loc, ip=ip)
    value = Uint32(NEG_ZERO_F32_BITS).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value, value, value, value],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_async_release_gpu_add_u32(
    pointer: cute.Pointer, value: Uint32, *, loc=None, ip=None
) -> None:
    """The exact SM100 arrival primitive used by upstream LamportFlags."""

    address = pointer.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            value.ir_value(loc=loc, ip=ip),
        ],
        "red.async.release.global.gpu.add.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_volatile_u32(pointer: cute.Pointer, *, loc=None, ip=None) -> Uint32:
    address = pointer.toint(loc=loc, ip=ip)
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def map_shared_to_peer(
    smem_ptr: cute.Pointer,
    peer_rank: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Map a local shared-memory slot to the same slot in a peer CTA."""

    smem_address = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_address, peer_rank.ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_shared_cluster_f32(
    remote_address: Int32,
    value: Float32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [
            remote_address.ir_value(loc=loc, ip=ip),
            value.ir_value(loc=loc, ip=ip),
        ],
        "st.shared::cluster.f32 [$0], $1;",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_f32x2(pointer: cute.Pointer, *, loc=None, ip=None):
    """Load two aligned FP32 DSM partials from local shared memory."""

    address = Int32(pointer.toint(loc=loc, ip=ip))
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [address.ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(2)
    )


@dsl_user_op
def load_shared_f32x4(pointer: cute.Pointer, *, loc=None, ip=None):
    """Load four aligned FP32 DSM partials from local shared memory."""

    address = Int32(pointer.toint(loc=loc, ip=ip))
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def packed_u32x4_to_bf16x8(packed, *, loc=None, ip=None):
    target_type = ir.VectorType.get([VEC_BF16], BFloat16.mlir_type, loc=loc)
    values = llvm.bitcast(
        target_type,
        packed.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(values, VEC_BF16, BFloat16)


@dsl_user_op
def bf16x8_to_packed_u32x4(values, *, loc=None, ip=None):
    target_type = ir.VectorType.get([4], T.i32(), loc=loc)
    packed = llvm.bitcast(
        target_type,
        values.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def bf16x4_to_packed_u32x2(values, *, loc=None, ip=None):
    target_type = ir.VectorType.get([2], T.i32(), loc=loc)
    packed = llvm.bitcast(
        target_type,
        values.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 2, Uint32)


@dsl_user_op
def bf16x2_to_u32(values, *, loc=None, ip=None):
    packed = llvm.bitcast(
        T.i32(),
        values.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return Uint32(packed)


@cute.jit
def sanitize_negative_zero_u32(word):
    low = Uint16(word & Uint32(0xFFFF))
    high = Uint16(word >> Uint32(16))
    if low == Uint16(NEG_ZERO_BF16_BITS):
        word = word & Uint32(0xFFFF0000)
    if high == Uint16(NEG_ZERO_BF16_BITS):
        word = word & Uint32(0x0000FFFF)
    return word


@cute.jit
def sanitize_negative_zero_u32x2(packed):
    result = cute.make_rmem_tensor(cute.make_layout((2,)), Uint32)
    for i in cutlass.range_constexpr(2):
        result[i] = sanitize_negative_zero_u32(packed[i])
    return result.load()


@cute.jit
def sanitize_negative_zero(packed):
    """Turn real BF16 -0 into +0 so it cannot equal the empty sentinel."""

    result = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
    for i in cutlass.range_constexpr(4):
        word = packed[i]
        low = Uint16(word & Uint32(0xFFFF))
        high = Uint16(word >> Uint32(16))
        if low == Uint16(NEG_ZERO_BF16_BITS):
            word = word & Uint32(0xFFFF0000)
        if high == Uint16(NEG_ZERO_BF16_BITS):
            word = word & Uint32(0x0000FFFF)
        result[i] = word
    return result.load()


@cute.jit
def fragment_is_dirty(packed):
    """Bit-exact upstream sentinel check: one comparison per 32-bit word."""

    dirty = packed[0] == Uint32(NEG_ZERO_F32_BITS)
    for i in cutlass.range_constexpr(1, 4):
        dirty = dirty | (packed[i] == Uint32(NEG_ZERO_F32_BITS))
    return dirty


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


@cute.jit
def block_sum_specialized(
    value: Float32,
    warp_sums: cute.Tensor,
    tidx: Int32,
    warps: cutlass.Constexpr[int],
    last_warp_lanes: cutlass.Constexpr[int],
    last_warp_mask: cutlass.Constexpr[int],
) -> Float32:
    """Upstream-equivalent FP32 block reduction."""

    lane = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    value = warp_sum_specialized(
        value, warp_idx, lane, warps, last_warp_lanes, last_warp_mask
    )
    if lane == 0:
        warp_sums[warp_idx] = value
    cute.arch.barrier()

    block_sum = Float32(0.0)
    if warp_idx == 0:
        if lane < Int32(warps):
            block_sum = warp_sums[lane]
        block_sum = cute.arch.warp_reduction_sum(block_sum)
        if lane == 0:
            warp_sums[0] = block_sum
    cute.arch.barrier()
    return warp_sums[0]
