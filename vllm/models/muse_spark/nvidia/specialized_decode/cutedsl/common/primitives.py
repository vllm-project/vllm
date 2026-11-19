# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
# The PTX primitives, launch helpers and Lamport stage constants are adapted
# from FlashInfer (flashinfer/comm/mnnvl_cutedsl/cute_dsl_primitives.py,
# runtime.py and kernel_ll/device_kernels.py,
# flashinfer-ai/flashinfer@14ba7aaa84718abe7a5648184fa9ad15362e2880,
# Apache-2.0), in the form of vLLM's DeepSeek V4.1 copy
# (vllm/models/deepseek_v41/nvidia/ops/cute_dsl/primitives.py).
# FlashInfer NOTICE: "FlashInfer / Copyright 2025-2026 NVIDIA / Copyright
# 2023-2026 FlashInfer community (https://flashinfer.ai/)".
# From FlashInfer: the Lamport stage and BF16 constants, load_global_u32x2/x4,
# load_global_bf16_as_f32, store_global_u32/_address/u32x2/u32x4,
# store_lamport_sentinel_u32x4, f32_to_bf16_bits, the packed BF16 and
# negative-zero helpers, map_shared_to_peer, store_shared_cluster_f32,
# store_shared_u32x4_address and stmc_bf16x2/x4/x8. The rest is not
# FlashInfer's.

"""PTX building blocks and launch helpers of the Muse Spark specialized decode.

Every Lamport kernel of the package (the boundary gathers, the attention
boundary, the MoE tail and the unpooler's zero-copy epilogue) is built from
these. BF16 negative zero (0x8000) marks a slot that has not been written yet;
a three-generation stage state (`NEXT_STAGE` written by the consumer,
`ACTIVE_STAGE` by the publisher) lets a fast rank publish the next call while
a slow one still reads the current one.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import (
    BFloat16,
    Float8E4M3FN,
    Float16,
    Float32,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
)
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, nvvm, vector
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

WARP_SIZE = 32
VEC_BF16 = 8
QUAD_BF16 = 4
NEGATIVE_ZERO_BF16_BITS = 0x8000
NEGATIVE_ZERO_BF16_PAIR = 0x80008000
LOG2E = 1.4426950408889634

# Lamport stage protocol: [3, tp, capacity, width] mailboxes and an int32[2]
# stage state per mailbox.
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


def _words(loaded, count, *, loc, ip):
    packed = vector.from_elements(
        ir.VectorType.get([count], T.i32(), loc=loc),
        [
            llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip)
            for index in range(count)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, count, Uint32)


def _words_of(loaded, count: int, *, loc, ip) -> tuple:
    """The `count` 32-bit words of the loaded value `loaded` (a scalar or
    vector of any element type), bits kept, in memory order."""
    words = llvm.bitcast(
        ir.VectorType.get([count], T.i32(), loc=loc), loaded, loc=loc, ip=ip
    )
    return tuple(
        Uint32(vector.extract(words, [], [i], loc=loc, ip=ip)) for i in range(count)
    )


def _bits32(value, *, loc, ip):
    """The 32 bits of `value` as an i32: an integer as is, an FP32 bit-cast
    (never converted by value); any other type fails at compile time."""
    if isinstance(value, int):
        value = Uint32(value)
    kind = type(value)
    if not (issubclass(kind, (cutlass.Integer, cutlass.Float)) and kind.width == 32):
        raise TypeError(f"expected a 32-bit word, got {kind.__name__}")
    bits = value.ir_value(loc=loc, ip=ip)
    if issubclass(kind, cutlass.Float):
        return llvm.bitcast(T.i32(), bits, loc=loc, ip=ip)
    return bits


def _word_vector(words, count: int | None = None, *, loc, ip):
    """The 32-bit `words` (a `TensorSSA`, or `count` indexable scalars) as
    one vector of i32, bits kept (`_bits32`)."""
    if isinstance(words, cute.TensorSSA) and words.dtype.width == 32:
        packed = words.ir_value(loc=loc, ip=ip)
        if issubclass(words.dtype, cutlass.Integer):
            return packed
        shape = ir.VectorType(packed.type).shape
        return llvm.bitcast(
            ir.VectorType.get(shape, T.i32(), loc=loc), packed, loc=loc, ip=ip
        )
    count = len(words) if count is None else count
    return vector.from_elements(
        ir.VectorType.get([count], T.i32(), loc=loc),
        [_bits32(words[i], loc=loc, ip=ip) for i in range(count)],
        loc=loc,
        ip=ip,
    )


def _unpack(struct, kind, count: int | None = None, *, loc, ip) -> tuple:
    """The fields of an asm result `struct`: `count` of type `kind`, or one
    per type of a tuple `kind`."""
    kinds = kind if isinstance(kind, tuple) else (kind,) * (count or 0)
    return tuple(
        k(llvm.extractvalue(k.mlir_type, struct, [index], loc=loc, ip=ip))
        for index, k in enumerate(kinds)
    )


def _ir(values, *, loc, ip) -> list:
    return [value.ir_value(loc=loc, ip=ip) for value in values]


def _intrinsic(name: str, result, operands, loc, ip):
    """A call of the LLVM intrinsic `name` on FP32 `operands`."""
    return llvm.call_intrinsic(
        result,
        name,
        _ir([Float32(x) for x in operands], loc=loc, ip=ip),
        [],
        [],
        loc=loc,
        ip=ip,
    )


def _ordered(text, constraints, operands, loc, ip, result=None):
    """Side-effecting asm: kept in program order with the other side-effecting
    operations."""
    return _asm(
        result,
        _ir(operands, loc=loc, ip=ip),
        text,
        constraints,
        side_effects=True,
        loc=loc,
        ip=ip,
    )


def _pure(text, constraints, operands, loc, ip, result=None):
    """Asm without side effects: the compiler may move or drop it."""
    return _asm(
        result,
        _ir(operands, loc=loc, ip=ip),
        text,
        constraints,
        side_effects=False,
        loc=loc,
        ip=ip,
    )


def predicated(predicate, body, zeros: tuple = (), *, loc, ip):
    """`body(ip)` when `predicate` is None or nonzero; otherwise nothing, or
    `zeros` for a `body` that returns that many values of their types."""
    if predicate is None:
        return body(ip)
    results = cutlass.if_generate(
        Int32(predicate) != Int32(0),
        lambda: body(None),
        (lambda: zeros) if zeros else None,
        return_types=[type(z) for z in zeros] or None,
        loc=loc,
        ip=ip,
    )
    return tuple(results) if len(zeros) > 1 else results


def global_ptr(dtype, address: Int64, align: int | None = None) -> cute.Pointer:
    """A global-memory `dtype` pointer at the integer `address`, aligned to
    `align` bytes (default: the element size)."""
    return cute.make_ptr(
        dtype,
        address,
        cute.AddressSpace.gmem,
        assumed_align=align or max(dtype.width // 8, 1),
    )


def shared_ptr(dtype, address: Int32, align: int | None = None) -> cute.Pointer:
    """A shared-memory `dtype` pointer at the integer (shared::cta) `address`,
    aligned to `align` bytes (default: the element size)."""
    return cute.make_ptr(
        dtype,
        address,
        cute.AddressSpace.smem,
        assumed_align=align or max(dtype.width // 8, 1),
    )


def _addr_ptr(address, space: cute.AddressSpace, *, loc, ip):
    """A raw LLVM pointer in `space` at the integer `address` (no alignment
    assumed): an operand of an NVVM op."""
    return llvm.inttoptr(
        llvm.PointerType.get(int(space)),
        address.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


def _global_addr(address: Int64, *, loc, ip):
    return _addr_ptr(Int64(address), cute.AddressSpace.gmem, loc=loc, ip=ip)


def _shared_addr(address: Int32, *, loc, ip):
    return _addr_ptr(Int32(address), cute.AddressSpace.smem, loc=loc, ip=ip)


def _cluster_addr(address: Int32, *, loc, ip):
    """The shared::cluster `address` (`map_shared_address_to_peer`)."""
    return _addr_ptr(Int32(address), cute.AddressSpace.dsmem, loc=loc, ip=ip)


@dsl_user_op
def load_global_u32x4(
    pointer: cute.Pointer,
    predicate: Int32 | None = None,
    *,
    volatile: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
):
    """Load four words; an omitted predicate loads normally, false returns zeros.
    Kept as asm: `cute.arch.load` has no predicated, zero-filled form, and the
    unpredicated `nvvm.load_ext` (volatile or weak) reshapes the Lamport poll
    loops (all_reduce_dual_norm +32 SASS, draft_all_reduce_boundary +56)."""
    predicate = Int32(1) if predicate is None else Int32(predicate)
    address = pointer.toint(loc=loc, ip=ip)
    opcode = "ld.volatile.global.v4.u32" if volatile else "ld.global.v4.u32"
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value(loc=loc, ip=ip), predicate.ir_value(loc=loc, ip=ip)],
        (
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.s32 p, $5, 0;\n\t"
            "@!p mov.u32 $0, 0;\n\t"
            "@!p mov.u32 $1, 0;\n\t"
            "@!p mov.u32 $2, 0;\n\t"
            "@!p mov.u32 $3, 0;\n\t"
            f"@p {opcode} {{$0, $1, $2, $3}}, [$4];\n\t"
            "}"
        ),
        "=r,=r,=r,=r,l,r",
        side_effects=volatile,
        loc=loc,
        ip=ip,
    )
    return _words(loaded, 4, loc=loc, ip=ip)


@dsl_user_op
def load_global_u32x2(pointer: cute.Pointer, *, loc=None, ip=None):
    """Two words at the 8-byte aligned global `pointer`, in one load."""
    loaded = cute.arch.load(
        global_ptr(Uint32, pointer.toint(loc=loc, ip=ip), 8),
        ir.VectorType.get([2], T.i32(), loc=loc),
        loc=loc,
        ip=ip,
    )
    return _words_of(loaded, 2, loc=loc, ip=ip)


@dsl_user_op
def load_global_bf16_as_f32(address: Int64, *, loc=None, ip=None) -> Float32:
    """The BF16 at global `address`, widened to FP32 (exact)."""
    bits = cute.arch.load(global_ptr(Uint16, Int64(address)), Uint16, loc=loc, ip=ip)
    bf16 = llvm.bitcast(
        BFloat16.mlir_type, bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    return Float32(arith.extf(T.f32(), bf16, loc=loc, ip=ip))


@dsl_user_op
def store_global_u32x4(address: Int64, packed, *, loc=None, ip=None) -> None:
    """Four words at the 16-byte aligned global `address`, in one store."""
    words = _word_vector(packed, 4, loc=loc, ip=ip)
    cute.arch.store(global_ptr(Uint32, address, 16), words, loc=loc, ip=ip)


@dsl_user_op
def store_global_u32x2(address: Int64, packed, *, loc=None, ip=None) -> None:
    """Two words at the 8-byte aligned global `address`, in one store."""
    words = _word_vector(packed, 2, loc=loc, ip=ip)
    cute.arch.store(global_ptr(Uint32, address, 8), words, loc=loc, ip=ip)


@dsl_user_op
def store_global_u32(
    pointer: cute.Pointer, value: Uint32, *, loc=None, ip=None
) -> None:
    """A weak store of one word; see `store_global_u32_address`."""
    cute.arch.store(pointer, Uint32(value), loc=loc, ip=ip)


@dsl_user_op
def store_global_u32_address(
    address: Int64, value: Uint32, *, loc=None, ip=None
) -> None:
    """A weak store of one word. The protocol words stored this way (stage
    states, tickets, sequence numbers, arrival counts) are read by a later
    grid after its `griddepcontrol.wait`, or are published by a release
    (fence, barrier and acq_rel atomic) that follows them; none relies on
    program order."""
    cute.arch.store(
        global_ptr(Uint32, Int64(address)),
        _bits32(value, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_lamport_sentinel_u32x4(address: Int64, *, loc=None, ip=None) -> None:
    """Re-arm a 16-byte Lamport slot with BF16 -0 (a weak store: the slot's
    next writer is a peer two generations on, after this grid)."""
    sentinel = cute.full(4, NEGATIVE_ZERO_BF16_PAIR, Uint32, loc=loc, ip=ip)
    store_global_u32x4(address, sentinel, loc=loc, ip=ip)


@dsl_user_op
def f32_to_bf16_bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    """`value` rounded to nearest-even BF16, as bits 0-15 of a word."""
    bf16 = _intrinsic("llvm.nvvm.f2bf16.rn", BFloat16.mlir_type, (value,), loc, ip)
    bits = llvm.bitcast(T.i16(), bf16, loc=loc, ip=ip)
    return Uint32(llvm.zext(T.i32(), bits, loc=loc, ip=ip))


@dsl_user_op
def pack_bf16x2_rn(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """Two FP32 rounded to nearest-even BF16, lo in bits 0-15, hi in bits 16-31."""
    pair = _intrinsic(
        "llvm.nvvm.ff2bf16x2.rn",
        ir.VectorType.get([2], BFloat16.mlir_type, loc=loc),
        (hi, lo),
        loc,
        ip,
    )
    return Uint32(llvm.bitcast(T.i32(), pair, loc=loc, ip=ip))


@dsl_user_op
def bf16x2_to_f32x2(word: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    """The BF16 pair in `word` widened to FP32 (exact): (bits 0-15, 16-31)."""
    pair = llvm.bitcast(
        ir.VectorType.get([2], BFloat16.mlir_type, loc=loc),
        Uint32(word).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    widened = arith.extf(ir.VectorType.get([2], T.f32(), loc=loc), pair, loc=loc, ip=ip)
    lo, hi = (vector.extract(widened, [], [i], loc=loc, ip=ip) for i in range(2))
    return Float32(lo), Float32(hi)


@dsl_user_op
def e4m3x4_to_bf16x4(word: Uint32, *, loc=None, ip=None) -> tuple[Uint32, Uint32]:
    """The four E4M3 bytes of `word` widened to BF16 (exact): (bytes 0-1,
    bytes 2-3) as BF16 pairs, the lower byte in the lower half.
    Kept as asm: `nvvm.convert_f8x2_to_bf16x2` takes only E8M0 and E5M3
    sources."""
    pair = _pure(
        "{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, $2;\n\t"
        "cvt.rn.bf16x2.e4m3x2 $0, lo;\n\tcvt.rn.bf16x2.e4m3x2 $1, hi;\n\t}",
        "=r,=r,r",
        (Uint32(word),),
        loc,
        ip,
        llvm.StructType.get_literal([T.i32()] * 2),
    )
    return _unpack(pair, Uint32, 2, loc=loc, ip=ip)


@dsl_user_op
def quant_e4m3x2(
    lo: Float32, hi: Float32, inv_scale: Float32, *, loc=None, ip=None
) -> Uint16:
    """`lo * inv_scale` and `hi * inv_scale` (round to nearest) as saturated
    E4M3 bytes, lo in bits 0-7 and hi in bits 8-15: the FP8 KV cache store of
    `reshape_and_cache_flash`."""
    return Uint16(
        nvvm.convert_f32x2_to_f8x2(
            T.i16(),
            mul_rn(hi, inv_scale, loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            mul_rn(lo, inv_scale, loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Float8E4M3FN.mlir_type,
            rnd=nvvm.FPRoundingMode.RN,
            sat=nvvm.SaturationMode.SATFINITE,
            loc=loc,
            ip=ip,
        )
    )


# FP32 arithmetic as NVVM ops and intrinsics: the rounding of each op is the one
# named, and LLVM neither contracts, reassociates nor narrows across them.
def _f32_asm(text: str, constraints: str, operands, loc, ip) -> Float32:
    """One FP32 PTX instruction as asm (opaque to LLVM)."""
    return Float32(
        _asm(
            T.f32(),
            _ir([Float32(x) for x in operands], loc=loc, ip=ip),
            text,
            constraints,
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


def _f32_intrinsic(name: str, operands, loc, ip) -> Float32:
    return Float32(_intrinsic(name, T.f32(), operands, loc, ip))


def _mul_contract(a, b, loc, ip) -> Float32:
    """A * b as `mul.f32`: ptxas may fuse it into the instruction it feeds."""
    return Float32(
        arith.mulf(
            *_ir([Float32(a), Float32(b)], loc=loc, ip=ip),
            fastmath=arith.FastMathFlags.contract,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """A + b, round to nearest, subnormals kept."""
    return Float32(
        nvvm.addf(
            *_ir([Float32(a), Float32(b)], loc=loc, ip=ip),
            rnd=nvvm.FPRoundingMode.RN,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def add_ftz_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """A + b, round to nearest, subnormal inputs and result flushed to a zero
    of the same sign (the add of code built with fast math).
    Kept as asm: with the typed FTZ ops (nvvm.addf/subf, mul.rn.ftz, fmax,
    ex2) attn2's split kernel unrolls less and runs slower."""
    return _f32_asm("add.rn.ftz.f32 $0, $1, $2;", "=f,f,f", (a, b), loc, ip)


@dsl_user_op
def mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """A * b, round to nearest, subnormals kept; never contracted."""
    return _f32_intrinsic("llvm.nvvm.mul.rn.f", (a, b), loc, ip)


@dsl_user_op
def fma_rn(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """A * b + c with one rounding."""
    return Float32(
        nvvm.fma(
            *_ir([Float32(a), Float32(b), Float32(c)], loc=loc, ip=ip),
            ir.Attribute.parse("#nvvm.fp_rnd_mode<rn>"),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def rsqrt_approx_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    """rsqrt.approx with subnormal inputs kept (`rsqrtf` without fast math)."""
    return Float32(nvvm.rsqrt(Float32(a).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def fms_rn(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """A * b - c with one rounding (the negation is exact, -0 included)."""
    return fma_rn(a, b, -Float32(c), loc=loc, ip=ip)


@dsl_user_op
def max_ftz_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """The larger of a and b (a NaN operand yields the other), subnormals
    flushed. Kept as asm: see `add_ftz_f32`."""
    return _f32_asm("max.ftz.f32 $0, $1, $2;", "=f,f,f", (a, b), loc, ip)


@dsl_user_op
def ex2_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """2 ** a by ex2.approx, subnormals flushed. Kept as asm: see
    `add_ftz_f32`."""
    return _f32_asm("ex2.approx.ftz.f32 $0, $1;", "=f,f", (a,), loc, ip)


@dsl_user_op
def div_full_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """A / b as `div.full.f32` (Triton's FP32 division)."""
    return _f32_intrinsic("llvm.nvvm.div.full", (a, b), loc, ip)


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
def packed_u32x4_to_f16x8(packed, *, loc=None, ip=None):
    values = llvm.bitcast(
        ir.VectorType.get([VEC_BF16], Float16.mlir_type, loc=loc),
        packed.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(values, VEC_BF16, Float16)


@dsl_user_op
def bf16x8_to_packed_u32x4(values, *, loc=None, ip=None):
    packed = llvm.bitcast(
        ir.VectorType.get([4], T.i32(), loc=loc),
        values.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@cute.jit
def sanitize_negative_zero_u32x4(packed):
    sanitized = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
    for index in cutlass.range_constexpr(4):
        sanitized[index] = sanitize_negative_zero_u32(packed[index])
    return sanitized.load()


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
def fragment_has_negative_zero(packed):
    dirty = False
    for index in cutlass.range_constexpr(4):
        word = packed[index]
        dirty = (
            dirty
            | (Uint16(word & Uint32(0xFFFF)) == Uint16(NEGATIVE_ZERO_BF16_BITS))
            | (Uint16(word >> Uint32(16)) == Uint16(NEGATIVE_ZERO_BF16_BITS))
        )
    return dirty


@dsl_user_op
def map_shared_to_peer(
    smem_pointer: cute.Pointer, peer_rank: Int32, *, loc=None, ip=None
) -> Int32:
    return map_shared_address_to_peer(
        smem_pointer.toint(loc=loc, ip=ip), peer_rank, loc=loc, ip=ip
    )


@dsl_user_op
def map_shared_address_to_peer(
    address: Int32, peer_rank: Int32, *, loc=None, ip=None
) -> Int32:
    """The shared::cluster address of this CTA's shared `address` in the
    cluster's CTA `peer_rank`."""
    peer = cute.arch.map_dsmem_ptr(
        shared_ptr(Uint8, address), peer_rank, loc=loc, ip=ip
    )
    return Int32(peer.toint(loc=loc, ip=ip))


@dsl_user_op
def store_shared_cluster_f32(
    remote_address: Int32, value: Float32, *, loc=None, ip=None
) -> None:
    """Store `value` at the shared::cluster address `remote_address`."""
    llvm.store(
        Float32(value).ir_value(loc=loc, ip=ip),
        _cluster_addr(remote_address, loc=loc, ip=ip),
        alignment=4,
        loc=loc,
        ip=ip,
    )


def _load_shared_u32x4(address, loc, ip):
    """Four words at the shared `address`, the struct of one ld.shared.v4;
    side-effecting, so it does not move across the barriers and mbarrier waits
    that guard the data."""
    return _ordered(
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        (Int32(address),),
        loc,
        ip,
        result=llvm.StructType.get_literal([T.i32()] * 4),
    )


@dsl_user_op
def load_shared_u32x4_address(address: Int32, *, loc=None, ip=None):
    """`load_shared_u32x4` by address.
    Kept as asm: the typed load halves LLVM's unroll of qkv_front's K loop,
    which runs slower."""
    return _words(_load_shared_u32x4(address, loc, ip), 4, loc=loc, ip=ip)


@dsl_user_op
def load_shared_u32x4_unpacked(address: Int32, *, loc=None, ip=None) -> tuple:
    """`load_shared_u32x4_address` as a tuple of four Uint32.
    Kept as asm: see `load_shared_u32x4_address`."""
    return _unpack(_load_shared_u32x4(address, loc, ip), Uint32, 4, loc=loc, ip=ip)


@dsl_user_op
def store_shared_u32x4_address(address: Int32, words, *, loc=None, ip=None) -> None:
    """Four words at the 16-byte aligned shared `address`, in one store."""
    packed = _word_vector(words, 4, loc=loc, ip=ip)
    cute.arch.store(shared_ptr(Uint32, address, 16), packed, loc=loc, ip=ip)


def _stmc_words(
    address: Int64, words, nwords: int | None = None, dtype=Float32, *, loc, ip
) -> None:
    """Multicast-store 32-bit `words` (as for `_word_vector`) with one
    multimem.st.relaxed.sys of `dtype` elements (the type only names the
    width: the bits are stored as is)."""
    packed = _word_vector(words, nwords, loc=loc, ip=ip)
    count = 32 * ir.VectorType(packed.type).shape[0] // dtype.width
    value_type = (
        dtype.mlir_type
        if count == 1
        else ir.VectorType.get([count], dtype.mlir_type, loc=loc)
    )
    nvvm.multimem_st(
        llvm.bitcast(value_type, packed, loc=loc, ip=ip),
        _global_addr(address, loc=loc, ip=ip),
        ir.Attribute.parse("#nvvm.mem_order<relaxed>"),
        nvvm.MemScopeKind.SYS,
        dtype.mlir_type,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stmc_bf16x2(address: Int64, packed: Uint32, *, loc=None, ip=None) -> None:
    """Multicast-store one packed BF16 pair."""
    _stmc_words(address, [packed], loc=loc, ip=ip)


@dsl_user_op
def stmc_bf16x4(address: Int64, values, *, loc=None, ip=None) -> None:
    """Multicast-store four BF16s (two packed words)."""
    _stmc_words(address, values, 2, loc=loc, ip=ip)


@dsl_user_op
def stmc_bf16x8(
    address: Int64, values, predicate: Int32 | None = None, *, loc=None, ip=None
) -> None:
    """Multicast-store eight BF16s; an omitted predicate stores, false skips."""
    predicated(
        predicate,
        lambda ip: _stmc_words(address, values, 4, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def global_timer_elapsed(
    now: Int64, start: Int64, ns: Int64, *, loc=None, ip=None
) -> Int32:
    """1 when `now - start`, wrapped to 64 bits, is at least `ns`, else 0.

    Wrap-safe at any timer value, and opaque to the compiler: a load
    predicated on it is ordered after the timer read `now` without a
    timer-derived address term.
    Kept as asm: its order against the timer read is deliberate.
    """
    return Int32(
        _asm(
            T.i32(),
            [
                Int64(now).ir_value(loc=loc, ip=ip),
                Int64(start).ir_value(loc=loc, ip=ip),
                Int64(ns).ir_value(loc=loc, ip=ip),
            ],
            (
                "{\n\t"
                ".reg .s64 t;\n\t"
                ".reg .pred p;\n\t"
                "sub.s64 t, $1, $2;\n\t"
                "setp.ge.s64 p, t, $3;\n\t"
                "selp.u32 $0, 1, 0, p;\n\t"
                "}"
            ),
            "=r,l,l,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def paced_ready(
    start: Int64, block: cutlass.Constexpr[int], pace_ns: cutlass.Constexpr[int]
) -> Int32:
    """The predicate of K block `block`'s paced weight loads: spins until
    `block * pace_ns` after `start` (at once for block 0 or no pacing),
    then 1, from the clock read that ends the spin, so the loads predicated
    on it cannot be issued before it."""
    ready = Int32(1)
    if cutlass.const_expr(pace_ns > 0 and block > 0):
        delay = Int64(block * pace_ns)
        ready = global_timer_elapsed(cute.arch.globaltimer(), start, delay)
        while ready == Int32(0):
            ready = global_timer_elapsed(cute.arch.globaltimer(), start, delay)
    return ready


@dsl_user_op
def trap(*, loc=None, ip=None) -> None:
    """Abort the kernel: the launch fails instead of hanging."""
    llvm.intr_trap(loc=loc, ip=ip)


@cute.jit
def load_f32x8(base: cute.Tensor, offset: Int64):
    pointer = cute.make_ptr(
        Float32,
        (base.iterator + offset).llvm_ptr,
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    fragment = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
    cute.autovec_copy(
        cute.make_tensor(pointer, cute.make_layout((VEC_BF16,))), fragment
    )
    return fragment.load()


@cute.jit
def store_f32x8(base: cute.Tensor, offset: Int64, values) -> None:
    pointer = cute.make_ptr(
        Float32,
        (base.iterator + offset).llvm_ptr,
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    fragment = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
    fragment.store(values)
    cute.autovec_copy(
        fragment, cute.make_tensor(pointer, cute.make_layout((VEC_BF16,)))
    )


@cute.jit
def load_bf16x8_as_f32(base: cute.Tensor, offset: Int64):
    pointer = cute.make_ptr(
        BFloat16,
        (base.iterator + offset).llvm_ptr,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return packed_u32x4_to_bf16x8(load_global_u32x4(pointer)).to(Float32)


@cute.jit
def load_f16x8_as_f32(base: cute.Tensor, offset: Int64):
    """8 FP16 elements from element `offset` of `base` (one 16-byte
    access), converted exactly to FP32."""
    pointer = cute.make_ptr(
        Float16,
        (base.iterator + offset).llvm_ptr,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return packed_u32x4_to_f16x8(load_global_u32x4(pointer)).to(Float32)


def regs(values) -> list:
    """An 8-wide FP32 TensorSSA (`load_f32x8`, `load_bf16x8_as_f32`) as
    its 8 scalars (traced inline)."""
    fragment = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
    fragment.store(values)
    return [fragment[i] for i in range(VEC_BF16)]


def pack_bf16x8_rn(values: list):
    """8 FP32 scalars rounded to nearest-even BF16, as 4 packed words (an
    rmem tensor; traced inline)."""
    words = cute.make_rmem_tensor(cute.make_layout((VEC_BF16 // 2,)), Uint32)
    for q in range(VEC_BF16 // 2):
        words[q] = pack_bf16x2_rn(values[2 * q], values[2 * q + 1])
    return words


def store_bf16x8_rn(base: cute.Tensor, offset: Int64, values: list) -> None:
    """8 FP32 scalars rounded to nearest-even BF16 and stored as one 16-byte
    access at element `offset` of the BF16 `base` (traced inline)."""
    store_global_u32x4(
        Int64((base.iterator + offset).toint()), pack_bf16x8_rn(values).load()
    )


class _GraphSafeDLPack:
    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def __dlpack__(self, stream=None):
        # stream=-1 skips producer sync; CuTe launches on the current stream,
        # which is the capturing one under CUDA graphs.
        return self.tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


def to_cute(tensor: torch.Tensor, alignment: int) -> cute.Tensor:
    if tensor.dtype == torch.float8_e4m3fn:
        # from_dlpack has no Float8E4M3FN entry: export the bytes and restore
        # the element type.
        result = from_dlpack(
            _GraphSafeDLPack(tensor.view(torch.uint8).detach()),
            assumed_align=alignment,
        )
        result.element_type = Float8E4M3FN
        return result
    return from_dlpack(_GraphSafeDLPack(tensor.detach()), assumed_align=alignment)


def to_cute_dynamic(
    tensor: torch.Tensor, alignment: int, *, divisibility: int
) -> cute.Tensor:
    return to_cute(tensor, alignment).mark_compact_shape_dynamic(
        mode=0, divisibility=divisibility
    )


def make_fake_dynamic_compact_tensor(dtype, *, alignment: int, divisibility: int):
    return make_fake_compact_tensor(
        dtype, (cute.sym_int32(divisibility=divisibility),), assumed_align=alignment
    )


def current_cu_stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


# Global memory by byte address (an Int64). An ``ordered`` load is
# side-effecting asm, kept in program order with the other side-effecting
# operations (stores, atomics, barriers, the PDL wait); an unordered load may be
# moved by the compiler. With a ``predicate``, a false one loads zero.


def _load_global_address(
    kind, zero, result, constraint, address, predicate, ordered, loc, ip
):
    operands = [address.ir_value(loc=loc, ip=ip)]
    if predicate is None:
        text = f"ld.global.{kind} $0, [$1];"
        constraints = f"={constraint},l"
    else:
        operands.append(Int32(predicate).ir_value(loc=loc, ip=ip))
        text = (
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t"
            f"mov.{zero} $0, 0;\n\t@p ld.global.{kind} $0, [$1];\n\t}}"
        )
        constraints = f"={constraint},l,r"
    return _asm(
        result, operands, text, constraints, side_effects=ordered, loc=loc, ip=ip
    )


@dsl_user_op
def load_global_u32_address(
    address: Int64,
    predicate: Int32 | None = None,
    *,
    ordered: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
) -> Uint32:
    """Kept as asm: the ordered form stays after griddepcontrol.wait."""
    return Uint32(
        _load_global_address(
            "u32", "u32", T.i32(), "r", address, predicate, ordered, loc, ip
        )
    )


@dsl_user_op
def load_global_s64_address(
    address: Int64,
    predicate: Int32 | None = None,
    *,
    ordered: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
) -> Int64:
    """Kept as asm: the ordered form stays after griddepcontrol.wait."""
    return Int64(
        _load_global_address(
            "s64", "u64", T.i64(), "l", address, predicate, ordered, loc, ip
        )
    )


@dsl_user_op
def load_global_f32x4_address(
    address: Int64,
    *,
    ordered: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
) -> tuple[Float32, Float32, Float32, Float32]:
    """Four FP32 at the 16-byte aligned `address` in one load; `ordered`
    as for `load_global_u32_address`. A coherent load, not `ld.global.nc`:
    ptxas moves read-only-path loads across `griddepcontrol.wait` either way,
    so they cannot read what the PDL primary grid writes.
    Kept as asm: the ordered form stays after griddepcontrol.wait."""
    loaded = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        side_effects=ordered,
        loc=loc,
        ip=ip,
    )
    a, b, c, d = _unpack(loaded, Float32, 4, loc=loc, ip=ip)
    return a, b, c, d


@dsl_user_op
def f32_bits(value: Float32, *, loc=None, ip=None) -> Int32:
    """The bit pattern of an FP32 value."""
    return Int32(
        llvm.bitcast(T.i32(), Float32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


@dsl_user_op
def f32_from_bits(bits: Uint32, *, loc=None, ip=None) -> Float32:
    """The FP32 value of a bit pattern (`f32_bits`'s inverse)."""
    return Float32(
        llvm.bitcast(T.f32(), Uint32(bits).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


# Triton's FP32 operations as its NVIDIA backend lowers them (round to nearest,
# subnormals kept, never contracted).


@dsl_user_op
def lowest_set_bit(mask: Uint32, *, loc=None, ip=None) -> Int32:
    """The index of the lowest set bit of a nonzero `mask` (CUDA's
    `__ffs(mask) - 1`).
    Kept as asm: LLVM folds ctlz(bitreverse) to cttz, which NVPTX lowers to
    a popc sequence, not brev + bfind.shiftamt (routing +56 SASS)."""
    return Int32(
        _asm(
            T.i32(),
            [Uint32(mask).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 r;\n\tbrev.b32 r, $1;\n\tbfind.shiftamt.u32 $0, r;\n\t}",
            "=r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def sigmoid_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    """`tl.sigmoid(x)`: x * -log2(e), ex2.approx, + 1, then 1 / that with
    div.full.
    Kept as asm: `llvm.nvvm.div.full` takes no immediate numerator, and
    ptxas expands div.full by a register 1.0 longer (routing +48 SASS)."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .f32 a, e, d;\n\tmul.f32 a, $1, 0fBFB8AA3B;\n\t"
            "ex2.approx.f32 e, a;\n\tadd.f32 d, e, 0f3F800000;\n\t"
            "div.full.f32 $0, 0f3F800000, d;\n\t}",
            "=f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def silu_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    """`x / (1 + expf(-x))` as the CUDA C++ ops built with fast math
    compute it: x * -log2(e), ex2.approx, + 1, then x / that with
    div.full."""
    e = ex2_approx_ftz(_mul_contract(x, Float32(-LOG2E), loc, ip), loc=loc, ip=ip)
    d = add_ftz_f32(e, Float32(1.0), loc=loc, ip=ip)
    return div_full_f32(x, d, loc=loc, ip=ip)


# Shared-memory pipeline, cluster and TMA instructions by address: a shared
# address is the Int32 of the shared window, a global one an Int64, a tensor map
# the Int64 generic address of a TensorMap kernel parameter. Each wrapper is one
# PTX instruction or a try_wait spin loop. Those that touch memory are NVVM ops
# with memory effects or side-effecting asm, so LLVM keeps them in order with
# the accesses they may alias; a st.async store with ``clobber_memory`` is
# also followed by a compiler memory barrier.


@dsl_user_op
def _mbarrier_try_wait_cluster(
    bar: Int32, parity: Int32, *, loc=None, ip=None
) -> cutlass.Boolean:
    """True once the phase of `parity` of the shared mbarrier at `bar` has
    completed (acquire at cluster scope: the stores that completed it are
    visible), false when the bounded wait ends first."""
    done = nvvm.mbarrier_wait_parity(
        _shared_addr(bar, loc=loc, ip=ip),
        Int32(parity).ir_value(loc=loc, ip=ip),
        nvvm.MBarrierWaitKind.TRY,
        scope=nvvm.MBarrierScopeKind.CLUSTER,
        order=nvvm.MemOrderKind.ACQUIRE,
        loc=loc,
        ip=ip,
    )
    return cutlass.Boolean(done)


@dsl_user_op
def mbarrier_wait_address(bar: Int32, parity: Int32, *, loc=None, ip=None) -> None:
    """Spin until the phase of `parity` completes (acquire, CTA scope).
    Kept as asm: the spin loop inside one asm keeps qkv_front's K loop
    innermost; a typed try_wait loop stops its full unroll, which runs
    slower."""
    _ordered(
        "{\n\t.reg .pred done;\n\tspin:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 done, [$0], $1;\n\t"
        "@!done bra spin;\n\t}",
        "r,r",
        (Int32(bar), Int32(parity)),
        loc,
        ip,
    )


@cute.jit
def mbarrier_wait_cluster_address(bar: Int32, parity: Int32) -> None:
    """Spin on `_mbarrier_try_wait_cluster` until the phase of `parity`
    completes: for a phase that peers complete (st.async complete_tx,
    release at cluster scope)."""
    done = cutlass.Boolean(False)
    while not done:
        done = _mbarrier_try_wait_cluster(bar, parity)


@dsl_user_op
def tma_load_2d(
    dst: Int32,
    tensor_map: Int64,
    c0: Int32,
    c1: Int32,
    bar: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """The box of `tensor_map` at (c0, c1) (inner coordinate first) into
    shared `dst`, its bytes completing on mbarrier `bar`.
    Kept as asm: the typed nvvm.cp_async_bulk_tensor_shared_cluster_global
    reloads the tensor map address per load in qkv_front (LDCU 7 -> 14, +8
    SASS)."""
    _ordered(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx"
        "::bytes [$0], [$1, {$2, $3}], [$4];",
        "r,l,r,r,r",
        (Int32(dst), Int64(tensor_map), Int32(c0), Int32(c1), Int32(bar)),
        loc,
        ip,
    )


@dsl_user_op
def tma_prefetch_2d_l2(
    tensor_map: Int64, c0: Int32, c1: Int32, *, loc=None, ip=None
) -> None:
    """Prefetch the box of `tensor_map` at (c0, c1) into L2."""
    nvvm.cp_async_bulk_tensor_prefetch(
        _addr_ptr(Int64(tensor_map), cute.AddressSpace.generic, loc=loc, ip=ip),
        _ir([Int32(c0), Int32(c1)], loc=loc, ip=ip),
        [],
        loc=loc,
        ip=ip,
    )


def _memory_barrier(loc, ip) -> None:
    """A compiler-only memory barrier (empty asm with a memory clobber): LLVM
    moves no memory access across it; it emits no instruction."""
    _ordered("", "~{memory}", (), loc, ip)


def _store_async_cluster(remote, values, remote_bar, loc, ip) -> None:
    packed = vector.from_elements(
        ir.VectorType.get([len(values)], T.f32(), loc=loc),
        _ir([Float32(v) for v in values], loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    nvvm.store_async(
        _cluster_addr(remote, loc=loc, ip=ip),
        llvm.bitcast(
            ir.VectorType.get([len(values)], T.i32(), loc=loc), packed, loc=loc, ip=ip
        ),
        _cluster_addr(remote_bar, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_async_cluster_f32x2(
    remote: Int32,
    lo: Float32,
    hi: Float32,
    remote_bar: Int32,
    predicate: Int32 | None = None,
    *,
    clobber_memory: bool = False,
    loc=None,
    ip=None,
) -> None:
    """Store (lo, hi) at cluster address `remote`
    (`map_shared_address_to_peer`), completing 8 transaction bytes on the
    peer's mbarrier at cluster address `remote_bar`; with a `predicate`,
    only if it is true. `clobber_memory` follows it with a compiler memory
    barrier."""
    predicated(
        predicate,
        lambda ip: _store_async_cluster(remote, (lo, hi), remote_bar, loc, ip),
        loc=loc,
        ip=ip,
    )
    if clobber_memory:
        _memory_barrier(loc, ip)


@dsl_user_op
def store_async_cluster_f32x4(
    address: Int32,
    values,
    mbarrier: Int32,
    *,
    clobber_memory: bool = False,
    loc=None,
    ip=None,
) -> None:
    """Four FP32 into a cluster CTA's shared memory
    (`map_shared_address_to_peer`), completing 16 transaction bytes on that
    CTA's mbarrier (`mbarrier`, mapped the same way). `clobber_memory`
    follows it with a compiler memory barrier."""
    _store_async_cluster(address, tuple(values), mbarrier, loc, ip)
    if clobber_memory:
        _memory_barrier(loc, ip)


@dsl_user_op
def bulk_prefetch_l2(address: Int64, nbytes: Int32, *, loc=None, ip=None) -> None:
    """Prefetch `nbytes` (a multiple of 16) at the 16-byte aligned `address`
    into L2."""
    nvvm.cp_async_bulk_prefetch(
        _global_addr(address, loc=loc, ip=ip),
        Int32(nbytes).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


# The QKV front's global load, FP32 and mma wrappers (the mma is also attn2's
# and the unpooler's).


@dsl_user_op
def load_global_nc_f32_address(address: Int64, *, loc=None, ip=None) -> Float32:
    """A read-only FP32 load through the non-coherent path, in program order.
    Kept as asm: CUTLASS has no `.nc` load."""
    return Float32(
        _ordered(
            "ld.global.nc.f32 $0, [$1];", "=f,l", (Int64(address),), loc, ip, T.f32()
        )
    )


def _mma_m16n8k16(operands: str, a, b0, b1, c, ordered: bool, loc, ip) -> tuple:
    struct = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        _ir([Uint32(w) for w in (*a, b0, b1)], loc=loc, ip=ip)
        + _ir([Float32(x) for x in c], loc=loc, ip=ip),
        f"mma.sync.aligned.m16n8k16.row.col.f32.{operands}.{operands}.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        side_effects=ordered,
        loc=loc,
        ip=ip,
    )
    return _unpack(struct, Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def mma_m16n8k16_bf16(
    a,
    b0: Uint32,
    b1: Uint32,
    c,
    *,
    ordered: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
) -> tuple:
    """mma.sync m16n8k16 row.col with BF16 operands and FP32 accumulation: the
    four accumulators of A (four words) x B (b0, b1) + C (four FP32).
    `ordered` makes it side-effecting asm, kept in program order with the
    other side-effecting operations; otherwise the compiler schedules it.
    Kept as asm: `ordered` pins it in program order."""
    return _mma_m16n8k16("bf16", a, b0, b1, c, ordered, loc, ip)


@dsl_user_op
def mma_m16n8k16_f16(a, b0: Uint32, b1: Uint32, c, *, loc=None, ip=None) -> tuple:
    """`mma_m16n8k16_bf16` with F16 operands.
    Kept as asm: callers pass hand-built fragments."""
    return _mma_m16n8k16("f16", a, b0, b1, c, False, loc, ip)


# The paged decode attention's building blocks (attention/attn2.py): FP32 ops
# with subnormals flushed and the approximate ex2, rcp and rsqrt (each one
# explicit, nothing contracts across them), shared and global memory by
# address, and the cluster operations of its distributed-shared-memory merge.


@dsl_user_op
def sub_ftz_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """A - b, round to nearest, subnormals flushed.
    Kept as asm: see `add_ftz_f32`."""
    return _f32_asm("sub.rn.ftz.f32 $0, $1, $2;", "=f,f,f", (a, b), loc, ip)


@dsl_user_op
def mul_ftz_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """A * b, round to nearest, subnormals flushed; never contracted.
    Kept as asm: see `add_ftz_f32`."""
    return _f32_asm("mul.rn.ftz.f32 $0, $1, $2;", "=f,f,f", (a, b), loc, ip)


@dsl_user_op
def scale_or_neg_inf(
    x: Float32,
    scale: Float32,
    key: Int32,
    last: Int32,
    first: Int32 | None = None,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """`scale * x` (round to nearest, subnormals flushed) for a `key` in
    [first, last] (no lower bound without `first`), else -inf."""
    keep = Int32(key) <= Int32(last)
    if first is not None:
        keep = keep & (Int32(key) >= Int32(first))
    scaled = mul_ftz_f32(scale, x, loc=loc, ip=ip)
    return Float32(
        arith.select(
            *_ir([keep, scaled, Float32(float("-inf"))], loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def movmatrix_trans_b16(word: Uint32, *, loc=None, ip=None) -> Uint32:
    """This lane's word of the warp's 8x8 b16 matrix, transposed."""
    return Uint32(
        nvvm.movmatrix(
            Uint32(word).ir_value(loc=loc, ip=ip),
            ir.Attribute.parse("#nvvm.ld_st_matrix_shape<m = 8, n = 8>"),
            nvvm.LdStMatrixEltType.B16,
            results=[T.i32()],
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_global_nc_u32x4_address(address: Int64, *, loc=None, ip=None) -> tuple:
    """Four words of data no thread of the grid writes (the read-only path),
    in program order with the other side-effecting operations.
    Kept as asm: CUTLASS has no `.nc` load."""
    struct = _ordered(
        "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        (Int64(address),),
        loc,
        ip,
        llvm.StructType.get_literal([T.i32()] * 4),
    )
    return _unpack(struct, Uint32, 4, loc=loc, ip=ip)


@dsl_user_op
def load_global_nc_s32_or(
    address: Int64, predicate: Int32, other: Int32, *, loc=None, ip=None
) -> Int32:
    """The read-only word at `address` if `predicate`, else `other`.
    Kept as asm: CUTLASS has no `.nc` load."""
    return Int32(
        _ordered(
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\tmov.b32 $0, $3;\n\t"
            "@p ld.global.nc.s32 $0, [$1];\n\t}",
            "=r,l,r,r",
            (Int64(address), Int32(predicate), Int32(other)),
            loc,
            ip,
            T.i32(),
        )
    )


@dsl_user_op
def load_global_cg_f32x2_address(address: Int64, *, loc=None, ip=None) -> tuple:
    """Two FP32 at the 8-byte aligned global `address`, cached in L2 only."""
    loaded = cute.arch.load(
        global_ptr(Float32, address, 8),
        ir.VectorType.get([2], T.f32(), loc=loc),
        cop="cg",
        loc=loc,
        ip=ip,
    )
    values = cute.TensorSSA(loaded, 2, Float32)
    return Float32(values[0]), Float32(values[1])


@dsl_user_op
def store_global_f32_address(
    address: Int64, value: Float32, predicate: Int32, *, loc=None, ip=None
) -> None:
    """Store `value` at global `address` if `predicate`.
    Kept as asm: `cute.arch.store` has no predicated form."""
    _ordered(
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t"
        "@p st.global.f32 [$0], $1;\n\t}",
        "l,f,r",
        (Int64(address), Float32(value), Int32(predicate)),
        loc,
        ip,
    )


@dsl_user_op
def store_global_f32x2_address(
    address: Int64, x: Float32, y: Float32, *, loc=None, ip=None
) -> None:
    """Two FP32 at the 8-byte aligned global `address`, in one store."""
    pair = vector.from_elements(
        ir.VectorType.get([2], T.f32(), loc=loc),
        [Float32(x).ir_value(loc=loc, ip=ip), Float32(y).ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )
    cute.arch.store(global_ptr(Float32, address, 8), pair, loc=loc, ip=ip)


@dsl_user_op
def membar_gl(*, loc=None, ip=None) -> None:
    """The fence of `__threadfence()`."""
    nvvm.MembarOp(nvvm.MemScopeKind.GPU, loc=loc, ip=ip)


@dsl_user_op
def load_shared_f32x4_address(address: Int32, *, loc=None, ip=None) -> tuple:
    """Four FP32 at the 16-byte aligned shared `address`, in one load."""
    loaded = cute.arch.load(
        shared_ptr(Float32, address, 16),
        ir.VectorType.get([4], T.f32(), loc=loc),
        loc=loc,
        ip=ip,
    )
    values = cute.TensorSSA(loaded, 4, Float32)
    return tuple(Float32(values[i]) for i in range(4))


@dsl_user_op
def load_shared_f32x2_address(address: Int32, *, loc=None, ip=None) -> tuple:
    """Two FP32 at the 8-byte aligned shared `address`, in one load."""
    loaded = cute.arch.load(
        shared_ptr(Float32, address, 8),
        ir.VectorType.get([2], T.f32(), loc=loc),
        loc=loc,
        ip=ip,
    )
    values = cute.TensorSSA(loaded, 2, Float32)
    return Float32(values[0]), Float32(values[1])


@dsl_user_op
def load_shared_f32_or(
    address: Int32, predicate: Int32, other: Float32, *, loc=None, ip=None
) -> Float32:
    """The FP32 at shared `address` if `predicate`, else `other`.
    Kept as asm: `cute.arch.load` has no predicated form."""
    return Float32(
        _ordered(
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\tmov.f32 $0, $3;\n\t"
            "@p ld.shared.f32 $0, [$1];\n\t}",
            "=f,r,r,f",
            (Int32(address), Int32(predicate), Float32(other)),
            loc,
            ip,
            T.f32(),
        )
    )


@dsl_user_op
def store_shared_f32x4_address(address: Int32, values, *, loc=None, ip=None) -> None:
    """Four FP32 at the 16-byte aligned shared `address`, in one store."""
    packed = vector.from_elements(
        ir.VectorType.get([4], T.f32(), loc=loc),
        [Float32(v).ir_value(loc=loc, ip=ip) for v in values],
        loc=loc,
        ip=ip,
    )
    cute.arch.store(shared_ptr(Float32, address, 16), packed, loc=loc, ip=ip)


@dsl_user_op
def store_shared_f32_address(
    address: Int32, value: Float32, predicate: Int32, *, loc=None, ip=None
) -> None:
    """Store `value` at shared `address` if `predicate`.
    Kept as asm: `cute.arch.store` has no predicated form."""
    _ordered(
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t"
        "@p st.shared.f32 [$0], $1;\n\t}",
        "r,f,r",
        (Int32(address), Float32(value), Int32(predicate)),
        loc,
        ip,
    )


@dsl_user_op
def cp_async_cg_16(
    destination: Int32, source: Int64, predicate: Int32, *, loc=None, ip=None
) -> None:
    """Copy 16 bytes from global `source` to this CTA's shared
    `destination` asynchronously, bypassing L1, if `predicate`.
    Kept as asm: `CopyG2SOp` predication zero-fills, not skips."""
    _ordered(
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t"
        "@p cp.async.cg.shared.global [$0], [$1], 16;\n\t}",
        "r,l,r,~{memory}",
        (Int32(destination), Int64(source), Int32(predicate)),
        loc,
        ip,
    )


@dsl_user_op
def load_global_bf16x8_as_f32_address(
    address: Int64,
    predicate: Int32,
    *,
    fill_bits: cutlass.Constexpr[int] = 0,
    nc: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
) -> tuple:
    """8 BF16 at the 16-byte aligned global `address` widened to FP32 (an
    ordered load, `.nc` if `nc`); where not `predicate`, the BF16 pairs of
    the 32-bit `fill_bits` (0: zeros, -8323200: -inf).
    Kept as asm: an ordered, predicated fill load."""
    space = "global.nc" if nc else "global"
    fill = "".join(f"mov.b32 w{i}, {fill_bits};\n\t" for i in range(4))
    struct = _ordered(
        "{\n\t.reg .pred p;\n\t.reg .b32 w0, w1, w2, w3;\n\t"
        ".reg .b16 h0, h1, h2, h3, h4, h5, h6, h7;\n\t"
        f"setp.ne.s32 p, $9, 0;\n\t{fill}"
        f"@p ld.{space}.v4.b32 {{w0, w1, w2, w3}}, [$8];\n\t"
        "mov.b32 {h0, h1}, w0;\n\tmov.b32 {h2, h3}, w1;\n\t"
        "mov.b32 {h4, h5}, w2;\n\tmov.b32 {h6, h7}, w3;\n\t"
        "cvt.f32.bf16 $0, h0;\n\tcvt.f32.bf16 $1, h1;\n\t"
        "cvt.f32.bf16 $2, h2;\n\tcvt.f32.bf16 $3, h3;\n\t"
        "cvt.f32.bf16 $4, h4;\n\tcvt.f32.bf16 $5, h5;\n\t"
        "cvt.f32.bf16 $6, h6;\n\tcvt.f32.bf16 $7, h7;\n\t}",
        "=f,=f,=f,=f,=f,=f,=f,=f,l,r",
        (Int64(address), Int32(predicate)),
        loc,
        ip,
        llvm.StructType.get_literal([T.f32()] * 8),
    )
    return _unpack(struct, Float32, 8, loc=loc, ip=ip)


# Shared and predicated accesses and key compares of the sampling kernels.


@dsl_user_op
def store_shared_u32(address: Int32, value: Uint32, *, loc=None, ip=None):
    """Kept as asm: side-effecting, so it keeps its order against the
    other asm shared accesses."""
    _ordered("st.shared.u32 [$0], $1;", "r,r", (Int32(address), Uint32(value)), loc, ip)


@dsl_user_op
def load_shared_u32(address: Int32, *, loc=None, ip=None):
    """Kept as asm: side-effecting, so it keeps its order against the
    other asm shared accesses."""
    return Uint32(
        _ordered("ld.shared.u32 $0, [$1];", "=r,r", (Int32(address),), loc, ip, T.i32())
    )


@dsl_user_op
def order_key(value: Float32, valid: Int32, *, loc=None, ip=None):
    """The FP32 bits mapped to uint32 preserving order (+0.0 above -0.0, as
    FlashInfer's top-k ranks them); 0, at or below every valid key, if not
    `valid`.
    Kept as asm: moved unchanged from the shard top-k; a `cutlass.select_`
    form is not checked for SASS identity."""
    return Uint32(
        _pure(
            "{\n\t.reg .pred pn, pv;\n\t.reg .b32 b, n;\n\t"
            "mov.b32 b, $1;\n\tsetp.lt.s32 pn, b, 0;\n\tnot.b32 n, b;\n\t"
            "or.b32 b, b, -2147483648;\n\tselp.b32 $0, n, b, pn;\n\t"
            "setp.ne.s32 pv, $2, 0;\n\t@!pv mov.b32 $0, 0;\n\t}",
            "=r,f,r",
            (Float32(value), Int32(valid)),
            loc,
            ip,
            T.i32(),
        )
    )


@dsl_user_op
def select_max(a: Uint32, b: Uint32, take_max: Int32, *, loc=None, ip=None):
    """max(a, b) if `take_max` else min(a, b).
    Kept as asm: moved unchanged from the shard top-k; a `cutlass.select_`
    form is not checked for SASS identity."""
    return Uint32(
        _pure(
            "{\n\t.reg .pred p;\n\t.reg .b32 hi, lo;\n\t"
            "max.u32 hi, $1, $2;\n\tmin.u32 lo, $1, $2;\n\t"
            "setp.ne.s32 p, $3, 0;\n\tselp.b32 $0, hi, lo, p;\n\t}",
            "=r,r,r,r",
            (Uint32(a), Uint32(b), Int32(take_max)),
            loc,
            ip,
            T.i32(),
        )
    )


@dsl_user_op
def store_u32x2_pred(
    address: Int64, a: Uint32, b: Uint32, predicate: Int32, *, loc=None, ip=None
):
    """Kept as asm: `cute.arch.store` has no predicated form."""
    _ordered(
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $3, 0;\n\t"
        "@p st.global.v2.u32 [$0], {$1, $2};\n\t}",
        "l,r,r,r",
        (Int64(address), Uint32(a), Uint32(b), Int32(predicate)),
        loc,
        ip,
    )


@dsl_user_op
def store_shared_u32x2_pred(
    address: Int32, a: Uint32, b: Uint32, predicate: Int32, *, loc=None, ip=None
):
    """Kept as asm: `cute.arch.store` has no predicated form."""
    _ordered(
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $3, 0;\n\t"
        "@p st.shared.v2.u32 [$0], {$1, $2};\n\t}",
        "r,r,r,r",
        (Int32(address), Uint32(a), Uint32(b), Int32(predicate)),
        loc,
        ip,
    )


@dsl_user_op
def store_shared_u32x4_pred(address: Int32, w, predicate: Int32, *, loc=None, ip=None):
    """Kept as asm: `cute.arch.store` has no predicated form."""
    _ordered(
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $5, 0;\n\t"
        "@p st.shared.v4.u32 [$0], {$1, $2, $3, $4};\n\t}",
        "r,r,r,r,r,r",
        (Int32(address), *[Uint32(v) for v in w], Int32(predicate)),
        loc,
        ip,
    )


@dsl_user_op
def load_shared_u32x2(address: Int32, *, loc=None, ip=None) -> tuple:
    """Kept as asm: side-effecting, so it keeps its order against the
    other asm shared accesses."""
    struct = _ordered(
        "ld.shared.v2.u32 {$0, $1}, [$2];",
        "=r,=r,r",
        (Int32(address),),
        loc,
        ip,
        llvm.StructType.get_literal([T.i32()] * 2),
    )
    return _unpack(struct, Uint32, 2, loc=loc, ip=ip)


@dsl_user_op
def load_cg_u32x4(address: Int64, predicate: Int32, *, loc=None, ip=None) -> tuple:
    """Four words cached in L2 only (another CTA's stores); zeros if not
    `predicate`.
    Kept as asm: `cute.arch.load` has no predicated, zero-filled form."""
    struct = _ordered(
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $5, 0;\n\t"
        "mov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\tmov.b32 $2, 0;\n\tmov.b32 $3, 0;\n\t"
        "@p ld.global.cg.v4.u32 {$0, $1, $2, $3}, [$4];\n\t}",
        "=r,=r,=r,=r,l,r",
        (Int64(address), Int32(predicate)),
        loc,
        ip,
        llvm.StructType.get_literal([T.i32()] * 4),
    )
    return _unpack(struct, Uint32, 4, loc=loc, ip=ip)


@dsl_user_op
def sync_warp(*, loc=None, ip=None):
    """`bar.warp.sync` of the full warp (a memory-effecting intrinsic)."""
    nvvm.bar_warp_sync(Int32(-1).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)


@dsl_user_op
def above_u32(a: Uint32, b: Uint32, *, loc=None, ip=None):
    """1 if `a` > `b` (unsigned), else 0.
    Kept as asm: moved unchanged from the shard top-k; a `cutlass.select_`
    form is not checked for SASS identity."""
    return Int32(
        _pure(
            "{\n\t.reg .pred p;\n\tsetp.gt.u32 p, $1, $2;\n\t"
            "selp.s32 $0, 1, 0, p;\n\t}",
            "=r,r,r",
            (Uint32(a), Uint32(b)),
            loc,
            ip,
            T.i32(),
        )
    )
