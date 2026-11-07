# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from cutlass import BFloat16, Constexpr, Float32, Uint16, Uint32, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def fp32x2_to_bf16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)


@dsl_user_op
def bf16x2_to_fp32x2(data, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    if isinstance(data, Uint32):
        out = llvm.inline_asm(
            llvm.StructType.get_literal([T.f32(), T.f32()]),
            [data.ir_value(loc=loc, ip=ip)],
            "shl.b32 $0, $2, 16;\n\tand.b32 $1, $2, 0xFFFF0000;",
            "=f,=f,r",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
        return (
            Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip)),
        )

    elif isinstance(data, (cute.Tensor, cute.TensorSSA)):
        # TensorSSA in, TensorSSA out (it can't be recast: stage it in rmem)
        is_ssa = isinstance(data, cute.TensorSSA)
        if is_ssa:
            frag = cute.make_rmem_tensor(data.shape, data.dtype)
            frag.store(data)
            data = frag
        # recast to 32-bit registers if needed
        if data.element_type == BFloat16:
            data = cute.recast_tensor(data, Uint32)
        assert data.element_type == Uint32

        # NOTE: the output is always 1D
        size = cute.size(data.shape)
        out = cute.make_rmem_tensor(size * 2, Float32)
        for i in range(size):
            out[i * 2], out[i * 2 + 1] = bf16x2_to_fp32x2(data[i])
        return out.load() if is_ssa else out

    else:
        raise ValueError(f"Unsupported type {type(data)}")


@dsl_user_op
def fp8x4_to_bf16x4(x: Uint32, *, loc=None, ip=None) -> cute.TensorSSA:
    # there is only fp8->fp16 conversion, hence we need to go
    # round trip through fp16.
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [x.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 x0, x1;\n\t"
        ".reg .b16 t00, t01, t10, t11;\n\t"
        "mov.b32 {x0, x1}, $2;\n\t"
        "cvt.rn.f16x2.e4m3x2 $0, x0;\n\t"
        "cvt.rn.f16x2.e4m3x2 $1, x1;\n\t"
        "mov.b32 {t00, t01}, $0;\n\t"
        "mov.b32 {t10, t11}, $1;\n\t"
        "cvt.rn.bf16.f16 t00, t00;\n\t"
        "cvt.rn.bf16.f16 t01, t01;\n\t"
        "cvt.rn.bf16.f16 t10, t10;\n\t"
        "cvt.rn.bf16.f16 t11, t11;\n\t"
        "mov.b32 $0, {t00, t01};\n\t"
        "mov.b32 $1, {t10, t11};\n\t"
        "}\n",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    vec = vector.from_elements(
        ir.VectorType.get([2], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(2)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 2, Uint32)


@dsl_user_op
def fp32x2_to_fp8x2(a0: Float32, a1: Float32, *, loc=None, ip=None) -> Uint16:
    out = llvm.inline_asm(
        T.i16(),
        [
            a0.ir_value(loc=loc, ip=ip),
            a1.ir_value(loc=loc, ip=ip),
        ],
        "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;",
        "=h,f,f",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint16(out)


@dsl_user_op
def fp8x4_to_fp16x4(x: Uint32, *, loc=None, ip=None) -> cute.TensorSSA:
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [x.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, $2;\n\t"
        "cvt.rn.f16x2.e4m3x2 $0, lo;\n\t"
        "cvt.rn.f16x2.e4m3x2 $1, hi;\n\t"
        "}\n",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([2], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(2)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 2, Uint32)


@dsl_user_op
def nvfp4x8_to_fp16x8(x: Uint32, scale: Uint32, *, loc=None, ip=None) -> cute.TensorSSA:
    # Eight e2m1 codes, low nibble first, times one {scale, scale} f16x2 pair.
    # e2m1 x e4m3 is exact in f16 (<= 6 significant bits, 2^-10..2688).
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [x.ir_value(loc=loc, ip=ip), scale.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b8 x0, x1, x2, x3;\n\t"
        ".reg .b32 h0, h1, h2, h3;\n\t"
        "mov.b32 {x0, x1, x2, x3}, $4;\n\t"
        "cvt.rn.f16x2.e2m1x2 h0, x0;\n\t"
        "cvt.rn.f16x2.e2m1x2 h1, x1;\n\t"
        "cvt.rn.f16x2.e2m1x2 h2, x2;\n\t"
        "cvt.rn.f16x2.e2m1x2 h3, x3;\n\t"
        "mul.rn.f16x2 $0, h0, $5;\n\t"
        "mul.rn.f16x2 $1, h1, $5;\n\t"
        "mul.rn.f16x2 $2, h2, $5;\n\t"
        "mul.rn.f16x2 $3, h3, $5;\n\t"
        "}\n",
        "=r,=r,=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 4, Uint32)


@dsl_user_op
def nvfp4x8_to_e4m3_fp16x8(
    x: Uint32, scale: Uint32, *, loc=None, ip=None
) -> cute.TensorSSA:
    # Eight e2m1 codes, low nibble first, times one {scale, scale} f16x2 pair,
    # rounded to e4m3 (RN, saturating at 448) and returned as f16. The f16
    # product is exact (<= 6 significant bits, 2^-10..2688), so this rounds once,
    # like the Q8KV4 kernel's e2m1 x e4m3 -> e4m3 multiply.
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [x.ir_value(loc=loc, ip=ip), scale.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b8 x0, x1, x2, x3;\n\t"
        ".reg .b16 e0, e1, e2, e3;\n\t"
        ".reg .b32 h0, h1, h2, h3;\n\t"
        "mov.b32 {x0, x1, x2, x3}, $4;\n\t"
        "cvt.rn.f16x2.e2m1x2 h0, x0;\n\t"
        "cvt.rn.f16x2.e2m1x2 h1, x1;\n\t"
        "cvt.rn.f16x2.e2m1x2 h2, x2;\n\t"
        "cvt.rn.f16x2.e2m1x2 h3, x3;\n\t"
        "mul.rn.f16x2 h0, h0, $5;\n\t"
        "mul.rn.f16x2 h1, h1, $5;\n\t"
        "mul.rn.f16x2 h2, h2, $5;\n\t"
        "mul.rn.f16x2 h3, h3, $5;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n\t"
        "cvt.rn.f16x2.e4m3x2 $0, e0;\n\t"
        "cvt.rn.f16x2.e4m3x2 $1, e1;\n\t"
        "cvt.rn.f16x2.e4m3x2 $2, e2;\n\t"
        "cvt.rn.f16x2.e4m3x2 $3, e3;\n\t"
        "}\n",
        "=r,=r,=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 4, Uint32)


@dsl_user_op
def e4m3x4_select_to_fp16x2x4(
    a: Uint32, b: Uint32, sel: Uint32, *, loc=None, ip=None
) -> tuple[Uint32, Uint32, Uint32, Uint32]:
    # Bytes sel[3:0], sel[7:4], sel[11:8] and sel[15:12] of {b, a} (a holds bytes
    # 0-3) as e4m3, each broadcast to an f16x2 pair.
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            sel.ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .b32 picked, h01, h23;\n\t"
        ".reg .b16 lo, hi, s0, s1, s2, s3;\n\t"
        "prmt.b32 picked, $4, $5, $6;\n\t"
        "mov.b32 {lo, hi}, picked;\n\t"
        "cvt.rn.f16x2.e4m3x2 h01, lo;\n\t"
        "cvt.rn.f16x2.e4m3x2 h23, hi;\n\t"
        "mov.b32 {s0, s1}, h01;\n\t"
        "mov.b32 {s2, s3}, h23;\n\t"
        "mov.b32 $0, {s0, s0};\n\t"
        "mov.b32 $1, {s1, s1};\n\t"
        "mov.b32 $2, {s2, s2};\n\t"
        "mov.b32 $3, {s3, s3};\n\t"
        "}\n",
        "=r,=r,=r,=r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def fp32x4_to_fp8x4(
    a0: Float32,
    a1: Float32,
    a2: Float32,
    a3: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    # Pack four FP32 values into one b32 of four e4m3 bytes, byte order
    # {a0, a1, a2, a3} from low to high address.
    out = llvm.inline_asm(
        T.i32(),
        [
            a0.ir_value(loc=loc, ip=ip),
            a1.ir_value(loc=loc, ip=ip),
            a2.ir_value(loc=loc, ip=ip),
            a3.ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .b16 t0, t1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 t0, $2, $1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 t1, $4, $3;\n\t"
        "mov.b32 $0, {t0, t1};\n\t"
        "}\n",
        "=r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)


@dsl_user_op
def fp32x8_to_fp4x8(
    vals: cute.Tensor,
    offset: Constexpr[int],
    *,
    loc=None,
    ip=None,
) -> Uint32:
    # Pack eight scaled FP32 values into four E2M1x2 bytes, returned as one b32.
    assert vals.element_type is Float32
    out = llvm.inline_asm(
        T.i32(),
        [vals[offset + i].ir_value(loc=loc, ip=ip) for i in range(8)],
        "{\n\t"
        ".reg .b8 x0, x1, x2, x3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 x0, $2, $1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 x1, $4, $3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 x2, $6, $5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 x3, $8, $7;\n\t"
        "mov.b32 $0, {x0, x1, x2, x3};\n\t"
        "}\n",
        "=r,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)
