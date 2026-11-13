# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified software fp8e4m3 <-> {fp16, bf16, fp32} conversion for pre-SM89 Triton.

The public Triton helpers dispatch on dtype at compile time. Conversion code
lives in an always-inline CUDA C++ helper linked from portable SM75 LLVM
bitcode. Scalar adapters support Triton layouts with partial packs. Inference assumes
finite activations; pass propagate_nan=True to produce NaNs for NaN inputs.

Contract: encode accepts FP16, BF16, or FP32 and returns uint8 E4M3
bytes; decode accepts uint8 E4M3 bytes and returns the requested floating
dtype. Unsupported floating dtypes are rejected. Encoding uses saturating
round-to-nearest, ties-to-even, clamping overflow and infinities to +/-448.
Decoding all finite E4M3 values is exact. Subnormals and signed zeros are
preserved. NaN handling is disabled by default; NaN inputs then have
unspecified outputs. Enabling propagate_nan guarantees a NaN result but
does not preserve NaN sign or payload. The compile-time policy selects
separate bitcode entry points, so the default has no NaN checking cost.
SM89+ compilation requires FORCE_SOFTWARE_CONVERSION=True; prefer native conversion.
"""

from pathlib import Path

from vllm.triton_utils import tl, triton

_HELPER_PATH = Path(__file__).with_name("fp8e4nv_helper_sm75.bc")
_HELPER_PATH_STR = str(_HELPER_PATH)
FP8E4NV_EXTERN_LIBS = {"fp8e4nv": _HELPER_PATH_STR}


@tl.core.builtin
def _check_software_conversion(FORCE_SOFTWARE_CONVERSION, _semantic=None):
    """Catch accidental software conversion on native-FP8 CUDA targets."""
    arch = _semantic.builder.options.arch
    if arch.startswith("sm"):
        capability = int("".join(c for c in arch[2:] if c.isdigit()))
        if capability >= 89 and not tl.core._unwrap_if_constexpr(
            FORCE_SOFTWARE_CONVERSION
        ):
            raise ValueError(
                f"Triton is compiling for {arch}, which supports native FP8 E4M3 "
                "conversion. Use native conversion, or set "
                "FORCE_SOFTWARE_CONVERSION=True to deliberately use "
                "software conversion."
            )


@tl.core.extern
def _fp16x1_to_fp8e4m3(arg0, propagate_nan=False, _semantic=None):
    """Link the scalar FP16-to-FP8 bitcode conversion."""
    u8 = tl.core.dtype("uint8")
    u16 = tl.core.dtype("uint16")
    return tl.core.extern_elementwise(
        "fp8e4nv",
        _HELPER_PATH_STR,
        [arg0],
        {
            (u16,): (
                "fp16x1_to_fp8e4m3"
                + ("_nan" if tl.core._unwrap_if_constexpr(propagate_nan) else ""),
                u8,
            )
        },
        is_pure=True,
        _semantic=_semantic,
    )


@tl.core.extern
def _bf16x1_to_fp8e4m3(arg0, propagate_nan=False, _semantic=None):
    """Link the scalar BF16-to-FP8 bitcode conversion."""
    u8 = tl.core.dtype("uint8")
    u16 = tl.core.dtype("uint16")
    return tl.core.extern_elementwise(
        "fp8e4nv",
        _HELPER_PATH_STR,
        [arg0],
        {
            (u16,): (
                "bf16x1_to_fp8e4m3"
                + ("_nan" if tl.core._unwrap_if_constexpr(propagate_nan) else ""),
                u8,
            )
        },
        is_pure=True,
        _semantic=_semantic,
    )


@tl.core.extern
def _fp32x1_to_fp8e4m3(arg0, propagate_nan=False, _semantic=None):
    """Link direct scalar FP32-to-FP8 conversion without intermediate rounding."""
    u8 = tl.core.dtype("uint8")
    u32 = tl.core.dtype("uint32")
    return tl.core.extern_elementwise(
        "fp8e4nv",
        _HELPER_PATH_STR,
        [arg0],
        {
            (u32,): (
                "fp32x1_to_fp8e4m3"
                + ("_nan" if tl.core._unwrap_if_constexpr(propagate_nan) else ""),
                u8,
            )
        },
        is_pure=True,
        _semantic=_semantic,
    )


@tl.core.extern
def _fp8e4m3x1_to_float(arg0, dtype, propagate_nan=False, _semantic=None):
    """Link direct scalar decoding to the requested floating-point dtype."""
    dtype = tl.core._unwrap_if_constexpr(dtype)
    name = {tl.float16: "fp16", tl.bfloat16: "bf16", tl.float32: "fp32"}[dtype]
    bits = tl.core.dtype("uint32" if dtype == tl.float32 else "uint16")
    return tl.core.extern_elementwise(
        "fp8e4nv",
        _HELPER_PATH_STR,
        [arg0],
        {
            (tl.core.dtype("uint8"),): (
                f"fp8e4m3x1_to_{name}x1"
                + ("_nan" if tl.core._unwrap_if_constexpr(propagate_nan) else ""),
                bits,
            )
        },
        is_pure=True,
        _semantic=_semantic,
    )


@tl.core.extern
def _fp8e4m3x4_to_fp16x4(arg0, propagate_nan=False, _semantic=None):
    """Link packed conversion of four FP8 values to FP16."""
    u32 = tl.core.dtype("uint32")
    u64 = tl.core.dtype("uint64")
    return tl.core.extern_elementwise(
        "fp8e4nv",
        _HELPER_PATH_STR,
        [arg0],
        {
            (u32,): (
                "fp8e4m3x4_to_fp16x4"
                + ("_nan" if tl.core._unwrap_if_constexpr(propagate_nan) else ""),
                u64,
            )
        },
        is_pure=True,
        _semantic=_semantic,
    )


@tl.core.extern
def _fp8e4m3x4_to_bf16x4(arg0, propagate_nan=False, _semantic=None):
    """Link packed conversion of four FP8 values to BF16."""
    u32 = tl.core.dtype("uint32")
    u64 = tl.core.dtype("uint64")
    return tl.core.extern_elementwise(
        "fp8e4nv",
        _HELPER_PATH_STR,
        [arg0],
        {
            (u32,): (
                "fp8e4m3x4_to_bf16x4"
                + ("_nan" if tl.core._unwrap_if_constexpr(propagate_nan) else ""),
                u64,
            )
        },
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _pack_fp8x4(x0, x1, x2, x3):
    """Pack four FP8 bytes into one unsigned 32-bit value."""
    return (
        x0.to(tl.uint32)
        | (x1.to(tl.uint32) << 8)
        | (x2.to(tl.uint32) << 16)
        | (x3.to(tl.uint32) << 24)
    )


@triton.jit
def _decode_fp16_pack4(x0, x1, x2, x3, propagate_nan: tl.constexpr = False):
    """Decode a four-byte FP8 pack into four FP16 values."""
    decoded = _fp8e4m3x4_to_fp16x4(_pack_fp8x4(x0, x1, x2, x3), propagate_nan)
    return (
        (decoded & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True),
        ((decoded >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True),
        ((decoded >> 32) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True),
        (decoded >> 48).to(tl.uint16).to(tl.float16, bitcast=True),
    )


@triton.jit
def _decode_bf16_pack4(x0, x1, x2, x3, propagate_nan: tl.constexpr = False):
    """Decode a four-byte FP8 pack into four BF16 values."""
    decoded = _fp8e4m3x4_to_bf16x4(_pack_fp8x4(x0, x1, x2, x3), propagate_nan)
    return (
        (decoded & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True),
        ((decoded >> 16) & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True),
        ((decoded >> 32) & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True),
        (decoded >> 48).to(tl.uint16).to(tl.bfloat16, bitcast=True),
    )


@tl.core.extern
def _fp16x4_to_fp8e4m3x4(arg0, arg1, propagate_nan=False, _semantic=None):
    """Encode four FP16 values from two packed words into four FP8 bytes."""
    u32 = tl.core.dtype("uint32")
    return tl.core.extern_elementwise(
        "fp8e4nv",
        _HELPER_PATH_STR,
        [arg0, arg1],
        {
            (u32, u32): (
                "fp16x4_to_fp8e4m3x4"
                + ("_nan" if tl.core._unwrap_if_constexpr(propagate_nan) else ""),
                u32,
            )
        },
        is_pure=True,
        _semantic=_semantic,
    )


@tl.core.extern
def _bf16x4_to_fp8e4m3x4(arg0, arg1, propagate_nan=False, _semantic=None):
    """Encode four BF16 values from two packed words into four FP8 bytes."""
    u32 = tl.core.dtype("uint32")
    return tl.core.extern_elementwise(
        "fp8e4nv",
        _HELPER_PATH_STR,
        [arg0, arg1],
        {
            (u32, u32): (
                "bf16x4_to_fp8e4m3x4"
                + ("_nan" if tl.core._unwrap_if_constexpr(propagate_nan) else ""),
                u32,
            )
        },
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _encode_pack4(x0, x1, x2, x3, propagate_nan: tl.constexpr = False):
    """Encode four FP16/BF16/FP32 values into four E4M3 bytes."""
    if x0.dtype == tl.float32:
        b0 = x0.to(tl.uint32, bitcast=True)
        b1 = x1.to(tl.uint32, bitcast=True)
        b2 = x2.to(tl.uint32, bitcast=True)
        b3 = x3.to(tl.uint32, bitcast=True)
        # Round-to-odd preserves FP32-to-FP8 RNE without double rounding.
        b0 = (b0 >> 16) | ((b0 & 0xFFFF) != 0).to(tl.uint32)
        b1 = (b1 >> 16) | ((b1 & 0xFFFF) != 0).to(tl.uint32)
        b2 = (b2 >> 16) | ((b2 & 0xFFFF) != 0).to(tl.uint32)
        b3 = (b3 >> 16) | ((b3 & 0xFFFF) != 0).to(tl.uint32)
    else:
        b0 = x0.to(tl.uint16, bitcast=True).to(tl.uint32)
        b1 = x1.to(tl.uint16, bitcast=True).to(tl.uint32)
        b2 = x2.to(tl.uint16, bitcast=True).to(tl.uint32)
        b3 = x3.to(tl.uint16, bitcast=True).to(tl.uint32)
    if x0.dtype == tl.float16:
        encoded = _fp16x4_to_fp8e4m3x4(b0 | (b1 << 16), b2 | (b3 << 16), propagate_nan)
    else:
        encoded = _bf16x4_to_fp8e4m3x4(b0 | (b1 << 16), b2 | (b3 << 16), propagate_nan)
    return (
        encoded.to(tl.uint8),
        (encoded >> 8).to(tl.uint8),
        (encoded >> 16).to(tl.uint8),
        (encoded >> 24).to(tl.uint8),
    )


@triton.jit
def _decode_fp16_pack4_nan(x0, x1, x2, x3):
    """Decode four E4M3 bytes to FP16 with explicit NaN propagation."""
    return _decode_fp16_pack4(x0, x1, x2, x3, True)


@triton.jit
def _decode_bf16_pack4_nan(x0, x1, x2, x3):
    """Decode four E4M3 bytes to BF16 with explicit NaN propagation."""
    return _decode_bf16_pack4(x0, x1, x2, x3, True)


@triton.jit
def _encode_pack4_nan(x0, x1, x2, x3):
    """Encode four floating values with explicit NaN propagation."""
    return _encode_pack4(x0, x1, x2, x3, True)


@triton.jit
def convert_to_fp8e4m3(
    x,
    propagate_nan: tl.constexpr = False,
    FORCE_SOFTWARE_CONVERSION: tl.constexpr = False,
):
    """Encode FP16/BF16/FP32 to uint8 E4M3 bytes using saturating RNE.

    Finite overflow and infinities saturate to +/-448; ties round to even.
    Subnormals and signed zeros are preserved. propagate_nan defaults to
    False, leaving NaN inputs unspecified without NaN-checking overhead.
    True maps NaN inputs to NaN outputs, without a sign/payload guarantee.
    FORCE_SOFTWARE_CONVERSION=False rejects software conversion on SM89+
    to catch accidental use where native conversion is available. Setting
    it to True permits deliberate software-path tests on those targets.
    """
    tl.static_assert(
        (x.dtype == tl.float16) or (x.dtype == tl.bfloat16) or (x.dtype == tl.float32),
        "convert_to_fp8e4m3 expects fp16, bf16, or fp32 input",
    )
    # NaN sign/payload need not be preserved; opt-in handling requires NaN output.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
    _check_software_conversion(FORCE_SOFTWARE_CONVERSION)
    if x.numel >= 4 * tl.extra.cuda.num_threads():
        if propagate_nan:
            return tl.map_elementwise(_encode_pack4_nan, x, pack=4)[0]
        else:
            return tl.map_elementwise(_encode_pack4, x, pack=4)[0]
    elif x.dtype == tl.float32:
        return _fp32x1_to_fp8e4m3(x.to(tl.uint32, bitcast=True), propagate_nan)
    elif x.dtype == tl.float16:
        return _fp16x1_to_fp8e4m3(x.to(tl.uint16, bitcast=True), propagate_nan)
    else:
        return _bf16x1_to_fp8e4m3(x.to(tl.uint16, bitcast=True), propagate_nan)


@triton.jit
def convert_from_fp8e4m3(
    x,
    dtype: tl.constexpr,
    propagate_nan: tl.constexpr = False,
    FORCE_SOFTWARE_CONVERSION: tl.constexpr = False,
):
    """Decode uint8 fp8e4m3 bytes to fp16, bf16, or fp32.

    Every finite E4M3 encoding decodes exactly, including subnormals and
    signed zeros. Scalar adapters accept partial per-thread packs.
    propagate_nan=False leaves NaN inputs unspecified and incurs no NaN
    checking; True guarantees NaN output without sign/payload preservation.
    FORCE_SOFTWARE_CONVERSION defaults to False; set True only to explicitly
    permit software conversion on CUDA targets with native FP8 support.
    """
    tl.static_assert(
        (dtype == tl.float16) or (dtype == tl.bfloat16) or (dtype == tl.float32),
        "convert_from_fp8e4m3 expects fp16 or bf16, or fp32 output",
    )
    # NaN sign/payload need not be preserved; opt-in handling requires NaN output.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
    _check_software_conversion(FORCE_SOFTWARE_CONVERSION)
    # Keep packed decoding when each thread has at least one complete pack.
    if dtype != tl.float32 and x.numel >= 4 * tl.extra.cuda.num_threads():
        if dtype == tl.float16:
            if propagate_nan:
                return tl.map_elementwise(_decode_fp16_pack4_nan, x, pack=4)[0]
            else:
                return tl.map_elementwise(_decode_fp16_pack4, x, pack=4)[0]
        else:
            if propagate_nan:
                return tl.map_elementwise(_decode_bf16_pack4_nan, x, pack=4)[0]
            else:
                return tl.map_elementwise(_decode_bf16_pack4, x, pack=4)[0]
    else:
        return _fp8e4m3x1_to_float(x, dtype, propagate_nan).to(dtype, bitcast=True)
