# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified software fp8e4m3 <-> {fp16, bf16, fp32} conversion for pre-SM89 Triton.

The public Triton helpers dispatch on dtype at compile time. Conversion code
lives in an always-inline CUDA C++ helper linked from portable SM75 LLVM
bitcode. Scalar adapters support Triton layouts with partial packs.
"""

from pathlib import Path

from vllm.triton_utils import tl, triton

_HELPER_PATH = Path(__file__).with_name("fp8e4nv_helper_sm75.bc")
_HELPER_PATH_STR = str(_HELPER_PATH)
FP8E4NV_EXTERN_LIBS = {"fp8e4nv": _HELPER_PATH_STR}


@tl.core.extern
def _fp16x1_to_fp8e4m3(arg0, _semantic=None):
    """Link the scalar FP16-to-FP8 bitcode conversion."""
    u8 = tl.core.dtype("uint8")
    u16 = tl.core.dtype("uint16")
    return tl.core.extern_elementwise(
        "fp8e4nv",
        _HELPER_PATH_STR,
        [arg0],
        {(u16,): ("fp16x1_to_fp8e4m3", u8)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl.core.extern
def _bf16x1_to_fp8e4m3(arg0, _semantic=None):
    """Link the scalar BF16-to-FP8 bitcode conversion."""
    u8 = tl.core.dtype("uint8")
    u16 = tl.core.dtype("uint16")
    return tl.core.extern_elementwise(
        "fp8e4nv",
        _HELPER_PATH_STR,
        [arg0],
        {(u16,): ("bf16x1_to_fp8e4m3", u8)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl.core.extern
def _fp32x1_to_fp8e4m3(arg0, _semantic=None):
    """Link direct scalar FP32-to-FP8 conversion without intermediate rounding."""
    u8 = tl.core.dtype("uint8")
    u32 = tl.core.dtype("uint32")
    return tl.core.extern_elementwise(
        "fp8e4nv",
        _HELPER_PATH_STR,
        [arg0],
        {(u32,): ("fp32x1_to_fp8e4m3", u8)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl.core.extern
def _fp8e4m3x1_to_fp32x1(arg0, _semantic=None):
    """Link the scalar FP8-to-FP32 bitcode conversion."""
    u8 = tl.core.dtype("uint8")
    u32 = tl.core.dtype("uint32")
    return tl.core.extern_elementwise(
        "fp8e4nv",
        _HELPER_PATH_STR,
        [arg0],
        {(u8,): ("fp8e4m3x1_to_fp32x1", u32)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl.core.extern
def _fp8e4m3x4_to_fp16x4(arg0, _semantic=None):
    """Link packed conversion of four FP8 values to FP16."""
    u32 = tl.core.dtype("uint32")
    u64 = tl.core.dtype("uint64")
    return tl.core.extern_elementwise(
        "fp8e4nv",
        _HELPER_PATH_STR,
        [arg0],
        {(u32,): ("fp8e4m3x4_to_fp16x4", u64)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl.core.extern
def _fp8e4m3x4_to_bf16x4(arg0, _semantic=None):
    """Link packed conversion of four FP8 values to BF16."""
    u32 = tl.core.dtype("uint32")
    u64 = tl.core.dtype("uint64")
    return tl.core.extern_elementwise(
        "fp8e4nv",
        _HELPER_PATH_STR,
        [arg0],
        {(u32,): ("fp8e4m3x4_to_bf16x4", u64)},
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
def _decode_fp16_pack4(x0, x1, x2, x3):
    """Decode a four-byte FP8 pack into four FP16 values."""
    decoded = _fp8e4m3x4_to_fp16x4(_pack_fp8x4(x0, x1, x2, x3))
    return (
        (decoded & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True),
        ((decoded >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True),
        ((decoded >> 32) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True),
        (decoded >> 48).to(tl.uint16).to(tl.float16, bitcast=True),
    )


@triton.jit
def _decode_bf16_pack4(x0, x1, x2, x3):
    """Decode a four-byte FP8 pack into four BF16 values."""
    decoded = _fp8e4m3x4_to_bf16x4(_pack_fp8x4(x0, x1, x2, x3))
    return (
        (decoded & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True),
        ((decoded >> 16) & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True),
        ((decoded >> 32) & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True),
        (decoded >> 48).to(tl.uint16).to(tl.bfloat16, bitcast=True),
    )


@tl.core.extern
def _fp16x4_to_fp8e4m3x4(arg0, arg1, _semantic=None):
    u32 = tl.core.dtype("uint32")
    return tl.core.extern_elementwise(
        "fp8e4nv",
        _HELPER_PATH_STR,
        [arg0, arg1],
        {(u32, u32): ("fp16x4_to_fp8e4m3x4", u32)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl.core.extern
def _bf16x4_to_fp8e4m3x4(arg0, arg1, _semantic=None):
    u32 = tl.core.dtype("uint32")
    return tl.core.extern_elementwise(
        "fp8e4nv",
        _HELPER_PATH_STR,
        [arg0, arg1],
        {(u32, u32): ("bf16x4_to_fp8e4m3x4", u32)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _encode_pack4(x0, x1, x2, x3):
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
        encoded = _fp16x4_to_fp8e4m3x4(b0 | (b1 << 16), b2 | (b3 << 16))
    else:
        encoded = _bf16x4_to_fp8e4m3x4(b0 | (b1 << 16), b2 | (b3 << 16))
    return (
        encoded.to(tl.uint8),
        (encoded >> 8).to(tl.uint8),
        (encoded >> 16).to(tl.uint8),
        (encoded >> 24).to(tl.uint8),
    )


@triton.jit
def convert_to_fp8e4m3(x):
    """Encode float -> uint8 fp8e4m3 bytes (saturating RNE)."""
    tl.static_assert(
        (x.dtype == tl.float16) or (x.dtype == tl.bfloat16) or (x.dtype == tl.float32),
        "convert_to_fp8e4m3 expects fp16, bf16, or fp32 input",
    )
    if x.numel >= 4 * tl.extra.cuda.num_threads():
        return tl.map_elementwise(_encode_pack4, x, pack=4)[0]
    elif x.dtype == tl.float32:
        return _fp32x1_to_fp8e4m3(x.to(tl.uint32, bitcast=True))
    elif x.dtype == tl.float16:
        return _fp16x1_to_fp8e4m3(x.to(tl.uint16, bitcast=True))
    else:
        return _bf16x1_to_fp8e4m3(x.to(tl.uint16, bitcast=True))


@triton.jit
def convert_from_fp8e4m3(x, dtype: tl.constexpr):
    """Decode uint8 fp8e4m3 bytes to fp16, bf16, or fp32.

    Scalar adapters accept partial per-thread packs.
    """
    tl.static_assert(
        (dtype == tl.float16) or (dtype == tl.bfloat16) or (dtype == tl.float32),
        "convert_from_fp8e4m3 expects fp16 or bf16, or fp32 output",
    )
    # Keep packed decoding when each thread has at least one complete pack.
    if dtype != tl.float32 and x.numel >= 4 * tl.extra.cuda.num_threads():
        if dtype == tl.float16:
            return tl.map_elementwise(_decode_fp16_pack4, x, pack=4)[0]
        else:
            return tl.map_elementwise(_decode_bf16_pack4, x, pack=4)[0]
    else:
        return _fp8e4m3x1_to_fp32x1(x).to(tl.float32, bitcast=True).to(dtype)
