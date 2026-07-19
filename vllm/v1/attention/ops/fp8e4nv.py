# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified software fp8e4m3 <-> {fp16, bf16} conversion for pre-SM89 Triton.

The public Triton helpers dispatch on dtype at compile time. Conversion code
lives in an always-inline CUDA C++ helper linked from portable SM75 LLVM
bitcode. Scalar encode adapters preserve Triton's partial pack-4 lowering;
decode maps complete packs directly.
"""

from pathlib import Path

from vllm.triton_utils import tl, triton

_HELPER_PATH = Path(__file__).with_name("fp8e4nv_helper_sm75.bc")
_HELPER_PATH_STR = str(_HELPER_PATH)
FP8E4NV_EXTERN_LIBS = {"fp8e4nv": _HELPER_PATH_STR}


@tl.core.extern
def _fp16x1_to_fp8e4m3(arg0, _semantic=None):
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
def _fp8e4m3x4_to_fp16x4(arg0, _semantic=None):
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
    return (
        x0.to(tl.uint32)
        | (x1.to(tl.uint32) << 8)
        | (x2.to(tl.uint32) << 16)
        | (x3.to(tl.uint32) << 24)
    )


@triton.jit
def _decode_fp16_pack4(x0, x1, x2, x3):
    decoded = _fp8e4m3x4_to_fp16x4(_pack_fp8x4(x0, x1, x2, x3))
    return (
        (decoded & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True),
        ((decoded >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True),
        ((decoded >> 32) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True),
        (decoded >> 48).to(tl.uint16).to(tl.float16, bitcast=True),
    )


@triton.jit
def _decode_bf16_pack4(x0, x1, x2, x3):
    decoded = _fp8e4m3x4_to_bf16x4(_pack_fp8x4(x0, x1, x2, x3))
    return (
        (decoded & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True),
        ((decoded >> 16) & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True),
        ((decoded >> 32) & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True),
        (decoded >> 48).to(tl.uint16).to(tl.bfloat16, bitcast=True),
    )


@triton.jit
def convert_to_fp8e4m3(x):
    """Encode fp16/bf16 -> 4 packed uint8 fp8e4m3 bytes (pack-4, saturating RNE).

    x MUST already be fp16 or bf16 (the activation dtype); this is asserted, not
    cast, so the caller owns handing in the right 16-bit representation.
    """
    tl.static_assert(
        (x.dtype == tl.float16) or (x.dtype == tl.bfloat16),
        "convert_to_fp8e4m3 expects fp16 or bf16 input",
    )
    bits = x.to(tl.uint16, bitcast=True)
    if x.dtype == tl.float16:
        return _fp16x1_to_fp8e4m3(bits)
    return _bf16x1_to_fp8e4m3(bits)


@triton.jit
def convert_from_fp8e4m3(x, dtype: tl.constexpr):
    """Decode packed uint8 fp8e4m3 bytes to fp16 or bf16.

    The input layout must assign a multiple of four elements to each thread.
    """
    if dtype == tl.float16:
        return tl.map_elementwise(_decode_fp16_pack4, x, pack=4)[0]
    else:
        return tl.map_elementwise(_decode_bf16_pack4, x, pack=4)[0]
