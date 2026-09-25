# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified software fp8e4m3 <-> {fp16, bf16} conversion for pre-SM89 Triton.

Single merged entry points dispatching on dtype at Triton compile time. The
encode REQUIRES an fp16/bf16 input -- asserted, never silently cast -- so the
encoder always matches the data and a wider accumulator (e.g. an fp32 scaled
tile) can never slip through unnoticed; the caller hands in the activation
representation. fp16 serves SM75 (no bf16 hardware type); bf16 serves SM80-88;
SM89+ uses the native cvt and skips this path. Encode is round-to-nearest-even
(bit-exact vs torch.float8_e4m3fn); decode is exact (prmt byte-LUT). The PTX
literals live in fp8e4nv_fp16.py / fp8e4nv_bf16.py.
"""

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.fp8e4nv_bf16 import (
    _BF16_TO_FP8E4M3_RNE_ASM,
    _FP8E4M3_TO_BF16_ASM,
)
from vllm.v1.attention.ops.fp8e4nv_fp16 import (
    _FP8E4M3_TO_FP16_ASM,
    _FP16_TO_FP8E4M3_RNE_ASM,
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
    if x.dtype == tl.float16:
        return tl.inline_asm_elementwise(
            _FP16_TO_FP8E4M3_RNE_ASM,
            "=r,r,r",
            [x],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )
    else:
        return tl.inline_asm_elementwise(
            _BF16_TO_FP8E4M3_RNE_ASM,
            "=r,r,r",
            [x],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )


@triton.jit
def convert_from_fp8e4m3(x, dtype: tl.constexpr):
    """Decode 4 packed uint8 fp8e4m3 bytes -> dtype (tl.float16 or tl.bfloat16)."""
    if dtype == tl.float16:
        return tl.inline_asm_elementwise(
            _FP8E4M3_TO_FP16_ASM,
            "=r,=r,r",
            [x],
            dtype=tl.float16,
            is_pure=True,
            pack=4,
        )
    else:
        # Explicit else is required: the two branches return different dtypes
        # (fp16 vs bf16), and Triton type-checks a bare trailing return even when
        # an earlier branch already returned -- rejecting it as "inconsistent
        # return types". Mutually-exclusive if/else gives one type per specialization.
        return tl.inline_asm_elementwise(
            _FP8E4M3_TO_BF16_ASM,
            "=r,=r,r",
            [x],
            dtype=tl.bfloat16,
            is_pure=True,
            pack=4,
        )
