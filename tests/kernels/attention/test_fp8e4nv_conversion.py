# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the software fp8e4m3 <-> {fp16, bf16} Triton conversions.

These back the pre-SM89 fp8 KV cache path of the Triton attention backend:
fp8 <-> bf16 on SM80/86 (``fp8e4nv_sm80``) and fp8 <-> fp16 on SM75
(``fp8e4nv_fp16_sm75``), since bf16 is an SM80+ hardware type. Each helper is a
``pack=4`` inline-asm function; the tests drive it through a tiny wrapper kernel
on a tensor and compare against ``torch.float8_e4m3fn`` as the reference, and
additionally against a native fp8 cast on SM89+.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.fp8e4nv_fp16_sm75 import (
    fp8e4m3_to_fp16,
    fp16_to_fp8e4m3,
    fp16_to_fp8e4m3_trunc,
)
from vllm.v1.attention.ops.fp8e4nv_sm80 import bf16_to_fp8e4m3, fp8e4m3_to_bf16

if not current_platform.is_cuda():
    pytest.skip("fp8e4nv software conversions require CUDA", allow_module_level=True)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


@triton.jit
def _decode_kernel(x_ptr, out_ptr, n, IS_FP16: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    y = fp8e4m3_to_fp16(x) if IS_FP16 else fp8e4m3_to_bf16(x)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _encode_kernel(x_ptr, out_ptr, n, KIND: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if KIND == 0:
        y = fp16_to_fp8e4m3(x)
    elif KIND == 1:
        y = fp16_to_fp8e4m3_trunc(x)
    else:
        y = bf16_to_fp8e4m3(x)
    tl.store(out_ptr + offs, y, mask=mask)


def _finite_fp8_bytes() -> torch.Tensor:
    """All 254 finite fp8e4m3 bytes (excludes the two NaN encodings 0x7f/0xff)."""
    vals = [b for b in range(256) if (b & 0x7F) != 0x7F]
    return torch.tensor(vals, dtype=torch.uint8, device="cuda")


def _run_decode(x_u8: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    out = torch.empty(x_u8.numel(), dtype=dtype, device="cuda")
    n = x_u8.numel()
    _decode_kernel[(triton.cdiv(n, 256),)](
        x_u8, out, n, IS_FP16=(dtype == torch.float16), BLOCK=256
    )
    return out


def _run_encode(x: torch.Tensor, kind: int) -> torch.Tensor:
    out = torch.empty(x.numel(), dtype=torch.uint8, device="cuda")
    n = x.numel()
    _encode_kernel[(triton.cdiv(n, 256),)](x, out, n, KIND=kind, BLOCK=256)
    return out


@pytest.mark.parametrize(
    "dtype,min_cap",
    [(torch.float16, 75), (torch.bfloat16, 80)],
)
def test_decode_exact_all_bytes(dtype: torch.dtype, min_cap: int):
    """fp8 -> {fp16, bf16} is exact for every finite byte (incl. denorms)."""
    if not current_platform.has_device_capability(min_cap):
        pytest.skip(f"requires SM{min_cap}+")
    x_u8 = _finite_fp8_bytes()
    actual = _run_decode(x_u8, dtype)
    # Reference: reinterpret bytes as fp8e4m3 and cast to the target float.
    expected = x_u8.view(FP8_DTYPE).to(dtype)
    # Bit-exact (signed zero is fungible -> compare values).
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "kind,dtype,min_cap,exact",
    [
        (0, torch.float16, 75, True),  # fp16 RNE (saturating) -> bit-exact
        (1, torch.float16, 75, False),  # fp16 trunc (saturating) -> <= 2 fp8-code ULP
        (2, torch.bfloat16, 80, True),  # bf16 RNE -> bit-exact
    ],
)
def test_encode_matches_reference(
    kind: int, dtype: torch.dtype, min_cap: int, exact: bool
):
    """Encode matches a saturating torch.float8_e4m3fn reference over finite inputs."""
    if not current_platform.has_device_capability(min_cap):
        pytest.skip(f"requires SM{min_cap}+")
    torch.manual_seed(0)
    # Mixed coverage: normals, subnormal-range, and overflow (to exercise saturation).
    x = torch.cat(
        [
            torch.linspace(-FP8_MAX * 2, FP8_MAX * 2, 4096, device="cuda"),
            (torch.rand(4096, device="cuda") - 0.5) * 2e-2,  # denormal-range magnitudes
        ]
    ).to(dtype)
    actual = _run_encode(x, kind).view(FP8_DTYPE)
    # Saturating reference (overflow clamps to +-448, never NaN).
    ref = x.clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    if exact:
        torch.testing.assert_close(actual.float(), ref.float(), atol=0.0, rtol=0.0)
    else:
        # Truncation: within 2 fp8-code ULP -> bounded relative error after decode.
        torch.testing.assert_close(actual.float(), ref.float(), atol=0.0, rtol=0.30)


@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="native fp8e4nv cast cross-check requires SM89+",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_decode_matches_native_on_sm89(dtype: torch.dtype):
    """On SM89+, the software decode must equal the native fp8 -> float cast."""
    x_u8 = _finite_fp8_bytes()
    actual = _run_decode(x_u8, dtype)
    native = x_u8.view(FP8_DTYPE).to(dtype)  # native hardware cvt on SM89+
    torch.testing.assert_close(actual.float(), native.float(), atol=0.0, rtol=0.0)


@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="native fp8e4nv cast cross-check requires SM89+",
)
@pytest.mark.parametrize(
    "kind,dtype",
    [(0, torch.float16), (2, torch.bfloat16)],  # RNE encoders only (trunc is <=2 ULP)
)
def test_encode_matches_native_on_sm89(kind: int, dtype: torch.dtype):
    """On SM89+, the software RNE encode must equal the native float -> fp8 cvt.

    The input is clamped to the fp8 representable range (+-448) BEFORE the cast, so
    the native hardware cvt saturates to fp8 max (0x7e) at overflow -- matching our
    saturating encode -- instead of the NaN a bare native cvt would emit on >448.
    """
    torch.manual_seed(0)
    x = torch.cat(
        [
            torch.linspace(-FP8_MAX * 2, FP8_MAX * 2, 4096, device="cuda"),
            (torch.rand(4096, device="cuda") - 0.5) * 2e-2,  # denormal-range magnitudes
        ]
    ).to(dtype)
    actual = _run_encode(x, kind).view(FP8_DTYPE)
    native = x.clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)  # native hardware cvt on SM89+
    torch.testing.assert_close(actual.float(), native.float(), atol=0.0, rtol=0.0)
