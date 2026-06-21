# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the software fp8e4m3 <-> {fp16, bf16} Triton conversions.

These back the pre-SM89 fp8 KV cache path of the Triton attention backend: fp8
<-> bf16 on SM80/86 and fp8 <-> fp16 on SM75 (bf16 is an SM80+ hardware type).
The unified ``convert_to_fp8e4m3`` / ``convert_from_fp8e4m3`` (fp8e4nv.py)
dispatch on dtype; encode is round-to-nearest-even, decode is exact.

Oracle (per the test plan):
  * SM75-SM88: compare against a PyTorch reference. The reference SATURATES
    overflow (and +-inf / NaN) to the fp8 representable max (+-448), never NaN --
    matching our kernels, which treat anything past the fp8 range as overflow and
    do not spend cycles distinguishing NaN (NaN must not occur in KV activations).
  * SM89+: the same sampled set, plus a FULL barrage over every one of the 65,536
    fp16/bf16 bit patterns, cross-checked against the native hardware fp8 cast
    (which the saturating reference lowers to once the input is clamped in range).
Decode is exact; the RNE encode is bit-exact vs the saturating reference.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.fp8e4nv import convert_from_fp8e4m3, convert_to_fp8e4m3

if not current_platform.is_cuda():
    pytest.skip("fp8e4nv software conversions require CUDA", allow_module_level=True)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0  # largest finite fp8 e4m3fn magnitude


@triton.jit
def _decode_kernel(x_ptr, out_ptr, n, IS_FP16: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    dt = tl.float16 if IS_FP16 else tl.bfloat16
    tl.store(out_ptr + offs, convert_from_fp8e4m3(x, dt), mask=mask)


@triton.jit
def _encode_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, convert_to_fp8e4m3(x), mask=mask)


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


def _run_encode(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.numel(), dtype=torch.uint8, device="cuda")
    n = x.numel()
    _encode_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
    return out


def _saturating_fp8_ref(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference matching our kernels' saturating instruction.

    Overflow -- finite ``|x| > 448``, ``+-inf``, and NaN -- saturates to the fp8
    max (``+-448``), sign preserved, never NaN. For in-range finite inputs this is
    exactly ``clamp(+-448).to(fp8)`` (which is bit-exact vs our encode over the
    whole finite domain); clamping the input in range before ``.to(fp8)`` makes the
    cast -- torch software emulation on pre-SM89, native hardware cvt on SM89+ --
    produce the saturated 0x7e at overflow instead of a NaN byte. Returns fp8 bytes.

    The sign is read from the raw 16-bit pattern (not ``signbit()``), so it matches
    the kernel's sign-bit OR for every input including NaN -- ``signbit()`` does not
    recover the sign of a NaN, which would spuriously fail the negative-NaN inputs.
    """
    neg = x.view(torch.int16) < 0
    mag = x.abs()
    over = mag.isnan() | (mag > FP8_MAX)
    mag = torch.where(over, torch.full_like(mag, FP8_MAX), mag)
    sat = torch.where(neg, -mag, mag)
    return sat.to(FP8_DTYPE).view(torch.uint8)


def _edge_case_inputs(dtype: torch.dtype) -> torch.Tensor:
    """A sampled set deliberately including the interesting cases: signed zeros,
    subnormals, the fp8 max, just-over-max and far-over-max (overflow), +-inf, and
    +-NaN -- plus dense normal/subnormal-range random coverage."""
    inf_v = float("inf")
    nan_v = float("nan")
    specials = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        FP8_MAX,
        -FP8_MAX,  # exact fp8 max
        449.0,
        -449.0,  # just over max -> saturate
        1.0e4,
        -1.0e4,  # far over max -> saturate
        inf_v,
        -inf_v,
        nan_v,
        -nan_v,  # non-finite -> saturate, sign preserved
        2.0**-9,
        -(2.0**-9),  # fp8 smallest normal-ish
        2.0**-10,
        2.0**-12,  # fp8 subnormal range
    ]
    sp = torch.tensor(specials, dtype=dtype, device="cuda")
    torch.manual_seed(0)
    normals = torch.linspace(-FP8_MAX * 2, FP8_MAX * 2, 8192, device="cuda")
    subs = (torch.rand(8192, device="cuda") - 0.5) * 4.0e-2  # denormal-range
    return torch.cat([sp, normals.to(dtype), subs.to(dtype)])


def _all_uint16_as(dtype: torch.dtype) -> torch.Tensor:
    """Every one of the 65,536 16-bit patterns, reinterpreted as ``dtype``."""
    pats = torch.arange(0, 65536, dtype=torch.int32, device="cuda").to(torch.int16)
    return pats.view(torch.uint16).view(dtype)


# --------------------------- decode (read path) ----------------------------
@pytest.mark.parametrize(
    "dtype,min_cap",
    [(torch.float16, 75), (torch.bfloat16, 80)],
)
def test_decode_exact_all_bytes(dtype: torch.dtype, min_cap: int):
    """fp8 -> {fp16, bf16} is exact for every finite byte (incl. denorms).

    The decode input domain is only 256 bytes, so 'all finite bytes' is already
    exhaustive. The reference is native hardware cvt on SM89+, torch emulation
    below it.
    """
    if not current_platform.has_device_capability(min_cap):
        pytest.skip(f"requires SM{min_cap}+")
    x_u8 = _finite_fp8_bytes()
    actual = _run_decode(x_u8, dtype)
    expected = x_u8.view(FP8_DTYPE).to(dtype)
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.0, rtol=0.0)


# --------------------------- encode (write path) ---------------------------
@pytest.mark.parametrize(
    "dtype,min_cap",
    [(torch.float16, 75), (torch.bfloat16, 80)],
)
def test_encode_sampled_edge_cases(dtype: torch.dtype, min_cap: int):
    """RNE encode over a sampled set incl. edge cases, bit-exact vs the saturating
    reference. Runs on SM75-SM88 (reference oracle) and SM89+ (reference lowers to
    native)."""
    if not current_platform.has_device_capability(min_cap):
        pytest.skip(f"requires SM{min_cap}+")
    x = _edge_case_inputs(dtype)
    actual = _run_encode(x)
    ref = _saturating_fp8_ref(x)
    torch.testing.assert_close(
        actual.view(FP8_DTYPE).float(),
        ref.view(FP8_DTYPE).float(),
        atol=0.0,
        rtol=0.0,
    )


# ----------------- SM89+ exhaustive cross-check vs native ------------------
@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="native fp8e4nv cast cross-check requires SM89+",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_decode_matches_native_on_sm89(dtype: torch.dtype):
    """On SM89+, the software decode equals the native fp8 -> float cast,
    exhaustively over every finite fp8 byte."""
    x_u8 = _finite_fp8_bytes()
    actual = _run_decode(x_u8, dtype)
    native = x_u8.view(FP8_DTYPE).to(dtype)  # native hardware cvt on SM89+
    torch.testing.assert_close(actual.float(), native.float(), atol=0.0, rtol=0.0)


@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="native fp8e4nv cast cross-check requires SM89+",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_encode_full_barrage_matches_native_on_sm89(dtype: torch.dtype):
    """On SM89+, the RNE encode is cross-checked against the native float -> fp8 cvt
    over EVERY one of the 65,536 input bit patterns (normals, subnormals, signed
    zeros, overflow, +-inf, +-NaN -- all saturating, never NaN)."""
    x = _all_uint16_as(dtype)
    actual = _run_encode(x)
    native = _saturating_fp8_ref(x)  # native hardware cvt on SM89+ (clamped input)
    torch.testing.assert_close(
        actual.view(FP8_DTYPE).float(),
        native.view(FP8_DTYPE).float(),
        atol=0.0,
        rtol=0.0,
    )
