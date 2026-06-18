# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the software fp8e4m3 <-> {fp16, bf16} Triton conversions.

These back the pre-SM89 fp8 KV cache path of the Triton attention backend:
fp8 <-> bf16 on SM80/86 (``fp8e4nv_sm80``) and fp8 <-> fp16 on SM75
(``fp8e4nv_fp16_sm75``), since bf16 is an SM80+ hardware type. Each helper is a
``pack=4`` inline-asm function driven here through a tiny wrapper kernel.

Oracle (per the test plan):
  * SM75-SM88: compare against a PyTorch reference. The reference SATURATES
    overflow (and +-inf / NaN) to the fp8 representable max (+-448), never NaN --
    matching our kernels, which treat anything past the fp8 range as overflow and
    do not spend cycles distinguishing NaN (NaN must not occur in KV activations).
  * SM89+: the same sampled set, plus a FULL barrage over every one of the 65,536
    fp16/bf16 bit patterns, cross-checked against the native hardware fp8 cast
    (which the saturating reference lowers to once the input is clamped in range).
Decode is exact; RNE encode is exact; truncating encode is within 2 fp8-code ULP.
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
FP8_MAX = 448.0  # largest finite fp8 e4m3fn magnitude

# Encode KIND codes for the wrapper kernel.
KIND_FP16_RNE = 0
KIND_FP16_TRUNC = 1
KIND_BF16_RNE = 2


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
    if KIND == KIND_FP16_RNE:
        y = fp16_to_fp8e4m3(x)
    elif KIND == KIND_FP16_TRUNC:
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


def _assert_code_ulp(actual_u8: torch.Tensor, ref_u8: torch.Tensor, max_ulp: int):
    """Assert |actual - ref| <= ``max_ulp`` in fp8-code space, with matching sign.

    For the truncating encode, which rounds toward zero, the magnitude code is <=
    the saturating-RNE reference code by at most ``max_ulp`` steps.
    """
    a = actual_u8.to(torch.int32)
    r = ref_u8.to(torch.int32)
    sign_a, sign_r = a & 0x80, r & 0x80
    mag_a, mag_r = a & 0x7F, r & 0x7F
    # Sign must match except for the +0/-0 pair (code 0), which is value-fungible.
    sign_ok = (sign_a == sign_r) | ((mag_a == 0) & (mag_r == 0))
    assert bool(sign_ok.all()), "sign mismatch in truncating encode"
    ulp = (mag_a - mag_r).abs()
    assert int(ulp.max()) <= max_ulp, f"max fp8-code ULP {int(ulp.max())} > {max_ulp}"


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
    "kind,dtype,min_cap",
    [
        (KIND_FP16_RNE, torch.float16, 75),
        (KIND_FP16_TRUNC, torch.float16, 75),
        (KIND_BF16_RNE, torch.bfloat16, 80),
    ],
)
def test_encode_sampled_edge_cases(kind: int, dtype: torch.dtype, min_cap: int):
    """Encode over a sampled set incl. edge cases, vs the saturating reference.

    Runs on SM75-SM88 (reference oracle) and SM89+ (reference lowers to native).
    RNE is bit-exact; truncation is within 2 fp8-code ULP.
    """
    if not current_platform.has_device_capability(min_cap):
        pytest.skip(f"requires SM{min_cap}+")
    x = _edge_case_inputs(dtype)
    actual = _run_encode(x, kind)
    ref = _saturating_fp8_ref(x)
    if kind == KIND_FP16_TRUNC:
        _assert_code_ulp(actual, ref, max_ulp=2)
    else:
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
@pytest.mark.parametrize(
    "kind,dtype,exact",
    [
        (KIND_FP16_RNE, torch.float16, True),  # RNE: bit-exact vs native
        (KIND_FP16_TRUNC, torch.float16, False),  # trunc: <=2 fp8-code ULP (closeness)
        (KIND_BF16_RNE, torch.bfloat16, True),  # RNE: bit-exact vs native
    ],
)
def test_encode_full_barrage_matches_native_on_sm89(
    kind: int, dtype: torch.dtype, exact: bool
):
    """On SM89+, encode is cross-checked against the native float -> fp8 cvt over
    EVERY one of the 65,536 input bit patterns (normals, subnormals, signed zeros,
    overflow, +-inf, +-NaN -- all saturating, never NaN). RNE must match the native
    cvt exactly; truncation is checked for CLOSENESS (within 2 fp8-code ULP), since
    round-toward-zero legitimately differs from the native round-to-nearest."""
    x = _all_uint16_as(dtype)
    actual = _run_encode(x, kind)
    native = _saturating_fp8_ref(x)  # native hardware cvt on SM89+ (clamped input)
    if exact:
        torch.testing.assert_close(
            actual.view(FP8_DTYPE).float(),
            native.view(FP8_DTYPE).float(),
            atol=0.0,
            rtol=0.0,
        )
    else:
        _assert_code_ulp(actual, native, max_ulp=2)
