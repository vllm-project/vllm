# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for fp8e4m3 <-> {fp16, bf16, fp32} conversion.

The unified ``convert_to_fp8e4m3`` / ``convert_from_fp8e4m3`` helpers dispatch
on dtype; encode is round-to-nearest-even, and decode is exact.

Oracle (per the test plan):
  * SM75-SM88: compare against a PyTorch reference. The reference SATURATES
    overflow (and +-inf) to the fp8 representable max (+-448), preserving
    signed E4M3 NaNs only with propagate_nan enabled.
  * SM89+: the same sampled set, plus a FULL barrage over every one of the 65,536
    fp16/bf16 bit patterns, cross-checked against the native hardware fp8 cast
    (which the saturating reference lowers to once the input is clamped in range).
Decode is exact; the RNE encode is bit-exact vs the saturating reference.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

if not (current_platform.is_cuda() and HAS_TRITON):
    pytest.skip(
        "fp8e4nv software conversions require CUDA with Triton",
        allow_module_level=True,
    )

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.fp8e4nv import (
    FP8E4NV_EXTERN_LIBS,
    convert_from_fp8e4m3,
    convert_to_fp8e4m3,
)
from vllm.v1.attention.ops.fp8e4nv_ptx import (
    convert_from_fp8e4m3 as ptx_convert_from,
)
from vllm.v1.attention.ops.fp8e4nv_ptx import (
    convert_to_fp8e4m3 as ptx_convert_to,
)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0  # largest finite fp8 e4m3fn magnitude


@triton.jit
def _decode_kernel(
    x_ptr,
    out_ptr,
    n,
    IS_FP16: tl.constexpr,
    IS_FP32: tl.constexpr,
    BLOCK: tl.constexpr,
    propagate_nan: tl.constexpr = False,
    FORCE_SOFTWARE_CONVERSION: tl.constexpr = True,
):
    """Decode FP8 test inputs into the selected floating-point dtype."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    dt = tl.float16 if IS_FP16 else tl.float32 if IS_FP32 else tl.bfloat16
    tl.store(
        out_ptr + offs,
        convert_from_fp8e4m3(x, dt, propagate_nan, FORCE_SOFTWARE_CONVERSION),
        mask=mask,
    )


@triton.jit
def _encode_kernel(
    x_ptr,
    out_ptr,
    n,
    BLOCK: tl.constexpr,
    propagate_nan: tl.constexpr = False,
    FORCE_SOFTWARE_CONVERSION: tl.constexpr = True,
):
    """Encode floating-point test inputs into FP8 bytes."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(
        out_ptr + offs,
        convert_to_fp8e4m3(x, propagate_nan, FORCE_SOFTWARE_CONVERSION),
        mask=mask,
    )


def _finite_fp8_bytes() -> torch.Tensor:
    """All 254 finite fp8e4m3 bytes (excludes the two NaN encodings 0x7f/0xff)."""
    vals = [b for b in range(256) if (b & 0x7F) != 0x7F]
    return torch.tensor(vals, dtype=torch.uint8, device="cuda")


def _run_decode(
    x_u8: torch.Tensor, dtype: torch.dtype, propagate_nan: bool = False
) -> torch.Tensor:
    """Launch software decoding and return its output tensor."""
    out = torch.empty(x_u8.numel(), dtype=dtype, device="cuda")
    n = x_u8.numel()
    _decode_kernel[(triton.cdiv(n, 256),)](
        x_u8,
        out,
        n,
        propagate_nan=propagate_nan,
        IS_FP16=(dtype == torch.float16),
        IS_FP32=(dtype == torch.float32),
        BLOCK=256,
        num_warps=2,
        extern_libs=FP8E4NV_EXTERN_LIBS,
    )
    return out


def _run_encode(
    x: torch.Tensor, block: int = 256, propagate_nan: bool = False
) -> torch.Tensor:
    """Launch software encoding and return the resulting FP8 bytes."""
    out = torch.empty(x.numel(), dtype=torch.uint8, device="cuda")
    n = x.numel()
    _encode_kernel[(triton.cdiv(n, block),)](
        x,
        out,
        n,
        BLOCK=block,
        num_warps=4,
        propagate_nan=propagate_nan,
        extern_libs=FP8E4NV_EXTERN_LIBS,
    )
    return out


def _saturating_fp8_ref(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference matching our kernels' saturating instruction.

    Overflow -- finite ``|x| > 448`` and ``+-inf`` -- saturates to the fp8
    max (``+-448``), sign preserved. NaN encodes as signed 0x7f/0xff.
    For in-range finite inputs this is
    exactly ``clamp(+-448).to(fp8)`` (which is bit-exact vs our encode over the
    whole finite domain); clamping the input in range before ``.to(fp8)`` makes the
    cast -- torch software emulation on pre-SM89, native hardware cvt on SM89+ --
    produce the saturated 0x7e at overflow instead of a NaN byte. Returns fp8 bytes.

    The sign is read from the raw element-width bit pattern (not ``signbit()``), so
    it matches the kernel's sign-bit OR for every input including NaN.
    """
    neg = x.view(torch.int32 if x.element_size() == 4 else torch.int16) < 0
    mag = x.abs()
    over = mag > FP8_MAX
    mag = torch.where(over, torch.full_like(mag, FP8_MAX), mag)
    sat = torch.where(neg, -mag, mag)
    encoded = sat.to(FP8_DTYPE).view(torch.uint8)
    return torch.where(x.isnan(), 0x7F | (neg.to(torch.uint8) << 7), encoded)


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
        -nan_v,  # NaN encodes as 0x7f/0xff; infinities saturate
        2.0**-9,
        -(2.0**-9),  # fp8 smallest normal-ish
        2.0**-10,
        2.0**-12,  # fp8 subnormal range
    ]
    if dtype == torch.float32:
        # Direct FP32->FP8 differs from a prior FP16/BF16 rounding.
        specials += [37.98868179321289, 91.80638885498047]
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
    [(torch.float16, 75), (torch.bfloat16, 80), (torch.float32, 75)],
)
@pytest.mark.parametrize("propagate_nan", [False, True])
def test_decode_exact_all_bytes(dtype: torch.dtype, min_cap: int, propagate_nan: bool):
    """fp8 -> {fp16, bf16, fp32} is exact for every finite byte (incl. denorms).

    The decode input domain is only 256 bytes, so 'all finite bytes' is already
    exhaustive. The reference is native hardware cvt on SM89+, torch emulation
    below it.
    """
    if not current_platform.has_device_capability(min_cap):
        pytest.skip(f"requires SM{min_cap}+")
    if propagate_nan:
        nan_bytes = torch.tensor([0x7F, 0xFF], dtype=torch.uint8, device="cuda")
        decoded_nan = _run_decode(nan_bytes, dtype, propagate_nan=True)
        # NaN sign and payload are not part of this contract; only NaN output
        # is required.
        assert torch.isnan(decoded_nan).all()
    x_u8 = _finite_fp8_bytes()
    actual = _run_decode(x_u8, dtype, propagate_nan=propagate_nan)
    expected = x_u8.view(FP8_DTYPE).to(dtype)
    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), atol=0.0, rtol=0.0
    )


# --------------------------- encode (write path) ---------------------------
@pytest.mark.parametrize(
    "dtype,min_cap",
    [(torch.float16, 75), (torch.bfloat16, 80), (torch.float32, 75)],
)
@pytest.mark.parametrize("block", [64, 128, 256, 512])
@pytest.mark.parametrize("propagate_nan", [False, True])
def test_encode_sampled_edge_cases(
    dtype: torch.dtype, min_cap: int, block: int, propagate_nan: bool
):
    """RNE encode over a sampled set incl. edge cases, bit-exact vs the saturating
    reference. Runs on SM75-SM88 (reference oracle) and SM89+ (reference lowers to
    native)."""
    if not current_platform.has_device_capability(min_cap):
        pytest.skip(f"requires SM{min_cap}+")
    x = _edge_case_inputs(dtype)
    if not propagate_nan:
        x = x[~x.isnan()]
    actual = _run_encode(x, block, propagate_nan)
    ref = _saturating_fp8_ref(x)
    torch.testing.assert_close(
        actual.view(torch.uint8),
        ref.view(torch.uint8),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "value,intermediate,direct_byte,rounded_byte,min_cap",
    [
        (37.98868179321289, torch.float16, 0x61, 0x62, 75),
        (91.80638885498047, torch.bfloat16, 0x6B, 0x6C, 80),
    ],
)
def test_encode_fp32_avoids_16bit_double_rounding(
    value: float,
    intermediate: torch.dtype,
    direct_byte: int,
    rounded_byte: int,
    min_cap: int,
):
    """Verify direct FP32 encoding avoids rounding through FP16 or BF16."""
    if not current_platform.has_device_capability(min_cap):
        pytest.skip(f"requires SM{min_cap}+")
    x = torch.tensor([value], dtype=torch.float32, device="cuda")
    assert _run_encode(x).item() == direct_byte
    assert _run_encode(x.to(intermediate)).item() == rounded_byte


# ----------------- SM89+ exhaustive encode cross-check ---------------------
@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="native fp8e4nv cast cross-check requires SM89+",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block", [256, 512])
@pytest.mark.parametrize("propagate_nan", [False, True])
def test_encode_full_barrage_matches_native_on_sm89(
    dtype: torch.dtype, block: int, propagate_nan: bool
):
    """On SM89+, the RNE encode is cross-checked against the native float -> fp8 cvt
    over EVERY one of the 65,536 input bit patterns (normals, subnormals, signed
    zeros, overflow, +-inf, +-NaN -- overflow saturates, NaNs remain NaNs
    when handling is enabled)."""
    x = _all_uint16_as(dtype)
    if not propagate_nan:
        x = x[~x.isnan()]
    actual = _run_encode(x, block, propagate_nan)
    native = _saturating_fp8_ref(x)  # native hardware cvt on SM89+ (clamped input)
    torch.testing.assert_close(
        actual.view(torch.uint8),
        native.view(torch.uint8),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block", [64, 128, 256, 512])
@pytest.mark.parametrize("propagate_nan", [False, True])
def test_decode_partial_per_thread_packs(dtype, block, propagate_nan):
    if dtype == torch.bfloat16 and not current_platform.has_device_capability(80):
        pytest.skip("BF16 requires SM80+")
    raw = torch.arange(block, dtype=torch.int32, device="cuda").to(torch.uint8)
    if not propagate_nan:
        raw = torch.where((raw & 0x7F) == 0x7F, 0, raw)
    out = torch.empty(block, dtype=dtype, device="cuda")
    _decode_kernel[(1,)](
        raw,
        out,
        block,
        IS_FP16=(dtype == torch.float16),
        IS_FP32=False,
        BLOCK=block,
        propagate_nan=propagate_nan,
        num_warps=4,
        extern_libs=FP8E4NV_EXTERN_LIBS,
    )
    expected = raw.view(FP8_DTYPE).to(dtype)
    torch.testing.assert_close(out, expected, atol=0, rtol=0, equal_nan=True)
    finite = ~torch.isnan(expected)
    torch.testing.assert_close(
        out[finite].view(torch.uint8),
        expected[finite].view(torch.uint8),
        atol=0,
        rtol=0,
    )


def test_packaged_bitcode_matches_cuda_source(tmp_path):
    """Verify packaged LLVM bitcode matches a rebuild of its CUDA source."""
    compiler = shutil.which("clang++-18")
    if compiler is None:
        pytest.skip("bitcode regeneration requires the documented clang++-18 compiler")
    assert compiler is not None
    helper = Path(FP8E4NV_EXTERN_LIBS["fp8e4nv"])
    rebuilt = tmp_path / helper.name
    subprocess.run(
        [
            compiler,
            "--cuda-device-only",
            "-nocudainc",
            "-nocudalib",
            "--cuda-gpu-arch=sm_75",
            "-fcuda-flush-denormals-to-zero",
            "-O3",
            "-emit-llvm",
            "-c",
            "fp8e4nv_helper.cu",
            "-o",
            str(rebuilt),
        ],
        cwd=helper.parent,
        check=True,
    )
    assert rebuilt.read_bytes() == helper.read_bytes()


@pytest.mark.parametrize("direction", ["encode", "decode"])
@pytest.mark.parametrize("arch", [75, 80, 89, 90])
@pytest.mark.parametrize("force", [False, True])
def test_software_conversion_requires_explicit_override_on_native_targets(
    direction: str, arch: int, force: bool
):
    """Older targets work by default; native targets diagnose accidental use."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    signature = {
        "x_ptr": "*fp16" if direction == "encode" else "*u8",
        "out_ptr": "*u8" if direction == "encode" else "*fp16",
        "n": "i32",
    }
    constants = {
        "BLOCK": 128,
        "propagate_nan": False,
        "FORCE_SOFTWARE_CONVERSION": force,
    }
    kernel = _encode_kernel if direction == "encode" else _decode_kernel
    if direction == "decode":
        constants.update(IS_FP16=True, IS_FP32=False)
    source = ASTSource(kernel, signature, constants)
    options = {
        "arch": f"sm{arch}",
        "num_warps": 4,
        "extern_libs": FP8E4NV_EXTERN_LIBS,
    }
    if arch >= 89 and not force:
        with pytest.raises(triton.CompilationError) as exc:
            triton.compile(source, target=GPUTarget("cuda", arch, 32), options=options)
        cause = exc.value
        while cause.__cause__ is not None:
            cause = cause.__cause__
        assert isinstance(cause, ValueError)
        assert f"Triton is compiling for sm{arch}" in str(cause)
        assert "Use native conversion" in str(cause)
        assert "FORCE_SOFTWARE_CONVERSION=True" in str(cause)
    else:
        triton.compile(source, target=GPUTarget("cuda", arch, 32), options=options)


@triton.jit
def _policy_kernel(
    src,
    dst,
    n,
    ENCODE: tl.constexpr,
    BF16: tl.constexpr,
    PACK: tl.constexpr,
    propagate_nan: tl.constexpr,
    enable_ftz: tl.constexpr,
):
    """Exercise the direct PTX interface without caller-side packing."""
    offsets = tl.program_id(0) * 512 + tl.arange(0, 512)
    x = tl.load(src + offsets, offsets < n, other=0)
    if ENCODE:
        y = ptx_convert_to(x, PACK, propagate_nan, True, enable_ftz)
    else:
        y = ptx_convert_from(
            x,
            tl.bfloat16 if BF16 else tl.float16,
            PACK,
            propagate_nan,
            True,
            enable_ftz,
        )
    tl.store(dst + offsets, y, offsets < n)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pack", [1, 2, 4])
@pytest.mark.parametrize("encode", [False, True])
@pytest.mark.parametrize("propagate_nan", [False, True])
@pytest.mark.parametrize("enable_ftz", [False, True])
def test_packed_conversion_policies(dtype, pack, encode, propagate_nan, enable_ftz):
    """Check independent policies over every input encoding, including signed zeros."""
    if dtype == torch.bfloat16 and not current_platform.has_device_capability(80):
        pytest.skip("bf16 needs SM80+")
    if encode:
        bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.uint16)
        src = bits.view(dtype)
        expected = _saturating_fp8_ref(src)
        finite = ~src.isnan()
        if enable_ftz:
            expected = torch.where(src.abs() < 2**-6, 0, expected)
    else:
        src = torch.arange(256, device="cuda", dtype=torch.int32).to(torch.uint8)
        expected = src.view(FP8_DTYPE).to(dtype)
        finite = (src & 0x7F) != 0x7F
        if enable_ftz:
            expected = torch.where(
                (src & 0x78) == 0, torch.zeros_like(expected), expected
            )
    actual = torch.empty_like(expected)
    _policy_kernel[(triton.cdiv(src.numel(), 512),)](
        src,
        actual,
        src.numel(),
        encode,
        dtype == torch.bfloat16,
        pack,
        propagate_nan,
        enable_ftz,
        num_warps=4,
    )
    assert torch.equal(
        actual[finite].view(torch.uint8), expected[finite].view(torch.uint8)
    )
    if propagate_nan:
        # PTX requires NaN output, not preservation of NaN sign or payload.
        if encode:
            assert torch.all((actual[~finite] & 0x7F) == 0x7F)
        else:
            assert torch.all(actual[~finite].isnan())
