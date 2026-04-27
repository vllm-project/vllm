# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm AITER FP4 and MXFP4 tests.

This file keeps the ROCm-specific FP4 coverage:
- ROCm AITER FP4 env and enablement gates
- large-shape MXFP4 wrapper checks through vLLM's public helper op
- AITER Triton MXFP4 quant format and determinism
- FP4 GEMM, preshuffled-scale, hardware-quant, and skinny decode
  paths

Generic MXFP4 wrapper/reference checks already live in
``tests/kernels/quantization/test_rocm_quark.py``.
"""

import importlib
import warnings

import pytest
import torch

from tests.kernels.utils import _assert_accurate, _assert_deterministic
from tests.quantization.reference_mxfp4 import dq_mxfp4_torch, qdq_mxfp4_torch
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

LLAMA_SHAPES = [
    (64, 4096),
    (64, 11008),
    (32, 14336),
]
GEMM_ATOL = 0.5
GEMM_RTOL = 0.0
GEMM_MAX_VIOLATION_FACTOR = 2.0
A4W4_GEMM_PASS_RATES = {
    (64, 128, 64): 0.9999,
    (128, 256, 128): 0.9999,
    (128, 4096, 4096): 0.9999,
    (256, 4096, 11008): 0.9999,
    (64, 8192, 28672): 0.99765,
}
SKINNY_GEMM_PASS_RATES = {
    (4096, 4096): 0.9999,
    (4096, 11008): 0.99235,
    (8192, 8192): 0.99765,
}
PRESHUFFLED_SHAPES = [
    (64, 4096, 8192),
    (32, 8192, 8192),
]


def _reload_envs():
    import vllm.envs as envs

    return importlib.reload(envs)


def _assert_aiter_supported() -> None:
    from vllm._aiter_ops import is_aiter_found_and_supported

    assert is_aiter_found_and_supported(), (
        "aiter is required on supported ROCm hardware for this test"
    )


def _format_observed_rate(count: int, total: int) -> str:
    return f"{count / total:.4%} ({count}/{total})"


def _format_allowed_rate(rate: float, total: int) -> str:
    allowed_count = int(rate * total)
    return f"{rate:.4%} (<= {allowed_count}/{total})"


def _print_close_stats(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    pass_rate: float = 1.0,
    max_atol: float | None = None,
) -> None:
    abs_diff = (actual - expected).abs().float().flatten()
    expected_abs = expected.abs().float().flatten()
    allowed = atol + rtol * expected_abs
    within = abs_diff <= allowed

    total = abs_diff.numel()
    passed = int(within.sum().item())
    failed = total - passed
    allowed_fail_rate = 1.0 - pass_rate

    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    p99_abs = torch.quantile(abs_diff, 0.99).item()
    p999_abs = torch.quantile(abs_diff, 0.999).item()
    worst_ratio = (abs_diff / allowed.clamp_min(1e-12)).max().item()

    msg = (
        "[rocm_aiter_fp4] "
        f"{label}: "
        f"pass={passed / total:.4%} ({passed}/{total}) "
        f"fail={_format_observed_rate(failed, total)} "
        f"allowed_fail={_format_allowed_rate(allowed_fail_rate, total)} "
        f"atol={atol:g} "
        f"rtol={rtol:g} "
    )
    if max_atol is not None:
        above_max_count = int((abs_diff > max_atol).sum().item())
        msg += (
            f"abs>{max_atol:g}={_format_observed_rate(above_max_count, total)} "
            f"allowed_above_max={_format_allowed_rate(0.0, total)} "
        )
    msg += (
        f"max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g} "
        f"p99_abs={p99_abs:.6g} "
        f"p999_abs={p999_abs:.6g} "
        f"worst_ratio={worst_ratio:.6g}"
    )
    print(msg)
    if failed > 0:
        warnings.warn(msg, stacklevel=2)


# Env and enablement tests ------------------------------------------------


def test_fp4_env_defaults():
    """ROCm FP4 env defaults should stay stable for the AITER gates."""
    import vllm.envs as envs

    assert envs.VLLM_ROCM_USE_AITER_FP4_ASM_GEMM is False
    assert envs.VLLM_ROCM_USE_AITER_FP4BMM is True


@pytest.mark.parametrize(
    (
        "use_aiter",
        "use_fp4_asm_gemm",
        "use_fp4bmm",
        "on_gfx950_value",
        "expected_asm_gemm",
        "expected_fp4bmm",
    ),
    [
        (True, True, True, True, True, True),
        (True, True, True, False, False, False),
        (True, True, False, True, True, False),
        (True, False, True, True, False, True),
        (False, True, True, True, False, False),
    ],
)
def test_rocm_aiter_fp4_enablement_follows_env_and_arch(
    use_aiter,
    use_fp4_asm_gemm,
    use_fp4bmm,
    on_gfx950_value,
    expected_asm_gemm,
    expected_fp4bmm,
    monkeypatch,
):
    """The ROCm FP4 AITER gates should depend only on the env toggles and the
    gfx950 hardware check."""
    import vllm._aiter_ops as aiter_ops
    from vllm._aiter_ops import rocm_aiter_ops

    _assert_aiter_supported()

    with monkeypatch.context() as mp:
        mp.setenv("VLLM_ROCM_USE_AITER", "1" if use_aiter else "0")
        mp.setenv(
            "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM",
            "1" if use_fp4_asm_gemm else "0",
        )
        mp.setenv("VLLM_ROCM_USE_AITER_FP4BMM", "1" if use_fp4bmm else "0")
        mp.setattr("vllm.platforms.rocm.on_gfx950", lambda: on_gfx950_value)
        _reload_envs()
        rocm_aiter_ops.refresh_env_variables()

        assert (
            rocm_aiter_ops.is_asm_fp4_gemm_dynamic_quant_enabled() is expected_asm_gemm
        )
        assert rocm_aiter_ops.is_fp4bmm_enabled() is expected_fp4bmm

    _reload_envs()
    aiter_ops.rocm_aiter_ops.refresh_env_variables()


# Large-shape MXFP4 wrapper tests -----------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", LLAMA_SHAPES)
def test_vllm_quant_dequant_mxfp4_matches_reference_on_large_shapes(shape, dtype):
    """The public vLLM MXFP4 QDQ helper should stay exact against the torch
    reference even on larger Llama-like shapes."""
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        quant_dequant_mxfp4,
    )

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    rows, cols = shape
    x = torch.randn(rows, cols, dtype=dtype)
    out = quant_dequant_mxfp4(x)
    ref = qdq_mxfp4_torch(x, "even")

    assert out.shape == x.shape
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


def test_vllm_quant_dequant_mxfp4_is_deterministic():
    """The public vLLM MXFP4 QDQ helper should stay bitwise deterministic."""
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        quant_dequant_mxfp4,
    )

    torch.set_default_device("cuda")
    torch.manual_seed(3)

    x = torch.randn(64, 128, dtype=torch.bfloat16)

    _assert_deterministic(quant_dequant_mxfp4, x, n_runs=4)


# Triton MXFP4 quant tests ------------------------------------------------


def test_aiter_dynamic_mxfp4_quant_output_format():
    """dynamic_mxfp4_quant returns packed uint8 FP4 values and E8M0 uint8 scales.

    OCP MXFP4: block_size=32, 2 FP4 E2M1 values packed per byte.
    Scale shape: (M, K // 32); one E8M0 exponent byte per 32-element block.
    """
    _assert_aiter_supported()
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    torch.set_default_device("cuda")
    M, K = 64, 256
    x = torch.randn(M, K, dtype=torch.bfloat16)

    x_fp4, x_scale = dynamic_mxfp4_quant(x)

    # FP4 values packed 2-per-byte; shape (M, K // 2)
    assert x_fp4.shape == (M, K // 2), (
        f"Expected fp4 shape ({M}, {K // 2}), got {x_fp4.shape}"
    )
    assert x_fp4.dtype == torch.uint8

    # One E8M0 scale byte per 32-element block; shape (M, K // 32)
    assert x_scale.shape == (M, K // 32), (
        f"Expected scale shape ({M}, {K // 32}), got {x_scale.shape}"
    )
    assert x_scale.dtype == torch.uint8


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (128, 4096),  # Llama-7B hidden
        (256, 11008),  # Llama-7B FFN
        (64, 14336),  # Llama-70B FFN
        (512, 8192),  # Llama-70B hidden
        (32, 28672),  # DeepSeek-style large FFN
    ],
)
def test_aiter_dynamic_mxfp4_quant_llama_shapes(shape, dtype):
    """MXFP4 quantization output format is correct for Llama-class weight shapes.

    Targets B200 parity: NVIDIA nvfp4 tests parametrize over Llama shapes
    (7168, 14336, 28672...). This covers gfx950 MXFP4 quant format validation.
    """
    _assert_aiter_supported()
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    torch.set_default_device("cuda")
    M, K = shape
    x = torch.randn(M, K, dtype=dtype)

    x_fp4, x_scale = dynamic_mxfp4_quant(x)

    assert x_fp4.shape == (M, K // 2)
    assert x_fp4.dtype == torch.uint8
    assert x_scale.shape == (M, K // 32)
    assert x_scale.dtype == torch.uint8
    assert not torch.any(torch.isnan(x_fp4.float()))


def test_aiter_dynamic_mxfp4_quant_determinism():
    """dynamic_mxfp4_quant is bitwise deterministic across repeated runs."""
    _assert_aiter_supported()
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    torch.set_default_device("cuda")
    torch.manual_seed(7)
    x = torch.randn(128, 256, dtype=torch.bfloat16)

    fp4_results = []
    scale_results = []
    for _ in range(4):
        fp4, scale = dynamic_mxfp4_quant(x)
        fp4_results.append(fp4)
        scale_results.append(scale)

    for i in range(1, 4):
        assert torch.equal(fp4_results[0], fp4_results[i]), (
            f"dynamic_mxfp4_quant FP4 output not deterministic on run {i}"
        )
        assert torch.equal(scale_results[0], scale_results[i]), (
            f"dynamic_mxfp4_quant scale not deterministic on run {i}"
        )


# gfx950 hardware FP4 GEMM tests ------------------------------------------


@pytest.mark.skipif(not on_gfx950(), reason="gfx950 ROCm only")
@pytest.mark.parametrize(
    "shape",
    [
        (64, 128, 64),  # small square-ish
        (128, 256, 128),  # medium
        (128, 4096, 4096),  # Llama-7B hidden square
        (256, 4096, 11008),  # Llama-7B FFN
        (64, 8192, 28672),  # Llama-70B FFN
    ],
)
def test_aiter_fp4_gemm_a4w4_accuracy(shape):
    """AITER A4W4 FP4 GEMM output is close to matmul on dequantized weights.

    Tests gfx950 parity with B200 test_nvfp4_scaled_mm.py: E2M1 x E2M1 GEMM
    with block-scaled quantization, multiple Llama-class shapes.
    Requires gfx950 hardware; gracefully skips elsewhere.
    """
    atol = GEMM_ATOL
    rtol = GEMM_RTOL
    pass_rate = A4W4_GEMM_PASS_RATES[shape]
    _assert_aiter_supported()
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        quant_dequant_mxfp4,
    )

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    M, K, N = shape

    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(N, K, dtype=torch.bfloat16)  # weight stored as (N, K)

    # Quantize
    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    B_fp4, B_scale = dynamic_mxfp4_quant(B)

    # FP4 GEMM computes A_fp4 (M, K) @ B_fp4.T (K, N) into (M, N).
    out = gemm_afp4wfp4(A_fp4, B_fp4, A_scale, B_scale)

    assert out.shape == (M, N), f"Expected ({M}, {N}), got {out.shape}"
    assert out.dtype == torch.bfloat16
    assert not torch.any(torch.isnan(out))

    # Reference: matmul on dequantized FP4 inputs (quark roundtrip)
    A_dq = quant_dequant_mxfp4(A)
    B_dq = quant_dequant_mxfp4(B)
    ref = torch.matmul(A_dq.float(), B_dq.t().float())

    # FP4 has only 3 mantissa bits; large K amplifies the accumulation error.
    # These budgets were measured on gfx950 and keep p99/p999 drift visible
    # without pretending the kernel is close to BF16 matmul everywhere.
    _print_close_stats(
        f"a4w4_gemm shape={shape}",
        out.float(),
        ref,
        atol=atol,
        rtol=rtol,
        pass_rate=pass_rate,
        max_atol=GEMM_MAX_VIOLATION_FACTOR * atol,
    )
    _assert_accurate(
        out.float(),
        ref,
        atol=atol,
        rtol=rtol,
        pass_rate=pass_rate,
        max_violation_factor=GEMM_MAX_VIOLATION_FACTOR,
    )


@pytest.mark.skipif(not on_gfx950(), reason="gfx950 ROCm only")
@pytest.mark.parametrize(
    "shape",
    PRESHUFFLED_SHAPES,
)
def test_aiter_fp4_gemm_preshuffled_tuned_shapes(shape):
    """The preshuffled FP4 GEMM path should run on tuned shapes with the same
    packed inputs vLLM uses in production.

    This kernel only advertises a narrow tuned `(N, K)` set on gfx950. The
    important contract here is that those tuned shapes execute with the
    production-style shuffled weights and stay deterministic.
    """
    _assert_aiter_supported()
    from aiter import per_1x32_f4_quant_hip
    from aiter.ops.shuffle import shuffle_weight
    from aiter.ops.triton.gemm_afp4wfp4 import (
        gemm_afp4wfp4_preshuffled_weight_scales,
    )
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from vllm._aiter_ops import rocm_aiter_ops

    torch.set_default_device("cuda")
    torch.manual_seed(1)
    M, K, N = shape

    assert M <= 64
    assert rocm_aiter_ops.is_triton_gemm_afp4wfp4_presh_ws_tuned(N, K)

    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(N, K, dtype=torch.bfloat16)

    B_fp4, B_scale = dynamic_mxfp4_quant(B)
    A_q, A_s = per_1x32_f4_quant_hip(A, shuffle=M >= 32)

    scale_rows, scale_cols = B_scale.shape
    B_scale = (
        B_scale.view(scale_rows // 32, 2, 16, scale_cols // 8, 2, 4, 1)
        .permute(0, 3, 5, 2, 4, 1, 6)
        .contiguous()
        .view(scale_rows, scale_cols)
    )
    B_fp4 = shuffle_weight(B_fp4, layout=(16, 16))

    if M >= 32:
        A_s = A_s.contiguous().view(torch.uint8).reshape(A_s.shape[0] // 32, -1)
    else:
        A_s = A_s[:M, ...].contiguous().view(torch.uint8)
    B_scale = B_scale.contiguous().view(torch.uint8).reshape(B_scale.shape[0] // 32, -1)

    def run_preshuffled() -> torch.Tensor:
        y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        return gemm_afp4wfp4_preshuffled_weight_scales(
            A_q.contiguous().view(torch.uint8),
            B_fp4.contiguous().view(torch.uint8).reshape(B_fp4.shape[0] // 16, -1),
            A_s,
            B_scale,
            torch.bfloat16,
            y,
        )

    out = run_preshuffled()
    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()

    _assert_deterministic(run_preshuffled, n_runs=3)


@pytest.mark.skipif(not on_gfx950(), reason="gfx950 ROCm only")
def test_aiter_fp4_gemm_a4w4_determinism():
    """AITER FP4 A4W4 GEMM is bitwise deterministic across repeated runs.

    Targets parity with B200 FP4 determinism requirements.
    Requires gfx950.
    """
    _assert_aiter_supported()
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    torch.set_default_device("cuda")
    torch.manual_seed(2)

    M, K, N = 128, 256, 128
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(N, K, dtype=torch.bfloat16)

    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    B_fp4, B_scale = dynamic_mxfp4_quant(B)

    def run_gemm():
        return gemm_afp4wfp4(A_fp4, B_fp4, A_scale, B_scale)

    _assert_deterministic(run_gemm, n_runs=4)


# gfx950 hardware FP4 dynamic quantization --------------------------------


@pytest.mark.skipif(not on_gfx950(), reason="gfx950 ROCm only")
@pytest.mark.parametrize(
    "shape",
    [
        (128, 256),
        (256, 4096),
        (64, 14336),
    ],
)
def test_aiter_hardware_fp4_dynamic_quant_format(shape):
    """aiter hardware FP4 dynamic quant produces correct output format.

    Tests gfx950 hardware-accelerated FP4 quantization (OCP MXFP4 E2M1).
    Parity with B200 scaled_fp4_quant: block_size=32, packed uint8 output.
    Requires gfx950.
    """
    _assert_aiter_supported()

    from aiter import dynamic_per_group_scaled_quant_fp4

    torch.set_default_device("cuda")
    M, K = shape
    group_size = 32

    x = torch.randn(M, K, dtype=torch.bfloat16)
    out_fp4 = torch.empty(M, K // 2, dtype=torch.uint8)
    scales = torch.empty(M, K // group_size, dtype=torch.uint8)

    dynamic_per_group_scaled_quant_fp4(out_fp4, x, scales, group_size)

    assert out_fp4.shape == (M, K // 2), (
        f"Shape {shape}: expected fp4 ({M}, {K // 2}), got {out_fp4.shape}"
    )
    assert scales.shape == (M, K // group_size), (
        f"Shape {shape}: expected scale ({M}, {K // group_size}), got {scales.shape}"
    )


@pytest.mark.skipif(not on_gfx950(), reason="gfx950 ROCm only")
def test_aiter_hardware_fp4_quant_vs_triton():
    """Hardware FP4 quant should dequantize back to the same MXFP4 values as
    the Triton path.

    The packed bytes are not identical on gfx950, but with
    ``shuffle_scale=False`` the dequantized results match tightly.
    """
    _assert_aiter_supported()

    from aiter import dynamic_per_group_scaled_quant_fp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    torch.set_default_device("cuda")
    torch.manual_seed(5)

    M, K = 128, 256
    group_size = 32
    x = torch.randn(M, K, dtype=torch.bfloat16)

    # Hardware path
    out_hw = torch.empty(M, K // 2, dtype=torch.uint8)
    scales_hw = torch.empty(M, K // group_size, dtype=torch.uint8)
    dynamic_per_group_scaled_quant_fp4(
        out_hw, x, scales_hw, group_size, shuffle_scale=False
    )

    # Triton path
    out_triton, scales_triton = dynamic_mxfp4_quant(x)

    assert out_hw.shape == out_triton.shape
    assert scales_hw.shape == scales_triton.shape

    dq_hw = dq_mxfp4_torch(out_hw, scales_hw, torch.bfloat16)
    dq_triton = dq_mxfp4_torch(out_triton, scales_triton, torch.bfloat16)

    _print_close_stats(
        "hardware_quant_vs_triton",
        dq_hw.float(),
        dq_triton.float(),
        atol=0.25,
        rtol=0.0,
    )
    torch.testing.assert_close(dq_hw, dq_triton, atol=0.25, rtol=0.0)


# ROCm skinny GEMM FP4 tests ----------------------------------------------


@pytest.mark.skipif(not on_gfx950(), reason="gfx950 ROCm only")
@pytest.mark.parametrize("M", [1, 2, 4, 8])  # decode / skinny batch sizes
@pytest.mark.parametrize("N, K", [(4096, 4096), (4096, 11008), (8192, 8192)])
def test_aiter_fp4_gemm_skinny_shapes(M, N, K):
    """FP4 GEMM accuracy for skinny (small-M) shapes (decode phase).

    Skinny GEMMs (M=1..8) are the bottleneck in decode phase.
    Tests parity with B200 nvfp4_scaled_mm tests at decode batch sizes.
    Requires gfx950.
    """
    atol = GEMM_ATOL
    rtol = GEMM_RTOL
    pass_rate = SKINNY_GEMM_PASS_RATES[(N, K)]
    _assert_aiter_supported()
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        quant_dequant_mxfp4,
    )

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(N, K, dtype=torch.bfloat16)

    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    B_fp4, B_scale = dynamic_mxfp4_quant(B)

    out = gemm_afp4wfp4(A_fp4, B_fp4, A_scale, B_scale)

    assert out.shape == (M, N)
    assert not torch.any(torch.isnan(out))

    A_dq = quant_dequant_mxfp4(A)
    B_dq = quant_dequant_mxfp4(B)
    ref = torch.matmul(A_dq.float(), B_dq.t().float())

    _print_close_stats(
        f"skinny_gemm M={M} N={N} K={K}",
        out.float(),
        ref,
        atol=atol,
        rtol=rtol,
        pass_rate=pass_rate,
        max_atol=GEMM_MAX_VIOLATION_FACTOR * atol,
    )
    _assert_accurate(
        out.float(),
        ref,
        atol=atol,
        rtol=rtol,
        pass_rate=pass_rate,
        max_violation_factor=GEMM_MAX_VIOLATION_FACTOR,
    )
