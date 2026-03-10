# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fused CUTLASS GEMM + static FP8 output quantization."""

import pytest
import torch

from tests.kernels.utils import baseline_scaled_mm, opcheck, to_fp8
from vllm import _custom_ops as ops
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("These tests require CUDA", allow_module_level=True)

FP8_DTYPE = current_platform.fp8_dtype()
OUTPUT_SCALE_VALUE = 1.25
FP8_CLOSE_RTOL = 1e-1
FP8_CLOSE_ATOL = 1e-1


def _make_positive_scale(
    shape: tuple[int, int],
    start: float,
    end: float,
) -> torch.Tensor:
    numel = shape[0] * shape[1]
    return torch.linspace(
        start,
        end,
        steps=numel,
        device="cuda",
        dtype=torch.float32,
    ).reshape(shape)


def _make_scales(
    m: int,
    n: int,
    per_token_a: bool,
    per_out_ch_b: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_shape = (m, 1) if per_token_a else (1, 1)
    b_shape = (1, n) if per_out_ch_b else (1, 1)
    a_scales = _make_positive_scale(a_shape, 0.75, 1.25)
    b_scales = _make_positive_scale(b_shape, 0.9, 1.4)
    return a_scales, b_scales


def _reference_fused_output(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    reference = baseline_scaled_mm(a, b, a_scales, b_scales, torch.float32)
    fp8_info = torch.finfo(FP8_DTYPE)
    return (reference / output_scale).clamp(fp8_info.min, fp8_info.max).to(FP8_DTYPE)


@pytest.mark.parametrize("m,n,k", [(32, 128, 128), (128, 256, 128)])
@pytest.mark.parametrize("per_token_a", [False, True])
@pytest.mark.parametrize("per_out_ch_b", [False, True])
@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="Fused CUTLASS FP8 output quantization requires SM89+",
)
def test_cutlass_scaled_mm_static_fp8_quant(
    m: int,
    n: int,
    k: int,
    per_token_a: bool,
    per_out_ch_b: bool,
) -> None:
    a = to_fp8(torch.randn((m, k), device="cuda") / 4)
    b = to_fp8(torch.randn((n, k), device="cuda").t() / 4)
    a_scales, b_scales = _make_scales(m, n, per_token_a, per_out_ch_b)
    # Keep this non-unit so the output-scale folding path is exercised.
    output_scale = torch.tensor(OUTPUT_SCALE_VALUE, device="cuda",
                                dtype=torch.float32)

    fused = ops.cutlass_scaled_mm_static_fp8_quant(
        a, b, a_scales, b_scales, output_scale
    )
    reference = _reference_fused_output(a, b, a_scales, b_scales, output_scale)

    assert not torch.isnan(fused.to(torch.float32)).any(), "Fused output contains NaN"
    assert not torch.isinf(fused.to(torch.float32)).any(), "Fused output contains Inf"
    assert fused.dtype == FP8_DTYPE
    assert fused.shape == (m, n)

    # Compare in float32 space with tolerances that account for:
    # - scale folding rounding (a_scales * reciprocal(output_scale) vs / output_scale)
    # - float non-associativity in CUTLASS epilogue vs reference multiply order
    # - FP8 cast boundary rounding differences (CUTLASS round_to_nearest vs PyTorch)
    # These tolerances allow ~1 FP8 E4M3 quantum of rounding difference.
    torch.testing.assert_close(
        fused.to(torch.float32),
        reference.to(torch.float32),
        rtol=FP8_CLOSE_RTOL,
        atol=FP8_CLOSE_ATOL,
    )

    out = torch.empty((m, n), device="cuda", dtype=FP8_DTYPE)
    opcheck(
        torch.ops._C.cutlass_scaled_mm_static_fp8_quant,
        (out, a, b, a_scales, b_scales, output_scale, None),
    )
