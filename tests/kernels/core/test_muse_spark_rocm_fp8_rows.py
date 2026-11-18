# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark ROCm FP8 rows for the token-sharded all-gathers."""

import pytest
import torch

from vllm.models.muse_spark.amd.ops.fp8_rows import dequant_rows, quant_rows
from vllm.models.muse_spark.amd.ops.fused_norm_expand import fused_rmsnorm_expand
from vllm.models.muse_spark.common.ops.fused_rmsnorm import (
    fused_norm_residual_norm_fwd,
    fused_rmsnorm_fwd,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="requires a ROCm GPU"
)


def assert_fp8_rows_close(
    q: torch.Tensor, scale: torch.Tensor, ref: torch.Tensor
) -> None:
    """Scales are `max|row| / fp8_max` and each value is within half an FP8 step
    of the fp64 `ref`: relative `2^-4` for normal codes, plus a subnormal term."""
    dtype = current_platform.fp8_dtype()
    ref = ref.double()
    expected_scale = ref.abs().amax(dim=1).clamp_min(1e-12) / torch.finfo(dtype).max
    torch.testing.assert_close(scale.double(), expected_scale, rtol=1e-6, atol=0)
    got = dequant_rows(q, scale, torch.float32).double()
    bound = (
        ref.abs() * 2**-4 * (1 + 2**-10)
        + scale.double()[:, None] * torch.finfo(dtype).tiny
    )
    assert ((got - ref).abs() <= bound).all()


@pytest.mark.parametrize("num_tokens,hidden", [(1, 4096), (37, 8192), (513, 4096)])
@torch.inference_mode()
def test_quant_rows_round_trip_is_within_half_an_fp8_step(num_tokens, hidden):
    x = torch.randn(num_tokens, hidden, device="cuda")
    x = (x * torch.logspace(-3, 3, num_tokens, device="cuda")[:, None]).bfloat16()
    q, scale = quant_rows(x)
    assert q.dtype == current_platform.fp8_dtype() and scale.shape == (num_tokens,)
    assert_fp8_rows_close(q, scale, x)


@pytest.mark.parametrize("num_tokens", [1, 7, 300])
@torch.inference_mode()
def test_residual_boundary_fp8_output_matches_its_fp32_output(num_tokens):
    """The boundary kernel's FP8 rows quantize the output it would store in fp32."""
    hidden = 8192
    branch = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    stream = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.float32) * 4
    pre_w = torch.randn(hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    beta = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    args = (branch, stream, None, pre_w, 1e-6, 1e-6)
    kwargs = dict(alpha=1.0, beta=beta, pre_gain_center=1.0)

    ref, ref_stream = fused_norm_residual_norm_fwd(
        *args, **kwargs, out_dtype=torch.float32
    )
    scale = torch.empty(num_tokens, device="cuda", dtype=torch.float32)
    q, new_stream = fused_norm_residual_norm_fwd(
        *args, **kwargs, out_dtype=current_platform.fp8_dtype(), fp8_scale=scale
    )
    torch.testing.assert_close(new_stream, ref_stream, atol=0, rtol=0)
    assert_fp8_rows_close(q, scale, ref)


@torch.inference_mode()
def test_rmsnorm_expand_dequantizes_fp8_rows_on_load():
    num_tokens, top_k, hidden = 9, 8, 4096
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16) * 3
    w = torch.randn(hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    q, scale = quant_rows(x)
    ref = fused_rmsnorm_fwd(
        dequant_rows(q, scale, torch.float32),
        w,
        1e-6,
        gain_center=1.0,
        out_dtype=torch.bfloat16,
    )
    out = fused_rmsnorm_expand(q, w, 1e-6, 1.0, False, top_k, x_scale=scale)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref)
