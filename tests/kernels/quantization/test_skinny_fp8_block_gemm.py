# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CuTe DSL skinny block-FP8 GEMM matches the CUTLASS block-FP8 GEMM."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.cute_dsl.skinny_fp8_block_gemm import (
    SkinnyFp8BlockGemmConfig,
    shape_dynamic_skinny_fp8_block_gemm,
)
from vllm.model_executor.kernels.linear.cute_dsl.skinny_fp8_block_gemm_plans import (
    SM103_SKINNY_FP8_BLOCK_GEMM_PLANS,
    get_skinny_fp8_block_gemm_plan,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.has_device_capability(100)
    or not shape_dynamic_skinny_fp8_block_gemm.is_available(),
    reason="skinny block-FP8 GEMM needs SM100+ and the CuTe DSL",
)

PLANNED = [
    (n, k, cfg)
    for (n, k), plan in SM103_SKINNY_FP8_BLOCK_GEMM_PLANS.items()
    for cfg in plan.values()
]


def _operands(m: int, n: int, k: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device="cuda") * 0.05).to(torch.float8_e4m3fn)
    w_scale = torch.rand(n // 128, k // 128, generator=g, device="cuda") * 0.01
    w_scale += 0.001
    x = torch.randn(m, k, generator=g, device="cuda", dtype=torch.bfloat16)
    x[0, 3] = 30.0  # an outlier group
    a, a_scale = per_token_group_quant_fp8(
        x, 128, column_major_scales=True, use_ue8m0=False
    )
    return a, a_scale, w, w_scale


@pytest.mark.parametrize(
    "n,k,config", PLANNED, ids=[f"{n}x{k}-m{c.num_rows}" for n, k, c in PLANNED]
)
@torch.inference_mode()
def test_skinny_fp8_block_gemm_matches_cutlass(
    n: int, k: int, config: SkinnyFp8BlockGemmConfig
):
    m = config.num_rows
    a, a_scale, w, w_scale = _operands(m, n, k, seed=m)
    out = shape_dynamic_skinny_fp8_block_gemm(
        a, a_scale, w, w_scale, config, torch.bfloat16
    )
    ref = ops.cutlass_scaled_mm(
        a, w.T, out_dtype=torch.bfloat16, scale_a=a_scale, scale_b=w_scale.T
    )
    # Exact product of the dequantized operands.
    exact = (a.float() * a_scale.repeat_interleave(128, 1)) @ (
        w.float() * w_scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    ).T
    scale = exact.abs().max()
    err = ((out.float() - exact).abs().max() / scale).item()
    err_ref = ((ref.float() - exact).abs().max() / scale).item()
    # Both round the FP32 sum to BF16 once; the summation order differs.
    assert err <= max(2 * err_ref, 4e-3), (err, err_ref)
    torch.testing.assert_close(out, ref, rtol=2**-7, atol=scale.item() * 2**-8)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 16, 33])
@torch.inference_mode()
def test_small_m_dispatch_uses_skinny_plan(m: int):
    """The small-M dispatch op takes the skinny GEMM for planned token counts
    and keeps its other kernels (bit-identical to CUTLASS) otherwise."""
    import vllm.model_executor.kernels.linear.scaled_mm.cutlass  # noqa: F401

    n, k = 2048, 4096
    a, a_scale, w, w_scale = _operands(m, n, k, seed=m)
    out = torch.ops.vllm.fp8_block_scaled_mm_small_m(a, w, a_scale, w_scale)
    ref = ops.cutlass_scaled_mm(
        a, w.T, out_dtype=torch.bfloat16, scale_a=a_scale, scale_b=w_scale.T
    )
    plan = get_skinny_fp8_block_gemm_plan(n, k)
    if plan is not None and m in plan:
        torch.testing.assert_close(
            out, ref, rtol=2**-7, atol=ref.float().abs().max().item() * 2**-8
        )
    else:
        assert torch.equal(out, ref)
