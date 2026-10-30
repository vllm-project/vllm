# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 fused MoE tail vs the CuTeDSL chain it replaces."""

import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import convert_swizzled_to_linear
from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(100):
    pytest.skip("Requires SM100", allow_module_level=True)

H, TOPK, EPS = 6144, 4, 1e-6


@pytest.mark.parametrize("m", [1, 37, 1024, 8192])
@torch.inference_mode()
def test_moe_finalize_add_correctness(m: int):
    """Bitwise vs moe_unpermute without a shared expert; within rounding with
    one (the chain rounds more often) and with the fused norm + MXFP8 quant."""
    from flashinfer.fused_moe.cute_dsl.moe_utils import moe_unpermute

    from vllm.models.minimax_m3.nvidia.ops.moe_finalize_add import (
        moe_finalize_add,
        moe_finalize_add_norm_mxfp8,
    )
    from vllm.models.minimax_m3.nvidia.ops.norm_mxfp8 import (
        gemma_fused_add_rmsnorm_mxfp8,
    )

    # What the deferred CuTeDSL experts hand over: expanded GEMM2 rows with
    # the route scales unapplied (the tail multiplies them in, as
    # moe_unpermute would) and an identity row map.
    g = torch.Generator(device="cuda").manual_seed(7)
    rows = (torch.randn(m * TOPK, H, device="cuda", generator=g) / 10).bfloat16()
    scales = torch.rand(m, TOPK, device="cuda", generator=g)
    row_map = torch.arange(m * TOPK, device="cuda", dtype=torch.int32).view(m, TOPK)
    shared = (torch.randn(m, H, device="cuda", generator=g) / 10).bfloat16()
    residual = torch.randn(m, H, device="cuda", generator=g).bfloat16()
    weight = (torch.randn(H, device="cuda", generator=g) * 0.2).bfloat16()

    routed = rows.new_empty(m, H)
    moe_unpermute(rows, routed, row_map, scales, m, TOPK, input_is_expanded=True)

    # Finalize + add
    out = moe_finalize_add(rows, scales, row_map, None)
    torch.testing.assert_close(out, routed, atol=0, rtol=0)
    out = moe_finalize_add(rows, scales, row_map, shared)
    torch.testing.assert_close(out, routed + shared, atol=1e-2, rtol=2**-7)

    # Finalize + add + norm + quant
    residual_ref = residual.clone()
    q_ref, sf_ref = gemma_fused_add_rmsnorm_mxfp8(
        routed + shared, residual_ref, weight, EPS
    )
    residual_out, q, sf = moe_finalize_add_norm_mxfp8(
        rows, scales, row_map, shared, residual, weight, EPS
    )
    torch.testing.assert_close(residual_out, residual_ref, atol=1e-2, rtol=2**-7)

    def dequant(q_, sf_):
        e = convert_swizzled_to_linear(sf_, m, H, 32).to(torch.int32)
        scale = torch.exp2(e.float() - 127).unsqueeze(-1)
        return q_.float().view(m, -1, 32) * scale, e

    (deq, e), (deq_ref, e_ref) = dequant(q, sf), dequant(q_ref, sf_ref)
    # Two e4m3 steps (one more if the block scale flips), plus the residual
    # drift relative to the block max.
    amax = 448 * torch.exp2(e_ref.float() - 127).unsqueeze(-1)
    err = (deq - deq_ref).abs()
    assert (err <= 2**-2 * deq_ref.abs() + 2**-6 * amax).all()
    assert err.mean() < 1e-3
    # Block scales may only flip by one at a UE8M0 boundary, and rarely.
    assert (e - e_ref).abs().max() <= 1
    assert (e != e_ref).float().mean() < 1e-3
