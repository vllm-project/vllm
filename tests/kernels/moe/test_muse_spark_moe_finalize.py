# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x post-expert finalize kernels: CuTe-DSL and Triton."""

import pytest
import torch

from vllm.models.muse_spark.common.ops.fused_finalize import (
    fused_post_expert_finalize,
)
from vllm.models.muse_spark.nvidia.ops.cute_dsl.moe_finalize_rmsnorm import (
    moe_finalize_with_rmsnorm,
    supports_moe_finalize,
)
from vllm.platforms import current_platform

requires_cutedsl_finalize = pytest.mark.skipif(
    not (
        current_platform.has_device_capability(100)
        and supports_moe_finalize(torch.bfloat16, 4096)
    ),
    reason="Requires the SM100+ CuTe-DSL MoE finalize",
)


def _ref_finalize(rows, norm_weight, weights, eps, weight_first, gain_center=1.0):
    """``sum_k weights * norm(rows[:, k])`` in fp32."""
    x = rows.float()
    gain = 1 if norm_weight is None else norm_weight.float() + gain_center
    if weight_first:
        x = x * gain
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    if not weight_first:
        y = y * gain
    return (y * weights.float()[..., None]).sum(1)


# 2048 is the widest row served by a single warp; wider rows reduce through
# shared memory. MuseSpark1x's compressed MoE hidden dim is 4096.
@requires_cutedsl_finalize
@pytest.mark.parametrize("hidden_dim", [2048, 4096])
@pytest.mark.parametrize("weight_before_norm", [False, True])
def test_masked_slots_never_produce_nan(
    hidden_dim: int, weight_before_norm: bool
) -> None:
    """An EP rank that owns no expert in the batch must contribute exactly zero.

    Its unwritten workspace holds arbitrary bits and ``NaN * 0`` is NaN, so
    zero-weighting the rows is not enough.
    """
    dtype = torch.bfloat16
    topk = 8
    num_tokens, rows_total = 5, 40
    # Uniform bf16 bit patterns: ~0.4% are NaN/Inf, so a row this wide almost
    # surely holds one.
    gemm2_output = (
        torch.randint(
            0, 65536, (rows_total, hidden_dim), device="cuda", dtype=torch.int32
        )
        .to(torch.int16)
        .view(dtype)
    )
    expert_weights = torch.randn((num_tokens, topk), device="cuda", dtype=dtype)
    permuted_idx = torch.full((num_tokens, topk), -1, device="cuda", dtype=torch.int32)
    rms_norm_weight = torch.randn(hidden_dim, device="cuda", dtype=dtype) * 0.1
    output = torch.empty((num_tokens, hidden_dim), device="cuda", dtype=dtype)

    moe_finalize_with_rmsnorm(
        gemm2_output,
        expert_weights,
        permuted_idx,
        rms_norm_weight,
        output,
        topk=topk,
        eps=1e-6,
        weight_before_norm=weight_before_norm,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)


@requires_cutedsl_finalize
@pytest.mark.parametrize("weight_before_norm", [False, True])
def test_fp32_routing_weights_match_bf16(weight_before_norm: bool) -> None:
    """fp32 routing weights are read as-is, not rounded to the activation dtype."""
    dtype = torch.bfloat16
    hidden_dim, topk, rows = 4096, 8, 17
    torch.manual_seed(0)
    gemm2_output = torch.randn(rows * topk, hidden_dim, device="cuda", dtype=dtype)
    weights_fp32 = torch.rand((rows, topk), device="cuda", dtype=torch.float32)
    permuted_idx = torch.arange(rows * topk, device="cuda", dtype=torch.int32).view(
        rows, topk
    )
    rms_norm_weight = torch.randn(hidden_dim, device="cuda", dtype=dtype) * 0.1

    def run(weights: torch.Tensor) -> torch.Tensor:
        out = torch.empty((rows, hidden_dim), device="cuda", dtype=dtype)
        moe_finalize_with_rmsnorm(
            gemm2_output,
            weights,
            permuted_idx,
            rms_norm_weight,
            out,
            topk=topk,
            eps=1e-6,
            weight_before_norm=weight_before_norm,
        )
        return out

    got = run(weights_fp32)
    rounded = run(weights_fp32.to(dtype))
    ref = _ref_finalize(
        gemm2_output.view(rows, topk, hidden_dim),
        rms_norm_weight,
        weights_fp32,
        1e-6,
        weight_before_norm,
        gain_center=0.0,
    )

    err_fp32 = (got.float() - ref).abs().max().item()
    err_rounded = (rounded.float() - ref).abs().max().item()
    # bf16 weights carry ~3 decimal digits, so rounding them is the larger error.
    assert err_fp32 <= err_rounded, f"fp32 {err_fp32} should beat bf16 {err_rounded}"
    torch.testing.assert_close(got, ref.to(dtype), rtol=0.02, atol=0.02)


_SHAPES = [(4096, 8), (8192, 9), (513, 3)]  # tiled, looped, masked-column kernel
_LAYOUTS = ["dense", "expert_map", "permuted"]
_WEIGHTS = [None, "first", "after"]


# A Latin square: every shape/layout, shape/weight and layout/weight pair once.
@pytest.mark.parametrize(
    "tokens,hidden,topk,layout,weight_mode",
    [
        (33, *_SHAPES[i], _LAYOUTS[j], _WEIGHTS[(i + j) % 3])
        for i in range(3)
        for j in range(3)
    ]
    + [(0, 4096, 8, "permuted", "after")],
)
@torch.inference_mode()
def test_triton_finalize_masks_remote_experts(
    tokens, hidden, topk, layout, weight_mode
):
    """Remote route workspaces can be NaN and must never enter normalization."""
    torch.manual_seed(42)
    routed = torch.randn(tokens, topk, hidden, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(tokens, topk, device="cuda")
    ids = torch.arange(topk, device="cuda").expand(tokens, -1).contiguous()
    # Deliberately non-contiguous ownership, so range-based masking is wrong.
    expert_map = torch.arange(topk, device="cuda").roll(2)
    expert_map[1::2] = -1
    if layout == "dense":
        local = torch.ones_like(ids, dtype=torch.bool)
    else:
        local = expert_map[ids] >= 0
        if tokens:
            local[0] = False
            ids[0] = 1
    routed.masked_fill_(~local[..., None], float("nan"))
    norm_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16) / 10
    output = torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    if layout == "permuted":
        idx = torch.arange(tokens * topk, device="cuda", dtype=torch.int32)
        idx = idx.view(tokens, topk).masked_fill(~local, -1)
        rows, kwargs = routed.view(tokens * topk, hidden), {"permuted_idx": idx}
    else:
        mapped = expert_map if layout == "expert_map" else None
        rows, kwargs = routed, {"topk_ids": ids, "expert_map": mapped}
    gain = None if weight_mode is None else norm_weight
    weight_first = weight_mode == "first"

    fused_post_expert_finalize(
        rows, gain, weights, 1e-8, 1.0, weight_first, output, **kwargs
    )

    expected = _ref_finalize(
        routed.masked_fill(~local[..., None], 0), gain, weights, 1e-8, weight_first
    ).to(output.dtype)
    torch.testing.assert_close(output, expected, atol=0.015625, rtol=0.01)
    assert torch.isfinite(output).all()
