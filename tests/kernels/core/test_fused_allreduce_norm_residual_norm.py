# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.envs as envs
from tests.kernels.core.test_muse_spark_norms import make_residual_boundary
from vllm.models.common.amd.ops.fused_allreduce_norm_residual_norm import (
    fused_allreduce_norm_residual_norm_partials,
)
from vllm.models.muse_spark.common.ops.fused_rmsnorm import fused_norm_residual_norm_fwd
from vllm.platforms import current_platform

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="the fused boundary is a Triton kernel",
)


def test_fused_ar_boundary_is_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_MUSE_FUSED_AR_BOUNDARY", raising=False)
    assert envs.VLLM_ROCM_MUSE_FUSED_AR_BOUNDARY is False
    monkeypatch.setenv("VLLM_ROCM_MUSE_FUSED_AR_BOUNDARY", "1")
    assert envs.VLLM_ROCM_MUSE_FUSED_AR_BOUNDARY is True


@requires_cuda
@pytest.mark.parametrize("world", [1, 4])
@pytest.mark.parametrize("post_input_scale", [False, True])
def test_partial_sum_matches_the_unfused_boundary(world, post_input_scale):
    """Summing the partials in the boundary matches summing them first."""
    torch.manual_seed(world + int(post_input_scale))
    rows = 3
    boundary, pre_norm = make_residual_boundary(rows, post_input_scale=post_input_scale)
    partials = torch.randn(
        world, rows, boundary.branch.shape[-1], device="cuda", dtype=torch.bfloat16
    )
    post_norm = boundary.post_norm
    connection = boundary.connection
    kwargs = dict(
        post_weight=post_norm.weight if post_norm.has_weight else None,
        pre_weight=pre_norm.weight,
        post_eps=post_norm.variance_epsilon,
        pre_eps=pre_norm.variance_epsilon,
        alpha=connection.alpha,
        beta=connection.beta,
        post_gain_center=post_norm.fused_gain_center,
        pre_gain_center=pre_norm.fused_gain_center,
        post_input_scale=(
            post_norm.applies_weight_before_norm and post_norm.has_weight
        ),
        pre_input_scale=(pre_norm.applies_weight_before_norm and pre_norm.has_weight),
        out_dtype=torch.bfloat16,
    )
    actual_out, actual_stream = fused_allreduce_norm_residual_norm_partials(
        partials, boundary.stream, **kwargs
    )
    reduced = partials.float().sum(0)
    expected_out, expected_stream = fused_norm_residual_norm_fwd(
        reduced, boundary.stream, **kwargs
    )

    torch.testing.assert_close(actual_out, expected_out, rtol=0.015625, atol=0.03125)
    torch.testing.assert_close(actual_stream, expected_stream, rtol=1e-4, atol=1e-4)
