# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark's attention all-reduce, folded into the residual boundary."""

import torch

import vllm.envs as envs
from vllm.models.common.amd.ops.fused_allreduce_norm_residual_norm import (
    fused_allreduce_norm_residual_norm,
)
from vllm.models.muse_spark.common.layers import (
    MuseSparkNormPreservingResidualConnection,
    MuseSparkResidualStream,
    MuseSparkRMSNorm,
)
from vllm.models.muse_spark.common.ops.fused_rmsnorm import supports_norm_residual_norm
from vllm.triton_utils import triton

# Decode steps are CUDA-graphed and too small for the host barrier in the
# common op. Prefill and mixed steps are eager and sit above this.
_MIN_ROWS = 1024
_MAX_HIDDEN = 8192


def fuse_allreduce_norm_residual() -> bool:
    """Whether to skip ``o_proj``'s all-reduce and sum the partials here."""
    return envs.VLLM_ROCM_MUSE_FUSED_AR_BOUNDARY


def fused_allreduce_norm_residual_then_norm(
    boundary: MuseSparkResidualStream,
    pre_norm: MuseSparkRMSNorm,
    *,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Sum this rank's o_proj partial with its peers and apply the boundary.

    Returns ``None`` when the fused collective cannot run. The caller
    all-reduces the partial and uses the unfused boundary.
    """
    post_norm, connection = boundary.post_norm, boundary.connection
    branch, stream = boundary.branch, boundary.stream
    if not (
        isinstance(connection, MuseSparkNormPreservingResidualConnection)
        and getattr(connection, "_coefficients_ready", False)
        and post_norm.supports_fused_kernel(branch)
        and pre_norm.supports_fused_kernel(stream)
        and supports_norm_residual_norm(branch, stream)
        and branch.shape[0] >= _MIN_ROWS
        and triton.next_power_of_2(branch.shape[-1]) <= _MAX_HIDDEN
    ):
        return None
    flat_branch = branch.reshape(-1, branch.shape[-1])
    flat_stream = stream.reshape(-1, stream.shape[-1])
    fused = fused_allreduce_norm_residual_norm(
        flat_branch,
        flat_stream,
        post_norm.weight if post_norm.has_weight else None,
        pre_norm.weight,
        post_norm.variance_epsilon,
        pre_norm.variance_epsilon,
        alpha=connection.alpha,
        beta=connection.beta,
        post_gain_center=post_norm.fused_gain_center,
        pre_gain_center=pre_norm.fused_gain_center,
        post_input_scale=(
            post_norm.applies_weight_before_norm and post_norm.has_weight
        ),
        pre_input_scale=(pre_norm.applies_weight_before_norm and pre_norm.has_weight),
        out_dtype=out_dtype,
    )
    if fused is None:
        return None
    out, new_stream = fused
    return out.view(branch.shape), new_stream.view(stream.shape)
