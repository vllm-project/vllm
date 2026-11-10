# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark's attention all-reduce, fused into the residual boundary.

The kernel is AITER's FlyDSL QuickReduce epilogue. It runs when that
QuickReduce would have handled the o_proj partial. No separate switch.
"""

import torch

from vllm.models.muse_spark.common.layers import (
    MuseSparkNormPreservingResidualConnection,
    MuseSparkResidualStream,
    MuseSparkRMSNorm,
)
from vllm.models.muse_spark.common.ops.fused_rmsnorm import supports_norm_residual_norm


def _qr_comm():
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    if group.world_size <= 1:
        return None
    comm = getattr(group, "device_communicator", None)
    qr = getattr(comm, "qr_comm", None) if comm is not None else None
    if qr is None or getattr(qr, "disabled", True):
        return None
    return qr


def o_proj_skips_reduce(rows: int, hidden: int, *, stream_dtype, out_dtype) -> bool:
    """Whether o_proj should return its partial for the fused boundary."""
    if stream_dtype is not torch.float32 or out_dtype is not torch.bfloat16:
        return False
    if rows < 1:
        return False
    qr = _qr_comm()
    if qr is None or not hasattr(qr, "flydsl_boundary_bf16"):
        return False
    return bool(qr.flydsl_boundary_bf16(rows, hidden))


def fused_allreduce_norm_residual_then_norm(
    boundary: MuseSparkResidualStream,
    pre_norm: MuseSparkRMSNorm,
    *,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """All-reduce the o_proj partial inside the residual boundary.

    Returns ``None`` when FlyDSL QuickReduce would not take the partial, or
    the installed AITER has no boundary epilogue. The caller all-reduces and
    uses the unfused boundary.
    """
    post_norm, connection = boundary.post_norm, boundary.connection
    branch, stream = boundary.branch, boundary.stream
    if not isinstance(connection, MuseSparkNormPreservingResidualConnection):
        return None
    alpha, beta = connection.alpha, connection.beta
    if not (
        getattr(connection, "_coefficients_ready", False)
        and post_norm.supports_fused_kernel(branch)
        and pre_norm.supports_fused_kernel(stream)
        and supports_norm_residual_norm(branch, stream)
        and isinstance(beta, torch.Tensor)
        and beta.is_cuda
        and (not isinstance(alpha, torch.Tensor) or alpha.is_cuda)
    ):
        return None
    qr = _qr_comm()
    if qr is None or not hasattr(qr, "allreduce_norm_residual_norm"):
        return None
    flat_branch = branch.reshape(-1, branch.shape[-1])
    flat_stream = stream.reshape(-1, stream.shape[-1])
    fused = qr.allreduce_norm_residual_norm(
        flat_branch,
        flat_stream,
        post_norm.weight if post_norm.has_weight else None,
        pre_norm.weight if pre_norm.has_weight else None,
        post_norm.variance_epsilon,
        pre_norm.variance_epsilon,
        alpha=alpha,
        beta=beta,
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
