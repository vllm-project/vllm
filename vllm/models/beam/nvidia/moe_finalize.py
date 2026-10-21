# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Finalize live MegaMoE tokens with Beam's post normalization."""

import torch

from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput

from .allreduce import BeamAllReduceWorkspace


def beam_moe_finalize(
    expert_outputs: torch.Tensor,
    expert_weights: torch.Tensor,
    output: torch.Tensor,
    routed_scale: float = 1.0,
    post_weight: torch.Tensor | None = None,
    post_eps: float | None = None,
) -> torch.Tensor:
    """Reduce only output rows, preserving BF16 casts before scale and RMSNorm.

    Args:
        expert_outputs: Contiguous BF16 expert results [capacity, top_k, hidden].
        expert_weights: Contiguous FP32 routing weights [capacity, top_k].
        output: Contiguous BF16 destination [live_tokens, hidden].
        routed_scale: Multiplier applied after rounding the expert reduction.
        post_weight: Optional BF16 RMSNorm affine weight [hidden].
        post_eps: RMSNorm epsilon, or None to omit normalization.

    """
    assert expert_outputs.ndim == 3 and expert_weights.ndim == output.ndim == 2
    capacity, top_k, hidden = expert_outputs.shape
    assert 1 <= top_k <= 32
    assert 256 <= hidden <= 8192 and hidden % 256 == 0
    assert output.shape[1] == hidden and output.shape[0] <= capacity
    assert expert_weights.shape[1] == top_k
    assert expert_weights.shape[0] >= output.shape[0]
    assert expert_outputs.dtype == output.dtype == torch.bfloat16
    assert expert_weights.dtype == torch.float32
    assert output.is_cuda
    for tensor in (expert_outputs, expert_weights, output):
        assert tensor.is_contiguous() and tensor.device == output.device
    if post_weight is not None:
        assert post_eps is not None
        assert post_weight.shape == (hidden,) and post_weight.is_contiguous()
        assert post_weight.dtype == torch.bfloat16
        assert post_weight.device == output.device
    if output.shape[0]:
        torch.ops._C.beam_moe_finalize(
            expert_outputs,
            expert_weights,
            output,
            post_weight,
            routed_scale,
            post_eps or 0.0,
            post_eps is not None,
        )
    return output


def beam_moe_finalize_gather_norm_residual_norm(
    x: UnfinalizedMoEOutput,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    post_eps: float,
    next_weight: torch.Tensor | None,
    next_eps: float | None,
    scales: tuple[float, float],
    workspace: BeamAllReduceWorkspace,
    routed_scale: float = 1.0,
    *,
    strategy: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Finalize MegaMoE output and restore the replicated token boundary.

    FlashInfer's ``forward_unfinalized`` guarantees token-major expert rows and
    an identity route map, so a view recovers its [tokens, top_k, hidden] layout.
    This consumer requires that producer contract.

    Each rank owns ceil(global_tokens / TP) consecutive tokens. Capacity rows
    and the final rank's token padding do not contribute to the returned tensors.
    ``strategy`` overrides the token-count heuristic for validation and benchmarking.
    """
    assert residual.ndim == 2 and residual.is_cuda
    tokens, hidden_size = residual.shape
    local_tokens = (tokens + workspace.world_size - 1) // workspace.world_size
    assert tokens <= workspace.max_tokens and hidden_size == workspace.hidden_size
    expert_weights = x.expert_weights
    assert expert_weights.ndim == 2
    assert x.gemm2_permuted.ndim == 2 and x.gemm2_permuted.is_contiguous()
    assert x.gemm2_permuted.shape[1] == hidden_size
    expert_outputs = x.gemm2_permuted.view(*expert_weights.shape, hidden_size)
    assert expert_weights.shape[0] >= local_tokens
    assert 1 <= expert_weights.shape[1] <= 32
    assert expert_outputs.dtype == residual.dtype == torch.bfloat16
    assert expert_weights.dtype == torch.float32
    for tensor in (expert_outputs, expert_weights, residual):
        assert tensor.is_contiguous() and tensor.device == residual.device
    for weight, eps in ((post_weight, post_eps), (next_weight, next_eps)):
        if weight is not None:
            assert eps is not None
            assert weight.shape == (hidden_size,) and weight.is_contiguous()
            assert weight.dtype == torch.bfloat16 and weight.device == residual.device
    if strategy is None:
        if tokens <= workspace.moe_gather_oneshot_max_tokens:
            strategy = "oneshot"
        elif (
            workspace.moe_gather_separate_norm_min_tokens is not None
            and tokens >= workspace.moe_gather_separate_norm_min_tokens
        ):
            strategy = "unfused"
        else:
            strategy = "twoshot"
    if strategy not in ("unfused", "oneshot", "twoshot"):
        raise ValueError(f"Unknown MegaMoE gather strategy: {strategy}")
    if not tokens:
        hidden = torch.empty_like(residual)
        return hidden, torch.empty_like(residual) if next_eps is not None else None
    if strategy == "unfused":
        from .layernorm import beam_norm_residual_norm

        local = torch.empty(
            (local_tokens, hidden_size), dtype=residual.dtype, device=residual.device
        )
        beam_moe_finalize(
            expert_outputs, expert_weights, local, routed_scale, post_weight, post_eps
        )
        if workspace.all_gather is not None:
            gathered = workspace.all_gather(local)
        else:
            gathered = torch.empty(
                (local_tokens * workspace.world_size, hidden_size),
                dtype=residual.dtype,
                device=residual.device,
            )
            torch.distributed.all_gather_into_tensor(
                gathered, local, group=workspace.device_group
            )
        return beam_norm_residual_norm(
            gathered[:tokens], residual, None, None, next_weight, next_eps, scales
        )
    hidden = torch.empty_like(residual)
    normalized = torch.empty_like(residual) if next_eps is not None else None
    torch.ops._C.beam_moe_gather_norm(
        expert_outputs,
        expert_weights,
        residual,
        post_weight,
        next_weight,
        hidden,
        normalized,
        workspace.buffer_ptrs,
        workspace.flags,
        workspace.multicast_ptr,
        workspace.local_ptr,
        workspace.world_size,
        workspace.rank,
        post_eps or 0.0,
        next_eps or 0.0,
        *scales,
        post_eps is not None,
        next_eps is not None,
        scales is not None,
        strategy == "oneshot",
        routed_scale,
    )
    return hidden, normalized


def beam_moe_finalize_norm_residual_norm(
    x: UnfinalizedMoEOutput,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    post_eps: float,
    next_weight: torch.Tensor | None,
    next_eps: float | None,
    scales: tuple[float, float],
    routed_scale: float = 1.0,
    *,
    strategy: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Finalize local MegaMoE tokens through the next normalization boundary.

    Requires FlashInfer's token-major expert rows and identity route map.
    Only the live rows in ``residual`` are read from the workspace capacity.
    ``strategy`` can force the unfused reference for validation and benchmarking.
    """
    assert residual.ndim == 2 and residual.is_cuda
    tokens, hidden_size = residual.shape
    assert 256 <= hidden_size <= 8192 and hidden_size % 256 == 0
    expert_weights = x.expert_weights
    assert expert_weights.ndim == 2 and expert_weights.shape[0] >= tokens
    assert 1 <= expert_weights.shape[1] <= 32
    assert x.gemm2_permuted.ndim == 2 and x.gemm2_permuted.is_contiguous()
    assert x.gemm2_permuted.shape[1] == hidden_size
    expert_outputs = x.gemm2_permuted.view(*expert_weights.shape, hidden_size)
    assert expert_outputs.dtype == residual.dtype == torch.bfloat16
    assert expert_weights.dtype == torch.float32
    for tensor in (expert_outputs, expert_weights, residual):
        assert tensor.is_contiguous() and tensor.device == residual.device
    for weight, eps in ((post_weight, post_eps), (next_weight, next_eps)):
        if weight is not None:
            assert eps is not None
            assert weight.shape == (hidden_size,) and weight.is_contiguous()
            assert weight.dtype == torch.bfloat16 and weight.device == residual.device
    if strategy not in (None, "fused", "unfused"):
        raise ValueError(f"Unknown local MegaMoE finalize strategy: {strategy}")
    if strategy == "unfused":
        from .layernorm import beam_norm_residual_norm

        finalized = torch.empty_like(residual)
        beam_moe_finalize(
            expert_outputs,
            expert_weights,
            finalized,
            routed_scale,
            post_weight,
            post_eps,
        )
        return beam_norm_residual_norm(
            finalized, residual, None, None, next_weight, next_eps, scales
        )
    hidden = torch.empty_like(residual)
    normalized = torch.empty_like(residual) if next_eps is not None else None
    if tokens:
        torch.ops._C.beam_moe_finalize_norm(
            expert_outputs,
            expert_weights,
            residual,
            post_weight,
            next_weight,
            hidden,
            normalized,
            routed_scale,
            post_eps or 0.0,
            next_eps or 0.0,
            *scales,
            post_eps is not None,
            next_eps is not None,
            scales is not None,
        )
    return hidden, normalized
