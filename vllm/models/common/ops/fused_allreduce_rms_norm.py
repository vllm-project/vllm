# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused all-reduce + residual-add + RMSNorm for breakable CUDA graph paths.

This recovers a fusion that vLLM's torch.compile passes would normally do but
that cannot fire under a breakable CUDA graph (CompilationMode.NONE).
"""

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    _AR_RESIDUAL_RMS_NORM,
    _can_use_aiter_fused_ar_rms,
    _can_use_flashinfer,
    flashinfer_trtllm_fused_allreduce_norm,
)
from vllm.model_executor.layers.layernorm import RMSNorm

# Cached zero residual for kernels that require a residual buffer even when
# the model path is ``RMSNorm(all_reduce(x))`` with no residual add.
_zero_residual: torch.Tensor | None = None


def _get_zero_residual(
    hidden_states: torch.Tensor, min_numel: int | None = None
) -> torch.Tensor:
    """Read-only zeros matching ``hidden_states``, grown to ``min_numel``."""
    global _zero_residual
    needed = hidden_states.numel()
    if min_numel is not None:
        needed = max(min_numel, needed)
    buf = _zero_residual
    if (
        buf is None
        or buf.dtype != hidden_states.dtype
        or buf.device != hidden_states.device
        or buf.numel() < needed
    ):
        buf = torch.zeros(
            needed,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        _zero_residual = buf
    return buf[: hidden_states.numel()].view_as(hidden_states)


def _flashinfer_fused_ar_rms(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: RMSNorm,
    tp_size: int,
    max_token_num: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    norm_out = torch.empty_like(hidden_states)
    # With norm_out provided, the kernel writes the new residual
    # (all_reduce(hidden_states) + residual) into the hidden_states
    # buffer and the normalized result into norm_out.
    flashinfer_trtllm_fused_allreduce_norm(
        allreduce_in=hidden_states,
        residual=residual,
        rms_gamma=norm.weight,
        rms_eps=norm.variance_epsilon,
        world_size=tp_size,
        weight_bias=0.0,  # standard RMSNorm (Gemma would use 1.0)
        launch_with_pdl=True,
        fp32_acc=True,
        max_token_num=max_token_num,
        pattern_code=_AR_RESIDUAL_RMS_NORM,
        norm_out=norm_out,
    )
    return norm_out, hidden_states


def _aiter_fused_ar_rms(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm._aiter_ops import rocm_aiter_ops

    return rocm_aiter_ops.get_fused_allreduce_rmsnorm_op()(
        input_=hidden_states,
        residual=residual,
        weight=norm.weight,
        epsilon=norm.variance_epsilon,
        gemma_norm=False,
    )


def fused_allreduce_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-reduce + add residual + (standard) RMSNorm, fused when possible.

    ``hidden_states`` is the per-rank *partial* output of a row-parallel linear
    run with ``reduce_results=False``; ``norm`` is the RMSNorm applied right
    after. Returns ``(normed_output, new_residual)``, equivalent to
    ``norm(all_reduce(hidden_states), residual)``. Falls back to an explicit
    all-reduce + RMSNorm when neither the flashinfer nor the AITER 1-stage
    fast path applies.
    """
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return norm(hidden_states, residual)

    if flashinfer_trtllm_fused_allreduce_norm is not None:
        ok, max_token_num = _can_use_flashinfer(hidden_states, tp_size)
        if ok:
            return _flashinfer_fused_ar_rms(
                hidden_states, residual, norm, tp_size, max_token_num
            )

    if _can_use_aiter_fused_ar_rms(hidden_states):
        return _aiter_fused_ar_rms(hidden_states, residual, norm)

    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return norm(reduced, residual)


def fused_allreduce_rms_norm_out(
    hidden_states: torch.Tensor,
    norm: RMSNorm,
) -> torch.Tensor:
    """All-reduce + (standard) RMSNorm with no residual add.

    Equivalent to ``norm(all_reduce(hidden_states))``. Used by Kimi-K3 latent
    MoE, which RMSNorms the reduced latent before the up-projection. Fused
    kernels still require a residual buffer; a cached zero tensor is passed.
    Prefill-sized tensors miss the AITER 1-stage gate and fall back, matching
    the unfused QuickReduce + RMSNorm path.
    """
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return norm(hidden_states)

    if flashinfer_trtllm_fused_allreduce_norm is not None:
        ok, max_token_num = _can_use_flashinfer(hidden_states, tp_size)
        if ok:
            zero = _get_zero_residual(
                hidden_states, max_token_num * hidden_states.shape[-1]
            )
            norm_out, _ = _flashinfer_fused_ar_rms(
                hidden_states, zero, norm, tp_size, max_token_num
            )
            return norm_out

    if _can_use_aiter_fused_ar_rms(hidden_states):
        norm_out, _ = _aiter_fused_ar_rms(
            hidden_states, _get_zero_residual(hidden_states), norm
        )
        return norm_out

    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return norm(reduced)
