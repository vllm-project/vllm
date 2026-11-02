# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER fused all-reduce + RMSNorm (no residual)."""

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.layernorm import RMSNorm

# Zero residual, cached per shape so CUDA-graph captures keep a stable pointer.
_zero_residuals: dict[tuple[torch.Size, torch.dtype, torch.device], torch.Tensor] = {}


def _zero_residual(hidden_states: torch.Tensor) -> torch.Tensor:
    key = (hidden_states.shape, hidden_states.dtype, hidden_states.device)
    buf = _zero_residuals.get(key)
    if buf is None:
        buf = torch.zeros_like(hidden_states)
        _zero_residuals[key] = buf
    return buf


def can_fuse_allreduce_rms_norm(hidden_states: torch.Tensor) -> bool:
    """True when AITER's one-stage fused AR+RMSNorm covers this tensor."""
    if not rocm_aiter_ops.is_custom_all_reduce_enabled():
        return False
    # gfx1250 custom AR has no fused_allreduce_rmsnorm:
    # https://github.com/ROCm/aiter/blob/v0.1.22/csrc/include/custom_all_reduce_gfx1250.h
    # https://github.com/ROCm/aiter/blob/v0.1.22/csrc/include/custom_all_reduce.h#L69
    from vllm.platforms.rocm import on_gfx1250

    if on_gfx1250():
        return False
    if hidden_states.dim() != 2 or not hidden_states.is_contiguous():
        return False
    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        return False

    aiter_ar = rocm_aiter_ops.get_aiter_allreduce()
    if aiter_ar is None or aiter_ar.disabled:
        return False
    if getattr(aiter_ar.aiter_ca, "_is_gfx1250", False):
        return False
    if not aiter_ar.supports_dynamic_hidden_dim:
        return False
    if not aiter_ar.should_custom_ar(hidden_states):
        return False
    return aiter_ar.use_1stage_fused_ar_rms(hidden_states)


def fused_allreduce_rms_norm_out(
    hidden_states: torch.Tensor,
    norm: RMSNorm,
) -> torch.Tensor:
    """Equivalent to ``norm(all_reduce(hidden_states))``."""
    if get_tensor_model_parallel_world_size() == 1:
        return norm(hidden_states)

    if not can_fuse_allreduce_rms_norm(hidden_states):
        return norm(tensor_model_parallel_all_reduce(hidden_states))

    norm_out, _ = rocm_aiter_ops.get_fused_allreduce_rmsnorm_op()(
        input_=hidden_states,
        residual=_zero_residual(hidden_states),
        weight=norm.weight,
        epsilon=norm.variance_epsilon,
        gemma_norm=False,
    )
    return norm_out
