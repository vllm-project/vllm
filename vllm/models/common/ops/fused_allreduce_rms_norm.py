# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused all-reduce + residual-add + RMSNorm for eager model paths.

This recovers a fusion that vLLM's torch.compile passes would normally do but
that doesn't fire for models running eager (or under a breakable CUDA graph).
"""

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    _AR_RESIDUAL_RMS_NORM,
    _can_use_flashinfer,
    flashinfer_trtllm_fused_allreduce_norm,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform


def _try_aiter_fused_allreduce_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """AITER CustomAR + residual-add + RMSNorm, or None to fall back.

    Same contract as the compile-pass replacement in
    ``AiterAllreduceFusedAddRMSNormPattern``: returns
    ``(norm(all_reduce(x) + residual), all_reduce(x) + residual)``.
    """
    if not current_platform.is_rocm():
        return None
    aiter_ar = rocm_aiter_ops.get_aiter_allreduce()
    if aiter_ar is None:
        return None
    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        return None
    if hidden_states.numel() == 0 or residual.shape != hidden_states.shape:
        return None

    max_bytes = aiter_ar.effective_max_size()
    # numel() * element_size() rather than .nbytes: under torch.compile the
    # shape is symbolic and .nbytes calls numel() eagerly in C++, which throws
    # on symbolic sizes. numel() from Python returns a SymInt and guards.
    if hidden_states.numel() * hidden_states.element_size() > max_bytes:
        return None

    try:
        fused_op = rocm_aiter_ops.get_fused_allreduce_rmsnorm_op()
    except (AttributeError, RuntimeError):
        return None

    orig_shape = hidden_states.shape
    hidden_2d = hidden_states.reshape(-1, orig_shape[-1]).contiguous()
    residual_2d = residual.reshape(-1, orig_shape[-1]).contiguous()
    norm_out, residual_out = fused_op(
        input_=hidden_2d,
        residual=residual_2d,
        weight=norm.weight.to(dtype=hidden_states.dtype),
        epsilon=norm.variance_epsilon,
    )
    return norm_out.view(orig_shape), residual_out.view(orig_shape)


def fused_allreduce_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-reduce + add residual + (standard) RMSNorm.

    ``hidden_states`` is the per-rank *partial* output of a row-parallel linear
    run with ``reduce_results=False``; ``norm`` is the RMSNorm applied right
    after. Returns ``(normed_output, new_residual)``, equivalent to
    ``norm(all_reduce(hidden_states), residual)``.

    Fast paths: AITER CustomAR fusion on ROCm, flashinfer on CUDA. Falls back
    to an explicit all-reduce + RMSNorm when neither applies.
    """
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return norm(hidden_states, residual)

    aiter_result = _try_aiter_fused_allreduce_rms_norm(hidden_states, residual, norm)
    if aiter_result is not None:
        return aiter_result

    if flashinfer_trtllm_fused_allreduce_norm is not None:
        ok, max_token_num = _can_use_flashinfer(hidden_states, tp_size)
        if ok:
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

    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return norm(reduced, residual)
