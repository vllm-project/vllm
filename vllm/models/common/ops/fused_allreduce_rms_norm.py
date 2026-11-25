# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused all-reduce + residual-add + RMSNorm for eager model paths.

This recovers a fusion that vLLM's torch.compile passes would normally do but
that doesn't fire for models running eager (or under a breakable CUDA graph).
"""

import torch

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

_zero_residuals: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}


def _zero_residual(hidden_states: torch.Tensor, max_token_num: int) -> torch.Tensor:
    """A read-only zero residual shared by all calls: the fused kernel writes the
    new residual into `hidden_states`, never into `residual`."""
    key = (hidden_states.device, hidden_states.dtype, hidden_states.shape[1])
    zeros = _zero_residuals.get(key)
    if zeros is None:
        assert not torch.cuda.is_current_stream_capturing(), (
            "allocate the zero residual in an eager pass before graph capture"
        )
        zeros = hidden_states.new_zeros(max_token_num, hidden_states.shape[1])
        _zero_residuals[key] = zeros
    return zeros[: hidden_states.shape[0]]


def fused_allreduce_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    norm: RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-reduce + add residual + (standard) RMSNorm, fused via flashinfer
    (CUDA) or AITER's one-stage custom all-reduce kernel (ROCm).

    `hidden_states` is the per-rank *partial* output of a row-parallel linear
    run with `reduce_results=False`; `norm` is the RMSNorm applied right
    after. Returns `(normed_output, new_residual)`, equivalent to
    `norm(all_reduce(hidden_states), residual)`. Without `residual` this is
    `norm(all_reduce(hidden_states))` and the new residual is the reduced
    input. Falls back to an explicit all-reduce + RMSNorm when no fused path
    applies.
    """
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        if residual is None:
            return norm(hidden_states), hidden_states
        return norm(hidden_states, residual)

    if flashinfer_trtllm_fused_allreduce_norm is not None:
        ok, max_token_num = _can_use_flashinfer(hidden_states, tp_size)
        if ok:
            if residual is None:
                residual = _zero_residual(hidden_states, max_token_num)
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

    if current_platform.is_rocm():
        from vllm._aiter_ops import rocm_aiter_ops

        # decode-size inputs only: beyond the one-stage kernel's limits the
        # fused op's two-stage variant is slower than all-reduce + norm
        aiter_ar = rocm_aiter_ops.get_aiter_allreduce()
        if (
            residual is not None
            and aiter_ar is not None
            and aiter_ar.use_1stage_fused_ar_rms(hidden_states)
        ):
            return rocm_aiter_ops.get_fused_allreduce_rmsnorm_op()(
                hidden_states, residual, norm.weight, norm.variance_epsilon
            )

    reduced = tensor_model_parallel_all_reduce(hidden_states)
    if residual is None:
        return norm(reduced), reduced
    return norm(reduced, residual)
