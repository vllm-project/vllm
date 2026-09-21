# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager fused all-reduce + residual-add + RMSNorm.

Kimi-K3 AMD serving uses vLLM's own compilation:
``CompilationMode.NONE`` + breakable CUDA/HIP graphs, not torch.compile.
Write fusions as ordinary eager functions with no compile decorator and no
``eager_break_during_capture``. If the call sits on the main model stream,
breakable capture records it as a graph node.

Do not route this through inductor fusion passes or ``@support_torch_compile``.
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

    Equivalent to ``norm(all_reduce(hidden_states))``. Kimi-K3 latent MoE
    RMSNorms the reduced latent before the up-projection. Fused kernels still
    take a residual buffer; a same-shaped zero tensor is passed (eager
    ``zeros_like``, captured with the surrounding graph). Prefill-sized
    tensors miss the AITER 1-stage gate and fall back to unfused QR + RMSNorm.
    """
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return norm(hidden_states)

    if flashinfer_trtllm_fused_allreduce_norm is not None:
        ok, max_token_num = _can_use_flashinfer(hidden_states, tp_size)
        if ok:
            norm_out, _ = _flashinfer_fused_ar_rms(
                hidden_states,
                torch.zeros_like(hidden_states),
                norm,
                tp_size,
                max_token_num,
            )
            return norm_out

    if _can_use_aiter_fused_ar_rms(hidden_states):
        norm_out, _ = _aiter_fused_ar_rms(
            hidden_states, torch.zeros_like(hidden_states), norm
        )
        return norm_out

    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return norm(reduced)
