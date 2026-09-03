# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused all-reduce + residual-add + RMSNorm for eager model paths.

This recovers a fusion that vLLM's torch.compile passes would normally do but
that doesn't fire for models running eager (or under a breakable CUDA graph).
"""

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    _AR_RESIDUAL_RMS_NORM,
    _can_use_flashinfer,
    flashinfer_trtllm_fused_allreduce_norm,
)
from vllm.model_executor.layers.fusion.fused_act_quant import (
    maybe_allocate_fp8_block_quant,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
)
from vllm.model_executor.layers.layernorm import RMSNorm

try:
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        flashinfer_comm,
        get_fi_ar_packed_quant_workspace,
    )

    _AR_RESIDUAL_RMS_NORM_OUT_FP8_BLOCK_QUANT = getattr(
        flashinfer_comm.AllReduceFusionPattern,
        "kARResidualRMSNormOutPerTokenGroupFP8PackedQuant",
        None,
    )
except (ImportError, AttributeError):
    get_fi_ar_packed_quant_workspace = None  # type: ignore[assignment]
    _AR_RESIDUAL_RMS_NORM_OUT_FP8_BLOCK_QUANT = None


@torch.compiler.assume_constant_result
def _has_packed_quant_workspace(
    tp_size: int,
    max_token_num: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> bool:
    return (
        get_fi_ar_packed_quant_workspace(
            world_size=tp_size,
            rank=get_tensor_model_parallel_rank(),
            max_token_num=max_token_num,
            hidden_dim=hidden_size,
            dtype=dtype,
            group=get_tp_group().cpu_group,
        )
        is not None
    )


def fused_allreduce_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-reduce + add residual + (standard) RMSNorm, fused via flashinfer.

    ``hidden_states`` is the per-rank *partial* output of a row-parallel linear
    run with ``reduce_results=False``; ``norm`` is the RMSNorm applied right
    after. Returns ``(normed_output, new_residual)``, equivalent to
    ``norm(all_reduce(hidden_states), residual)``. Falls back to an explicit
    all-reduce + RMSNorm when the flashinfer fast path is unavailable.
    """
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return norm(hidden_states, residual)

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


def fused_allreduce_rms_norm_fp8_quant(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: RMSNorm,
    consumer: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, QuantizedActivation | None]:
    """Fuse all-reduce, residual RMSNorm, and DeepGEMM FP8 quantization.

    The normalized BF16 output remains available to unquantized consumers while
    the returned activation lets ``consumer`` skip its input quantization. The
    regular eager path is used when the required FlashInfer collective or
    activation contract is unavailable.
    """
    tp_size = get_tensor_model_parallel_world_size()
    if (
        tp_size == 1
        or flashinfer_trtllm_fused_allreduce_norm is None
        or get_fi_ar_packed_quant_workspace is None
        or _AR_RESIDUAL_RMS_NORM_OUT_FP8_BLOCK_QUANT is None
    ):
        norm_out, residual_out = fused_allreduce_rms_norm(hidden_states, residual, norm)
        return norm_out, residual_out, None

    ok, max_token_num = _can_use_flashinfer(hidden_states, tp_size)
    if ok:
        ok = _has_packed_quant_workspace(
            tp_size,
            max_token_num,
            hidden_states.shape[-1],
            hidden_states.dtype,
        )
    if not ok:
        norm_out, residual_out = fused_allreduce_rms_norm(hidden_states, residual, norm)
        return norm_out, residual_out, None

    quant = maybe_allocate_fp8_block_quant(hidden_states, consumer)
    if quant is None:
        norm_out, residual_out = fused_allreduce_rms_norm(hidden_states, residual, norm)
        return norm_out, residual_out, None

    norm_out = torch.empty_like(hidden_states)
    flashinfer_trtllm_fused_allreduce_norm(
        allreduce_in=hidden_states,
        residual=residual,
        rms_gamma=norm.weight,
        rms_eps=norm.variance_epsilon,
        world_size=tp_size,
        weight_bias=0.0,
        launch_with_pdl=True,
        fp32_acc=True,
        max_token_num=max_token_num,
        pattern_code=_AR_RESIDUAL_RMS_NORM_OUT_FP8_BLOCK_QUANT,
        norm_out=norm_out,
        quant_out=quant.data,
        scale_out=quant.scale,
        block_quant_group_size=128,
    )
    return norm_out, hidden_states, quant
