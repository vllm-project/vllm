# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused all-reduce + residual-add + RMSNorm for eager model paths.

This recovers a fusion that vLLM's torch.compile passes would normally do but
that doesn't fire for models running eager (or under a breakable CUDA graph).
A MoE output whose top-k reduction is still open (an ``UnfinalizedMoEOutput``)
has that reduction and the shared-expert add folded into the same kernel.
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
    MiB,
    _can_use_flashinfer,
    _max_token_num,
    flashinfer_trtllm_fused_allreduce_norm,
)
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform

# FlashInfer's MoE-finalize all-reduce is one-shot only and warns past this many
# tokens (kOneShotMaxToken in trtllm_moe_allreduce_fusion.cuh).
_MOE_FINALIZE_ONE_SHOT_MAX_TOKENS = 128


def fused_allreduce_rms_norm(
    hidden_states: torch.Tensor | MoEOutput,
    residual: torch.Tensor,
    norm: RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-reduce + add residual + (standard) RMSNorm, fused via flashinfer.

    ``hidden_states`` is the per-rank *partial* output of a row-parallel linear
    run with ``reduce_results=False``, or of a MoE layer whose reduction was
    left open (``MoEOutput``); ``norm`` is the RMSNorm applied right after.
    Returns ``(normed_output, new_residual)``, equivalent to
    ``norm(all_reduce(hidden_states), residual)``. Falls back to an explicit
    all-reduce + RMSNorm when the flashinfer fast path is unavailable.
    """
    if isinstance(hidden_states, MoEOutput):
        return _moe_finalize_allreduce_rms_norm(hidden_states, residual, norm)

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


def moe_finalize_allreduce_max_num_tokens(hidden_size: int, dtype: torch.dtype) -> int:
    """Most tokens a deferred MoE output can be finalized for by one fused
    finalize + all-reduce + RMSNorm kernel, or 0 if that kernel is unavailable.

    FlashInfer's MoE-finalize all-reduce is one-shot only and runs on the
    TRT-LLM workspace, so this stays within both its own one-shot bound and the
    size up to which vLLM picks the one-shot all-reduce anyway. Used by a MoE
    layer to decide whether, and up to which batch size, to leave its top-k
    reduction to ``fused_allreduce_rms_norm``.
    """
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1 or flashinfer_trtllm_fused_allreduce_norm is None:
        return 0
    from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
        _FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB,
    )
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        _resolve_fi_ar_backend,
    )

    backend, _ = _resolve_fi_ar_backend(get_tp_group().cpu_group)
    capability = current_platform.get_device_capability()
    if backend != "trtllm" or capability is None:
        return 0
    one_shot_mb = _FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB.get(capability.to_int(), {}).get(
        tp_size
    )
    max_token_num = _max_token_num(tp_size, hidden_size, dtype)
    if one_shot_mb is None or max_token_num is None:
        return 0
    element_size = torch.tensor([], dtype=dtype).element_size()
    return min(
        _MOE_FINALIZE_ONE_SHOT_MAX_TOKENS,
        int(one_shot_mb * MiB) // (hidden_size * element_size),
        max_token_num,
    )


def _moe_finalize_allreduce_rms_norm(
    moe_output: MoEOutput,
    residual: torch.Tensor,
    norm: RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    routed = moe_output.routed
    if not isinstance(routed, UnfinalizedMoEOutput):
        return fused_allreduce_rms_norm(
            _add_shared_output(routed, moe_output), residual, norm
        )

    # The fused kernel reads the routing weights in the activation dtype.
    expert_weights = routed.expert_weights.to(residual.dtype)
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size > 1 and flashinfer_trtllm_fused_allreduce_norm is not None:
        from vllm.distributed.device_communicators.flashinfer_all_reduce import (
            flashinfer_comm,
            get_fi_ar_workspace,
        )

        # The kernel sizes its all-reduce by the token output, i.e. residual.
        ok, max_token_num = _can_use_flashinfer(residual, tp_size)
        workspace = (
            get_fi_ar_workspace(
                world_size=tp_size,
                rank=get_tensor_model_parallel_rank(),
                max_token_num=max_token_num,
                hidden_dim=residual.shape[-1],
                dtype=residual.dtype,
                group=get_tp_group().cpu_group,
            )
            if ok
            else None
        )
        if workspace is not None and workspace.backend == "trtllm":
            norm_out = torch.empty_like(residual)
            residual_out = torch.empty_like(residual)
            flashinfer_comm.trtllm_moe_finalize_allreduce_fusion(
                allreduce_in=routed.gemm2_permuted,
                residual_in=residual,
                norm_weight=norm.weight,
                expanded_idx_to_permuted_idx=routed.expanded_idx_to_permuted_idx,
                norm_out=norm_out,
                residual_out=residual_out,
                quant_out=None,
                scale_out=None,
                workspace_ptrs=workspace.workspace_tensor,
                launch_with_pdl=True,
                world_rank=workspace.rank,
                world_size=workspace.world_size,
                eps=norm.variance_epsilon,
                shared_expert_output=moe_output.shared_output,
                expert_scale_factor=expert_weights,
                routed_scaling_factor=moe_output.routed_scaling_factor,
            )
            return norm_out, residual_out

    # Finalize the top-k reduction unfused, then reduce as a plain partial.
    permuted_idx = routed.expanded_idx_to_permuted_idx
    rows = routed.gemm2_permuted[permuted_idx.clamp(min=0).long()]
    weights = torch.where(permuted_idx >= 0, expert_weights.float(), 0.0)
    routed_output = (rows.float() * weights.unsqueeze(-1)).sum(dim=1)
    return fused_allreduce_rms_norm(
        _add_shared_output(routed_output.to(residual.dtype), moe_output),
        residual,
        norm,
    )


def _add_shared_output(routed: torch.Tensor, moe_output: MoEOutput) -> torch.Tensor:
    if moe_output.routed_scaling_factor != 1.0:
        routed = routed * moe_output.routed_scaling_factor
    if moe_output.shared_output is None:
        return routed
    return moe_output.shared_output + routed
