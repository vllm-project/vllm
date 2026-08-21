# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer for the MoE tail: finalize + all-reduce + residual add + RMSNorm.

A MoE layer running un-reduced hands its output to the RMSNorm that follows --
the next decoder layer's ``input_layernorm`` -- as a ``MoEOutput``. When the
routed half is still unfinalized, the whole tail (top-k reduction over the
permuted GEMM2 rows, shared-expert add, all-reduce, residual add, norm)
collapses into flashinfer's ``kMoEFinalizeARResidualRMSNorm``.

A batch the fused kernel cannot take -- one over its token ceiling, or any at
all on a deployment without the workspace -- is finalized in the MoE kernel as
it always was. Nothing reaches this consumer then: the routed half is a plain
tensor, so the layer closes out its tail the way a layer without deferral does.
"""

from functools import cache

import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

try:
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        flashinfer_comm,
        get_fi_ar_moe_finalize_workspace,
    )
except (ImportError, AttributeError):
    flashinfer_comm = None  # type: ignore[assignment]
    get_fi_ar_moe_finalize_workspace = None  # type: ignore[assignment]

# The CuTe DSL tail is a bf16 backend.
_FI_SUPPORTED_DTYPES = (torch.bfloat16,)

logger = init_logger(__name__)


@cache
def moe_tail_fusion_available() -> bool:
    """Whether a fused MoE tail can run here at all.

    The consumer is a Blackwell CuTe DSL kernel, and only the TRTLLM-Gen MoE
    experts ever hand back an unfinalized output, so all that is left to check
    is the switch, the hardware, and that there is a tensor-parallel all-reduce
    for the reduction to fuse into.
    """
    if not envs.VLLM_ENABLE_MOE_TAIL_FUSION:
        return False
    if flashinfer_comm is None:
        logger.debug_once("MoE tail fusion off: flashinfer.comm is unavailable")
        return False
    if not current_platform.is_device_capability_family(100):
        logger.debug_once("MoE tail fusion off: the fused tail is Blackwell-only")
        return False
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        logger.debug_once("MoE tail fusion off: needs TP>1, got tp_size=%d", tp_size)
        return False
    return True


def _finalize_workspace(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    *,
    max_token_num: int,
    top_k: int,
    routed_scaling_factor: float,
    rms_eps: float,
    weight_bias: float,
    include_shared_expert: bool,
):
    """The (globally cached) fusion workspace for a batch, or None.

    The mnnvl CuTe DSL workspace is compiled around ``top_k``, ``rms_eps``,
    ``weight_bias`` and whether a shared expert is folded in; the kernel
    rejects per-call values that disagree, so they are part of building it.
    """
    if get_fi_ar_moe_finalize_workspace is None:
        return None
    tp_size = get_tensor_model_parallel_world_size()
    workspace = get_fi_ar_moe_finalize_workspace(
        world_size=tp_size,
        rank=get_tensor_model_parallel_rank(),
        max_token_num=max_token_num,
        hidden_dim=hidden_size,
        dtype=dtype,
        group=get_tp_group().cpu_group,
        top_k=top_k,
        rms_eps=rms_eps,
        routed_scaling_factor=routed_scaling_factor,
        weight_bias=weight_bias,
        include_shared_expert=include_shared_expert,
    )
    if workspace is None or not workspace.is_buffer_size_sufficient(
        tp_size, num_tokens, hidden_size, dtype
    ):
        return None
    return workspace


def moe_tail_fusion_applies(
    num_tokens: int, hidden_size: int, dtype: torch.dtype
) -> bool:
    """Whether this consumer can take an unfinalized MoE output of this shape.

    The producer's gate. There is no token ceiling to respect -- the backend
    covers decode and prefill alike -- so this is the switch, the hardware, the
    dtype it has kernels for, and a batch that is not an idle rank's empty one.
    """
    return (
        num_tokens > 0 and dtype in _FI_SUPPORTED_DTYPES and moe_tail_fusion_available()
    )


def _moe_finalize_allreduce_rms_norm(
    gemm2_permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    shared_output: torch.Tensor | None,
    residual: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    weight_bias: float,
    routed_scaling_factor: float,
    max_num_tokens: int,
) -> list[torch.Tensor]:
    num_tokens, hidden_size = residual.shape
    workspace = _finalize_workspace(
        num_tokens,
        hidden_size,
        residual.dtype,
        max_token_num=max(max_num_tokens, num_tokens),
        top_k=expert_weights.shape[-1],
        routed_scaling_factor=routed_scaling_factor,
        rms_eps=rms_eps,
        weight_bias=weight_bias,
        include_shared_expert=shared_output is not None,
    )
    assert workspace is not None, (
        "no MoE finalize fusion workspace for this batch: the mnnvl CuTe DSL "
        "backend has no kernel for this (tp_size, hidden_size, top_k, dtype). "
        "Unset VLLM_ENABLE_MOE_TAIL_FUSION to fall back to finalizing in the "
        "MoE kernel."
    )

    norm_out = torch.empty_like(residual)
    residual_out = torch.empty_like(residual)
    # The unified entry point, not trtllm_moe_finalize_allreduce_fusion: that
    # one takes raw trtllm workspace pointers, while the backend here owns a
    # compiled CuTe DSL workspace and is reached by pattern.
    flashinfer_comm.allreduce_fusion(
        input=gemm2_permuted,
        workspace=workspace,
        pattern=flashinfer_comm.AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
        launch_with_pdl=True,
        residual_in=residual,
        residual_out=residual_out,
        norm_out=norm_out,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        # No scale_factor: this backend rejects a per-call scale and takes
        # routed_scaling_factor at workspace construction instead.
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        expert_scale_factor=expert_weights,
        shared_expert_output=shared_output,
        weight_bias=weight_bias,
    )
    return [norm_out, residual_out]


def _moe_finalize_allreduce_rms_norm_fake(
    gemm2_permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    shared_output: torch.Tensor | None,
    residual: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    weight_bias: float,
    routed_scaling_factor: float,
    max_num_tokens: int,
) -> list[torch.Tensor]:
    return [torch.empty_like(residual), torch.empty_like(residual)]


direct_register_custom_op(
    op_name="moe_finalize_allreduce_rms_norm",
    op_func=_moe_finalize_allreduce_rms_norm,
    fake_impl=_moe_finalize_allreduce_rms_norm_fake,
)


def rms_norm_weight_bias(norm: torch.nn.Module) -> float:
    """The `1 +` a Gemma-style RMSNorm folds into its weight, else 0.

    Models with their own Gemma-style norm -- one that scales by ``1 + w``
    without subclassing ``GemmaRMSNorm`` -- declare it by setting
    ``rms_weight_bias = 1.0`` on the class.
    """
    default = 1.0 if isinstance(norm, GemmaRMSNorm) else 0.0
    return getattr(norm, "rms_weight_bias", default)


def fused_moe_finalize_allreduce_rms_norm(
    moe_output: MoEOutput,
    residual: torch.Tensor,
    norm: RMSNorm | GemmaRMSNorm | torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Close out a MoE layer's tail into the RMSNorm that follows it.

    ``moe_output`` is the un-reduced output of a MoE layer running with
    ``skip_final_all_reduce``; ``norm`` is the RMSNorm applied right after.
    Returns ``(normed_output, new_residual)``, equivalent to
    ``norm(all_reduce(routed * scale + shared), residual)``.
    """
    routed = moe_output.routed
    assert isinstance(routed, UnfinalizedMoEOutput)
    norm_out, residual_out = torch.ops.vllm.moe_finalize_allreduce_rms_norm(
        routed.gemm2_permuted,
        routed.expert_weights,
        routed.expanded_idx_to_permuted_idx,
        moe_output.shared_output,
        residual,
        norm.weight,
        norm.variance_epsilon,
        rms_norm_weight_bias(norm),
        moe_output.routed_scaling_factor,
        moe_output.max_num_tokens,
    )
    return norm_out, residual_out
