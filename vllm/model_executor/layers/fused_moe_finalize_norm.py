# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer for the MoE tail: finalize + all-reduce + residual add + RMSNorm.

A MoE layer running un-reduced hands its output to the RMSNorm that follows --
the next decoder layer's ``input_layernorm`` -- as a ``MoEOutput``. When the
routed half is still unfinalized, the whole tail (top-k reduction over the
permuted GEMM2 rows, shared-expert add, all-reduce, residual add, norm)
collapses into flashinfer's ``kMoEFinalizeARResidualRMSNorm``.

A batch the fused kernel cannot take -- an idle rank's empty one, or any at
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
        existing_fi_ar_moe_finalize_workspace,
        flashinfer_comm,
        get_fi_ar_moe_finalize_workspace,
        has_fi_ar_moe_finalize_backend,
    )
except (ImportError, AttributeError):
    flashinfer_comm = None  # type: ignore[assignment]
    existing_fi_ar_moe_finalize_workspace = None  # type: ignore[assignment]
    get_fi_ar_moe_finalize_workspace = None  # type: ignore[assignment]
    has_fi_ar_moe_finalize_backend = None  # type: ignore[assignment]

# The CuTe DSL tail is a bf16 backend.
_FI_SUPPORTED_DTYPES = (torch.bfloat16,)

logger = init_logger(__name__)


@cache
def moe_tail_fusion_available() -> bool:
    """Whether a fused MoE tail can run here at all.

    The consumer is a Blackwell CuTe DSL kernel, and only the TRTLLM-Gen MoE
    experts ever hand back an unfinalized output, so all that is left to check
    is the switch, that flashinfer ships the backend, the hardware, and that
    there is a tensor-parallel all-reduce for the reduction to fuse into.
    """
    if not envs.VLLM_ENABLE_MOE_TAIL_FUSION:
        return False
    if flashinfer_comm is None or has_fi_ar_moe_finalize_backend is None:
        logger.debug_once("MoE tail fusion off: flashinfer.comm is unavailable")
        return False
    if not has_fi_ar_moe_finalize_backend():
        return False
    if not current_platform.is_device_capability_family(100):
        logger.debug_once("MoE tail fusion off: the fused tail is Blackwell-only")
        return False
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        logger.debug_once("MoE tail fusion off: needs TP>1, got tp_size=%d", tp_size)
        return False
    return True


def initialize_moe_tail_fusion(
    *,
    max_num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    top_k: int,
    rms_eps: float,
    weight_bias: float,
    routed_scaling_factor: float,
    include_shared_expert: bool,
) -> int:
    """Build the fused tail for a layer, returning its token capacity or 0.

    A layer calls this at construction to decide whether to ask its experts to
    defer, and sets what comes back on ``FusedMoEConfig``:
    ``defer_moe_finalize`` when it is positive and
    ``defer_moe_finalize_max_num_tokens`` to the value, so the per-batch gate
    stays ``FusedMoEConfig.should_defer_moe_finalize`` on the producer side.

    Building here rather than at forward time is what makes that possible: the
    mnnvl CuTe DSL workspace is compiled around ``top_k``, ``rms_eps``,
    ``weight_bias``, ``routed_scaling_factor`` and whether a shared expert is
    folded in -- the kernel rejects per-call values that disagree -- and it
    takes a collective over the TP group, which the consumer's custom op cannot
    run. 0 means no fused tail on this deployment, and the experts finalize as
    they always did.
    """
    if dtype not in _FI_SUPPORTED_DTYPES:
        logger.debug_once("MoE tail fusion off: no %s kernel", dtype)
        return 0
    if not moe_tail_fusion_available() or get_fi_ar_moe_finalize_workspace is None:
        return 0
    workspace = get_fi_ar_moe_finalize_workspace(
        world_size=get_tensor_model_parallel_world_size(),
        rank=get_tensor_model_parallel_rank(),
        max_token_num=max_num_tokens,
        hidden_dim=hidden_size,
        dtype=dtype,
        group=get_tp_group().cpu_group,
        top_k=top_k,
        rms_eps=rms_eps,
        routed_scaling_factor=routed_scaling_factor,
        weight_bias=weight_bias,
        include_shared_expert=include_shared_expert,
    )
    if workspace is None:
        return 0
    if not workspace.is_buffer_size_sufficient(
        get_tensor_model_parallel_world_size(), max_num_tokens, hidden_size, dtype
    ):
        logger.warning_once(
            "MoE tail fusion off: the workspace holds fewer than "
            "max_num_tokens=%d tokens.",
            max_num_tokens,
        )
        return 0
    return max_num_tokens


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
) -> list[torch.Tensor]:
    workspace = (
        None
        if existing_fi_ar_moe_finalize_workspace is None
        else existing_fi_ar_moe_finalize_workspace()
    )
    assert workspace is not None, (
        "no MoE finalize fusion workspace: an unfinalized MoE output only "
        "reaches here when the consuming layer built one at construction, so "
        "it was destroyed in between."
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
    )
    return norm_out, residual_out
