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

from functools import lru_cache

import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import (
    get_node_count,
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
from vllm.utils.torch_utils import direct_register_custom_op

try:
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        flashinfer_comm,
        get_fi_ar_moe_finalize_workspace,
    )

    # One-shot only, so the Lamport payload is a hard ceiling on the batch.
    _MAX_COMM_SIZE = flashinfer_comm.trtllm_ar.MAX_COMM_SIZE
except (ImportError, AttributeError):
    flashinfer_comm = None  # type: ignore[assignment]
    get_fi_ar_moe_finalize_workspace = None  # type: ignore[assignment]
    _MAX_COMM_SIZE = 0

_FI_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

logger = init_logger(__name__)


@lru_cache
def moe_tail_fusion_max_tokens(hidden_size: int, dtype: torch.dtype) -> int:
    """Largest batch whose MoE tail this consumer can take, 0 if none can.

    A ceiling, not a veto: a deployment whose *maximum* batch overflows the
    fusion workspace still spends most of its steps well under it, so the
    producer checks this per forward. Batches above it finalize in the MoE
    kernel as they always did, which costs nothing over the unfused model.
    """
    from vllm.config.compilation import PassConfig

    if not envs.VLLM_ENABLE_MOE_TAIL_FUSION:
        return 0
    if flashinfer_comm is None:
        logger.debug_once("MoE tail fusion off: flashinfer.comm is unavailable")
        return 0
    if dtype not in _FI_SUPPORTED_DTYPES:
        logger.debug_once("MoE tail fusion off: unsupported dtype %s", dtype)
        return 0
    tp_size = get_tensor_model_parallel_world_size()
    # Without TP there is no all-reduce to fuse into; the pattern has no mnnvl
    # implementation and trtllm all-reduce is single-node only.
    if tp_size <= 1 or get_node_count() > 1:
        logger.debug_once(
            "MoE tail fusion off: needs single-node TP>1, got tp_size=%d nodes=%d",
            tp_size,
            get_node_count(),
        )
        return 0

    max_size_mb = PassConfig.default_fi_allreduce_fusion_max_size_mb().get(tp_size)
    if not max_size_mb:
        logger.debug_once(
            "MoE tail fusion off: no workspace size for tp_size=%d", tp_size
        )
        return 0
    element_size = torch.tensor([], dtype=dtype).element_size()
    row_bytes = hidden_size * element_size
    max_tokens = min(
        int(max_size_mb * 1024 * 1024) // row_bytes,
        _MAX_COMM_SIZE // (row_bytes * tp_size),
    )
    # Worth a line: which batches fuse is a deployment-shaped property, and a
    # ceiling under the running batch size means the fusion silently never runs.
    logger.debug_once(
        "MoE tail fusion on: batches up to %d tokens fuse (hidden=%d %s tp=%d); "
        "larger batches finalize in the MoE kernel",
        max_tokens,
        hidden_size,
        dtype,
        tp_size,
    )
    return max_tokens


def _finalize_workspace(num_tokens: int, hidden_size: int, dtype: torch.dtype):
    """The (globally cached) fusion workspace for a batch, or None."""
    if get_fi_ar_moe_finalize_workspace is None:
        return None
    max_token_num = moe_tail_fusion_max_tokens(hidden_size, dtype)
    if not max_token_num or not 0 < num_tokens <= max_token_num:
        return None
    tp_size = get_tensor_model_parallel_world_size()
    workspace = get_fi_ar_moe_finalize_workspace(
        world_size=tp_size,
        rank=get_tensor_model_parallel_rank(),
        max_token_num=max_token_num,
        hidden_dim=hidden_size,
        dtype=dtype,
        group=get_tp_group().cpu_group,
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

    The producer's gate: with no fused kernel for the batch, finalizing in the
    MoE kernel is what the contract asks for, and the unfinalized form never
    appears.
    """
    return _finalize_workspace(num_tokens, hidden_size, dtype) is not None


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
    num_tokens, hidden_size = residual.shape
    workspace = _finalize_workspace(num_tokens, hidden_size, residual.dtype)
    # An unfinalized output only exists for a batch `moe_tail_fusion_applies`
    # accepted, which is this same workspace.
    assert workspace is not None, "no MoE finalize fusion workspace for this batch"

    norm_out = torch.empty_like(residual)
    residual_out = torch.empty_like(residual)
    flashinfer_comm.trtllm_moe_finalize_allreduce_fusion(
        allreduce_in=gemm2_permuted,
        residual_in=residual,
        norm_weight=rms_gamma,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        norm_out=norm_out,
        residual_out=residual_out,
        quant_out=None,
        scale_out=None,
        workspace_ptrs=workspace.workspace_tensor,
        launch_with_pdl=True,
        world_rank=workspace.rank,
        world_size=workspace.world_size,
        eps=rms_eps,
        shared_expert_output=shared_output,
        expert_scale_factor=expert_weights,
        routed_scaling_factor=routed_scaling_factor,
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
