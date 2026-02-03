# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

#
# Methods used by the oracle for kernel selection.
#


def _supports_current_device() -> bool:
    """Supports only Blackwell-family GPUs."""
    p = current_platform
    return p.is_cuda() and p.is_device_capability_family(100)


def _supports_no_act_and_mul() -> bool:
    """Does not support non-gated MoE (i.e. Nanotron-Mini)."""
    return False


def _supports_activation(activation: str) -> bool:
    """Supports silu activation only."""
    return activation in ["silu"]


def _supports_routing_method_bf16(
    routing_method: RoutingMethodType,
) -> bool:
    return routing_method in [
        RoutingMethodType.Default,
        RoutingMethodType.Renormalize,
        RoutingMethodType.DeepSeekV3,
        RoutingMethodType.Llama4,
        RoutingMethodType.RenormalizeNaive,
    ]


def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
    """Supports TRTLLM Kernel does not support EPLB."""
    return not moe_parallel_config.enable_eplb


def is_supported_config_trtllm_bf16(
    moe_config: FusedMoEConfig,
    activation_format: mk.FusedMoEActivationFormat,
) -> tuple[bool, str | None]:
    """
    This method mirrors mk.FusedMoEPermuteExpertsUnpermute.is_supported_config
    for BF16 unquantized kernels.
    """

    def _make_reason(reason: str) -> str:
        return f"kernel does not support {reason}"

    if not _supports_current_device():
        return False, _make_reason("current device")
    elif not (moe_config.is_act_and_mul or _supports_no_act_and_mul()):
        return False, _make_reason("no act_and_mul MLP layer")
    elif not _supports_activation(moe_config.activation):
        return False, _make_reason(f"{moe_config.activation} activation")
    elif not _supports_parallel_config(moe_config.moe_parallel_config):
        return False, _make_reason("parallel config")
    elif not _supports_routing_method_bf16(moe_config.routing_method):
        return False, _make_reason("routing method")
    elif activation_format != mk.FusedMoEActivationFormat.Standard:
        return False, _make_reason("activation format")

    return True, None


def flashinfer_fused_moe_bf16(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor | None,
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: int | None,
    topk_group: int | None,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routing_method_type: int,
    tune_max_num_tokens: int = 8192,
) -> torch.Tensor:
    from vllm.utils.flashinfer import flashinfer_trtllm_bf16_moe

    return flashinfer_trtllm_bf16_moe(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        gemm1_weights=gemm1_weights,
        gemm2_weights=gemm2_weights,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        intermediate_size=intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routing_method_type=routing_method_type,
        tune_max_num_tokens=tune_max_num_tokens,
    )


def flashinfer_fused_moe_bf16_fake(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor | None,
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: int | None,
    topk_group: int | None,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routing_method_type: int = RoutingMethodType.Renormalize,
    tune_max_num_tokens: int = 8192,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="flashinfer_fused_moe_bf16",
    op_func=flashinfer_fused_moe_bf16,
    fake_impl=flashinfer_fused_moe_bf16_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)
