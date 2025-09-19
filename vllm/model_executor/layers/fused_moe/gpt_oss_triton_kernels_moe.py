# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG, FusedMoEQuantConfig)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate)
from vllm.utils import has_triton_kernels

logger = init_logger(__name__)

if has_triton_kernels():
    try:
        import triton_kernels.swiglu
        from triton_kernels.matmul_ogs import (FnSpecs, FusedActivation,
                                               matmul_ogs)
        from triton_kernels.routing import routing
    except (ModuleNotFoundError, AttributeError) as e:
        logger.error(
            "Failed to import Triton kernels. Please make sure your triton "
            "version is compatible. Error: %s", e)


def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: str = "silu",
    quant_config: Optional[FusedMoEQuantConfig] = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    routing_data, gather_idx, scatter_idx = routing(gating_output,
                                                    topk,
                                                    sm_first=not renormalize)

    return triton_kernel_fused_experts(
        None,
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_idx,
        scatter_idx,
        activation=activation,
        quant_config=quant_config,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map)


# This is a triton implementation of the fused_experts function
def triton_kernel_fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    routing_data,  # RoutingData
    gather_indx,  # GatherIndx
    scatter_indx,  # ScatterIndx
    activation: str = "silu",
    quant_config: Optional[FusedMoEQuantConfig] = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    a1q_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if quant_config is None:
        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

    # type check, uint8 means mxfp4
    assert hidden_states.dtype == torch.bfloat16
    assert (quant_config.w1_bias is None
            or quant_config.w1_bias.dtype == torch.float32)
    assert (quant_config.w2_bias is None
            or quant_config.w2_bias.dtype == torch.float32)

    # Shape check, only check non-mxfp4
    assert hidden_states.shape[-1] == w1.shape[-2]
    assert w2.shape[-1] == w1.shape[1]

    E, _, N = w1.shape

    if global_num_experts == -1:
        global_num_experts = E

    act = FusedActivation(
        FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")),
        (swiglu_alpha, swiglu_limit), 2)
    gammas = routing_data.gate_scal if routing_data else None

    intermediate_cache1 = matmul_ogs(
        hidden_states,
        w1,
        quant_config.w1_bias,
        routing_data,
        gather_indx=gather_indx,
        precision_config=quant_config.w1_precision,
        gammas=gammas if apply_router_weight_on_input else None,
        fused_activation=act)

    intermediate_cache3 = matmul_ogs(
        intermediate_cache1,
        w2,
        quant_config.w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        precision_config=quant_config.w2_precision,
        gammas=None if apply_router_weight_on_input else gammas,
        y=output_tensor,
    )
    return intermediate_cache3


class BatchedOAITritonExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(
        self,
        max_num_tokens: int,
        num_dispatchers: int,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(quant_config)
        self.max_num_tokens = max_num_tokens
        self.num_dispatchers = num_dispatchers

    @property
    def activation_formats(
        self
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (mk.FusedMoEActivationFormat.BatchedExperts,
                mk.FusedMoEActivationFormat.BatchedExperts)

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(
        self, a: torch.Tensor, aq: torch.Tensor, M: int, N: int, K: int,
        topk: int, global_num_experts: int, local_num_experts: int,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata]
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        # workspace are allocated inside the kernel
        assert a.dim() == 2
        num_dp = self.num_dispatchers
        num_experts = local_num_experts
        max_num_tokens = self.max_num_tokens
        workspace2 = (0, 0, 0)
        output = (num_experts, max_num_tokens * num_dp, N)
        return (output, workspace2, output, a.dtype)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
    ):
        return triton_kernel_fused_experts(
            output,
            hidden_states,
            w1,
            w2,
            routing_data=None,
            gather_indx=None,
            scatter_indx=None,
            activation=activation,
            quant_config=self.quant_config,
            apply_router_weight_on_input=False,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=a1q_scale)
