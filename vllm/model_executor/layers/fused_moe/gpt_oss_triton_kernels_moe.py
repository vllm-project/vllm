# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any, Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate)
from vllm.model_executor.layers.fused_moe.utils import extract_required_args
from vllm.utils import has_triton_kernels

logger = init_logger(__name__)

if has_triton_kernels():
    try:
        import triton_kernels.swiglu
        from triton_kernels.matmul_ogs import (FnSpecs, FusedActivation,
                                               matmul_ogs)
        from triton_kernels.routing import routing
    except ModuleNotFoundError:
        logger.error(
            "Failed to import Triton kernels. Please make sure your triton "
            "version is compatible.")

if TYPE_CHECKING:
    from triton_kernels.matmul_ogs import PrecisionConfig


def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    w1_precision: Optional["PrecisionConfig"] = None,
    w2_precision: Optional["PrecisionConfig"] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
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
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        w1_precision=w1_precision,
        w2_precision=w2_precision,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape)


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
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    w1_precision: Optional["PrecisionConfig"] = None,
    w2_precision: Optional["PrecisionConfig"] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:

    # type check, uint8 means mxfp4
    assert hidden_states.dtype == torch.bfloat16
    assert w1_bias is None or w1_bias.dtype == torch.float32
    assert w2_bias is None or w2_bias.dtype == torch.float32

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
        w1_bias,
        routing_data,
        gather_indx=gather_indx,
        precision_config=w1_precision,
        gammas=gammas if apply_router_weight_on_input else None,
        fused_activation=act)

    intermediate_cache3 = matmul_ogs(
        intermediate_cache1,
        w2,
        w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        precision_config=w2_precision,
        gammas=None if apply_router_weight_on_input else gammas,
        y=output_tensor,
    )
    return intermediate_cache3


class BatchedOAITritonExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(
        self,
        quant_config,
        max_num_tokens: int,
        num_dispatchers: int,
        w1_precision: "PrecisionConfig",
        w2_precision: "PrecisionConfig",
    ):
        super().__init__(quant_config)
        self.max_num_tokens = max_num_tokens
        self.num_dispatchers = num_dispatchers
        self.w1_precision = w1_precision
        self.w2_precision = w2_precision

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
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ):
        w1_bias, w2_bias = (extract_required_args(extra_expert_args,
                                                  ["w1_bias", "w2_bias"]))

        return triton_kernel_fused_experts(
            output,
            hidden_states,
            w1,
            w2,
            None,
            None,
            None,
            activation=activation,
            apply_router_weight_on_input=False,
            use_fp8_w8a8=False,
            per_channel_quant=False,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            w1_precision=self.w1_precision,
            w2_precision=self.w2_precision,
            a1_scale=a1q_scale,
            a2_scale=a2_scale)
