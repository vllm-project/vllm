# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch
import triton_kernels.swiglu
from triton_kernels.matmul_ogs import (FnSpecs, FusedActivation,
                                       PrecisionConfig, matmul_ogs)
from triton_kernels.routing import (GatherIndx, RoutingData, ScatterIndx,
                                    routing)


def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
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
    w1_precision: Optional[PrecisionConfig] = None,
    w2_precision: Optional[PrecisionConfig] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:

    routing_data, gather_idx, scatter_idx = routing(gating_output,
                                                    topk,
                                                    sm_first=not renormalize)

    return triton_kernel_fused_experts(
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
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    routing_data: RoutingData,
    gather_indx: GatherIndx,
    scatter_indx: ScatterIndx,
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
    w1_precision: Optional[PrecisionConfig] = None,
    w2_precision: Optional[PrecisionConfig] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:

    # type check, uint8 means mxfp4
    #TODO: fp8 x mxfp4 on blackwell
    assert hidden_states.dtype == torch.bfloat16
    assert w1.dtype in (torch.bfloat16, torch.uint8)
    assert w2.dtype in (torch.bfloat16, torch.uint8)
    assert w1_bias.dtype == torch.float32
    assert w2_bias.dtype == torch.float32

    # Shape check, only check non-mxfp4
    if w1.dtype != torch.uint8:
        assert hidden_states.ndim == 2
        assert hidden_states.shape[-1] == w1.shape[-2]
        assert w2.shape[-1] == w1.shape[1]

    M, K = hidden_states.shape
    E, _, N = w1.shape
    n_expts_act = routing_data.n_expts_act
    dtype = hidden_states.dtype

    if global_num_experts == -1:
        global_num_experts = E

    act = FusedActivation(
        FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")),
        (1.702, None), 2)

    intermediate_cache1 = matmul_ogs(hidden_states,
                                     w1,
                                     w1_bias,
                                     routing_data,
                                     gather_indx=gather_indx,
                                     precision_config=w1_precision,
                                     gammas=routing_data.gate_scal
                                     if apply_router_weight_on_input else None,
                                     fused_activation=act)

    intermediate_cache3 = matmul_ogs(
        intermediate_cache1,
        w2,
        w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        precision_config=w2_precision,
        gammas=None
        if apply_router_weight_on_input else routing_data.gate_scal)

    return intermediate_cache3
