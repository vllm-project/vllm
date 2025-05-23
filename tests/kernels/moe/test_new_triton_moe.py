from typing import Callable, List, Optional
import pytest
from dataclasses import dataclass, fields

import torch
import triton
import triton.language as tl

from triton_kernels.matmul_ogs import matmul_ogs
from triton_kernels.routing import (routing, routing_torch, RoutingData, GatherIndx, ScatterIndx)
from triton_kernels.testing import assert_close

from vllm.model_executor.layers.fused_moe.fused_moe import ( fused_experts )
from vllm.model_executor.layers.fused_moe import FusedMoE

def forward_cuda_ref(x, w1, w2,
                    use_grouped_topk: bool,
                    top_k: int,
                    router_logits: torch.Tensor,
                    renormalize: bool,
                    topk_group: Optional[int] = None,
                    num_expert_group: Optional[int] = None,
                    global_num_experts: int = -1,
                    expert_map: Optional[torch.Tensor] = None,
                    # custom_routing_function: Optional[Callable] = None,   // custom routing kernel is not supported due to pytorch ops constraint
                    scoring_func: str = "softmax",
                    e_score_correction_bias: Optional[torch.Tensor] = None,
                    apply_router_weight_on_input: bool = False,
                    activation: str = "silu") -> torch.Tensor:

    topk_weights, topk_ids = FusedMoE.select_experts(
        hidden_states=x,
        router_logits=router_logits,
        use_grouped_topk=use_grouped_topk,
        top_k=top_k,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        e_score_correction_bias=e_score_correction_bias)

    return fused_experts(
            hidden_states=x,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map)

# This is a triton implementation of the fused_experts function
def fused_experts_triton_exp(hidden_states: torch.Tensor,
                  w1: torch.Tensor,
                  w2: torch.Tensor,
                  routing_data: RoutingData,
                  gather_indx: GatherIndx,
                  scatter_indx: ScatterIndx,
                  inplace: bool = False,
                  activation: str = "silu",
                  apply_router_weight_on_input: bool = False,
                  use_fp8_w8a8: bool = False,
                  use_int8_w8a8: bool = False,
                  use_int8_w8a16: bool = False,
                  use_int4_w4a16: bool = False,
                  per_channel_quant: bool = False,
                  global_num_experts: int = -1,
                  expert_map: Optional[torch.Tensor] = None,
                  w1_scale: Optional[torch.Tensor] = None,
                  w2_scale: Optional[torch.Tensor] = None,
                  w1_zp: Optional[torch.Tensor] = None,
                  w2_zp: Optional[torch.Tensor] = None,
                  a1_scale: Optional[torch.Tensor] = None,
                  a2_scale: Optional[torch.Tensor] = None,
                  block_shape: Optional[List[int]] = None,
                  allow_deep_gemm: bool = False )-> torch.Tensor:
    
    # type check
    assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bfloat16"
    assert w1.dtype == torch.bfloat16, "w1 must be bfloat16"
    assert w2.dtype == torch.bfloat16, "w2 must be bfloat16"

    # Shape check
    assert hidden_states.ndim == 2, "hidden_states must be 2D"
    assert hidden_states.shape[-1] == w1.shape[-2], "hidden_states shape[-1] must be equal to w1 shape[-1]"
    assert w2.shape[-1] == w1.shape[1], "w1 shape[-1] must be equal to w2 shape[1]"

    # feature check
    assert inplace == False, "Inplace is not supported in new triton MoE kernel"
    assert apply_router_weight_on_input == False, "apply_router_weight_on_input is not supported in new triton MoE kernel"
    assert use_fp8_w8a8 == False, "use_fp8_w8a8 is not supported in new triton MoE kernel"
    assert use_int8_w8a8 == False, "use_int8_w8a8 is not supported in new triton MoE kernel"
    assert use_int8_w8a16 == False, "use_int8_w8a16 is not supported in new triton MoE kernel"
    assert use_int4_w4a16 == False, "use_int4_w4a16 is not supported in new triton MoE kernel"
    assert per_channel_quant == False, "per_channel_quant is not supported in new triton MoE kernel"
    assert global_num_experts == -1, "global_num_experts is not supported in new triton MoE kernel"
    assert expert_map is None, "expert_map is not supported in new triton MoE kernel"
    assert w1_scale is None, "w1_scale is not supported in new triton MoE kernel"
    assert w2_scale is None, "w2_scale is not supported in new triton MoE kernel"
    assert w1_zp is None, "w1_zp is not supported in new triton MoE kernel"
    assert w2_zp is None, "w2_zp is not supported in new triton MoE kernel"
    assert a1_scale is None, "a1_scale is not supported in new triton MoE kernel"
    assert a2_scale is None, "a2_scale is not supported in new triton MoE kernel"
    assert block_shape is None, "block_shape is not supported in new triton MoE kernel"
    assert allow_deep_gemm == False, "allow_deep_gemm is not supported in new triton MoE kernel"

    M, K = hidden_states.shape
    N = w1.shape[2]
    n_expts_tot = routing_data.n_expts_tot
    n_expts_act = routing_data.n_expts_act
    
    # consistent with default implementation
    intermediate_cache2 = torch.empty((M * n_expts_act, N // 2),
                                      device="cuda",
                                      dtype=torch.bfloat16)    
    
    intermediate_cache1 = matmul_ogs(hidden_states, w1, None, routing_data, gather_indx=gather_indx)
    
    if activation == "silu":
        torch.ops._C.silu_and_mul(intermediate_cache2,
                                    intermediate_cache1.view(-1, N))
    elif activation == "gelu":
        torch.ops._C.gelu_and_mul(intermediate_cache2,
                                    intermediate_cache1.view(-1, N))
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    intermediate_cache3 = matmul_ogs(intermediate_cache2, w2, None, routing_data, scatter_indx=scatter_indx, gammas=routing_data.gate_scal)

    return intermediate_cache3

def forward_cuda_exp(x, w1, w2,
                    use_grouped_topk: bool,
                    top_k: int,
                    router_logits: torch.Tensor,
                    renormalize: bool,
                    topk_group: Optional[int] = None,
                    num_expert_group: Optional[int] = None,
                    global_num_experts: int = -1,
                    expert_map: Optional[torch.Tensor] = None,
                    # custom_routing_function: Optional[Callable] = None,
                    scoring_func: str = "softmax",
                    e_score_correction_bias: Optional[torch.Tensor] = None,
                    apply_router_weight_on_input: bool = False,
                    activation: str = "silu"
                    ):
    # feature check
    # TODO: need to implement renormalize in actual triton kernel
    # assert renormalize == True, "renormalize can only be True in new triton MoE kernel, false not supported"
    assert use_grouped_topk == False, "use_grouped_topk is not supported in new triton MoE kernel"
    assert topk_group is None, "topk_group is not supported in new triton MoE kernel"
    assert num_expert_group is None, "num_expert_group is not supported in new triton MoE kernel"
    assert custom_routing_function is None, "custom_routing_function is not supported in new triton MoE kernel"
    assert scoring_func == "softmax", "scoring_func is not supported in new triton MoE kernel"
    assert e_score_correction_bias is None, "e_score_correction_bias is not supported in new triton MoE kernel"

    routing_data, gather_idx, scatter_idx = routing(router_logits, top_k, renormalize)

    return fused_experts_triton_exp(
        hidden_states=x,
        w1=w1,
        w2=w2,
        routing_data=routing_data,
        gather_indx=gather_idx,
        scatter_indx=scatter_idx,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map
    )

@dataclass
class Case:
    num_token: int
    inter_size: int
    K: int
    num_expts_tot: int
    num_expts_act: int

@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case)) for case in [
            Case(num_token=32, inter_size=512, K=32, num_expts_tot=128, num_expts_act=4),
            Case(num_token=16, inter_size=512, K=32, num_expts_tot=128, num_expts_act=4),
            Case(num_token=1024, inter_size=2048, K=32, num_expts_tot=128, num_expts_act=4),
        ]
    ],
)
def test_equiv(num_token, inter_size, K, num_expts_tot, num_expts_act):

    randbits = [torch.randperm(num_expts_tot) for _ in range(num_token)]
    x = [(-1)**i * ((16384 + ((i * 512) % 4096) + bits).to(torch.int16).view(torch.bfloat16)) for i, bits in enumerate(randbits)]
    exp_data = torch.stack(x).to(device="cuda")
    # exp_data = torch.randn((num_token, num_expts_tot), dtype=torch.bfloat16, device="cuda")

    # create input tensor
    x = torch.randn((num_token, K), dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn((num_expts_tot, inter_size, K), dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn((num_expts_tot, K, inter_size // 2), dtype=torch.bfloat16, device="cuda")
    
    exp_data_tri = exp_data.clone()
    x_tri = x.clone()
    w1_tri = w1.clone()
    w2_tri = w2.clone()
    w1_tri = w1_tri.transpose(-2, -1).contiguous()
    w2_tri = w2_tri.transpose(-2, -1).contiguous()

    out_triton = forward_cuda_exp(x_tri, w1_tri, w2_tri, False, num_expts_act, exp_data_tri, True)

    out_ref = forward_cuda_ref(x, w1, w2, False, num_expts_act, exp_data, True)
    assert_close(ref=out_ref, tri=out_triton)
    
