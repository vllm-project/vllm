from typing import Callable, List, Optional
from dataclasses import dataclass, field

import torch

from vllm import _custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.fused_moe import get_config_qtype
from vllm.utils import direct_register_custom_op
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache, moe_kernel_quantize_input)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP)

from triton_kernels.matmul_ogs import matmul_ogs
from triton_kernels.routing import (routing, RoutingData, GatherIndx, ScatterIndx)

def forward_cuda_triton(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    # use_grouped_topk: bool = False,
    # num_expert_group: Optional[int] = None,
    # topk_group: Optional[int] = None,
    # custom_routing_function: Optional[Callable] = None,
    use_fp8_w8a8: bool = False,
    # use_int8_w8a8: bool = False,
    # use_int8_w8a16: bool = False,
    # use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    # w1_zp: Optional[torch.Tensor] = None,
    # w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    
    # feature check, TODO: move outside of the func
    # assert use_grouped_topk == False, "use_grouped_topk is not supported in new triton MoE kernel"
    # assert topk_group is None, "topk_group is not supported in new triton MoE kernel"
    # assert num_expert_group is None, "num_expert_group is not supported in new triton MoE kernel"
    # assert scoring_func == "softmax", "scoring_func is not supported in new triton MoE kernel"
    # assert e_score_correction_bias is None, "e_score_correction_bias is not supported in new triton MoE kernel"

    if not renormalize: 
        gating_output = torch.softmax(gating_output, dim=-1)
    routing_data, gather_idx, scatter_idx = routing(gating_output, topk, renormalize)

    return fused_experts_triton_exp(
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_idx,
        scatter_idx,
        inplace=inplace,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape
    )

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
                  per_channel_quant: bool = False,
                  global_num_experts: int = -1,
                  expert_map: Optional[torch.Tensor] = None,
                  w1_scale: Optional[torch.Tensor] = None,
                  w2_scale: Optional[torch.Tensor] = None,
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
    assert hidden_states.shape[-1] == w1.shape[-2], "hidden_states shape[-1] must be equal to w1 shape[-2]"
    assert w2.shape[-1] == w1.shape[1], "w2 shape[-1] must be equal to w1 shape[1]"

    # feature check
    # assert inplace == False, "Inplace is not supported in new triton MoE kernel"
    # assert use_fp8_w8a8 == False, "use_fp8_w8a8 is not supported in new triton MoE kernel"
    # assert use_int8_w8a8 == False, "use_int8_w8a8 is not supported in new triton MoE kernel"
    # assert use_int8_w8a16 == False, "use_int8_w8a16 is not supported in new triton MoE kernel"
    # assert use_int4_w4a16 == False, "use_int4_w4a16 is not supported in new triton MoE kernel"
    # assert per_channel_quant == False, "per_channel_quant is not supported in new triton MoE kernel"
    # # assert global_num_experts == -1, "global_num_experts is not supported in new triton MoE kernel"
    # assert expert_map is None, "expert_map is not supported in new triton MoE kernel"
    # assert w1_scale is None, "w1_scale is not supported in new triton MoE kernel"
    # assert w2_scale is None, "w2_scale is not supported in new triton MoE kernel"
    # assert w1_zp is None, "w1_zp is not supported in new triton MoE kernel"
    # assert w2_zp is None, "w2_zp is not supported in new triton MoE kernel"
    # assert a1_scale is None, "a1_scale is not supported in new triton MoE kernel"
    # assert a2_scale is None, "a2_scale is not supported in new triton MoE kernel"
    # assert block_shape is None, "block_shape is not supported in new triton MoE kernel"
    # assert allow_deep_gemm == False, "allow_deep_gemm is not supported in new triton MoE kernel"

    M, K = hidden_states.shape
    N = w1.shape[2]
    n_expts_tot = routing_data.n_expts_tot
    n_expts_act = routing_data.n_expts_act
    dtype = hidden_states.dtype
    
    # consistent with default implementation
    intermediate_cache2 = torch.empty((M * n_expts_act, N // 2),
                                      device="cuda",
                                      dtype=dtype)    
    
    intermediate_cache1 = matmul_ogs(hidden_states, w1, None, routing_data, gather_indx=gather_indx, gammas=routing_data.gate_scal if apply_router_weight_on_input else None)
    
    if activation == "silu":
        torch.ops._C.silu_and_mul(intermediate_cache2,
                                    intermediate_cache1.view(-1, N))
    elif activation == "gelu":
        torch.ops._C.gelu_and_mul(intermediate_cache2,
                                    intermediate_cache1.view(-1, N))
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    intermediate_cache3 = matmul_ogs(intermediate_cache2, w2, None, routing_data, scatter_indx=scatter_indx, gammas=None if apply_router_weight_on_input else routing_data.gate_scal)

    return intermediate_cache3


def forward_cuda_triton_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    # use_grouped_topk: bool = False,
    # num_expert_group: Optional[int] = None,
    # topk_group: Optional[int] = None,
    # custom_routing_function: Optional[Callable] = None,
    use_fp8_w8a8: bool = False,
    # use_int8_w8a8: bool = False,
    # use_int8_w8a16: bool = False,
    # use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    # w1_zp: Optional[torch.Tensor] = None,
    # w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)

direct_register_custom_op(
    op_name="forward_cuda_triton",
    op_func=forward_cuda_triton,
    mutates_args=[],
    fake_impl=forward_cuda_triton_fake,
    tags=(torch.Tag.needs_fixed_stride_order, ),
)
 