# SPDX-License-Identifier: Apache-2.0
"""Fused MoE kernel."""
from typing import Optional

import torch

from vllm import _custom_ops as ops


#TODO make the grouped gemm kernel consistent with scaled gemm kernel
def cutlass_moe_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids_: torch.Tensor,
    ab_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    ab_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.half,
    expert_map: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """
    This function computes a a8w8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    grouped gemm.

    Parameters:
    - a (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1_q (torch.Tensor): The first set of fp8-quantized expert weights.
        Shape: [num_experts, K, 2N] (the weights are passed transposed)
    - w2_q (torch.Tensor): The second set of fp8-quantized expert weights.
        Shape: [num_experts, N, K] (the weights are passed transposed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts] or [num_experts, 2N]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts] or [num_experts, K]
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - ab_strides1 (torch.Tensor): The input and weights strides of the first
        grouped gemm.
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    - ab_strides2 (torch.Tensor): The input and weights strides of the second
        grouped gemm.
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [M]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [M]
    - out_dtype (torch.Tensor): The output tensor type.
    - expert_map (Optional[torch.Tensor]): In the case of Expert parallel,
        every Rank is responsible for a subset of experts. expert_map is a
        mapping from global expert-id to local expert-id. When expert_map[i]
        is -1, it means that this Rank is not responsible for global
        expert-id i.
    - apply_router_weight_on_input (bool): When true, the topk weights are
        applied directly on the inputs. This is only applicable when topk is 1.

    Returns:
    - torch.Tensor: The fp16 output tensor after applying the MoE layer.
    """

    assert topk_weights.shape == topk_ids_.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.float8_e4m3fn
    assert w2_q.dtype == torch.float8_e4m3fn
    assert a.shape[1] == w1_q.shape[1], "Hidden size mismatch w1"
    assert w1_q.shape[2] == w2_q.shape[1] * 2, "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert a1_scale is None or a1_scale.dim(
    ) == 0 or a1_scale.shape[0] == 1 or a1_scale.shape[0] == a.shape[
        0], "Input scale shape mismatch"
    assert w1_scale.dim() == 1 or w1_scale.shape[1] == 1 or w1_scale.shape[
        1] == w1_q.shape[2], "W1 scale shape mismatch"
    assert w2_scale.dim() == 1 or w2_scale.shape[1] == 1 or w2_scale.shape[
        1] == w2_q.shape[2], "W2 scale shape mismatch"
    assert w1_q.shape[0] == w2_q.shape[0], "Weights expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[
        0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[
        0], "w2 scales expert number mismatch"
    assert a2_scale is None or a1_scale is None or a2_scale.shape == a1_scale.shape, "Intermediate scale shape mismatch"  # noqa: E501
    assert ab_strides1.shape[0] == w1_q.shape[
        0], "AB Strides 1 expert number mismatch"
    assert c_strides1.shape[0] == w1_q.shape[
        0], "C Strides 1 expert number mismatch"
    assert ab_strides2.shape[0] == w2_q.shape[
        0], "AB Strides 2 expert number  mismatch"
    assert c_strides2.shape[0] == w2_q.shape[
        0], "C Strides 2 expert number mismatch"
    assert out_dtype in [torch.half, torch.bfloat16], "Invalid output dtype"

    num_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(1)
    n = w2_q.size(1)

    local_topk_ids = topk_ids_
    if expert_map is not None:
        "Translate info from expert_map to topk_ids"
        local_topk_ids = torch.where(expert_map[topk_ids_] != -1,
                                     expert_map[topk_ids_], -1)

    topk = local_topk_ids.size(1)

    per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
        a2_scale.numel() != 1 if a2_scale is not None else False)
    if apply_router_weight_on_input:
        assert topk == 1, \
            "apply_router_weight_on_input is only implemented for topk=1"
        # TODO: this only works for topK=1, will need to update for topK>1
        a = a * topk_weights.to(out_dtype)

    a_q, a1_scale = ops.scaled_fp8_quant(
        a, a1_scale, use_per_token_if_dynamic=per_act_token)
    device = a_q.device

    expert_offsets = torch.empty((num_experts + 1),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes1 = torch.empty((num_experts, 3),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes2 = torch.empty((num_experts, 3),
                                 dtype=torch.int32,
                                 device=device)

    a_map_initializer = torch.empty
    c2_initializer = torch.empty
    if expert_map is not None:
        # With expert_map each Rank processes only a subset of experts. As
        # a result not all of a_map and c2 tensors are filled. We fill it
        # zeros for correctness.
        a_map_initializer = torch.zeros
        c2_initializer = torch.zeros

    a_map = a_map_initializer((local_topk_ids.numel()),
                              dtype=torch.int32,
                              device=device)
    c_map = torch.empty((local_topk_ids.numel()),
                        dtype=torch.int32,
                        device=device)

    ops.get_cutlass_moe_mm_data(local_topk_ids, expert_offsets, problem_sizes1,
                                problem_sizes2, a_map, c_map, num_experts, n,
                                k)

    rep_a_q = a_q.view(dtype=torch.uint8)[a_map].view(dtype=a_q.dtype)
    rep_a1_scales = a1_scale[a_map] if per_act_token else a1_scale

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=out_dtype)
    c2 = c2_initializer((m * topk, k), device=device, dtype=out_dtype)

    ops.cutlass_moe_mm(c1, rep_a_q, w1_q, rep_a1_scales, w1_scale,
                       expert_offsets[:-1], problem_sizes1, ab_strides1,
                       ab_strides1, c_strides1)

    intermediate = torch.empty((m * topk, n), device=device, dtype=out_dtype)
    torch.ops._C.silu_and_mul(intermediate, c1)

    intemediate_q, a2_scale = ops.scaled_fp8_quant(
        intermediate, a2_scale, use_per_token_if_dynamic=per_act_token)

    ops.cutlass_moe_mm(c2, intemediate_q, w2_q, a2_scale, w2_scale,
                       expert_offsets[:-1], problem_sizes2, ab_strides2,
                       ab_strides2, c_strides2)
    # Gather tokens
    c2 = c2[c_map].view(m, topk, k)
    if not apply_router_weight_on_input:
        c2 = c2 * topk_weights.view(m, topk, 1).to(out_dtype)
    return c2.sum(dim=1)

from vllm.scalar_type import scalar_types

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1fn.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

def mock_group_gemm(rep_a, a_gs, 
                     w1_fp4,w1_blockscale, w1_tensorscale,
                     problem_sizes, expert_offsets,
                     m,n,k,e,topk,device, out_shape, out_dtype):
    c1 = torch.empty(out_shape, device=device, dtype=out_dtype)
    for eIdx in range(0, e):
        expert_num_tokens = problem_sizes[eIdx,0]
        eL = expert_offsets[eIdx]
        eH = expert_offsets[eIdx+1]
        assert(eH-eL == expert_num_tokens)
        if not expert_num_tokens:
            continue
        mat_a = rep_a[eL:eH].reshape(-1, rep_a.shape[-1])
        if a_gs[eIdx]:
            mat_a_gs = a_gs[eIdx]
        else:
            expert_amax = torch.abs(mat_a).amax().to(torch.float32)
            mat_a_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / expert_amax
        mat_a_q, mat_a_bs = ops.scaled_fp4_quant(
                mat_a, mat_a_gs)
        mat_b = w1_fp4[eIdx]
        mat_b_bs = w1_blockscale[eIdx]
        alpha = 1 / (mat_a_gs * w1_tensorscale[eIdx])
        out = ops.cutlass_scaled_fp4_mm(mat_a_q, mat_b, mat_a_bs,
                                    mat_b_bs, alpha,out_dtype)
        assert(out.shape[0] == expert_num_tokens and out.shape[1] == w1_fp4.shape[1])
        c1[eL:eL + expert_num_tokens] = out
    return c1


def cutlass_fp4_moe(
    a: torch.Tensor,
    w1_fp4: torch.Tensor, 
    w1_blockscale: torch.Tensor,
    w1_tensorscale: torch.Tensor,
    w2_fp4: torch.Tensor, 
    w2_blockscale: torch.Tensor,
    w2_tensorscale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    m:int, n:int, k:int, e:int,
    gemm1_AB_strides: Optional[torch.Tensor] = None,
    gemm1_C_strides: Optional[torch.Tensor] = None,
    gemm2_AB_strides: Optional[torch.Tensor] = None,
    gemm2_C_strides: Optional[torch.Tensor] = None,
    gemm1_a_scale: Optional[torch.Tensor] = None,
    gemm2_a_scale: Optional[torch.Tensor] = None,
    ): 
    """
  
    
    """
    num_topk = topk_ids.shape[1]
    device = a.device
    out_dtype = a.dtype
    
    expert_offsets = torch.empty((e+ 1),
                                 dtype=torch.int32,
                                 device=device)
    # Problem size:  (num_experts, (m,2n,k)) 
    problem_sizes1 = torch.empty((e, 3),
                                 dtype=torch.int32,
                                 device=device)
    # Problem size:  (num_experts, (m,n,k)) 
    problem_sizes2 = torch.empty((e, 3),
                                 dtype=torch.int32,
                                 device=device)

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    ops.get_cutlass_moe_mm_data(topk_ids, expert_offsets, problem_sizes1,
                                problem_sizes2, a_map, c_map, e, n, k)
            
    rep_a = a[a_map]
    a_gs = torch.zeros((e,),dtype=torch.float32, device=device)
    FLOAT4_E2M1_MAX = scalar_types.float4_e2m1fn.max()
    FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
    m_topk, two_n = rep_a.shape
    assert m_topk == m * num_topk
    # rep_a = rep_a.reshape(num_topk,m, two_n)

    if gemm1_a_scale is not None:
        a_gs = gemm1_a_scale
    # First gemm is up projection. We the contracting dimension is 
    # smaller than n
    # X: [m * num_topk, intermediate_partition] 
    # Y: [E, hidden_size, intermediate_partitions]: hidden_size = hidden_size_gate + hidden_size_up
    # Y_FP4 = [E, hidden_size, intermediate_partition_size // 2]
    c1_shape =  (m * num_topk, w1_fp4.shape[1])
    
    c1 = mock_group_gemm(rep_a,a_gs, w1_fp4,w1_blockscale, w1_tensorscale,
                         problem_sizes1, expert_offsets,
                         m,n,k,e,num_topk,device,c1_shape,out_dtype)
    # hidden size dimension is split to one half sized tensor. 

    intermediate = torch.empty((m * num_topk, w1_fp4.shape[1] // 2),
                                device=device, dtype=out_dtype)
    
    torch.ops._C.silu_and_mul(intermediate, c1)

    # rep_int = intermediate.reshape(num_topk,m,k)
    int_gs = torch.zeros((e,),dtype=torch.float32, device=device)
    if gemm2_a_scale is not None:
       int_gs = gemm2_a_scale
    c2_shape = (m * num_topk, w2_fp4.shape[1])
    c2 = mock_group_gemm(intermediate, int_gs,
                         w2_fp4,w2_blockscale,
                         w2_tensorscale,
                         problem_sizes2,
                         expert_offsets,
                         m,n,k,e,num_topk,
                         device,c2_shape, out_dtype)
    
    out =  (c2[c_map].view(m, num_topk, k) *
           topk_weights.view(m, num_topk, 1).half()).sum(dim=1)
    
    return out
    
