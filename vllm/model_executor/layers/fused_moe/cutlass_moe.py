# SPDX-License-Identifier: Apache-2.0
""" CUTLASS based Fused MoE kernels."""
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

def cutlass_moe_fp4(
    a: torch.Tensor,
    a1_gscale: torch.Tensor,
    w1_fp4: torch.Tensor, 
    w1_blockscale: torch.Tensor,
    w1_tensorscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    w2_fp4: torch.Tensor, 
    w2_blockscale: torch.Tensor,
    w2_tensorscale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    m:int, n:int, k:int, e:int,
    ): 
    """
    MoE implementation for FP4 Inputs
    
    # Gemm 1
    a: Input tensor: [m, k] (half/bfloat16)
    a1_gscale: Activation scale per expert: [e]  (float32)
    w1(gate up) (not an argument to cutlass_moe_fp4): [e, 2 * n, k]
    w1_fp4: [e, 2 * n, k // 2], dtype: torch.uint8 (stacked fp4: E2M1)
    (Note: `n` is the up projection output dim, `k` is the input dim in
     full precision)
    w1_blockscale: [e, 2 * n, k // block_size] (float8_e4m3)
                   (Block size = 16 for NVFP4)
    
    # Gemm 2
    a2_gscale: Activation scale per expert: [e]
    w2(down projection) (not an argument to cutlass_moe_fp4): [e, k, n]
    w2_fp4: [e, k, n // 2], dtype: torch.uint8 (stacked E2M1)
    w2_blockscale: [e, k, n // block_size], dtype: float8_e4m3
    
    topk_weights: [m, topk] dtype: float8
    topk_ids: [m, topk] dtype: float8
    
    m, n, k: Unquantized weight shapes, dtype: int
    e: number of experts, dtype: int

    assumes that topk < k < n to satisfy - up/down projection expectations.
    """
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_fp4.dtype == torch.uint8, "weight 1 must be uint8"
    assert w2_fp4.dtype == torch.uint8, "weight 2 must be uint8"
    assert (w1_fp4.ndim == 3 and w2_fp4.ndim == 3 
            and w1_blockscale.ndim == 3 and w2_blockscale.ndim == 3),(
                    "All Weights must be of rank 3 for cutlass_moe_fp4")
    m_a, k_a = a.shape
    e_w1, nx2_w1, half_k_w1 = w1_fp4.shape
    e_w2, k_w2, half_n_w2 = w2_fp4.shape

    assert (e_w1 == e_w2 and e_w1 == e), ("Number of experts must match",
                                          " between weights.")
    assert (k_a // 2 == half_k_w1 and k == k_w2), (
                        "Hidden size mismatch between a, w1 and w2")
    assert (nx2_w1 == n * 2 and half_n_w2 == n // 2), ("mismatch in "
                                                       "expected `n`")
    assert (m == m_a), "input shape mismatch"
    assert 2 * half_k_w1 == k_w2, "Hidden size mismatch w2 and w1"
    assert a.dtype in [torch.half, torch.bfloat16], "Invalid input dtype"
    assert (topk_weights.shape[0] == m and topk_ids.shape[0] == m), (
            "topk must be provided for each row of a")
    
    out_dtype = a.dtype   
    num_topk = topk_ids.shape[1]
    device = a.device
    out_dtype = a.dtype

    # Step 1: Quantize a to fp4
    a_amax = torch.abs(a).amax().to(torch.float32)
    mat_a_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax
    a_fp4, a_blockscale = ops.scaled_fp4_quant(a, mat_a_gs)

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

    # Get the problem sizes separately because k, n change based on which gemm.
    # This passes the right metrics for problem sizes 1
    # TODO: @pavanimajety Modify the get_cutlass_moe_mm_data to take in w1_n, w2_n 
    # as two different parameters
    # problem shapes should have [m, n, k]
    ops.get_cutlass_moe_mm_data(topk_ids, expert_offsets, problem_sizes1,
                                problem_sizes2, a_map, c_map, 
                                num_experts=w1_fp4.shape[0],
                                n=n, 
                                k=w1_fp4.shape[2])
    # Fix problem_sizes2 for nvfp4 shapes to match w2_fp4
    problem_sizes2[:,1] *= 2
    problem_sizes2[:,2] //= 2

    # Replicated scales 
    rep_a_fp4 = a_fp4[a_map]
    rep_a_blockscale = a_blockscale.view(dtype=torch.uint8)[a_map].view(
                                                     dtype=a_blockscale.dtype)
    assert rep_a_fp4.shape[0] == m * num_topk, "map was expanded incorrectly"
    
    
    # First gemm is up projection. We the contracting dimension is 
    # smaller than n
    # X: [m * num_topk, intermediate_partition] 
    # Y: [E, hidden_size, intermediate_partitions]: hidden_size = hidden_size_gate + hidden_size_up
    # Y_FP4 = [E, hidden_size, intermediate_partition_size // 2]
    
    c1_shape =  (m * num_topk, nx2_w1)
    c1 = torch.empty(c1_shape, device=device, dtype=out_dtype)
    a1_sf_layout = torch.empty((e, 5), device=device, dtype=torch.int)
    w1_sf_layout = torch.empty((e, 5), device=device, dtype=torch.int)
    c_strides1 = torch.full((e, ),
                            c1.stride(0),
                            device="cuda",
                            dtype=torch.int64)
    a_strides1 = torch.full((e, ),
                            rep_a_fp4.stride(0),
                            device="cuda",
                            dtype=torch.int64)
    b_strides1 = torch.full((e, ),
                            w1_fp4.stride(0),
                            device="cuda",
                            dtype=torch.int64)
    alphas = torch.empty((e, ),
                         dtype=torch.float32,
                         device=device)
    if a1_gscale is not None:
        alphas = 1 / (a1_gscale * w1_tensorscale)
    else:
        alphas = 1 / (mat_a_gs * w1_tensorscale)
    print(f"{problem_sizes1=}")
    ops.cutlass_fp4_moe_mm(c1, rep_a_fp4, w1_fp4, rep_a_blockscale,
                        w1_blockscale, alphas, a_strides1, b_strides1,
                        c_strides1,a1_sf_layout, w1_sf_layout, problem_sizes1,
                        expert_offsets[:-1])
    print(f"{c1.to(dtype=a.dtype)=}")
    return
    # hidden size dimension is split to one half sized tensor. 
    intermediate = torch.empty((m * num_topk, w1_fp4.shape[1] // 2),
                                 device=device, dtype=out_dtype)
    
    torch.ops._C.silu_and_mul(intermediate, c1)

    print(f"{intermediate}")

    intermediate_amax = torch.abs(intermediate).amax().to(torch.float32)
    mat_int_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / intermediate_amax 
    int_fp4, int_blockscale = ops.scaled_fp4_quant(intermediate, mat_int_gs)
    alphas_2 = torch.empty((e, ),
                         dtype=torch.float32,
                         device=device)

    if a2_gscale is not None:
        alphas_2 = 1/ (a2_gscale * w2_tensorscale)
    else:
        alphas_2 = 1/ (mat_int_gs * w2_tensorscale)

    c2_shape = (m * num_topk, w2_fp4.shape[1])
    c2 = torch.empty(c2_shape, device=device, dtype=out_dtype)
    c_strides2 = torch.full((e, ),
                            c2.stride(0),
                            device="cuda",
                            dtype=torch.int64)
    a_strides2 = torch.full((e, ),
                            int_fp4.stride(0),
                            device="cuda",
                            dtype=torch.int64)
    b_strides2 = torch.full((e, ),
                            w2_fp4.stride(0),
                            device="cuda",
                            dtype=torch.int64)
    a2_sf_layout = torch.empty((e, 5), device=device, dtype=torch.int)
    w2_sf_layout = torch.empty((e, 5), device=device, dtype=torch.int)
    
    ops.cutlass_fp4_moe_mm(c2, int_fp4, w2_fp4, int_blockscale,
                        w2_blockscale, alphas_2, a_strides2, b_strides2,
                        c_strides2,a2_sf_layout, w2_sf_layout, problem_sizes2,
                        expert_offsets[:-1])
   
    print(f"{c2.to(dtype=a.dtype)=}")
    out =  (c2[c_map].view(m, num_topk, k) *
            topk_weights.view(m, num_topk, 1).half()).sum(dim=1)
    
    print(f"{out.to(dtype=a.dtype)=}")
    return out
    
