# SPDX-License-Identifier: Apache-2.0
"""Fused MoE kernel."""
from typing import Optional

import torch

from vllm import _custom_ops as ops


#TODO make the grouped gemm kernel consistent with scaled gemm kernel
def cutlass_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    ab_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    ab_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
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
    - w1 (torch.Tensor): The first set of expert weights.
        Shape: [num_experts, K, 2N] (the weights are passed transposed)
    - w2 (torch.Tensor): The second set of expert weights.
        Shape: [num_experts, N, K] (the weights are passed transposed)
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - topk_ids (torch.Tensor): The token->expert mapping.
    - ab_strides1 (torch.Tensor): The input and weights strides of the first
        grouped gemm.
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    - ab_strides2 (torch.Tensor): The input and weights strides of the second
        grouped gemm.
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    - w1_scale (Optional[torch.Tensor]): The optional fp32 scale
        to dequantize w1.
        Shape: [num_experts] or [num_experts, 2N]
    - w2_scale (Optional[torch.Tensor]): The optional fp32 scale
        to dequantize w2.
        Shape: [num_experts] or [num_experts, K]
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [M]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [M]
    - expert_map (Optional[torch.Tensor]): In the case of Expert parallel,
        every Rank is responsible for some experts. expert_map is a mapping
        from global expert-id to local expert-id. When expert_map[i] is -1,
        it means that this Rank is not responsible for global expert-id i.
    - apply_router_weight_on_input (bool): When true, the topk weights are
        applied directly on the inputs. This is only applicable when topk is 1.

    Returns:

    - torch.Tensor: The output tensor after applying the MoE layer.
    """

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert a.shape[1] == w1.shape[1], "Hidden size mismatch w1"
    assert w1.shape[2] == w2.shape[1] * 2, "Hidden size mismatch w2"
    assert w1.shape[0] == w2.shape[0], "Expert number mismatch"
    assert ab_strides1.shape[0] == w1.shape[0], \
            "AB Strides 1 expert number mismatch"
    assert c_strides1.shape[0] == w1.shape[0], \
           "C Strides 1 expert number mismatch"
    assert ab_strides2.shape[0] == w2.shape[0], \
           "AB Strides 2 expert number  mismatch"
    assert c_strides2.shape[0] == w2.shape[0], \
           "C Strides 2 expert number mismatch"

    assert a.dtype in [torch.half, torch.bfloat16], "Invalid input dtype"
    assert w1.dtype in [torch.float8_e4m3fn, torch.half,torch.bfloat16], \
           "Invalid weight type"
    assert w1.dtype == w2.dtype, "Weights type mismatch"

    if w1.dtype in [torch.half, torch.bfloat16]:
        assert w1.dtype == a.dtype, \
               "Unquantized input and weights type mismatch"
        assert w1_scale is None and w2_scale is None \
               and a1_scale is None and a2_scale is None, \
               "Received scales for unquantized input type"
    elif w1.dtype == torch.float8_e4m3fn:
        assert w1_scale is not None and w2_scale is not None, \
               "Missing scales for quantized input type"

    if w1_scale is not None:
        assert w1_scale.dim() == 1 or w1_scale.shape[1] == 1 \
               or w1_scale.shape[1] == w1.shape[2], "W1 scale shape mismatch"
        assert w1.shape[0] == w1_scale.shape[0], \
               "w1 scales expert number mismatch"
    if w2_scale is not None:
        assert w2_scale.dim() == 1 or w2_scale.shape[1] == 1 \
               or w2_scale.shape[1] == w2.shape[2], "W2 scale shape mismatch"
        assert w2.shape[0] == w2_scale.shape[0], \
               "w2 scales expert number mismatch"
    if a1_scale is not None:
        assert a1_scale.dim() == 0 or a1_scale.shape[0] == 1 \
               or a1_scale.shape[0] == a.shape[0], "Input scale shape mismatch"
    if a2_scale is not None:
        assert a1_scale is None or a2_scale.shape == a1_scale.shape, \
               "Intermediate scale shape mismatch"

    is_quantized = w1.dtype == torch.float8_e4m3fn

    device = a.device
    num_experts = w1.size(0)
    m = a.size(0)
    k = w1.size(1)
    n = w2.size(1)
    topk = topk_ids.size(1)
    out_dtype = a.dtype

    local_topk_ids = topk_ids
    if expert_map is not None:
        "Translate info from expert_map to topk_ids"
        local_topk_ids = torch.where(expert_map[topk_ids] != -1,
                                     expert_map[topk_ids], -1)

    if apply_router_weight_on_input:
        assert topk == 1, \
            "apply_router_weight_on_input is only implemented for topk=1"
        # TODO: this only works for topK=1, will need to update for topK>1
        a = a * topk_weights.to(out_dtype)

    if is_quantized:
        per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
            a2_scale.numel() != 1 if a2_scale is not None else False)
        a, a1_scale = ops.scaled_fp8_quant(
            a, a1_scale, use_per_token_if_dynamic=per_act_token)

    expert_offsets = torch.empty((num_experts + 1),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes1 = torch.empty((num_experts, 3),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes2 = torch.empty((num_experts, 3),
                                 dtype=torch.int32,
                                 device=device)

    a_map = torch.zeros((local_topk_ids.numel()),
                        dtype=torch.int32,
                        device=device)
    c_map = torch.zeros((local_topk_ids.numel()),
                        dtype=torch.int32,
                        device=device)

    ops.get_cutlass_moe_mm_data(local_topk_ids, expert_offsets, problem_sizes1,
                                problem_sizes2, a_map, c_map, num_experts, n,
                                k)

    if is_quantized:
        rep_a = a.view(dtype=torch.uint8)[a_map].view(dtype=a.dtype)
        rep_a1_scales = a1_scale[a_map] if per_act_token else a1_scale
    else:
        rep_a = a[a_map]
        rep_a1_scales = None

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=out_dtype)
    c2 = torch.zeros((m * topk, k), device=device, dtype=out_dtype)

    ops.cutlass_moe_mm(c1, rep_a, w1, rep_a1_scales, w1_scale,
                       expert_offsets[:-1], problem_sizes1, ab_strides1,
                       ab_strides1, c_strides1)

    intermediate = torch.empty((m * topk, n), device=device, dtype=out_dtype)
    torch.ops._C.silu_and_mul(intermediate, c1)

    if is_quantized:
        rep_a = a.view(dtype=torch.uint8)[a_map].view(dtype=a.dtype)
        rep_a1_scales = a1_scale[a_map] if per_act_token else a1_scale
        intermediate, a2_scale = ops.scaled_fp8_quant(
            intermediate, a2_scale, use_per_token_if_dynamic=per_act_token)

    ops.cutlass_moe_mm(c2, intermediate, w2, a2_scale, w2_scale,
                       expert_offsets[:-1], problem_sizes2, ab_strides2,
                       ab_strides2, c_strides2)
    # Gather tokens
    c2 = c2[c_map].view(m, topk, k)
    if not apply_router_weight_on_input:
        c2 = c2 * topk_weights.view(m, topk, 1).to(out_dtype)
    return c2.sum(dim=1)
