# SPDX-License-Identifier: Apache-2.0
"""Fused MoE kernel."""
from typing import Optional, Tuple

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.dispatch_combine import (
    StandardDispatchCombine)
from vllm.model_executor.layers.fused_moe.utils import _fp8_perm, _resize_cache


class CutlassExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(
        self,
        ab_strides1: torch.Tensor,
        c_strides1: torch.Tensor,
        ab_strides2: torch.Tensor,
        c_strides2: torch.Tensor,
        out_dtype: torch.dtype,
    ):
        super().__init__()
        self.ab_strides1 = ab_strides1
        self.c_strides1 = c_strides1
        self.ab_strides2 = ab_strides2
        self.c_strides2 = c_strides2
        self.out_dtype = out_dtype

    def workspace_shapes(self, a_dtype: torch.dtype, M: int, N: int, K: int,
                         topk: int,
                         num_experts: int) -> Tuple[int, int, torch.dtype]:
        # Note that K, N are transposed
        N, K = K, N
        workspace1 = M * topk * max(2 * N, K)
        workspace2 = M * topk * N
        return (workspace1, workspace2, self.out_dtype)

    def apply(
        self,
        a1q: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
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
    ) -> torch.Tensor:
        M = a1q.shape[0]
        _, N, K = w2.shape  # because w1 + w2 are transposed
        topk = topk_ids.shape[1]
        device = a1q.device

        assert w1.shape[1] == K
        assert global_num_experts != -1
        assert a1q_scale is not None

        per_act_token = a1q_scale.numel() != 1 if a1q_scale is not None else (
            a2_scale.numel() != 1 if a2_scale is not None else False)

        expert_offsets = torch.empty((global_num_experts + 1),
                                     dtype=torch.int32,
                                     device=device)
        problem_sizes1 = torch.empty((global_num_experts, 3),
                                     dtype=torch.int32,
                                     device=device)
        problem_sizes2 = torch.empty((global_num_experts, 3),
                                     dtype=torch.int32,
                                     device=device)

        a_map = torch.empty((topk_ids.numel()),
                            dtype=torch.int32,
                            device=device)
        c_map = torch.empty((topk_ids.numel()),
                            dtype=torch.int32,
                            device=device)

        ops.get_cutlass_moe_mm_data(topk_ids, expert_offsets, problem_sizes1,
                                    problem_sizes2, a_map, c_map,
                                    global_num_experts, N, K)

        a1q = _fp8_perm(a1q, a_map)
        a1q_scale = a1q_scale[a_map] if per_act_token else a1q_scale

        c1 = _resize_cache(workspace13, (M * topk, N * 2))
        c2 = _resize_cache(workspace2, (M * topk, N))
        c3 = _resize_cache(workspace13, (M * topk, K))

        ops.cutlass_moe_mm(c1, a1q, w1, a1q_scale, w1_scale,
                           expert_offsets[:-1], problem_sizes1,
                           self.ab_strides1, self.ab_strides1, self.c_strides1)

        if activation == "silu":
            torch.ops._C.silu_and_mul(c2, c1)
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(c2, c1)
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        a2q, a2q_scale = ops.scaled_fp8_quant(
            c2, a2_scale, use_per_token_if_dynamic=per_act_token)

        ops.cutlass_moe_mm(c3, a2q, w2, a2q_scale, w2_scale,
                           expert_offsets[:-1], problem_sizes2,
                           self.ab_strides2, self.ab_strides2, self.c_strides2)

        c3 = c3[c_map, ...]

        return c3


def modular_cutlass_moe_fp8(
    ab_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    ab_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    out_dtype: torch.dtype = torch.half,
) -> mk.FusedMoEModularKernel:
    return mk.FusedMoEModularKernel(
        StandardDispatchCombine(quant_dtype=torch.float8_e4m3fn),
        CutlassExperts(
            ab_strides1,
            c_strides1,
            ab_strides2,
            c_strides2,
            out_dtype,
        ),
    )


#TODO make the grouped gemm kernel consistent with scaled gemm kernel
def cutlass_moe_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    ab_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    ab_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.half,
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
    - out_dtype (torch.dtype): The output tensor type.

    Returns:
    - torch.Tensor: The fp16 output tensor after applying the MoE layer.
    """
    fn = modular_cutlass_moe_fp8(
        ab_strides1,
        c_strides1,
        ab_strides2,
        c_strides2,
        out_dtype,
    )
    return fn(
        a,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
    )
