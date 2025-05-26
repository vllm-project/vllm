# SPDX-License-Identifier: Apache-2.0
""" CUTLASS based Fused MoE kernels."""
from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.scalar_type import scalar_types


class CutlassExpertsFp8(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(
        self,
        max_experts_per_worker: int,
        out_dtype: torch.dtype,
        per_act_token: bool,
        per_out_ch: bool,
    ):
        super().__init__()
        self.max_experts_per_worker = max_experts_per_worker
        self.out_dtype = out_dtype
        self.per_act_token = per_act_token
        self.per_out_ch = per_out_ch

    def workspace_shapes(
        self,
        a: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        num_experts: int,
        padded_M: int,
    ) -> tuple[int, int, torch.dtype]:
        # Note that K, N are transposed
        N, K = K, N

        workspace1 = self.max_experts_per_worker * padded_M * max(2 * N, K)
        workspace2 = self.max_experts_per_worker * padded_M * N
        return (workspace1, workspace2, self.out_dtype)

    def apply(
        self,
        hidden_states: torch.Tensor,
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
        expert_num_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        a1q = hidden_states

        try_output_map = True

        assert w1_scale is not None
        assert w2_scale is not None
        assert w1.dtype == torch.float8_e4m3fn
        assert w2.dtype == torch.float8_e4m3fn
        assert a1q.shape[2] == w1.shape[2], "Hidden size mismatch w1"
        assert w1.shape[1] == w2.shape[2] * 2, "Hidden size mismatch w2"
        assert w1.shape[0] == w2.shape[0], "Expert number mismatch"
        assert a1q_scale is None or a1q_scale.dim(
        ) == 0 or a1q_scale.shape[0] == 1 or a1q_scale.shape[0] == a1q.shape[
            0], "Input scale shape mismatch"  #TODO adjust
        assert w1_scale.dim() == 1 or w1_scale.shape[1] == 1 or w1_scale.shape[
            1] == w1.shape[1], "W1 scale shape mismatch"
        assert w2_scale.dim() == 1 or w2_scale.shape[1] == 1 or w2_scale.shape[
            1] == w2.shape[1], "W2 scale shape mismatch"
        assert w1.shape[0] == w2.shape[0], "Weights expert number mismatch"
        assert w1.shape[0] == w1_scale.shape[
            0], "w1 scales expert number mismatch"
        assert w1.shape[0] == w2_scale.shape[
            0], "w2 scales expert number mismatch"
        assert a2_scale is None or a2_scale.dim(
        ) == 0 or a2_scale.shape[0] == 1 or a2_scale.shape[0] == a1q.shape[
            0], "Intermediate scale shape mismatch"
        assert self.out_dtype in [torch.half,
                                  torch.bfloat16], "Invalid output dtype"

        padded_M = a1q.shape[1]
        _, K, N = w2.shape  # because w1 + w2 are transposed
        device = a1q.device

        assert w1.shape[1] == N * 2
        assert global_num_experts != -1
        assert a1q_scale is not None

        if expert_map is not None:
            "Translate info from expert_map to topk_ids"
            local_topk_ids = torch.where(expert_map[topk_ids] != -1,
                                         expert_map[topk_ids], -1)
        else:
            local_topk_ids = topk_ids

        topk = local_topk_ids.shape[1]
        local_E = w1.shape[0]

        # TODO make expert_map work again
        # With expert_map each Rank processes only a subset of experts. As
        # a result not all of a_map and c2 tensors are filled. We fill it
        # zeros for correctness.
        # if expert_map is not None:
        #     a_map = torch.zeros((local_topk_ids.numel()),
        #                         dtype=torch.int32,
        #                         device=device)

        non_zero_mask = expert_num_tokens[:] != 0
        # print("NON ZERO MASK:", non_zero_mask)
        masked_local_E = int(non_zero_mask.sum().item())

        expert_offsets = torch.empty((masked_local_E + 1),
                                     dtype=torch.int32,
                                     device=device)
        problem_sizes1 = torch.empty((masked_local_E, 3),
                                     dtype=torch.int32,
                                     device=device)
        problem_sizes2 = torch.empty((masked_local_E, 3),
                                     dtype=torch.int32,
                                     device=device)

        # TODO write a new get_cutlass_moe_mm_data kernel for this
        masked_idx = 0
        prev_expert_num_tokens = 0
        for idx in range(local_E):
            if expert_num_tokens[idx] != 0:
                if try_output_map:
                    offset = 0 if masked_idx == 0 else prev_expert_num_tokens + expert_offsets[
                        masked_idx - 1]
                    expert_offsets[masked_idx] = offset
                else:
                    expert_offsets[masked_idx] = idx * a1q.shape[1]
                problem_sizes1[masked_idx][0] = expert_num_tokens[idx]
                problem_sizes1[masked_idx][1] = 2 * N
                problem_sizes1[masked_idx][2] = K
                problem_sizes2[masked_idx][0] = expert_num_tokens[idx]
                problem_sizes2[masked_idx][1] = K
                problem_sizes2[masked_idx][2] = N
                prev_expert_num_tokens = expert_num_tokens[idx]
                masked_idx += 1

        # Filter out problem sizes with 0 tokens
        # problem_sizes1 = problem_sizes1[non_zero_mask]
        # problem_sizes2 = problem_sizes2[non_zero_mask]
        # expert_offsets = (expert_offsets[:-1])[non_zero_mask]
        w1 = w1[non_zero_mask].contiguous()
        w2 = w2[non_zero_mask].contiguous()
        # print("w1 scale before mask:", w1_scale.shape)
        w1_scale = w1_scale[non_zero_mask].contiguous()
        w2_scale = w2_scale[non_zero_mask].contiguous()
        # a1q = a1q[non_zero_mask].contiguous()
        # a1q_scale = a1q_scale[non_zero_mask].contiguous()

        w1_scale = w1_scale.reshape(-1, w1_scale.shape[-1])
        w2_scale = w2_scale.reshape(-1, w2_scale.shape[-1])

        # print("--- expert_offsets:", expert_offsets[:-1])
        # print("--- problem_sizes1:", problem_sizes1)
        # print("--- problem_sizes2:", problem_sizes2)

        ab_strides1 = torch.full((w1.shape[0], ),
                                 K,
                                 device=device,
                                 dtype=torch.int64)
        c_strides1 = torch.full((w1.shape[0], ),
                                2 * N,
                                device=device,
                                dtype=torch.int64)
        ab_strides2 = torch.full((w1.shape[0], ),
                                 N,
                                 device=device,
                                 dtype=torch.int64)
        c_strides2 = torch.full((w1.shape[0], ),
                                K,
                                device=device,
                                dtype=torch.int64)

        # TODO try to make an output map that redistributes the output tokens
        # back to the input format
        if try_output_map:
            all_tokens = torch.sum(expert_num_tokens)
            if all_tokens == 0:
                return torch.zeros(a1q.shape,
                                   device=a1q.device,
                                   dtype=self.out_dtype)
            output_map = torch.empty((all_tokens),
                                     dtype=torch.int32,
                                     device=device)
            cumul_idx = 0
            for idx in range(local_E):
                e_tokens = expert_num_tokens[idx]
                for t in range(e_tokens):
                    output_map[cumul_idx] = idx * padded_M + t
                    cumul_idx += 1

        fp8_type = a1q.dtype
        if try_output_map:
            a1q = a1q.reshape(-1, a1q.shape[2]).view(
                dtype=torch.uint8)[output_map].contiguous().view(
                    dtype=fp8_type)
        else:
            a1q = a1q.reshape(-1, a1q.shape[2])

        if self.per_act_token:
            if try_output_map:
                a1q_scale = a1q_scale.reshape(-1,
                                              a1q_scale.shape[2])[output_map]
            else:
                a1q_scale = a1q_scale.reshape(-1, a1q_scale.shape[2])
        else:
            if try_output_map:
                a1q_scale = a1q_scale.reshape(
                    -1, a1q_scale.shape[2])[output_map][0]
            else:
                a1q_scale = a1q_scale.reshape(-1, a1q_scale.shape[2])

        # ops.get_cutlass_moe_mm_data(local_topk_ids, expert_offsets,
        #                             problem_sizes1, problem_sizes2, a_map,
        #                             c_map, global_num_experts, N, K)

        if try_output_map:
            c1 = _resize_cache(workspace13, (all_tokens, N * 2))
            c2 = _resize_cache(workspace2, (all_tokens, N))
            c3 = _resize_cache(workspace13, (all_tokens, K))
        else:
            c1 = _resize_cache(workspace13, (masked_local_E * padded_M, N * 2))
            c2 = _resize_cache(workspace2, (masked_local_E * padded_M, N))
            c3 = _resize_cache(workspace13, (masked_local_E * padded_M, K))

        ops.cutlass_moe_mm(c1, a1q, w1.contiguous(), a1q_scale.contiguous(),
                           w1_scale.contiguous(), expert_offsets[:-1],
                           problem_sizes1, ab_strides1, ab_strides1,
                           c_strides1, self.per_act_token, self.per_out_ch)

        self.activation(activation, c2, c1)

        a2q, a2q_scale = ops.scaled_fp8_quant(
            c2, a2_scale, use_per_token_if_dynamic=self.per_act_token)

        if expert_map is not None:
            c3.fill_(0)

        ops.cutlass_moe_mm(c3, a2q, w2.contiguous(), a2q_scale.contiguous(),
                           w2_scale.contiguous(), expert_offsets[:-1],
                           problem_sizes2, ab_strides2, ab_strides2,
                           c_strides2, self.per_act_token, self.per_out_ch)

        if try_output_map:
            out = torch.zeros((local_E * padded_M, K),
                              device=hidden_states.device,
                              dtype=self.out_dtype)
            out[output_map] = c3
            # print("out:", out.reshape(local_E, padded_M, K))
            return out.reshape(local_E, padded_M, K)
        else:
            # print(expert_num_tokens.shape, local_E)
            out = torch.zeros((local_E, padded_M, K),
                              device=hidden_states.device,
                              dtype=self.out_dtype)
            out[non_zero_mask] = c3.reshape(masked_local_E, padded_M, K)
            return out


#TODO make the grouped gemm kernel consistent with scaled gemm kernel
def cutlass_moe_fp8(
    max_experts_per_worker: int,
    max_num_tokens: int,
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
    expert_map: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    per_act_token: bool = False,
    per_out_ch: bool = False,
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
    # per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
    #     a2_scale.numel() != 1 if a2_scale is not None else False)

    fn = mk.FusedMoEModularKernel(
        MoEPrepareAndFinalizeNoEP(
            per_channel_quant=per_act_token,
            quant_dtype=torch.float8_e4m3fn,
        ),
        CutlassExpertsFp8(
            max_experts_per_worker,
            max_num_tokens,
            ab_strides1,
            c_strides1,
            ab_strides2,
            c_strides2,
            out_dtype,
            per_act_token,
            per_out_ch,
        ),
    )

    return fn(
        a,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )


FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def cutlass_moe_fp4(a: torch.Tensor, a1_gscale: torch.Tensor,
                    w1_fp4: torch.Tensor, w1_blockscale: torch.Tensor,
                    w1_alphas: torch.Tensor, a2_gscale: torch.Tensor,
                    w2_fp4: torch.Tensor, w2_blockscale: torch.Tensor,
                    w2_alphas: torch.Tensor, topk_weights: torch.Tensor,
                    topk_ids: torch.Tensor, m: int, n: int, k: int, e: int,
                    device: torch.device):
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
    assert (w1_fp4.ndim == 3 and w2_fp4.ndim == 3 and w1_blockscale.ndim == 3
            and w2_blockscale.ndim
            == 3), ("All Weights must be of rank 3 for cutlass_moe_fp4")
    m_a, k_a = a.shape
    e_w1, nx2_w1, half_k_w1 = w1_fp4.shape
    e_w2, k_w2, half_n_w2 = w2_fp4.shape

    assert (e_w1 == e_w2 and e_w1 == e), ("Number of experts must match",
                                          " between weights.")
    assert (k_a // 2 == half_k_w1
            and k == k_w2), ("Hidden size mismatch between a, w1 and w2")
    assert (nx2_w1 == n * 2 and half_n_w2 == n // 2), ("mismatch in "
                                                       "expected `n`")
    assert (m == m_a), "input shape mismatch"
    assert 2 * half_k_w1 == k_w2, "Hidden size mismatch w2 and w1"
    assert a.dtype in [torch.half, torch.bfloat16], "Invalid input dtype"
    assert (topk_weights.shape[0] == m and topk_ids.shape[0]
            == m), ("topk must be provided for each row of a")

    out_dtype = a.dtype
    num_topk = topk_ids.shape[1]

    expert_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    # Problem size:  (num_experts, (m,2n,k))
    problem_sizes1 = torch.empty((e, 3), dtype=torch.int32, device=device)
    # Problem size:  (num_experts, (m,n,k))
    problem_sizes2 = torch.empty((e, 3), dtype=torch.int32, device=device)

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    # problem shapes should have [m, n, k]
    # Note that problem sizes are based on logical number of elements.
    ops.get_cutlass_moe_mm_data(topk_ids, expert_offsets, problem_sizes1,
                                problem_sizes2, a_map, c_map, e, n, k)

    tokens_per_expert = problem_sizes1[:, 0]
    rounded_tokens_per_expert = (tokens_per_expert + (128 - 1)) // 128 * 128
    blockscale_offsets = torch.zeros(e + 1, dtype=torch.int32, device=device)
    blockscale_offsets[1:] = torch.cumsum(rounded_tokens_per_expert, dim=0)

    rep_a_fp4, rep_a_blockscale = ops.scaled_fp4_experts_quant(
        a,
        a1_gscale,
        expert_offsets,
        blockscale_offsets,
        num_topk,
        expert_map=a_map)

    c1 = ops.cutlass_fp4_moe_mm(rep_a_fp4, w1_fp4, rep_a_blockscale,
                                w1_blockscale, w1_alphas, problem_sizes1,
                                expert_offsets[:-1], blockscale_offsets[:-1],
                                out_dtype, device)
    del rep_a_fp4, rep_a_blockscale
    # hidden size dimension is split to one halfpytho sized tensor.
    intermediate = torch.empty((m * num_topk, w1_fp4.shape[1] // 2),
                               device=device,
                               dtype=out_dtype)

    torch.ops._C.silu_and_mul(intermediate, c1)

    int_fp4, int_blockscale = ops.scaled_fp4_experts_quant(
        intermediate, a2_gscale, expert_offsets, blockscale_offsets, num_topk)

    c2 = ops.cutlass_fp4_moe_mm(int_fp4, w2_fp4, int_blockscale, w2_blockscale,
                                w2_alphas, problem_sizes2, expert_offsets[:-1],
                                blockscale_offsets[:-1], out_dtype, device)
    del int_fp4, int_blockscale
    out = (c2[c_map].view(m, num_topk, k) *
           topk_weights.view(m, num_topk, 1).half()).sum(dim=1)
    return out.to(dtype=out_dtype)
