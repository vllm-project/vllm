# SPDX-License-Identifier: Apache-2.0
""" CUTLASS based Fused MoE kernels."""
import os
from typing import Optional

import torch

from vllm import _custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.utils import (_resize_cache,
                                                        _fp8_perm)
from vllm.scalar_type import scalar_types


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


FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
MAX_TOKENS_PER_EXPERT = int(
    os.environ.get('VLLM_MODELOPT_MAX_TOKENS_PER_EXPERT', '65536'))


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
    assert (m <= MAX_TOKENS_PER_EXPERT), (
        f"m must be less than MAX_TOKENS_PER_EXPERT({MAX_TOKENS_PER_EXPERT})"
        f" for cutlass_moe_fp4, observed m = {m}. Use"
        f" VLLM_MODELOPT_MAX_TOKENS_PER_EXPERT to set this value.")
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
        expert_map=a_map,
        MAX_TOKENS_PER_EXPERT=MAX_TOKENS_PER_EXPERT)

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
        intermediate,
        a2_gscale,
        expert_offsets,
        blockscale_offsets,
        num_topk,
        MAX_TOKENS_PER_EXPERT=MAX_TOKENS_PER_EXPERT)

    c2 = ops.cutlass_fp4_moe_mm(int_fp4, w2_fp4, int_blockscale, w2_blockscale,
                                w2_alphas, problem_sizes2, expert_offsets[:-1],
                                blockscale_offsets[:-1], out_dtype, device)
    del int_fp4, int_blockscale
    out = (c2[c_map].view(m, num_topk, k) *
           topk_weights.view(m, num_topk, 1).half()).sum(dim=1)
    return out.to(dtype=out_dtype)


class CutlassDispatchCombine(mk.FusedMoEQuantizeDispatchCombine):
    def __init__(self, out_dtype: torch.dtype):
        super().__init__()
        self.out_dtype = out_dtype

    def dispatch(
            self,
            a: torch.Tensor,
            a1_scale: Optional[torch.Tensor],
            a2_scale: Optional[torch.Tensor],
            topk_ids: torch.Tensor,
            num_experts: int,
            expert_map: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # why do we need to check a2_scale here?
        per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
            a2_scale.numel() != 1 if a2_scale is not None else False)

        a_q, a1_scale = ops.scaled_fp8_quant(
            a, a1_scale, use_per_token_if_dynamic=per_act_token)

        return a_q, a1_scale, topk_ids

    def combine(
            self,
            out: torch.Tensor,  #TBD
            hidden_states: torch.Tensor,
            topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        M, topk = topk_weights.shape
        K = hidden_states.shape[1]
        hidden_states = (hidden_states.view(-1, topk, K) * topk_weights.view(M, -1, 1).to(self.out_dtype)).sum(dim=1)
        # use moe_sum? to write into out?
        return hidden_states

        ops.get_cutlass_moe_mm_data(topk_ids,
                                    expert_offsets,
                                    problem_sizes1,
                                    problem_sizes2,
                                    a_map,
                                    c_map,
                                    num_experts,
                                    k,
                                    n)

        rep_a_q = _fp8_perm(a_q, a_map)
        rep_a1_scales = a1_scale[a_map] if per_act_token else a1_scale

        return rep_a_q, rep_a1_scales, expert_offsets, c_map, (problem_sizes1, problem_sizes2)


class CutlassExperts(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(
            self,
            ab_strides1: torch.Tensor,
            c_strides1: torch.Tensor,
            ab_strides2: torch.Tensor,
            c_strides2: torch.Tensor,
    ):
        super().__init__()
        self.ab_strides1 = ab_strides1
        self.c_strides1 = c_strides1
        self.ab_strides2 = ab_strides2
        self.c_strides2 = c_strides2

    def workspace_shapes(
            self,
            M: int,
            K: int,
            N: int,
            topk: int,
            num_experts: int
    ) -> Tuple[int, int]:
        workspace1 = M * topk * max(2 * N, K)
        workspace2 = M * topk * N
        # return tuples????
        return (workspace1, workspace2)

    def apply(
            self,
            out: torch.Tensor, # TBD
            q_hidden_states: torch.Tensor,
            w1: torch.Tensor,
            w2: torch.Tensor,
            topk_ids: torch.Tensor,
            inplace: bool,
            activation: str,
            expert_map: Optional[torch.Tensor],
            w1_scale: Optional[torch.Tensor],
            w2_scale: Optional[torch.Tensor],
            a1_scale: Optional[torch.Tensor],
            a2_scale: Optional[torch.Tensor],
            workspace13: torch.Tensor,
            workspace2: torch.Tensor,
    ) -> torch.Tensor: # or None?  assume inplace?
        # chunking in here or in ModularFusedMoEKernel? ignore for now
        M = q_hidden_states.shape[0]
        E, N, _ = w2.shape   # because w1 + w2 are transposed
        K = w1.shape[1]  #?
        assert K == w2.shape[-1]
        device = q_hidden_states.device

        per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
            a2_scale.numel() != 1 if a2_scale is not None else False)

        expert_offsets = torch.empty((E + 1),
                                     dtype=torch.int32,
                                     device=device)
        problem_sizes1 = torch.empty((E, 3),
                                     dtype=torch.int32,
                                     device=device)
        problem_sizes2 = torch.empty((E, 3),
                                     dtype=torch.int32,
                                     device=device)

        a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
        c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

        #print(f"prob {k}, {n}")

        ops.get_cutlass_moe_mm_data(topk_ids,
                                    expert_offsets,
                                    problem_sizes1,
                                    problem_sizes2,
                                    a_map,
                                    c_map,
                                    E,
                                    N,
                                    K)

        q_hidden_states  = _fp8_perm(q_hidden_states, a_map)
        a1_scale = a1_scale[a_map] if per_act_token else a1_scale

        # fix names
        c1 = _resize_cache(workspace13, (M, N * 2))
        c2 = _resize_cache(workspace2, (M, N))
        c3 = _resize_cache(workspace13, (M, K))

        ops.cutlass_moe_mm(
            c1,
            q_hidden_states,
            w1,
            a1_scale,
            w1_scale,
            expert_offsets[:-1],
            problem_sizes1,
            self.ab_strides1,
            self.ab_strides1,
            self.c_strides1
        )

        if activation == "silu":
            torch.ops._C.silu_and_mul(c2, c1)
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(c2, c1)
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        intemediate_q, a2_scale = ops.scaled_fp8_quant(
            c2, a2_scale, use_per_token_if_dynamic=per_act_token)

        ops.cutlass_moe_mm(
            c3,
            intemediate_q,
            w2,
            a2_scale,
            w2_scale,
            expert_offsets[:-1],
            problem_sizes2,
            self.ab_strides2,
            self.ab_strides2,
            self.c_strides2
        )

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
        CutlassDispatchCombine(out_dtype),
        CutlassExperts(
            ab_strides1,
            c_strides1,
            ab_strides2,
            c_strides2,
        ),
    )
