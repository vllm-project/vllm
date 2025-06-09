# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
""" CUTLASS based Fused MoE kernels."""
from typing import Callable, Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP)
from vllm.model_executor.layers.fused_moe.utils import _fp8_perm, _resize_cache
from vllm.scalar_type import scalar_types


def run_cutlass_moe_fp8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    activation_callable: Callable,
    global_num_experts: int,
    expert_map: Optional[torch.Tensor],
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    a1q_scale: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_num_tokens: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    per_act_token: bool,
    per_out_ch: bool,
) -> torch.Tensor:
    a1q = hidden_states

    assert w1_scale is not None
    assert w2_scale is not None
    assert w1.dtype == torch.float8_e4m3fn
    assert w2.dtype == torch.float8_e4m3fn
    if expert_num_tokens is None:
        assert a1q.shape[1] == w1.shape[2], "Hidden size mismatch w1"
    else:
        assert a1q.shape[2] == w1.shape[2], "Hidden size mismatch w1"
    assert w1.shape[1] == w2.shape[2] * 2, "Hidden size mismatch w2"
    assert w1_scale.dim() == 1 or w1_scale.shape[1] == 1 or w1_scale.shape[
        1] == w1.shape[1], "W1 scale shape mismatch"
    assert w2_scale.dim() == 1 or w2_scale.shape[1] == 1 or w2_scale.shape[
        1] == w2.shape[1], "W2 scale shape mismatch"
    assert w1.shape[0] == w2.shape[0], "Expert number mismatch"
    assert a1q_scale is None or a1q_scale.dim(
    ) == 0 or a1q_scale.shape[0] == 1 or a1q_scale.shape[0] == a1q.shape[
        0], "Input scale shape mismatch"
    assert w1.shape[0] == w2.shape[0], "Weights expert number mismatch"
    assert w1.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"
    assert a2_scale is None or a2_scale.dim(
    ) == 0 or a2_scale.shape[0] == 1 or a2_scale.shape[0] == a1q.shape[
        0], "Intermediate scale shape mismatch"
    assert out_dtype in [torch.half, torch.bfloat16], "Invalid output dtype"
    if expert_map is not None:
        assert expert_num_tokens is None

    # We have two modes: PPLX and non-PPLX. We differentiate them by checking
    # if expert_num_tokens is None (expert_num_tokens is a tensor which PPLX
    # uses to track the number of tokens per expert).
    # In the non-PPLX mode, the input tokens are not padded: thus, the shape
    # of the input is [total_num_tokens, hidden_size]. The input and output
    # require shuffling by a_map and c_map such that the tokens assigned to
    # each expert are contiguous.
    # In the PPLX mode, the input tokens are padded per expert to ensure that
    # the PPLX dispatch and combine functions work correctly: thus, the shape
    # of the input is [num_experts, max_num_tokens_per_expert, hidden_size].
    # The PPLX input and output require no shuffling by a_map and c_map since
    # their tokens are already contiguous for each expert as a result of
    # the dispatch function.
    is_pplx = expert_num_tokens is not None

    M = a1q.shape[0]  # no pplx
    padded_M = a1q.shape[1]  # pplx
    _, K, N = w2.shape
    device = a1q.device

    assert w1.shape[2] == K
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

    if is_pplx:
        expert_offsets = torch.empty((local_E),
                                     dtype=torch.int32,
                                     device=device)
        problem_sizes1 = torch.empty((local_E, 3),
                                     dtype=torch.int32,
                                     device=device)
        problem_sizes2 = torch.empty((local_E, 3),
                                     dtype=torch.int32,
                                     device=device)

        ops.get_cutlass_pplx_moe_mm_data(expert_offsets, problem_sizes1,
                                         problem_sizes2, expert_num_tokens,
                                         local_E, padded_M, N, K)

        w1_scale = w1_scale.reshape(w1_scale.shape[0], -1)
        w2_scale = w2_scale.reshape(w2_scale.shape[0], -1)
        a1q = a1q.reshape(-1, a1q.shape[2])
        a1q_scale = a1q_scale.reshape(-1, a1q_scale.shape[2]).contiguous()

    else:
        expert_offsets = torch.empty((global_num_experts + 1),
                                     dtype=torch.int32,
                                     device=device)
        problem_sizes1 = torch.empty((global_num_experts, 3),
                                     dtype=torch.int32,
                                     device=device)
        problem_sizes2 = torch.empty((global_num_experts, 3),
                                     dtype=torch.int32,
                                     device=device)

        # With expert_map each Rank processes only a subset of experts. As
        # a result not all of a_map and c2 tensors are filled. We fill it
        # zeros for correctness.
        if expert_map is not None:
            a_map = torch.zeros((local_topk_ids.numel()),
                                dtype=torch.int32,
                                device=device)
        else:
            a_map = torch.empty((local_topk_ids.numel()),
                                dtype=torch.int32,
                                device=device)

        c_map = torch.empty((local_topk_ids.numel()),
                            dtype=torch.int32,
                            device=device)

        ops.get_cutlass_moe_mm_data(local_topk_ids, expert_offsets,
                                    problem_sizes1, problem_sizes2, a_map,
                                    c_map, global_num_experts, N, K)

        a1q = _fp8_perm(a1q, a_map)
        a1q_scale = a1q_scale[a_map] if per_act_token else a1q_scale
        expert_offsets = expert_offsets[:-1]

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

    if is_pplx:
        c1 = _resize_cache(workspace13, (local_E * padded_M, N * 2))
        c2 = _resize_cache(workspace2, (local_E * padded_M, N))
        c3 = _resize_cache(workspace13, (local_E * padded_M, K))
    else:
        c1 = _resize_cache(workspace13, (M * topk, N * 2))
        c2 = _resize_cache(workspace2, (M * topk, N))
        c3 = _resize_cache(workspace13, (M * topk, K))

    ops.cutlass_moe_mm(c1, a1q, w1, a1q_scale, w1_scale, expert_offsets,
                       problem_sizes1, ab_strides1, ab_strides1, c_strides1,
                       per_act_token, per_out_ch)

    activation_callable(c2, c1)

    a2q, a2q_scale = ops.scaled_fp8_quant(
        c2, a2_scale, use_per_token_if_dynamic=per_act_token)

    if expert_map is not None:
        c3.fill_(0)

    ops.cutlass_moe_mm(c3, a2q, w2, a2q_scale, w2_scale, expert_offsets,
                       problem_sizes2, ab_strides2, ab_strides2, c_strides2,
                       per_act_token, per_out_ch)

    if is_pplx:
        return c3.reshape(local_E, padded_M, K)
    else:
        return c3[c_map].view(M, topk, K)


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
        aq: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        num_experts: int,
    ) -> tuple[int, int, torch.dtype]:
        padded_M = aq.shape[1]
        workspace1 = self.max_experts_per_worker * padded_M * max(N, K)
        workspace2 = self.max_experts_per_worker * padded_M * (N // 2)
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
        assert w1_zp is None, "w1_zp is not supported in CUTLASS MoE"
        assert w2_zp is None, "w2_zp is not supported in CUTLASS MoE"
        activation_callable = lambda i, o: self.activation(activation, i, o)
        return run_cutlass_moe_fp8(hidden_states, w1, w2, topk_ids,
                                   activation_callable, global_num_experts,
                                   expert_map, w1_scale, w2_scale, a1q_scale,
                                   a2_scale, workspace13, workspace2,
                                   expert_num_tokens, self.out_dtype,
                                   self.per_act_token, self.per_out_ch)


def cutlass_moe_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    activation: str = "silu",
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    expert_map: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
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
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - topk_ids (torch.Tensor): The token->expert mappings.
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts] or [num_experts, 2N]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts] or [num_experts, K]
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [M]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [M]
    - expert_map (Optional[torch.Tensor]): In the case of Expert parallel,
        every Rank is responsible for a subset of experts. expert_map is a
        mapping from global expert-id to local expert-id. When expert_map[i]
        is -1, it means that this Rank is not responsible for global
        expert-id i.
    - apply_router_weight_on_input (bool): When true, the topk weights are
        applied directly on the inputs. This is only applicable when topk is 1.
    - global_num_experts (int): The total number of experts.

    Returns:
    - torch.Tensor: The fp16 output tensor after applying the MoE layer.
    """
    per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
        a2_scale.numel() != 1 if a2_scale is not None else False)
    per_out_ch = w1_scale.numel() != w1_q.shape[0]

    out_dtype = a.dtype

    fn = mk.FusedMoEModularKernel(
        MoEPrepareAndFinalizeNoEP(
            quant_dtype=torch.float8_e4m3fn,
            per_channel_quant=per_act_token,
        ),
        CutlassExpertsFp8(
            max_experts_per_worker=global_num_experts,
            out_dtype=out_dtype,
            per_act_token=per_act_token,
            per_out_ch=per_out_ch,
        ),
    )

    return fn(
        a,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        False,
        activation,
        global_num_experts if global_num_experts != -1 else w1_q.size(0),
        expert_map,
        w1_scale,
        w2_scale,
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
    blockscale_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    # Problem size:  (num_experts, (m,2n,k))
    problem_sizes1 = torch.empty((e, 3), dtype=torch.int32, device=device)
    # Problem size:  (num_experts, (m,n,k))
    problem_sizes2 = torch.empty((e, 3), dtype=torch.int32, device=device)

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    # problem shapes should have [m, n, k]
    # Note that problem sizes are based on logical number of elements.
    ops.get_cutlass_moe_mm_data(topk_ids, expert_offsets, problem_sizes1,
                                problem_sizes2, a_map, c_map, e, n, k,
                                blockscale_offsets)

    a = ops.shuffle_rows(a, a_map)

    rep_a_fp4, rep_a_blockscale = ops.scaled_fp4_experts_quant(
        a,
        a1_gscale,
        expert_offsets,
        blockscale_offsets,
        num_topk,
    )

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

    c2 = ops.shuffle_rows(c2, c_map)
    out = (c2.view(m, num_topk, k) *
           topk_weights.view(m, num_topk, 1).half()).sum(dim=1)
    return out.to(dtype=out_dtype)
