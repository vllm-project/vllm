# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE utilities for GPTQ."""
from typing import Optional

import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new, maybe_warn_marlin_atomic_add)
from vllm.scalar_type import ScalarType, scalar_types
from vllm.utils import direct_register_custom_op


def fused_marlin_moe(hidden_states: torch.Tensor,
                     w1: torch.Tensor,
                     w2: torch.Tensor,
                     bias1: Optional[torch.Tensor],
                     bias2: Optional[torch.Tensor],
                     w1_scale: torch.Tensor,
                     w2_scale: torch.Tensor,
                     gating_output: torch.Tensor,
                     topk_weights: torch.Tensor,
                     topk_ids: torch.Tensor,
                     quant_type_id: int,
                     apply_router_weight_on_input: bool = False,
                     global_num_experts: int = -1,
                     activation: Optional[str] = "silu",
                     expert_map: Optional[torch.Tensor] = None,
                     global_scale1: Optional[torch.Tensor] = None,
                     global_scale2: Optional[torch.Tensor] = None,
                     g_idx1: Optional[torch.Tensor] = None,
                     g_idx2: Optional[torch.Tensor] = None,
                     sort_indices1: Optional[torch.Tensor] = None,
                     sort_indices2: Optional[torch.Tensor] = None,
                     w1_zeros: Optional[torch.Tensor] = None,
                     w2_zeros: Optional[torch.Tensor] = None,
                     workspace: Optional[torch.Tensor] = None,
                     intermediate_cache13: Optional[torch.Tensor] = None,
                     intermediate_cache2: Optional[torch.Tensor] = None,
                     is_k_full: bool = True,
                     inplace: bool = False) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - w1_scale (torch.Tensor): Scale to be used for w1.
    - w2_scale (torch.Tensor): Scale to be used for w2.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - g_idx1 (Optional[torch.Tensor]): The first set of act_order indices.
    - g_idx2 (Optional[torch.Tensor]): The second set of act_order indices.
    - sort_indices1 (Optional[torch.Tensor]): The first act_order input
        permutation.
    - sort_indices2 (Optional[torch.Tensor]): The second act_order input
        permutation.
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.
    - w1_zeros (Optional[torch.Tensor]): Optional zero points to be used for w1.
    - w2_zeros (Optional[torch.Tensor]): Optional zero points to be used for w2.
    - num_bits (bool): The number of bits in expert weights quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    quant_type = ScalarType.from_id(quant_type_id)
    assert quant_type in [
        scalar_types.uint4, scalar_types.uint8b128, scalar_types.uint4b8,
        scalar_types.float8_e4m3fn, scalar_types.float4_e2m1f
    ]

    bit4_scalar_types = [
        scalar_types.uint4, scalar_types.uint4b8, scalar_types.float4_e2m1f
    ]
    num_bits = 4 if quant_type in bit4_scalar_types else 8

    # TODO (Varun) : make gating output optional.
    # Check constraints.
    if gating_output is not None:
        assert hidden_states.shape[0] == gating_output.shape[
            0], "Number of tokens mismatch"
    assert hidden_states.shape[
        1] == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // (
        num_bits // 2), "Hidden size mismatch w2"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    assert num_bits in [4, 8]
    assert topk_weights.dtype == torch.float32

    #print (f"w1 - {w1.shape} | w2 - {w2.shape}")

    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[1] * 16
    topk = topk_ids.shape[1]

    # M block size selection logic
    # TODO: tune this further for specific models
    for block_size_m in [8, 16, 32, 48, 64]:
        if M * topk / E / block_size_m < 0.9:
            break

    if global_num_experts == -1:
        global_num_experts = E
    sorted_token_ids, expert_ids, num_tokens_post_padded = \
        moe_align_block_size(topk_ids, block_size_m, global_num_experts,
                             expert_map)

    if workspace is None:
        workspace = marlin_make_workspace_new(hidden_states.device, 4)

    #print (f"ic2 shape : M={M} Topk={topk_ids.shape[1]}, N={N}")
    ic2_shape = (M * topk_ids.shape[1], N)
    if intermediate_cache2 is None:
        intermediate_cache2 = torch.empty(
            ic2_shape,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    assert intermediate_cache2.shape == ic2_shape, (
        f"Expected {ic2_shape}, got {intermediate_cache2.shape}")

    ic13_shape = (M * topk_ids.shape[1] * max(2 * N, K), )
    if intermediate_cache13 is None:
        intermediate_cache13 = torch.empty(
            ic13_shape,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    assert intermediate_cache13.shape == ic13_shape, (
        f"Expected {ic13_shape}, got {intermediate_cache13.shape}")

    intermediate_cache1 = intermediate_cache13[:M * topk_ids.shape[1] * 2 * N]
    intermediate_cache1 = intermediate_cache1.view(-1, 2 * N)
    intermediate_cache3 = intermediate_cache13[:M * topk_ids.shape[1] * K]
    intermediate_cache3 = intermediate_cache3.view(-1, K)

    maybe_warn_marlin_atomic_add(hidden_states.device, hidden_states.dtype)
    use_atomic_add = hidden_states.dtype == torch.half or \
        torch.cuda.get_device_capability(hidden_states.device)[0] >= 9

    intermediate_cache1 = ops.moe_wna16_marlin_gemm(
        hidden_states,
        intermediate_cache1,
        w1,
        bias1,
        w1_scale,
        global_scale1,
        w1_zeros,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=apply_router_weight_on_input,
        is_ep=expert_map is not None,
        b_q_type=quant_type,
        size_m=M,
        size_n=2 * N,
        size_k=K,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False)

    if activation == "silu":
        torch.ops._C.silu_and_mul(intermediate_cache2,
                                  intermediate_cache1.view(-1, 2 * N))
    elif activation == "swigluoai":
        # alpha = 1.702, limit = 7.0
        torch.ops._C.swigluoai_and_mul(intermediate_cache2,
                                       intermediate_cache1.view(-1, 2 * N))
    else:
        raise ValueError(f"Unsupported activation: {activation}. "
                         "Only silu and swigluoai activations are supported.")

    if expert_map is not None:
        intermediate_cache3.zero_()

    intermediate_cache3 = ops.moe_wna16_marlin_gemm(
        intermediate_cache2,
        intermediate_cache3,
        w2,
        bias2,
        w2_scale,
        global_scale2,
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=not apply_router_weight_on_input,
        is_ep=expert_map is not None,
        b_q_type=quant_type,
        size_m=M * topk,
        size_n=K,
        size_k=N,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False).view(-1, topk, K)

    output = hidden_states if inplace else torch.empty_like(hidden_states)
    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                     dim=1,
                     out=output)


def fused_marlin_moe_fake(hidden_states: torch.Tensor,
                          w1: torch.Tensor,
                          w2: torch.Tensor,
                          w1_scale: torch.Tensor,
                          w2_scale: torch.Tensor,
                          gating_output: torch.Tensor,
                          topk_weights: torch.Tensor,
                          topk_ids: torch.Tensor,
                          quant_type_id: int,
                          apply_router_weight_on_input: bool = False,
                          global_num_experts: int = -1,
                          global_scale1: Optional[torch.Tensor] = None,
                          global_scale2: Optional[torch.Tensor] = None,
                          expert_map: Optional[torch.Tensor] = None,
                          g_idx1: Optional[torch.Tensor] = None,
                          g_idx2: Optional[torch.Tensor] = None,
                          sort_indices1: Optional[torch.Tensor] = None,
                          sort_indices2: Optional[torch.Tensor] = None,
                          w1_zeros: Optional[torch.Tensor] = None,
                          w2_zeros: Optional[torch.Tensor] = None,
                          workspace: Optional[torch.Tensor] = None,
                          intermediate_cache13: Optional[torch.Tensor] = None,
                          intermediate_cache2: Optional[torch.Tensor] = None,
                          is_k_full: bool = True,
                          inplace: bool = False) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="fused_marlin_moe",
    op_func=fused_marlin_moe,
    mutates_args=[],
    fake_impl=fused_marlin_moe_fake,
)


class MarlinExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(self, quant_config: FusedMoEQuantConfig):
        # TODO (varun) : Enable activation quantization
        assert quant_config.use_mxfp4_w4a16, "Supports only mxfp4_w4a16"
        super().__init__(quant_config)

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @property
    def activation_formats(
        self
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (mk.FusedMoEActivationFormat.Standard,
                mk.FusedMoEActivationFormat.Standard)

    def supports_chunking(self) -> bool:
        return True

    def workspace_shapes(
        self, a: torch.Tensor, aq: torch.Tensor, M: int, N: int, K: int,
        topk: int, global_num_experts: int, local_num_experts: int,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata]
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        # TODO (varun): Check actual workspace size
        # TODO (varun) : There is a case where output is used as workspace1
        #  - needs handling
        #print (f"workspace_shapes M={M}, N={N}, K={K}, topk={topk} ...")

        # TODO (varun) : Pass in workspaces
        #workspace1 = (M * topk * max(2 * N, K),)
        #workspace2 = (M * topk, N)
        workspace1 = (M, K)
        workspace2 = (0, 0)
        output = (M, K)

        return (workspace1, workspace2, output, a.dtype)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
    ):
        if expert_map is not None:
            topk_ids = expert_map[topk_ids]
        topk_ids = torch.where(topk_ids == -1, 0, topk_ids)

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        print(f"activation : {activation} ...")

        return torch.ops.vllm.fused_marlin_moe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            bias1=self.w1_bias,
            bias2=self.w2_bias,
            w1_scale=self.w1_scale,
            w2_scale=self.w2_scale,
            gating_output=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_type_id=scalar_types.float4_e2m1f.id,  # works only for w4a16
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            activation=activation,
            expert_map=None)
        #expert_map = expert_map)
        #intermediate_cache13 = workspace13,
        #intermediate_cache2 = workspace2)
