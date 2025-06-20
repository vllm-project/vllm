# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import importlib.util
from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    _moe_permute)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache, per_token_group_quant_fp8)
from vllm.utils import round_up

logger = init_logger(__name__)

has_deep_gemm = importlib.util.find_spec("deep_gemm") is not None


@functools.cache
def deep_gemm_block_shape() -> list[int]:
    # Lazy import to avoid CUDA initialization problems.
    import deep_gemm as dg
    block = dg.get_m_alignment_for_contiguous_layout()
    return [block, block]


def _valid_deep_gemm_shape(M: int, N: int, K: int):
    align = deep_gemm_block_shape()[0]
    return align <= M and N % align == 0 and K % align == 0


def _valid_deep_gemm(hidden_states: torch.Tensor, w1: torch.Tensor,
                     w2: torch.Tensor) -> bool:
    """
    Check if the given problem size is supported by the DeepGemm grouped
    gemm kernel.  All of M, N, K and the quantization block_shape must be
    aligned by `dg.get_m_alignment_for_contiguous_layout()`.
    """
    if not has_deep_gemm:
        logger.debug("DeepGemm disabled: deep_gemm not available.")
        return False

    M = hidden_states.size(0)
    _, K, N = w2.size()
    if not _valid_deep_gemm_shape(M, N, K):
        logger.debug("DeepGemm disabled: unalinged problem size.")
        return False

    if (w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn):
        logger.debug("DeepGemm disabled: invalid weight dtype(s).")
        return False

    if (not hidden_states.is_contiguous() or not w1.is_contiguous()
            or not w2.is_contiguous()):
        logger.debug(
            "DeepGemm disabled: weights or activations not contiguous.")
        return False

    return True


class DeepGemmExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(self):
        super().__init__()
        self.block_shape = deep_gemm_block_shape()

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

        block_m = self.block_shape[0]
        M_sum = (M * topk) + num_experts * (block_m - 1)
        M_sum = round_up(M_sum, block_m)
        workspace1 = M_sum * max(N * 2, K)
        workspace2 = M_sum * max(N, K)

        return (workspace1, workspace2, a.dtype)

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
        import deep_gemm as dg

        a1q = hidden_states
        _, N, K = w1.size()

        if global_num_experts == -1:
            global_num_experts = w1.size(0)

        assert w2.size(1) == K

        a1q, a1q_scale, _, expert_ids, inv_perm = _moe_permute(
            a1q,
            a1q_scale,
            topk_ids,
            global_num_experts,
            expert_map,
            self.block_shape[0],
        )

        if expert_map is not None:
            # DeepGemm (Grouped Contiguous) kernel needs a valid B index
            # for all rows of A. To that effect, simply compute with
            # the 0th weight matrix.
            # Note that this relies on the fact that corresponding topk
            # weights would be 0 during weight multiplication.
            expert_ids = torch.where(expert_ids == -1, 0, expert_ids)

        # Note: M_sum is different than the pre-permuted shape of a1q.
        M_sum = a1q.size(0)

        mm1_out = _resize_cache(workspace13, (M_sum, N))
        act_out = _resize_cache(workspace2, (M_sum, N // 2))
        quant_out = _resize_cache(workspace13.view(dtype=torch.float8_e4m3fn),
                                  (M_sum, N // 2))
        mm2_out = _resize_cache(workspace2, (M_sum, K))
        out = _resize_cache(workspace13, (inv_perm.size(0), K))

        dg.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (a1q, a1q_scale), (w1, w1_scale), mm1_out, expert_ids)

        self.activation(activation, act_out, mm1_out.view(-1, N))

        a2q_scale: Optional[torch.Tensor] = None
        a2q, a2q_scale = per_token_group_quant_fp8(act_out,
                                                   self.block_shape[1],
                                                   column_major_scales=True,
                                                   out_q=quant_out)

        dg.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (a2q, a2q_scale), (w2, w2_scale), mm2_out, expert_ids)

        torch.index_select(mm2_out, 0, inv_perm, out=out)

        return out


def deep_gemm_moe_fp8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    apply_router_weight_on_input=False,
) -> torch.Tensor:
    """
    This function computes a a8w8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with DeepGemm
    grouped gemm.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1 (torch.Tensor): The first set of fp8 quantized expert weights.
        Shape: [num_experts, K, 2N] (the weights are passed transposed)
    - w2 (torch.Tensor): The second set of fp8 quantized expert weights.
        Shape: [num_experts, N, K] (the weights are passed transposed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts] or [num_experts, 2N]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts] or [num_experts, K]
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - topk_ids (torch.Tensor): The token->expert mapping for topk_weights.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - activation (str): The activation function to apply after the first
        MoE layer.
    - global_num_experts (int): The total number of experts in the global
        expert space.
    - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
        from the global expert space to the local expert space of the expert
        parallel shard.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [M]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [M]

    Returns:
    - torch.Tensor: The bfloat16 output tensor after applying the MoE layer.
    """
    fn = mk.FusedMoEModularKernel(
        MoEPrepareAndFinalizeNoEP(quant_dtype=torch.float8_e4m3fn,
                                  block_shape=deep_gemm_block_shape()),
        DeepGemmExperts(),
    )
    return fn(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace,
        activation,
        global_num_experts,
        expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
