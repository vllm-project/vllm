# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate)
from vllm.model_executor.layers.fused_moe.utils import extract_required_args
from vllm.utils import next_power_of_2

if (envs.VLLM_USE_FLASHINFER_MXFP4_MOE
        or envs.VLLM_USE_FLASHINFER_MXFP4_BF16_MOE):
    # from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer import mxfp8_quantize, trtllm_fp4_block_scale_routed_moe


class TrtLlmGenExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(
        self,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(quant_config)

    @property
    def activation_formats(
        self
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (mk.FusedMoEActivationFormat.Standard,
                mk.FusedMoEActivationFormat.Standard)

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(
        self,
        a: torch.Tensor,
        aq: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        self.topk = topk
        self.num_experts = local_num_experts
        self.intermediate_size = K
        workspace1 = (M, topk, max(N // 2, K))
        workspace2 = (M, topk, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output, a.dtype)

    def _get_tile_tokens_dim(self, x: torch.Tensor, top_k: int):
        # Number of tokens in the input tensor.
        num_tokens = x.shape[0]
        # Factor to account for the imbalance of the experts.
        # factor equals to the
        # max_real_num_tokens_per_expert / perfect_num_tokens_per_expert
        # 1.0 means perfect expert distribution.
        # > 1.0 means some experts have more tokens than the perfect
        # distribution.
        # < 1.0 does not make sense.
        imbalance_factor = 1.3
        # Calculate the number of tokens per expert assuming perfect
        # distribution.
        num_tokens_per_expert = (num_tokens * top_k) // self.num_experts
        # Apply the imbalance factor.
        num_tokens_per_expert = int(num_tokens_per_expert * imbalance_factor)
        # And pad the number to the next power of 2.
        tile_tokens_dim = next_power_of_2(num_tokens_per_expert)
        # Cap to 8-64 tokens per CTA tile as it's the range supported by the
        #  kernel.
        tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)

        return tile_tokens_dim

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
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        w1_bias: Optional[torch.Tensor],
        w2_bias: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ):
        # rank = get_ep_group().rank_in_group
        # low = rank * self.num_experts
        # high = low + self.num_experts - 1

        # mask = (topk_ids >= low) & (topk_ids <= high)
        # topk_ids[mask] = -1
        # topk_weights[mask] = 0.0

        topk = topk_weights.shape[-1]
        required_keys = ['gemm1_alpha', 'gemm1_beta', 'gemm1_clamp_limit']

        gemm1_alpha, gemm1_beta, gemm1_clamp_limit = (extract_required_args(
            extra_expert_args, required_keys))

        packed_tensor = (topk_ids.to(torch.int32) << 16) | topk_weights.view(
            torch.int16).to(torch.int32)
        if envs.VLLM_USE_FLASHINFER_MXFP4_BF16_MOE:
            assert hidden_states.dtype == torch.bfloat16
            x_quant = hidden_states
            x_scale = None
        else:
            x_quant, x_scale = mxfp8_quantize(hidden_states, False)  # to mxfp8
            x_scale = x_scale.view(torch.float8_e4m3fn).reshape(-1)

        trtllm_gen_output = trtllm_fp4_block_scale_routed_moe(
            packed_tensor,
            None,  # routing_bias
            x_quant,
            x_scale,
            w1,  # uint8 (e2m1 x 2)
            w1_scale,  # uint8 (e4m3 x 2)
            w1_bias,  # fp32 per expert per channel
            gemm1_alpha,  # fp32 per expert
            gemm1_beta,  # fp32 per expert
            gemm1_clamp_limit,  # fp32 per expert
            w2,  # uint8 (e2m1 x 2)
            w2_scale,  # ue8m0
            w2_bias,  # fp32 per expert per channel
            None,  # output1_scale_scalar
            None,  # output1_scale_gate_scalar
            None,  # output2_scale_scalar
            self.num_experts,
            topk,
            None,  # n_group
            None,  # topk_group
            self.intermediate_size,  # padded to multiple of 256
            0,  # local_expert_offset
            self.num_experts,  # local num experts
            None,
            self._get_tile_tokens_dim(hidden_states, topk),
            1,  # routing_method_type, renormalize
            True,  # do finalize
            output,
        )[0]

        return trtllm_gen_output
