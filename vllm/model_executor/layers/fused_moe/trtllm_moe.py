# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (FusedMoEConfig,
                                                         FusedMoEQuantConfig)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP)
from vllm.utils import next_power_of_2


class TrtLlmGenExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(
        self,
        moe: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        max_capture_size,
    ):
        super().__init__(quant_config)
        self.moe = moe
        self.gemm1_alpha = gemm1_alpha
        self.gemm1_beta = gemm1_beta
        self.gemm1_clamp_limit = gemm1_clamp_limit
        self.max_capture_size = max_capture_size

    @property
    def activation_formats(
        self
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (mk.FusedMoEActivationFormat.Standard,
                mk.FusedMoEActivationFormat.Standard)

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

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
        # The workspaces for this implementation are managed by flashinfer.
        # TODO(varun) : workspace1 is could be used as the output tensor. This
        # is error-prone. Allow the `workspace_shapes` to return None workspaces
        workspace1 = (M, K)
        workspace2 = (0, 0)
        output = (M, K)
        return (workspace1, workspace2, output, a.dtype)

    def _get_tile_tokens_dim(self, x: torch.Tensor, top_k: int,
                             local_num_experts: int):
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
        num_tokens_per_expert = (num_tokens * top_k) // local_num_experts
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
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
    ):
        topk = topk_ids.size(-1)
        local_num_experts = w1.size(0)
        intermediate_size = w2.size(1)
        local_expert_offset = self.moe.ep_rank * local_num_experts

        x_quant = hidden_states
        x_scale = a1q_scale
        if x_scale is not None:
            x_scale = x_scale.view(torch.float8_e4m3fn).reshape(
                *x_quant.shape[:-1], -1)

        packed_tensor = (topk_ids.to(torch.int32) << 16) | topk_weights.to(
            torch.bfloat16).view(torch.int16)

        assert self.w1_scale is not None
        assert self.w2_scale is not None
        kwargs = {
            "topk_ids":
            packed_tensor,
            "routing_bias":
            None,
            "hidden_states":
            x_quant,
            "hidden_states_scale":
            x_scale,
            "gemm1_weights":
            w1,
            "gemm1_weights_scale":
            self.w1_scale,
            "gemm1_bias":
            self.w1_bias,
            "gemm1_alpha":
            self.gemm1_alpha,
            "gemm1_beta":
            self.gemm1_beta,
            "gemm1_clamp_limit":
            self.gemm1_clamp_limit,
            "gemm2_weights":
            w2,
            "gemm2_weights_scale":
            self.w2_scale,
            "gemm2_bias":
            self.w2_bias,
            "output1_scale_scalar":
            None,
            "output1_scale_gate_scalar":
            None,
            "output2_scale_scalar":
            None,
            "num_experts":
            global_num_experts,
            "top_k":
            topk,
            "n_group":
            None,
            "topk_group":
            None,
            "intermediate_size":
            intermediate_size,
            "local_expert_offset":
            local_expert_offset,
            "local_num_experts":
            local_num_experts,
            "routed_scaling_factor":
            None,
            "tile_tokens_dim":
            self._get_tile_tokens_dim(x_quant, topk, local_num_experts),
            "routing_method_type":
            1,
            "do_finalize":
            True,
            "output":
            output,
            "tune_max_num_tokens":
            self.max_capture_size,
        }

        from flashinfer import trtllm_fp4_block_scale_routed_moe
        trtllm_fp4_block_scale_routed_moe(**kwargs)
        return output
