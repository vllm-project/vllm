# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)


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
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (
            mk.FusedMoEActivationFormat.Standard,
            mk.FusedMoEActivationFormat.Standard,
        )

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # The workspaces for this implementation are managed by flashinfer.
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, K)
        return (workspace1, workspace2, output)

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
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        topk = topk_ids.size(-1)
        local_num_experts = w1.size(0)
        intermediate_size = w2.size(1)
        local_expert_offset = self.moe.ep_rank * local_num_experts

        x_quant = hidden_states
        x_scale = a1q_scale
        if x_scale is not None:
            x_scale = x_scale.view(torch.float8_e4m3fn).reshape(*x_quant.shape[:-1], -1)

        packed_tensor = (topk_ids.to(torch.int32) << 16) | topk_weights.to(
            torch.bfloat16
        ).view(torch.int16)

        assert self.w1_scale is not None
        assert self.w2_scale is not None
        kwargs = {
            "topk_ids": packed_tensor,
            "routing_bias": None,
            "hidden_states": x_quant,
            "hidden_states_scale": x_scale,
            "gemm1_weights": w1,
            "gemm1_weights_scale": self.w1_scale,
            "gemm1_bias": self.w1_bias,
            "gemm1_alpha": self.gemm1_alpha,
            "gemm1_beta": self.gemm1_beta,
            "gemm1_clamp_limit": self.gemm1_clamp_limit,
            "gemm2_weights": w2,
            "gemm2_weights_scale": self.w2_scale,
            "gemm2_bias": self.w2_bias,
            "output1_scale_scalar": None,
            "output1_scale_gate_scalar": None,
            "output2_scale_scalar": None,
            "num_experts": global_num_experts,
            "top_k": topk,
            "n_group": None,
            "topk_group": None,
            "intermediate_size": intermediate_size,
            "local_expert_offset": local_expert_offset,
            "local_num_experts": local_num_experts,
            "routed_scaling_factor": None,
            "tile_tokens_dim": None,
            "routing_method_type": 1,
            "do_finalize": True,
            "output": output,
            "tune_max_num_tokens": max(self.max_capture_size, 1),
        }

        from flashinfer import trtllm_fp4_block_scale_routed_moe

        from vllm.utils.flashinfer import autotune

        with autotune(False):
            # Enable autotune when,
            # https://github.com/flashinfer-ai/flashinfer/issues/2023 is
            # resolved.
            trtllm_fp4_block_scale_routed_moe(**kwargs)

        return output
