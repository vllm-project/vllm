# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8DynamicTensorSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

if current_platform.is_xpu():
    from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe


class XPUExperts(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        self.is_fp8 = False

    @staticmethod
    def expects_unquantized_inputs(
        fused_moe_config: mk.FusedMoEConfig, quant_config: FusedMoEQuantConfig
    ) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_xpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: str) -> bool:
        return activation in ["silu", "gelu", "swigluoai"]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (None, None),
            (kFp8StaticTensorSym, None),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    def supports_chunking(self) -> bool:
        return False

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
        xpu_fused_moe(
            hidden_states=hidden_states,
            w13=w1,
            w13_scales=self.w1_scale,
            w13_bias=self.w1_bias,
            w2=w2,
            w2_scales=self.w2_scale,
            w2_bias=self.w2_bias,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            n_experts_per_token=topk,
            activation=activation,
            num_experts=self.moe_config.num_local_experts,
            ep_rank=self.moe_config.ep_rank,
            ep_size=self.moe_config.ep_size,
            output=output,
            is_fp8=self.is_fp8,
        )


class XPUExpertsFp8(XPUExperts):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        self.is_fp8 = True
