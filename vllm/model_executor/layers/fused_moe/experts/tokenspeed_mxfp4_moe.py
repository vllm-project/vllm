# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kMxfp4Static,
)
from vllm.platforms import current_platform


class TokenSpeedMxfp4ExpertsMonolithic(mk.FusedMoEExpertsMonolithic):
    """Monolithic TokenSpeed GFX950 MXFP4 MoE backend."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.topk = moe_config.experts_per_token
        self.swiglu_alpha = (
            1.702 if moe_config.swiglu_alpha is None else moe_config.swiglu_alpha
        )
        self.swiglu_limit = (
            7.0 if moe_config.swiglu_limit is None else moe_config.swiglu_limit
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        if not current_platform.is_rocm():
            return False
        from vllm.platforms.rocm import on_gfx950

        return on_gfx950()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, kFp8StaticTensorSym)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SWIGLUOAI

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
            and moe_parallel_config.dp_size <= 1
        )

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in (
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        )

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        if expert_map is not None:
            raise NotImplementedError(
                "TokenSpeed MXFP4 MoE does not support expert_map/expert "
                "parallel routing yet."
            )
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "TokenSpeed MXFP4 MoE does not support apply_router_weight_on_input."
            )

        # TODO: Re-enable after debugging CUDA graph failures in the small-M
        # warp-decode path.
        enable_warp_decode = False
        w13_precision_config = self.quant_config._w1.scale
        w2_precision_config = self.quant_config._w2.scale
        w13_kernel_weight = getattr(w1, "_gluon_shuffled", None)
        w13_is_preshuffled = w13_kernel_weight is not None or bool(
            getattr(w1, "is_shuffled_for_gluon_dot", False)
        )
        if w13_kernel_weight is None:
            w13_kernel_weight = w1

        w2_kernel_weight = getattr(w2, "_gluon_shuffled", None)
        w2_is_preshuffled = w2_kernel_weight is not None or bool(
            getattr(w2, "is_shuffled_for_gluon_dot", False)
        )
        if w2_kernel_weight is None:
            w2_kernel_weight = w2

        w13_original_k_pk = int(
            getattr(w13_kernel_weight, "original_k_pk", hidden_states.shape[1] // 2)
        )
        w2_original_k_pk = int(
            getattr(w2_kernel_weight, "original_k_pk", w13_kernel_weight.shape[-1] // 4)
        )
        w2_original_n = int(
            getattr(w2_kernel_weight, "original_n", hidden_states.shape[1])
        )

        if w13_is_preshuffled:
            w13_kernel_weight.is_shuffled_for_gluon_dot = True
            w13_kernel_weight.original_k_pk = w13_original_k_pk
            w13_kernel_weight.gluon_dot_block_k_pk = 128
            w13_kernel_weight.gluon_dot_block_n = 128
        if w2_is_preshuffled:
            w2_kernel_weight.is_shuffled_for_gluon_dot = True
            w2_kernel_weight.original_k_pk = w2_original_k_pk
            w2_kernel_weight.original_n = w2_original_n
            w2_kernel_weight.gluon_dot_block_k_pk = 128
            w2_kernel_weight.gluon_dot_block_n = 128

        from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950 import (
            gluon_mxfp_fused_moe,
        )

        return gluon_mxfp_fused_moe(
            hidden_states,
            router_logits,
            w13_kernel_weight,
            w2_kernel_weight,
            w13_bias=self.w1_bias,
            w2_bias=self.w2_bias,
            w13_precision_config=w13_precision_config,
            w2_precision_config=w2_precision_config,
            w13_act_scale=self.a1_scale,
            w2_act_scale=self.a2_scale,
            top_k=self.topk,
            enable_warp_decode=enable_warp_decode,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_limit=self.swiglu_limit,
        )
