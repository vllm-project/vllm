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


class TokenSpeedMxfp4Experts(mk.FusedMoEExpertsMonolithic):
    """TokenSpeed GFX950 MXFP4 MoE backend."""

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
        self.swiglu_beta = (
            1.0 if moe_config.swiglu_beta is None else moe_config.swiglu_beta
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
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if moe_config.in_dtype not in (torch.float16, torch.bfloat16):
            return (
                False,
                f"kernel does not support {moe_config.in_dtype} input/output dtype",
            )

        return mk.FusedMoEExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )

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
            and not moe_parallel_config.use_ep
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
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "TokenSpeed MXFP4 MoE does not support apply_router_weight_on_input."
            )
        if self.a1_scale is None or self.a2_scale is None:
            raise ValueError(
                "TokenSpeed MXFP4 MoE requires static FP8 activation scales; "
                "w13_input_scale and w2_input_scale must be loaded before dispatch."
            )

        enable_warp_decode = True
        from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950 import (
            gluon_mxfp_fused_moe,
        )

        return gluon_mxfp_fused_moe(
            hidden_states,
            router_logits,
            w1,
            w2,
            w13_mx_scale=self.quant_config._w1.scale,
            w2_mx_scale=self.quant_config._w2.scale,
            w13_act_scale=self.a1_scale,
            w2_act_scale=self.a2_scale,
            top_k=self.topk,
            w13_bias=self.w1_bias,
            w2_bias=self.w2_bias,
            out_dtype=hidden_states.dtype,
            enable_warp_decode=enable_warp_decode,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_limit=self.swiglu_limit,
            swiglu_beta=self.swiglu_beta,
        )
