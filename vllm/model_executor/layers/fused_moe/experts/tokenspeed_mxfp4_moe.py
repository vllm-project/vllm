# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from tokenspeed_kernel_amd.ops.moe.fused_mxfp_gfx950 import gluon_mxfp_fused_moe

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
from vllm.utils.torch_utils import direct_register_custom_op

_DTYPE_TO_CODE = {
    torch.bfloat16: 0,
    torch.float16: 1,
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}


def _tokenspeed_mxfp4_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_mx_scale: torch.Tensor,
    w2_mx_scale: torch.Tensor,
    w13_act_scale: torch.Tensor,
    w2_act_scale: torch.Tensor,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    top_k: int,
    out_dtype_code: int,
    enable_warp_decode: bool,
    swiglu_alpha: float,
    swiglu_limit: float,
) -> torch.Tensor:
    return gluon_mxfp_fused_moe(
        hidden_states,
        router_logits,
        w13_weight,
        w2_weight,
        w13_mx_scale=w13_mx_scale,
        w2_mx_scale=w2_mx_scale,
        w13_act_scale=w13_act_scale,
        w2_act_scale=w2_act_scale,
        top_k=top_k,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        out_dtype=_CODE_TO_DTYPE[out_dtype_code],
        enable_warp_decode=enable_warp_decode,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
    )


def _tokenspeed_mxfp4_moe_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_mx_scale: torch.Tensor,
    w2_mx_scale: torch.Tensor,
    w13_act_scale: torch.Tensor,
    w2_act_scale: torch.Tensor,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    top_k: int,
    out_dtype_code: int,
    enable_warp_decode: bool,
    swiglu_alpha: float,
    swiglu_limit: float,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="tokenspeed_mxfp4_moe",
    op_func=_tokenspeed_mxfp4_moe,
    fake_impl=_tokenspeed_mxfp4_moe_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


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

        enable_warp_decode = True
        out_dtype_code = _DTYPE_TO_CODE[torch.bfloat16]

        return torch.ops.vllm.tokenspeed_mxfp4_moe(
            hidden_states,
            router_logits,
            w1,
            w2,
            self.quant_config._w1.scale,
            self.quant_config._w2.scale,
            self.a1_scale,
            self.a2_scale,
            self.w1_bias,
            self.w2_bias,
            self.topk,
            out_dtype_code,
            enable_warp_decode,
            self.swiglu_alpha,
            self.swiglu_limit,
        )
