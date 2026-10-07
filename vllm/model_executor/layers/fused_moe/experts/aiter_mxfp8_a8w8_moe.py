# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 MoE on gfx942 via AITER ``moe_gemm_a8w8``.

Expert weights stay MXFP8. gfx942 has no MX matrix cores, so AITER runs each
32-wide K step as an FP8 MFMA and applies the E8M0 scales to the accumulator.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.platforms import current_platform

_SIGMOID_ROUTING = (RoutingMethodType.DeepSeekV3, RoutingMethodType.MiniMax2)
_SOFTMAX_ROUTING = (RoutingMethodType.Renormalize, RoutingMethodType.RenormalizeNaive)


def prepare_mxfp8_moe_weights_for_aiter_a8w8(
    w: torch.Tensor, w_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert MXFP8 expert weights to the layout ``moe_gemm_a8w8`` expects.

    On fnuz platforms the e4m3fn bits are reused as e4m3fnuz, which halves
    their value, so each E8M0 exponent is incremented and 0x80 (fnuz NaN)
    is zeroed. Scales are stored N-contiguous so per-K-step loads coalesce.
    """
    scale = w_scale.view(torch.uint8)
    if current_platform.is_fp8_fnuz() and w.dtype == torch.float8_e4m3fn:
        w_i8 = w.view(torch.int8)
        w_i8.masked_fill_(w_i8 == -128, 0)
        w = w_i8.view(torch.float8_e4m3fnuz)
        scale = (scale.to(torch.int16) + 1).clamp(max=254).to(torch.uint8)
    return w, scale.transpose(1, 2).contiguous().transpose(1, 2)


def _mxfp8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.quant.quant import dynamic_mxfp8_quant

    return dynamic_mxfp8_quant(
        x, quant_dtype=current_platform.fp8_dtype(), backend="triton"
    )


def aiter_mxfp8_a8w8_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    routing_data,
    gather_indx: torch.Tensor,
    scatter_indx: torch.Tensor,
    activation: MoEActivation,
    activation_config: ApplyMoEActivationConfig,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    from aiter.ops.triton.moe.moe_op_gemm_a8w8 import moe_gemm_a8w8

    gammas = routing_data.gate_scal
    xq, xs = _mxfp8_quant(hidden_states)
    gate_up = moe_gemm_a8w8(
        xq,
        w13.transpose(1, 2),
        xs,
        w13_scale.transpose(1, 2),
        routing_data=routing_data,
        gather_indx=gather_indx,
        gammas=gammas if apply_router_weight_on_input else None,
        out_dtype=hidden_states.dtype,
    )
    act = gate_up.new_empty(gate_up.shape[0], gate_up.shape[1] // 2)
    apply_moe_activation(activation, act, gate_up, activation_config=activation_config)
    aq, a_s = _mxfp8_quant(act)
    return moe_gemm_a8w8(
        aq,
        w2.transpose(1, 2),
        a_s,
        w2_scale.transpose(1, 2),
        routing_data=routing_data,
        scatter_indx=scatter_indx,
        gammas=None if apply_router_weight_on_input else gammas,
        out_dtype=hidden_states.dtype,
    )


class AiterMxfp8A8W8ExpertsMonolithic(mk.FusedMoEExpertsMonolithic):
    """Router logits -> AITER fused top-k and sort -> two ``moe_gemm_a8w8``."""

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config, quant_config)
        self.topk = moe_config.experts_per_token

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        if not (current_platform.is_rocm() and rocm_aiter_ops.is_enabled()):
            return False
        from vllm.platforms.rocm import on_gfx942

        return on_gfx942()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp8Static, kMxfp8Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        )

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_ep
            and not moe_parallel_config.enable_eplb
            and moe_parallel_config.dp_size <= 1
        )

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in _SIGMOID_ROUTING + _SOFTMAX_ROUTING

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
        routing_replay_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from aiter.ops.triton.moe.moe_routing.routing import routing

        routing_method = self.moe_config.routing_method
        if routing_method in _SIGMOID_ROUTING:
            bias = e_score_correction_bias
            routing_data, gather_indx, scatter_indx = routing(
                router_logits.float(),
                self.topk,
                score_mode="sigmoid",
                bias=None if bias is None else bias.float(),
                renorm=True,
                routed_scaling_factor=routed_scaling_factor or 1.0,
                use_grouped_topk=(num_expert_group or 1) > 1,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
            )
        else:
            routing_data, gather_indx, scatter_indx = routing(
                router_logits,
                self.topk,
                sm_first=routing_method == RoutingMethodType.RenormalizeNaive,
            )
        return aiter_mxfp8_a8w8_moe(
            hidden_states,
            w1,
            self.quant_config.w1_scale,
            w2,
            self.quant_config.w2_scale,
            routing_data,
            gather_indx,
            scatter_indx,
            activation,
            self.activation_config,
            apply_router_weight_on_input,
        )
