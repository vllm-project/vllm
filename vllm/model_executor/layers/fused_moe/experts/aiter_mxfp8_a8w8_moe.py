# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 MoE on gfx942 via AITER `moe_gemm_a8w8`.

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
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3fn_to_e4m3fnuz,
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
    """Convert [E, N, K] MXFP8 expert weights to the [E, K, N] layout of
    `moe_gemm_a8w8`. The weight is a transposed view; the scale is made
    N-contiguous so per-K-step scale loads coalesce."""
    E, N, K = w.shape
    assert w_scale.shape == (E, N, K // MXFP8_BLOCK_SIZE), (
        f"expected [E, N, K] weights, got {tuple(w.shape)} / {tuple(w_scale.shape)}"
    )
    scale = w_scale.view(torch.uint8)
    if current_platform.is_fp8_fnuz() and w.dtype == torch.float8_e4m3fn:
        w, scale = mxfp8_e4m3fn_to_e4m3fnuz(w, scale)
    return w.transpose(1, 2), scale.transpose(1, 2).contiguous()


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
        w13,
        xs,
        w13_scale,
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
        w2,
        a_s,
        w2_scale,
        routing_data=routing_data,
        scatter_indx=scatter_indx,
        gammas=None if apply_router_weight_on_input else gammas,
        out_dtype=hidden_states.dtype,
    )


class AiterMxfp8A8W8ExpertsMonolithic(mk.FusedMoEExpertsMonolithic):
    """Router logits -> AITER fused top-k and sort -> two `moe_gemm_a8w8`."""

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config, quant_config)
        self.topk = moe_config.experts_per_token

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        if not rocm_aiter_ops.is_fused_moe_enabled():
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

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        # moe_gemm_a8w8 is called without expert biases, and AITER's top-k
        # cannot add the fused shared-expert slots the modular routers append.
        if moe_config.has_bias:
            return False, "kernel does not support bias"
        if moe_config.is_fused_shared_expert_requested:
            return False, "kernel does not support fused shared experts"
        return mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )

    def supports_routing_replay_capture(self) -> bool:
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

        if self.moe_config.routing_method in _SIGMOID_ROUTING:
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
            # Softmax over all experts then renormalizing the top-k equals a
            # softmax over the top-k logits, so both methods use sm_first=False.
            assert (num_expert_group or 1) == 1, "grouped softmax top-k unsupported"
            assert routed_scaling_factor in (None, 1.0), "softmax scaling unsupported"
            routing_data, gather_indx, scatter_indx = routing(
                router_logits, self.topk, sm_first=False
            )
        if routing_replay_out is not None:
            topk_ids = routing_data.topk_ids
            routing_replay_out[: topk_ids.shape[0]].copy_(topk_ids)
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
