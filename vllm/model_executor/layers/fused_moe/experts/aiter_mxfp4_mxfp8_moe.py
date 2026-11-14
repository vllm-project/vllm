# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kMxfp8Dynamic,
)

__all__ = [
    "AiterMxfp4Mxfp8ExpertsMonolithic",
    "aiter_mxfp4_mxfp8_moe_forward",
    "quark_mxfp4_to_a8w4_layout",
]


def _interleave_gate_up_rows(t: torch.Tensor) -> torch.Tensor:
    """Repack [E, 2n, X] from [all gates | all ups] to [g0, u0, g1, u1, ...]."""
    num_experts, two_n, inner = t.shape
    return (
        t.view(num_experts, 2, two_n // 2, inner)
        .transpose(1, 2)
        .contiguous()
        .view(num_experts, two_n, inner)
    )


def quark_mxfp4_to_a8w4_layout(
    weight: torch.Tensor,
    scale: torch.Tensor,
    interleave_gate_up: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn checkpoint MXFP4 into the layout ``moe_gemm_a8w4`` reads.

    The bytes are not re-quantized. Quark stores packed e2m1 as ``[E, N, K/2]``
    and e8m0 scales as ``[E, N, K/32]``. The kernel wants both column-major in
    K (``stride(-2) == 1`` for the weights, ``stride(1) == 1`` for the scales).
    ``SWIGLUOAI_UNINTERLEAVE`` checkpoints store gate and up as two halves; the
    fused SwiGLU reads even/odd columns, so those rows are interleaved first.
    """
    # dynamic_mxfp4_quant leaves scales as a transpose. A checkpoint is
    # contiguous, but pack here too so a non-contiguous view is not indexed
    # with the wrong K stride.
    weight = weight.view(torch.uint8).contiguous()
    scale = scale.view(torch.uint8).contiguous()
    if interleave_gate_up:
        weight = _interleave_gate_up_rows(weight)
        scale = _interleave_gate_up_rows(scale)
    weight = weight.transpose(1, 2)
    scale = scale.transpose(1, 2)
    if weight.stride(-2) != 1 or scale.stride(1) != 1:
        raise RuntimeError(
            "MXFP4 to a8w4 layout lost column-major K: "
            f"weight stride {weight.stride()}, scale stride {scale.stride()}"
        )
    return weight, scale


def _swiglu_args(
    quant_config: FusedMoEQuantConfig, activation: MoEActivation
) -> tuple[float, float, bool]:
    """Match the emulation activation, which the two kernels do not share.

    Interleaved SwiGLU-OAI goes through ``swigluoai_and_mul``, which always
    uses alpha 1.702, limit 7 and ``(up + 1)``. The uninterleaved op uses
    ``silu_and_mul_with_clamp`` and therefore the quant-config alpha, limit
    and beta. ``moe_gemm_a8w4`` can add either 0 or 1 to the up projection.
    """
    if activation == MoEActivation.SWIGLUOAI:
        return 1.702, 7.0, True
    alpha = 1.0 if quant_config.gemm1_alpha is None else float(quant_config.gemm1_alpha)
    limit = quant_config.gemm1_clamp_limit
    if limit is None:
        raise ValueError("SWIGLUOAI_UNINTERLEAVE requires gemm1_clamp_limit")
    beta = 0.0 if quant_config.gemm1_beta is None else float(quant_config.gemm1_beta)
    if math.isclose(beta, 0.0):
        add_residual = False
    elif math.isclose(beta, 1.0):
        add_residual = True
    else:
        raise ValueError(
            "moe_gemm_a8w4 SwiGLU can add 0 or 1 to the up projection, "
            f"got gemm1_beta={beta}"
        )
    return alpha, float(limit), add_residual


def aiter_mxfp4_mxfp8_moe_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    quant_config: FusedMoEQuantConfig,
    apply_router_weight_on_input: bool,
    activation: MoEActivation,
    score_mode: str | None = None,
    e_score_correction_bias: torch.Tensor | None = None,
    routed_scaling_factor: float | None = None,
) -> torch.Tensor:
    """Two ``moe_gemm_a8w4`` calls: SwiGLU writes MXFP8, the second GEMM consumes it."""
    from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
    from aiter.ops.triton.moe.moe_routing.routing import routing
    from aiter.ops.triton.quant import dynamic_mxfp8_quant

    if score_mode is None:
        routing_data, gather_idx, scatter_idx = routing(
            gating_output, topk, sm_first=not renormalize
        )
    else:
        # Sigmoid scores plus a per-expert correction bias (MiniMax2). The
        # bias only picks the experts; the returned weights stay unbiased.
        routing_data, gather_idx, scatter_idx = routing(
            gating_output,
            topk,
            score_mode=score_mode,
            bias=(
                e_score_correction_bias.float()
                if e_score_correction_bias is not None
                else None
            ),
            renorm=renormalize,
            routed_scaling_factor=(
                1.0 if routed_scaling_factor is None else routed_scaling_factor
            ),
        )
    gammas = routing_data.gate_scal
    alpha, limit, add_residual = _swiglu_args(quant_config, activation)
    w1_scale = quant_config.w1_scale
    w2_scale = quant_config.w2_scale
    assert w1_scale is not None and w2_scale is not None

    hidden_q, hidden_scale = dynamic_mxfp8_quant(
        hidden_states, quant_dtype=torch.float8_e4m3fn
    )
    intermediate, intermediate_scale = moe_gemm_a8w4(
        hidden_q,
        w1,
        hidden_scale,
        w1_scale,
        None,
        None,
        quant_config.w1_bias,
        routing_data,
        gather_indx=gather_idx,
        gammas=gammas if apply_router_weight_on_input else None,
        swizzle_mx_scale=None,
        apply_swiglu=True,
        alpha=alpha,
        limit=limit,
        swiglu_add_residual=add_residual,
        out_mx_quant=True,
    )
    return moe_gemm_a8w4(
        intermediate,
        w2,
        intermediate_scale,
        w2_scale,
        None,
        None,
        quant_config.w2_bias,
        routing_data,
        scatter_indx=scatter_idx,
        gammas=None if apply_router_weight_on_input else gammas,
        swizzle_mx_scale=None,
        out_dtype=torch.bfloat16,
    )


class AiterMxfp4Mxfp8ExpertsMonolithic(mk.FusedMoEExpertsMonolithic):
    """MXFP4 weights with dynamic per-group MXFP8 activations, via AITER.

    ``moe_gemm_a8w4`` quantizes the SwiGLU output to MXFP8 as it writes GEMM1,
    and GEMM2 consumes that pair. Weights stay the checkpoint's MXFP4 bytes;
    only the layout changes. gfx1250's gluon path transposes the weight, which
    would undo that layout, and has not been checked, so this is gfx950 only.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.topk = moe_config.experts_per_token
        self.renormalize = moe_config.routing_method in (
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
            RoutingMethodType.MiniMax2,
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        if not rocm_aiter_ops.is_enabled():
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
        return (weight_key, activation_key) == (kMxfp4Static, kMxfp8Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        )

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
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
        return routing_method in (
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
            RoutingMethodType.MiniMax2,
        )

    @staticmethod
    def is_supported_config(
        cls,
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = super().is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        beta = moe_config.swiglu_beta
        if (
            supported
            and moe_config.activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE
            and beta is not None
            and not (math.isclose(beta, 0.0) or math.isclose(beta, 1.0))
        ):
            return False, (
                "moe_gemm_a8w4 SwiGLU can add 0 or 1 to the up projection, "
                f"got swiglu_beta={beta}"
            )
        return supported, reason

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
        input_ids: torch.Tensor | None = None,
        hash_indices_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del (
            global_num_experts,
            a1q_scale,
            num_expert_group,
            topk_group,
            routing_replay_out,
            input_ids,
            hash_indices_table,
        )
        if expert_map is not None:
            raise NotImplementedError(
                "Expert parallelism is not supported for AITER MXFP4 MXFP8 MoE."
            )
        score_mode = None
        if self.moe_config.routing_method == RoutingMethodType.MiniMax2:
            assert e_score_correction_bias is not None, (
                "MiniMax2 routing requires e_score_correction_bias"
            )
            score_mode = "sigmoid"
        return aiter_mxfp4_mxfp8_moe_forward(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            gating_output=router_logits,
            topk=self.topk,
            renormalize=self.renormalize,
            quant_config=self.quant_config,
            apply_router_weight_on_input=apply_router_weight_on_input,
            activation=activation,
            score_mode=score_mode,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
        )
