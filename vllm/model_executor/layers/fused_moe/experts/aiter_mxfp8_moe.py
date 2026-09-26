# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 (1x32 block, E8M0) MoE via AITER's FlyDSL two-stage grouped GEMM
(gfx950); alternative to ``Mxfp8NativeTritonExperts``. Routes through
``aiter.fused_moe`` (per_1x32, gate_mode=INTERLEAVE); weights are preshuffled in
``convert_to_fp8_moe_kernel_format``.
"""

import math

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.mxfp8_emulation_moe import (
    Mxfp8TritonExpertsBase,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

# AITER spells the two gate/up activations as different ActivationTypes, and
# each pins its own (alpha, beta). Both clamp gate/up to +-swiglu_limit.
#
#   SWIGLUOAI_UNINTERLEAVE -> ActivationType.Swiglu (gpt-oss / MiniMax-M3)
#       gate*sigmoid(1.702*gate) * (up + 1.0)
#   SILU                   -> ActivationType.Silu   (Hy4)
#       silu(clamp(gate, max=L)) * clamp(up, -L, L)
#
# The SiLU form is SiluAndMulWithClamp at its alpha=1.0/beta=0.0 defaults, and
# is exactly what AITER evaluates for ActivationType.Silu with a finite
# swiglu_limit -- see aiter/ops/flydsl/moe_common.py::apply_gate_up.
_AITER_SWIGLU_ALPHA = 1.702
_AITER_SWIGLU_BETA = 1.0
_CLAMPED_SILU_ALPHA = 1.0
_CLAMPED_SILU_BETA = 0.0

# activation -> (alpha, beta, requires_swiglu_limit)
_SUPPORTED_ACTIVATIONS: dict[MoEActivation, tuple[float, float, bool]] = {
    MoEActivation.SWIGLUOAI_UNINTERLEAVE: (
        _AITER_SWIGLU_ALPHA,
        _AITER_SWIGLU_BETA,
        False,
    ),
    MoEActivation.SILU: (_CLAMPED_SILU_ALPHA, _CLAMPED_SILU_BETA, True),
}


def _check_activation(moe_config) -> str | None:
    """Why this activation cannot run, or None if it can.

    alpha/beta are matched against the values the chosen activation implies.
    ``None`` means "unset", which is only legal where it coincides with that
    activation's own defaults -- so a SwiGLU-OAI config that never set
    alpha/beta is still rejected rather than silently run as clamped SiLU.
    """
    spec = _SUPPORTED_ACTIVATIONS.get(moe_config.activation)
    if spec is None:
        return (
            f"kernel supports {sorted(a.value for a in _SUPPORTED_ACTIVATIONS)}; "
            f"got activation={moe_config.activation.value}"
        )
    alpha, beta, needs_limit = spec
    # SiluAndMulWithClamp's own defaults; an unset field means these.
    for name, want, got, unset_default in (
        ("swiglu_alpha", alpha, moe_config.swiglu_alpha, _CLAMPED_SILU_ALPHA),
        ("swiglu_beta", beta, moe_config.swiglu_beta, _CLAMPED_SILU_BETA),
    ):
        effective = unset_default if got is None else float(got)
        if not math.isclose(effective, want):
            return (
                f"activation={moe_config.activation.value} requires "
                f"{name}={want}; got {got}"
            )
    if needs_limit and moe_config.swiglu_limit is None:
        return (
            f"activation={moe_config.activation.value} is the clamped variant "
            "and requires a swiglu_limit; got None"
        )
    return None


class AiterMxfp8Experts(Mxfp8TritonExpertsBase):
    """MXFP8 MoE through AITER's FlyDSL two-stage grouped GEMM (gfx950)."""

    consumes_expert_mask = True

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return self.quant_config.quant_dtype

    @property
    def block_shape(self) -> list[int] | None:
        return self.quant_config.block_shape

    @property
    def expects_unquantized_inputs(self) -> bool:
        # aiter.fused_moe MXFP8-quantizes the activations internally.
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.supports_mx() and rocm_aiter_ops.is_fused_moe_enabled()

    @staticmethod
    def _supports_parallel_config(moe_parallel_config) -> bool:
        return True

    @staticmethod
    def is_supported_config(
        cls, moe_config, weight_key, activation_key, activation_format
    ):
        is_supported, reason = super().is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if is_supported:
            why = _check_activation(moe_config)
            if why is not None:
                return False, why
        return is_supported, reason

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        from aiter import ActivationType, QuantType
        from aiter.ops.flydsl.moe_common import GateMode

        from vllm._aiter_ops import rocm_aiter_ops

        # make_fp8_moe_quant_config maps swiglu_limit -> gemm1_clamp_limit for
        # block_shape [1, 32], so the limit is already here for MXFP8.
        limit = self.quant_config.gemm1_clamp_limit
        swiglu_limit = 0.0 if limit is None else float(limit)

        # is_supported_config has already pinned activation -> (alpha, beta),
        # so the enum alone selects the AITER activation.
        is_clamped_silu = self.moe_config.activation == MoEActivation.SILU

        # RoutedExperts.expert_map hands AITER experts the precomputed 0/1
        # expert_mask (with trailing sentinel) instead of the vLLM expert_map.
        expert_mask = expert_map

        # Route through the graph-safe ``rocm_aiter_fused_moe`` custom op so the
        # call is captured under HIP graphs / torch.compile (a direct
        # ``aiter.fused_moe`` is opaque to the dispatcher). aiter requires FP32
        # routing weights / INT32 ids.
        out = rocm_aiter_ops.fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weights.to(torch.float32),
            topk_ids.to(torch.int32),
            expert_mask=expert_mask,
            activation_method=(
                ActivationType.Silu.value
                if is_clamped_silu
                else ActivationType.Swiglu.value
            ),
            quant_method=QuantType.per_1x32.value,
            doweight_stage1=apply_router_weight_on_input,
            w1_scale=self.w1_scale_val,
            w2_scale=self.w2_scale_val,
            a1_scale=None,
            a2_scale=None,
            gate_mode=GateMode.INTERLEAVE.value,
            swiglu_limit=swiglu_limit,
            output_dtype=output.dtype,
        )
        output.copy_(out.to(output.dtype))
