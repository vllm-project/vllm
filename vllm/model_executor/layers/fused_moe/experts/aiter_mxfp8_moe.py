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

_AITER_SWIGLU_ALPHA = 1.702
_AITER_SWIGLU_BETA = 1.0


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
        if (
            is_supported
            and moe_config.activation != MoEActivation.SWIGLUOAI_UNINTERLEAVE
        ):
            return False, (
                "kernel hardcodes SwiGLU-OAI activation and requires "
                f"activation={MoEActivation.SWIGLUOAI_UNINTERLEAVE.value}; "
                f"got activation={moe_config.activation.value}"
            )
        if is_supported and (
            moe_config.swiglu_alpha is None
            or not math.isclose(float(moe_config.swiglu_alpha), _AITER_SWIGLU_ALPHA)
            or moe_config.swiglu_beta is None
            or not math.isclose(float(moe_config.swiglu_beta), _AITER_SWIGLU_BETA)
        ):
            return False, (
                "kernel hardcodes SwiGLU-OAI with "
                f"alpha={_AITER_SWIGLU_ALPHA} and beta={_AITER_SWIGLU_BETA}; "
                f"got swiglu_alpha={moe_config.swiglu_alpha} and "
                f"swiglu_beta={moe_config.swiglu_beta}"
            )
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

        limit = self.quant_config.gemm1_clamp_limit
        swiglu_limit = 0.0 if limit is None else float(limit)

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
            activation_method=ActivationType.Swiglu.value,
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
