# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 (1x32 block, E8M0 scale) MoE experts on Triton.

``Mxfp8TritonExpertsBase`` stashes E8M0 weight scales for checkpoint layout.
``Mxfp8EmulationTritonExperts`` dequantizes to BF16 and runs ``TritonExperts``
for devices without a native MXFP8 MoE kernel (e.g. ROCm gfx942 / MI300).
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
    kMxfp8Static,
)

logger = init_logger(__name__)


class Mxfp8TritonExpertsBase(TritonExperts):
    """Shared MXFP8 MoE setup: stash E8M0 scales, clear scales on ``quant_config``."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.w1_scale_val = self.quant_config.w1_scale
        self.w2_scale_val = self.quant_config.w2_scale
        self.quant_config._w1.scale = None
        self.quant_config._w2.scale = None

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp8Static, kMxfp8Dynamic)

    @staticmethod
    def _supports_activation(activation) -> bool:
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        if activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
            return True
        return TritonExperts._supports_activation(activation)


class Mxfp8EmulationTritonExperts(Mxfp8TritonExpertsBase):
    """Dequantize MXFP8 weights to BF16 on the fly and run ``TritonExperts``."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        logger.warning_once(
            "Using Mxfp8EmulationTritonExperts MoE backend. Weights are "
            "dequantized to BF16 on the fly; this is slower than a native "
            "MXFP8 MoE kernel and is intended for devices without one."
        )

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        # BF16 fallback: do not MXFP8-quantize activations in ``TritonExperts``.
        return None

    @property
    def block_shape(self) -> list[int] | None:
        return None

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        return True

    def activation(
        self,
        activation,
        output: torch.Tensor,
        input: torch.Tensor,
        **kwargs,
    ):
        """Apply GEMM1 activation with quant-config alpha/beta/clamp."""
        from vllm.model_executor.layers.fused_moe.activation import (
            MoEActivation,
            apply_moe_activation,
        )

        if activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
            limit = self.quant_config.gemm1_clamp_limit
            if limit is None:
                raise ValueError("SWIGLUOAI_UNINTERLEAVE requires gemm1_clamp_limit")
            alpha = self.quant_config.gemm1_alpha
            alpha = 1.702 if alpha is None else float(alpha)
            beta = self.quant_config.gemm1_beta
            beta = 1.0 if beta is None else float(beta)
            apply_moe_activation(
                activation,
                output,
                input,
                clamp_limit=float(limit),
                alpha=alpha,
                beta=beta,
            )
            return
        super().activation(activation, output, input)

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
        # If the weights were already dequantized to BF16 at load time
        # (process_weights_after_loading on devices without a native MXFP8 MoE
        # kernel), use them directly -- no per-step dequant. MXFP8 weights are
        # 1-byte FP8 (element_size 1); BF16/FP16 are >= 2 bytes.
        if w1.element_size() >= 2:
            # tl.dot requires w and activations share a dtype; .to() is a no-op
            # when they already match (e.g. both BF16).
            w1_bf16 = w1.to(hidden_states.dtype)
            w2_bf16 = w2.to(hidden_states.dtype)
        else:
            w1_bf16 = dequant_mxfp8_to_bf16(w1, self.w1_scale_val).to(
                hidden_states.dtype
            )
            w2_bf16 = dequant_mxfp8_to_bf16(w2, self.w2_scale_val).to(
                hidden_states.dtype
            )

        super().apply(
            output=output,
            hidden_states=hidden_states,
            w1=w1_bf16,
            w2=w2_bf16,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=None,
            a2_scale=None,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
