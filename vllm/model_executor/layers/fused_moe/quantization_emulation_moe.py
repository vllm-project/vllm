# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Quantization Emulation Experts for MoE.

This module provides emulation support for MOE quantization schemes that
don't have native hardware support. It dequantizes weights on the fly
and falls back to calling fused_experts with activation quantization.

Similar to QuarkOCP_MX_MoEMethod's emulation path but abstracted into
a reusable NvFp4MoeBackend.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)

logger = init_logger(__name__)


class Nvfp4QuantizationEmulationTritonExperts(TritonExperts):
    """
    Emulation backend for NVFP4 quantized MoE experts.

    It may be used for NVFP4 models when the device does not have
    native support for this dtype.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        logger.warning_once(
            "Using Nvfp4QuantizationEmulationTritonExperts MOE backend. This will"
            " dequantize weights on the fly and may be slower than native"
            " quantized MOE. Consider using a device with native quantization"
            " support (e.g. Nvidia Blackwell) for better performance."
        )

        # `TritonExperts.apply` expects pre-dequantized weights,
        # which we handle in `apply` below.
        self.w1_scale_val = self.quant_config.w1_scale
        self.w2_scale_val = self.quant_config.w2_scale

        self.quant_config._w1.scale = None
        self.quant_config._w2.scale = None

        self.emulation = True

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return "nvfp4"

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        """
        Apply emulated quantized MoE computation.

        This dequantizes the weights on the fly and calls fused_experts_impl
        with activation quantization support.
        """
        # Dequantize weights if they are quantized
        # For NVFP4, weights are packed in uint8 format
        # w1 shape: [num_experts, 2*intermediate_size, hidden_size//2]
        # w2 shape: [num_experts, hidden_size, intermediate_size//2]
        assert w1.dtype == torch.uint8
        assert w2.dtype == torch.uint8

        # Dequantize w1 from packed NVFP4 to fp16/bf16
        w13_global_scale = self.quant_config.g1_alphas

        w1_dequant = dequantize_to_dtype(
            tensor_fp4=w1,
            tensor_sf=self.w1_scale_val,
            global_scale=w13_global_scale,
            dtype=hidden_states.dtype,
            block_size=16,
            swizzle=False,
        )

        # Dequantize w2 from packed NVFP4 to fp16/bf16
        w2_global_scale = self.quant_config.g2_alphas

        w2_dequant = dequantize_to_dtype(
            tensor_fp4=w2,
            tensor_sf=self.w2_scale_val,
            global_scale=w2_global_scale,
            dtype=hidden_states.dtype,
            block_size=16,
            swizzle=False,
        )

        hidden_states, _ = moe_kernel_quantize_input(
            A=hidden_states,
            A_scale=self.quant_config.a1_gscale,
            quant_dtype="nvfp4",
            per_act_token_quant=False,
            emulation=True,
        )

        # Activation quantization/dequantization is deferred to
        # `moe_kernel_quantize_input` in TritonExperts.apply.
        super().apply(
            output=output,
            hidden_states=hidden_states,
            w1=w1_dequant,
            w2=w2_dequant,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=None,
            a2_scale=self.quant_config.a2_gscale,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
