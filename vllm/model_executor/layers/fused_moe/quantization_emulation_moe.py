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
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.utils import SUPPORTED_MOE_ACTIVATION
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)

logger = init_logger(__name__)


class QuantEmulationExperts(mk.FusedMoEPermuteExpertsUnpermute):
    """
    Emulation backend for quantized MoE experts.

    This backend dequantizes weights on the fly and falls back to
    calling fused_experts. It may be used for NVFP4 models when the device does not have
    native support for this dtype.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        logger.warning_once(
            "Using QuantEmulationExperts MOE backend. This will dequantize "
            "weights on the fly and may be slower than native quantized MOE. "
            "Consider using a device with native quantization support for "
            "better performance."
        )

        # `fused_experts` expects pre-dequantized weights, which we handle in `apply` below.
        self.w1_scale_val = self.quant_config.w1_scale
        self.w2_scale_val = self.quant_config.w2_scale

        self.quant_config._w1.scale = None
        self.quant_config._w2.scale = None

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return True

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: str) -> bool:
        return activation in SUPPORTED_MOE_ACTIVATION

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not moe_parallel_config.use_fi_all2allv_kernels

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceNoOP,
        )

        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """
        Return workspace shapes for the modular kernel.

        fused_experts_impl allocates its own workspaces internally,
        so workspace13 and workspace2 are not used. However, the output
        shape must be correct since it's pre-allocated by the modular kernel.
        """
        # Workspaces are not used, allocate minimal size
        workspace13 = (1,)
        workspace2 = (1,)
        # Output must match the actual output shape
        output = (M, K)
        return (workspace13, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
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
            device=w1.device,
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
            device=w2.device,
            block_size=16,
            swizzle=False,
        )

        assert w1_dequant.dtype == torch.bfloat16
        assert w2_dequant.dtype == torch.bfloat16

        # For emulation, we perform activation QDQ (quantize-dequantize) if needed
        # This simulates the quantization that would happen in native NVFP4
        # The scales are provided in a1_gscale and a2_gscale
        # For now, we call fused_experts_impl without activation quantization
        # TODO: Implement activation QDQ when needed for accuracy

        # Call fused_experts_impl with dequantized weights
        # Since weights are dequantized, we run in unquantized mode
        result = fused_experts(
            hidden_states=hidden_states,
            w1=w1_dequant,
            w2=w2_dequant,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            quant_config=self.quant_config,
        )

        assert result.shape == output.shape
        assert result.dtype == output.dtype

        # Copy result to pre-allocated output tensor
        output.copy_(result)
