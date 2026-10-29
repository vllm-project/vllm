# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 SiTU MoE via AITER's FlyDSL grouped GEMM (gfx1250). Routes through
``aiter.fused_moe`` with gate/up-interleaved w13; weights are preshuffled in
``convert_weight_to_mxfp4_moe_kernel_format``.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    AiterExperts,
    QuantMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)


class AiterFlydslMxfp4Experts(AiterExperts):
    """MXFP4 weights with BF16 activations through AITER's FlyDSL MoE on
    gfx1250. AITER quantizes activations internally to FP8 (a8w4) when
    AITER_FORCE_A8W4=1 is set, otherwise FP4 (a4w4)."""

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        if (weight_key, activation_key) != (kMxfp4Static, None):
            return False
        from vllm.platforms.rocm import on_gfx1250

        return on_gfx1250()

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SITU

    @staticmethod
    def is_supported_config(
        cls, moe_config, weight_key, activation_key, activation_format
    ):
        is_supported, reason = super().is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        situv2 = rocm_aiter_ops.get_fused_moe_situv2_activation()
        if is_supported and situv2 not in ("a8w4", "a4w4"):
            return False, (
                "gfx1250 FlyDSL SiTU MoE only has a8w4 and a4w4 kernels; got "
                f"VLLM_ROCM_USE_AITER_MOE_SITUV2={situv2}"
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
        from aiter.ops.flydsl.moe_common import GateMode

        moe_config = self.moe_config
        quant_config = self.quant_config
        assert moe_config.hidden_dim_unpadded is not None
        assert moe_config.intermediate_size_per_partition_unpadded is not None
        hidden_size = hidden_states.shape[1]
        if hidden_states.dtype == torch.float4_e2m1fn_x2:
            # Pre-quantized MXFP4 from MoRI dispatch, packed two per byte.
            hidden_size *= 2
        # SiTU's FlyDSL kernel pads per gate/up half, so pads stay unrounded.
        intermediate_pad = (
            moe_config.intermediate_size_per_partition
            - moe_config.intermediate_size_per_partition_unpadded
            if moe_config.intermediate_pad is None
            else moe_config.intermediate_pad
        )

        result = rocm_aiter_ops.fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weights.to(torch.float32),
            topk_ids.to(torch.int32),
            # RoutedExperts hands AITER experts the 0/1 expert_mask here.
            expert_mask=expert_map,
            quant_method=QuantMethod.BLOCK_1X32.value,
            activation_method=rocm_aiter_ops.get_aiter_activation_type("situ"),
            w1_scale=quant_config.w1_scale,
            w2_scale=quant_config.w2_scale,
            a1_scale=quant_config.a1_scale if a1q_scale is None else a1q_scale,
            a2_scale=quant_config.a2_scale,
            doweight_stage1=apply_router_weight_on_input,
            num_local_tokens=(
                expert_tokens_meta.expert_num_tokens
                if expert_tokens_meta is not None
                else None
            ),
            output_dtype=output.dtype,
            hidden_pad=hidden_size - moe_config.hidden_dim_unpadded,
            intermediate_pad=intermediate_pad,
            # gfx1250 FlyDSL only accepts gate/up-interleaved w13, matching the
            # shuffle in oracle/mxfp4.py.
            gate_mode=GateMode.INTERLEAVE.value,
            bias1=quant_config.w1_bias,
            bias2=quant_config.w2_bias,
            moe_sorting_dispatch_policy=rocm_aiter_ops.get_moe_dispatch_policy(),
            swiglu_limit=(
                0.0
                if moe_config.swiglu_limit is None
                else float(moe_config.swiglu_limit)
            ),
            beta=moe_config.activation_situ_beta,
            linear_beta=moe_config.activation_situ_linear_beta,
        )
        # avoid redundant copy when output is a view of the result
        if (
            output.shape == result.shape
            and output.dtype == result.dtype
            and output.device == result.device
            and output.is_contiguous()
            and result.is_contiguous()
            and output._base is None
        ):
            output.set_(result)
        else:
            output.copy_(result)
