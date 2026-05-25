# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU FP8 W8A16 and MXFP4 W4A16 fused MoE experts."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._custom_ops import CPUQuantMethod, fused_experts_cpu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kMxfp4Static,
)
from vllm.platforms import current_platform


def prepare_fp8_moe_layer_for_cpu(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VNNI-prepack FP8 MoE weights for CPU kernel."""
    packed_w13 = torch.ops._C.convert_weight_packed(w13)
    packed_w2 = torch.ops._C.convert_weight_packed(w2)
    return packed_w13, packed_w2


class CPUExpertsFp8(mk.FusedMoEExpertsMonolithic):
    """CPU FP8 W8A16 block-quantized monolithic MoE experts."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(
            moe_config,
            quant_config,
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
            RoutingMethodType.Default,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
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
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.cpu_fused_moe import (
            select_experts,
        )

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=num_expert_group is not None,
            top_k=self.moe_config.experts_per_token,
            renormalize=self.moe_config.routing_method
            in (
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
            ),
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func="softmax",
            routed_scaling_factor=(
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            ),
            e_score_correction_bias=e_score_correction_bias,
        )

        block_shape = (
            list(self.quant_config.block_shape)
            if self.quant_config.block_shape
            else (
                [self.quant_config._w1.shape.row, self.quant_config._w1.shape.col]
                if self.quant_config._w1.shape is not None
                else None
            )
        )

        return fused_experts_cpu(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            False,  # inplace
            CPUQuantMethod.FP8_W8A16,  # moe_comp_method
            self.w1_scale,  # w1_scale
            self.w2_scale,  # w2_scale
            None,  # w1_zero
            None,  # w2_zero
            block_shape,  # block_size
            None,  # w1_bias
            None,  # w2_bias
            None,  # alpha
            None,  # limit
            True,  # is_vnni
        )


def prepare_mxfp4_moe_layer_for_cpu(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """VNNI-prepack MXFP4 MoE weights and repack scales for CPU AMX kernel."""
    packed_w13 = torch.ops._C.convert_weight_packed(w13)
    packed_w2 = torch.ops._C.convert_weight_packed(w2)
    packed_w13_scale = torch.ops._C.convert_scale_packed(w13_scale)
    packed_w2_scale = torch.ops._C.convert_scale_packed(w2_scale)
    return packed_w13, packed_w2, packed_w13_scale, packed_w2_scale


class CPUExpertsMxfp4(mk.FusedMoEExpertsMonolithic):
    """CPU MXFP4 W4A16 monolithic MoE experts."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(
            moe_config,
            quant_config,
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def supports_swiglu_clamp_limit(activation: MoEActivation) -> bool:
        """CPUExpertsMxfp4 forwards `gemm1_clamp_limit` to `fused_experts_cpu`
        only on the SILU branch. However, the underlying CPU kernel at
        `csrc/cpu/sgl-kernels/moe.cpp:1051/1091` only activates
        `CPUAcTMethod::swiglu` (the clamp-aware path) when BOTH `alpha`
        and `limit` are provided; a SILU-only config has no `alpha`, so
        the kernel falls back to `silu_and_mul` and silently drops the
        limit. Until the CPU kernel grows a clamp path that accepts
        `limit` without `alpha`, declare False on every activation so the
        oracle does not route SwiGLU-clamp configs to CPU MXFP4.
        """
        return False

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI)

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, None),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
            RoutingMethodType.Default,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
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
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.cpu_fused_moe import (
            select_experts,
        )

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=num_expert_group is not None,
            top_k=self.moe_config.experts_per_token,
            renormalize=self.moe_config.routing_method
            in (
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
            ),
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func="softmax",
            routed_scaling_factor=(
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            ),
            e_score_correction_bias=e_score_correction_bias,
        )

        # Get bias and swiglu params from quant config
        w1_bias = self.quant_config.w1_bias
        w2_bias = self.quant_config.w2_bias
        alpha = getattr(self.quant_config, "gemm1_alpha", None)
        limit = getattr(self.quant_config, "gemm1_clamp_limit", None)

        return fused_experts_cpu(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            False,  # inplace
            CPUQuantMethod.MXFP4,  # moe_comp_method
            self.w1_scale,  # w1_scale
            self.w2_scale,  # w2_scale
            None,  # w1_zero
            None,  # w2_zero
            None,  # block_size
            w1_bias,
            w2_bias,
            alpha,
            limit,
            True,  # is_vnni
        )
