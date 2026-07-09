# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU INT4 W4A8 dynamic quantized fused MoE experts."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4W4A8StaticChannelSym,
    kInt4W4A8StaticGroup32Sym,
    kInt4W4A8StaticGroup64Sym,
    kInt4W4A8StaticGroup128Sym,
    kInt4W4A8StaticGroupSym,
)
from vllm.platforms import CpuArchEnum, current_platform


class CPUExpertsInt4(mk.FusedMoEExpertsMonolithic):
    """CPU INT4 W4A8 dynamic quantized monolithic MoE experts.

    Uses the dynamic_4bit_int_moe kernel for efficient 4-bit weight,
    8-bit activation MoE inference on CPU.
    """

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
        """Expects unquantized inputs (quantization happens in kernel)."""
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if (
            not current_platform.is_cpu()
            or current_platform.get_cpu_architecture() != CpuArchEnum.ARM
        ):
            return False, "kernel only supports Arm CPU"

        if moe_config.in_dtype not in (
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ):
            return (
                False,
                f"kernel does not support {moe_config.in_dtype} input/output dtype",
            )

        try:
            _ = torch.ops.aten._dyn_quant_matmul_4bit
            _ = torch.ops.aten._dyn_quant_pack_4bit_weight
        except AttributeError:
            return (
                False,
                f"PyTorch {torch.__version__} does not support "
                "_dyn_quant_* 4bit ops. Install a newer version",
            )

        return mk.FusedMoEExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        """Does not support no_act_and_mul (requires SwiGLU or SiLU)."""
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        """Supports SiLU and SwiGLU variants."""
        return activation in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUSTEP,
        )

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        """Currently does not support expert parallelism."""
        # Based on compressed_tensors implementation check
        return moe_parallel_config.ep_size == 1

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Supports INT4 weights with INT8 dynamic activations.

        This is W4A8 with:
        - Weights: 4-bit integer (stored as int8, packed to uint8 nibbles)
          Can be channel-wise or group-wise quantization
        - Activations: dynamic per-token 8-bit integer quantization
        """
        # channelwise or groupwise with group size being a multiple of 32
        SUPPORTED_W_A = [
            (kInt4W4A8StaticChannelSym, None),
            (kInt4W4A8StaticGroup128Sym, None),
            (kInt4W4A8StaticGroup64Sym, None),
            (kInt4W4A8StaticGroup32Sym, None),
            (kInt4W4A8StaticGroupSym, None),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Supports standard routing methods."""
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
        """Accepts any router logits dtype."""
        return True

    def supports_expert_map(self) -> bool:
        """Expert parallelism not yet supported."""
        return False

    @staticmethod
    def _activation_kind(activation: MoEActivation) -> int:
        """Convert MoEActivation to kernel activation kind integer.

        Returns:
            0 = SwiGLU_Gu (SiLU(g)*u)
            1 = SwiGLU_Ug (SiLU(u)*g)
            2 = SiLU
        """
        if activation == MoEActivation.SWIGLUSTEP:
            return 0
        if activation == MoEActivation.SWIGLUOAI:
            return 1
        if activation == MoEActivation.SILU:
            return 2
        raise ValueError(f"Unsupported activation '{activation}'")

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,  # w13_weight_packed
        w2: torch.Tensor,  # w2_weight_packed
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
        """Apply the monolithic 4-bit INT MoE forward pass.

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]
            w1: Packed w13 weights (w1+w3 gated weights)
            w2: Packed w2 weights (down projection)
            router_logits: Router output logits [num_tokens, num_experts]
            activation: Activation function type
            global_num_experts: Total number of experts
            expert_map: Expert mapping for EP (not supported)
            a1q_scale: Activation quantization scale (not used, dynamic)
            apply_router_weight_on_input: Whether to apply routing on input
            num_expert_group: For grouped topk
            e_score_correction_bias: Bias for expert scores
            routed_scaling_factor: Scaling factor for routing
            topk_group: Group size for topk

        Returns:
            Output tensor after MoE computation
        """
        from vllm.model_executor.layers.fused_moe.cpu_fused_moe import (
            select_experts,
        )

        renormalize = self.moe_config.routing_method in (
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        )

        # TODO(bnell): this could be factored into a CPURouter class and
        # turn this into a modular kernel
        # Perform topk selection
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=num_expert_group is not None,
            top_k=self.moe_config.experts_per_token,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func="softmax",
            routed_scaling_factor=(
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            ),
            e_score_correction_bias=e_score_correction_bias,
        )

        hidden_size = self.moe_config.hidden_dim
        intermediate_size = self.moe_config.intermediate_size_per_partition
        assert self.quant_config.block_shape is not None
        # C++ kernel expects an int: -1 for channelwise, and group size for groupwise
        if self.quant_config.block_shape == [-1, 1]:
            group_size = -1
        else:
            group_size = self.quant_config.block_shape[1]
        return torch.ops._C.dynamic_4bit_int_moe(
            hidden_states,
            topk_ids.to(torch.long),
            topk_weights,
            w1,  # w13_weight_packed
            w2,  # w2_weight_packed
            hidden_size,
            intermediate_size,
            group_size,
            apply_router_weight_on_input,
            self._activation_kind(activation),
        )
