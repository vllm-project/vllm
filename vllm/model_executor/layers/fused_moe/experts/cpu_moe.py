# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU fused MoE experts."""

import math
import sys
from typing import cast

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._custom_ops import (
    CPUQuantAlgo,
    CPUQuantMethod,
    convert_weight_packed_scale_zp,
    cpu_fused_moe,
    cpu_fused_moe_int8,
    cpu_prepack_moe_weight,
    cpu_prepack_moe_weight_int8,
    fused_experts_cpu,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.zentorch_utils import (
    _ZENTORCH_MOE_ACTIVATIONS,
    has_zentorch_op,
    is_zentorch_moe_config_supported,
    is_zentorch_moe_supported,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kInt4Static,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
    kMxfp4Static,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)
# ===========================================================================
# Unquantized (BF16/FP16/FP32) MoE
# ===========================================================================


class CPUUnquantizedExperts(mk.FusedMoEExpertsModular):
    """Portable vector grouped-gemm unquantized MoE experts."""

    isa = "vec"
    output_alignment = 32
    reduction_alignment = 1

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self._use_zentorch = False

    @classmethod
    def _intermediate_alignment(cls) -> int:
        return math.lcm(cls.output_alignment, cls.reduction_alignment)

    @classmethod
    def _padded_intermediate_size(cls, moe_config: FusedMoEConfig) -> int:
        intermediate_size = moe_config.intermediate_size_per_partition
        if moe_config.activation == MoEActivation.SWIGLUOAI:
            return intermediate_size
        return round_up(intermediate_size, cls._intermediate_alignment())

    @classmethod
    def _supports_grouped_gemm(
        cls,
        moe_config: FusedMoEConfig,
    ) -> tuple[bool, str | None]:
        intermediate_size = cls._padded_intermediate_size(moe_config)
        if (
            moe_config.hidden_dim % cls.output_alignment != 0
            or intermediate_size % cls.output_alignment != 0
        ):
            return False, (
                "kernel requires hidden and intermediate dimensions divisible by "
                f"{cls.output_alignment}"
            )
        if (
            moe_config.hidden_dim % cls.reduction_alignment != 0
            or intermediate_size % cls.reduction_alignment != 0
        ):
            return False, (
                "kernel requires reduction dimensions divisible by "
                f"{cls.reduction_alignment}"
            )
        return True, None

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
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return supported, reason
        if is_zentorch_moe_config_supported(moe_config):
            return True, None
        cpu_cls = cast(type[CPUUnquantizedExperts], cls)
        return cpu_cls._supports_grouped_gemm(moe_config)

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (None, None)

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # Routing runs in the CPURouter, which covers every routing method a
        # layer can be configured with, including custom routing functions.
        return True

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._use_zentorch = is_zentorch_moe_supported(layer)
        if self._use_zentorch:
            return
        self._pad_moe_intermediate(layer)
        replace_parameter(
            layer, "w13_weight", cpu_prepack_moe_weight(layer.w13_weight, self.isa)
        )
        replace_parameter(
            layer, "w2_weight", cpu_prepack_moe_weight(layer.w2_weight, self.isa)
        )

    def _pad_moe_intermediate(self, layer: torch.nn.Module) -> None:
        """Zero-pad the per-partition MoE intermediate dim of both weights and
        the expert bias, see `_padded_intermediate_size`."""
        intermediate_size = self.moe_config.intermediate_size_per_partition
        padded_size = self._padded_intermediate_size(self.moe_config)
        if padded_size == intermediate_size:
            return

        num_experts, _, hidden_size = layer.w13_weight.shape

        new_w13 = layer.w13_weight.new_zeros(num_experts, 2 * padded_size, hidden_size)
        new_w13[:, :intermediate_size] = layer.w13_weight[:, :intermediate_size]
        new_w13[:, padded_size : padded_size + intermediate_size] = layer.w13_weight[
            :, intermediate_size:
        ]
        replace_parameter(layer, "w13_weight", new_w13)

        new_w2 = layer.w2_weight.new_zeros(num_experts, hidden_size, padded_size)
        new_w2[:, :, :intermediate_size] = layer.w2_weight
        replace_parameter(layer, "w2_weight", new_w2)

        if hasattr(layer, "w13_bias"):
            new_bias = layer.w13_bias.new_zeros(num_experts, 2 * padded_size)
            new_bias[:, :intermediate_size] = layer.w13_bias[:, :intermediate_size]
            new_bias[:, padded_size : padded_size + intermediate_size] = layer.w13_bias[
                :, intermediate_size:
            ]
            # Assign through .data rather than replacing the Parameter: the
            # quant config is built before this runs and holds a reference to
            # this very object, which is what feeds self.w1_bias in apply().
            layer.w13_bias.data = new_bias

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # cpu_fused_moe manages its own scratch space.
        return (0,), (0,), (M, K)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

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
    ) -> None:
        # apply_router_weight_on_input (topk=1 only) is already applied to
        # hidden_states by MoEPrepareAndFinalizeNoDPEPModular.prepare().
        if self._use_zentorch:
            torch.ops.zentorch.zentorch_fused_moe(
                output,
                hidden_states,
                w1,
                w2,
                self.w1_bias,
                self.w2_bias,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                str(activation.value).lower(),
            )
            return
        cpu_fused_moe(
            output,
            hidden_states,
            w1,
            w2,
            self.w1_bias,
            self.w2_bias,
            topk_weights,
            topk_ids,
            activation.value,
            self.isa,
            apply_router_weight_on_input,
        )


class X86CPUUnquantizedExperts(CPUUnquantizedExperts):
    """x86 AMX grouped-gemm unquantized MoE experts."""

    isa = "amx"
    output_alignment = 32
    reduction_alignment = 32

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cpu()
            and current_platform.get_cpu_architecture() == CpuArchEnum.X86
            and torch.cpu._is_amx_tile_supported()
        )

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return supported, reason
        if moe_config.in_dtype != torch.bfloat16:
            return False, "kernel requires bfloat16 activations"
        cpu_cls = cast(type[CPUUnquantizedExperts], cls)
        return cpu_cls._supports_grouped_gemm(moe_config)


class ArmCPUUnquantizedExperts(CPUUnquantizedExperts):
    """Arm NEON grouped-gemm unquantized MoE experts."""

    isa = "neon"
    output_alignment = 32
    reduction_alignment = 4

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cpu()
            and current_platform.get_cpu_architecture() == CpuArchEnum.ARM
            and sys.platform != "darwin"
        )

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return supported, reason
        if moe_config.in_dtype != torch.bfloat16:
            return False, "kernel requires bfloat16 activations"
        cpu_cls = cast(type[CPUUnquantizedExperts], cls)
        return cpu_cls._supports_grouped_gemm(moe_config)


class PowerCPUUnquantizedExperts(CPUUnquantizedExperts):
    """PowerPC VSX grouped-gemm unquantized MoE experts."""

    isa = "vsx"
    output_alignment = 16
    reduction_alignment = 2

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cpu()
            and current_platform.get_cpu_architecture() == CpuArchEnum.POWERPC
        )

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return supported, reason
        if moe_config.in_dtype != torch.bfloat16:
            return False, "kernel requires bfloat16 activations"
        cpu_cls = cast(type[CPUUnquantizedExperts], cls)
        return cpu_cls._supports_grouped_gemm(moe_config)


# ===========================================================================
# FP8 W8A16 MoE
# ===========================================================================


def prepare_fp8_moe_layer_for_cpu(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VNNI-prepack FP8 MoE weights for CPU kernel."""
    packed_w13 = torch.ops._C.convert_weight_packed(w13)
    packed_w2 = torch.ops._C.convert_weight_packed(w2)
    return packed_w13, packed_w2


class CPUExpertsFp8(mk.FusedMoEExpertsModular):
    """CPU FP8 W8A16 block-quantized modular MoE experts."""

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
        return (
            current_platform.is_cpu()
            and current_platform.get_cpu_architecture() == CpuArchEnum.X86
            and torch.cpu._is_amx_tile_supported()
        )

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
            RoutingMethodType.DeepseekV4,
        ]

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # fused_experts_cpu manages its own scratch space.
        return (0,), (0,), (M, K)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

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
    ) -> None:
        if apply_router_weight_on_input:
            # fused_experts_cpu always applies topk_weights internally on
            # combine; MoEPrepareAndFinalizeNoDPEPModular.prepare() would
            # also pre-apply it to hidden_states, double-weighting the
            # output. Not needed by any CPU FP8 model today.
            raise NotImplementedError(
                "CPUExpertsFp8 does not support apply_router_weight_on_input=True."
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

        fused_experts_cpu(
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
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


# ===========================================================================
# MXFP4 W4A16 MoE
# ===========================================================================


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


class CPUExpertsMxfp4(mk.FusedMoEExpertsModular):
    """CPU MXFP4 W4A16 modular MoE experts."""

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
        return (
            current_platform.is_cpu()
            and current_platform.get_cpu_architecture() == CpuArchEnum.X86
            and torch.cpu._is_amx_tile_supported()
        )

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
            RoutingMethodType.DeepseekV4,
        ]

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # fused_experts_cpu manages its own scratch space.
        return (0,), (0,), (M, K)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

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
    ) -> None:
        if apply_router_weight_on_input:
            # fused_experts_cpu always applies topk_weights internally on
            # combine; MoEPrepareAndFinalizeNoDPEPModular.prepare() would
            # also pre-apply it to hidden_states, double-weighting the
            # output. Not needed by any CPU MXFP4 model today.
            raise NotImplementedError(
                "CPUExpertsMxfp4 does not support apply_router_weight_on_input=True."
            )
        # Get bias and swiglu params from quant config
        w1_bias = self.quant_config.w1_bias
        w2_bias = self.quant_config.w2_bias
        alpha = getattr(self.quant_config, "gemm1_alpha", None)
        limit = getattr(self.quant_config, "gemm1_clamp_limit", None)

        fused_experts_cpu(
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
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


# ===========================================================================
# INT4 W4A16 MoE
# ===========================================================================


def prepare_int4_moe_layer_for_cpu(
    w13_packed: torch.Tensor,
    w2_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    quant_algo: CPUQuantAlgo = CPUQuantAlgo.GPTQ,
    w13_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Repack INT4 MoE weights via convert_weight_packed_scale_zp for CPU.

    Args:
        w13_packed: [E, K//8, 2*I] int32 (packed int4)
        w2_packed: [E, I//8, K] int32 (packed int4)
        w13_scale: [E, num_groups, 2*I] float16/bf16
        w2_scale: [E, num_groups, K] float16/bf16
        quant_algo: CPUQuantAlgo.GPTQ or CPUQuantAlgo.AWQ
        w13_zeros: optional [E, num_groups, N//8] int32 packed zeros.
                   If None, synthetic zeros are created for symmetric quant.
        w2_zeros: optional [E, num_groups, N//8] int32 packed zeros.
                  If None, synthetic zeros are created for symmetric quant.

    Returns:
        (blocked_w13, blocked_w2, blocked_s13, blocked_s2, blocked_z13, blocked_z2)
    """
    E = w13_packed.size(0)

    # No qzeros are available in compressed-tensors symmetric checkpoints.
    # The GPTQ unpack kernel (unpack_4bit_to_32bit_signed) adds +1 to stored zeros,
    # so we store 7 per nibble: 0x77777777 → +1 → 8.
    if w13_zeros is None:
        num_groups_w13 = w13_scale.size(1)
        N_w13 = w13_scale.size(2)  # 2*I
        _zp = 0x77777777
        w13_zeros = torch.full(
            (E, num_groups_w13, N_w13 // 8),
            _zp,
            dtype=torch.int32,
        )

    if w2_zeros is None:
        num_groups_w2 = w2_scale.size(1)
        N_w2 = w2_scale.size(2)  # K
        _zp = 0x77777777
        w2_zeros = torch.full(
            (E, num_groups_w2, N_w2 // 8),
            _zp,
            dtype=torch.int32,
        )

    blocked_w13, blocked_z13, blocked_s13 = convert_weight_packed_scale_zp(
        w13_packed, w13_zeros, w13_scale, quant_algo
    )
    blocked_w2, blocked_z2, blocked_s2 = convert_weight_packed_scale_zp(
        w2_packed, w2_zeros, w2_scale, quant_algo
    )
    return (blocked_w13, blocked_w2, blocked_s13, blocked_s2, blocked_z13, blocked_z2)


class CPUExpertsInt4(mk.FusedMoEExpertsModular):
    """CPU INT4 W4A16 group-quantized modular MoE experts.

    Weights are int4 (packed), activations are bf16/fp16.
    Internally uses int8 compute via fused_experts_cpu with INT4_W4A8.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
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
        return (
            current_platform.is_cpu()
            and current_platform.get_cpu_architecture() == CpuArchEnum.X86
            and torch.cpu._is_amx_tile_supported()
        )

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
            (kInt4Static, None),
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

    def supports_expert_map(self) -> bool:
        return False

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        # convert_weight_packed_scale_zp blocks w1/w2 into AMX tiles
        # (E, Nc, Kc, buffer_bytes), not the (E, N, K) layout the base
        # implementation assumes -- N/K aren't recoverable from that shape,
        # so read them from moe_config instead.
        E = w1.shape[0]
        K = a1.size(-1)
        N = (
            self.moe_config.intermediate_size_per_partition
            * self.moe_config.w13_num_shards
        )
        M = a1.size(0) if a1.dim() == 2 else a1.size(1)
        topk = topk_ids.size(1)
        return E, M, N, K, topk

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # fused_experts_cpu manages its own scratch space.
        return (0,), (0,), (M, K)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

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
    ) -> None:
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "CPUExpertsInt4 (W4A16) does not support "
                "apply_router_weight_on_input=True. "
            )

        fused_experts_cpu(
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            CPUQuantMethod.INT4_W4A8,
            self.w1_scale,
            self.w2_scale,
            self.w1_zp,
            self.w2_zp,
            None,  # block_size
            None,  # w1_bias
            None,  # w2_bias
            None,  # alpha
            None,  # limit
            True,  # is_vnni
        )


# ===========================================================================
# INT8 W8A8 MoE
# ===========================================================================


class CPUExpertsInt8(mk.FusedMoEExpertsModular):
    """CPU INT8 W8A8 per-channel weight / dynamic per-token activation
    modular MoE experts."""

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
        return (
            current_platform.is_cpu()
            and current_platform.get_cpu_architecture() == CpuArchEnum.X86
            and torch.cpu._is_amx_tile_supported()
        )

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )
        if not supported:
            return supported, reason
        # convert_weight_packed (shared VNNI prepack) requires the w13
        # OC/IC and w2 OC/IC to be multiples of TILE_N=16/TILE_K=32; the
        # w1 gate-up kernel additionally requires the intermediate size
        # itself (not 2x) to be a multiple of 32 (moe_int8.cpp), which
        # dominates. Net effect: both dims must be multiples of 32.
        if moe_config.hidden_dim % 32 != 0:
            return False, "kernel requires hidden dim divisible by 32"
        if moe_config.intermediate_size_per_partition % 32 != 0:
            return False, "kernel requires intermediate dim divisible by 32"
        return True, None

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
            (kInt8StaticChannelSym, kInt8DynamicTokenSym),
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

    def supports_expert_map(self) -> bool:
        return False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """VNNI-prepack INT8 MoE weights for CPU kernel."""

        w13 = torch.ops._C.convert_weight_packed(layer.w13_weight)
        w2 = torch.ops._C.convert_weight_packed(layer.w2_weight)
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # fused_experts_cpu manages its own scratch space.
        return (0,), (0,), (M, K)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

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
    ) -> None:
        if apply_router_weight_on_input:
            # fused_experts_cpu always applies topk_weights internally on
            # combine; MoEPrepareAndFinalizeNoDPEPModular.prepare() would
            # also pre-apply it to hidden_states, double-weighting the
            # output. Not needed by any CPU INT8 W8A8 model today.
            raise NotImplementedError(
                "CPUExpertsInt8 does not support apply_router_weight_on_input=True."
            )
        fused_experts_cpu(
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            CPUQuantMethod.INT8_W8A8,
            self.w1_scale,
            self.w2_scale,
            None,  # w1_zero
            None,  # w2_zero
            None,  # block_size (per-channel, no block)
            None,  # w1_bias
            None,  # w2_bias
            None,  # alpha
            None,  # limit
            True,  # is_vnni
        )


class ArmCPUExpertsInt8(mk.FusedMoEExpertsModular):
    """Arm INT8 MoE with per-token activation and channelwise weight quantization."""

    @property
    def expects_unquantized_inputs(self) -> bool:
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
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )
        if not supported:
            return supported, reason
        if moe_config.in_dtype not in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ):
            return False, "kernel requires float32, float16, or bfloat16 activations"
        if moe_config.hidden_dim % 32 != 0:
            return False, "kernel requires hidden dim divisible by 32"
        if moe_config.intermediate_size_per_partition % 32 != 0:
            return False, "kernel requires intermediate dim divisible by 32"
        return True, None

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cpu()
            and current_platform.get_cpu_architecture() == CpuArchEnum.ARM
            and hasattr(torch.ops._C, "cpu_fused_moe_int8")
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
        )

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return not moe_parallel_config.use_ep

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (
            kInt8StaticChannelSym,
            kInt8DynamicTokenSym,
        )

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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13 = cpu_prepack_moe_weight_int8(layer.w13_weight, "neon")
        w2 = cpu_prepack_moe_weight_int8(layer.w2_weight, "neon")
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # cpu_fused_moe_int8 manages its own scratch space.
        return (0,), (0,), (M, K)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

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
    ) -> None:
        # apply_router_weight_on_input (topk=1 only) is already applied to
        # hidden_states by MoEPrepareAndFinalizeNoDPEPModular.prepare().
        assert self.w1_scale is not None
        assert self.w2_scale is not None
        cpu_fused_moe_int8(
            output,
            hidden_states,
            w1,
            w2,
            self.w1_scale,
            self.w2_scale,
            self.w1_bias,
            self.w2_bias,
            topk_weights,
            topk_ids,
            activation.value,
            "neon",
            skip_weighted=apply_router_weight_on_input,
        )


class ZenCPUExpertsInt8(mk.FusedMoEExpertsModular):
    """AMD Zen INT8 MoE with per-token activation and channelwise weight
    quantization, dispatched through zentorch."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        assert self.w1_scale is not None and self.w2_scale is not None, (
            "ZenCPUExpertsInt8 requires per-channel weight scales on the layer."
        )
        num_experts = self.w1_scale.shape[0]
        self._w1_scale_bf16 = (
            self.w1_scale.detach()
            .to(torch.bfloat16)
            .reshape(num_experts, -1)
            .contiguous()
        )
        self._w2_scale_bf16 = (
            self.w2_scale.detach()
            .to(torch.bfloat16)
            .reshape(num_experts, -1)
            .contiguous()
        )
        self._w1_bias_bf16 = (
            None
            if self.w1_bias is None
            else self.w1_bias.detach().to(torch.bfloat16).contiguous()
        )
        self._w2_bias_bf16 = (
            None
            if self.w2_bias is None
            else self.w2_bias.detach().to(torch.bfloat16).contiguous()
        )
        logger.info_once("[zen_cpu] Using zentorch_fused_moe for W8A8 INT8 MoE")

    @property
    def expects_unquantized_inputs(self) -> bool:
        # zentorch_fused_moe quantizes activations itself.
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return has_zentorch_op(["zentorch_fused_moe"])

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return str(activation.value).lower() in _ZENTORCH_MOE_ACTIVATIONS

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return not moe_parallel_config.use_ep

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (
            kInt8StaticChannelSym,
            kInt8DynamicTokenSym,
        )

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

    def supports_expert_map(self) -> bool:
        return False

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # zentorch_fused_moe manages its own scratch space.
        return (0,), (0,), (M, K)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

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
    ) -> None:
        if apply_router_weight_on_input:
            assert topk_ids.size(1) == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            hidden_states = hidden_states.mul(topk_weights.to(hidden_states.dtype))

        torch.ops.zentorch.zentorch_fused_moe(
            output,
            hidden_states,
            w1,
            w2,
            self._w1_bias_bf16,
            self._w2_bias_bf16,
            topk_weights.to(torch.float32).contiguous(),
            topk_ids.to(torch.int32).contiguous(),
            apply_router_weight_on_input,  # skip_weighted
            str(activation.value).lower(),
            self._w1_scale_bf16,
            self._w2_scale_bf16,
        )
