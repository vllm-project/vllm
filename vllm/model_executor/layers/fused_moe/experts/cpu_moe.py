# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU quantized fused MoE experts."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._custom_ops import (
    CPUQuantAlgo,
    CPUQuantMethod,
    convert_weight_packed_scale_zp,
    fused_experts_cpu,
)
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
    kInt4Static,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
    kMxfp4Static,
)
from vllm.platforms import current_platform

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


def _fused_experts_cpu_local_skip_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    moe_comp_method: int,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    w1_zero: torch.Tensor | None,
    w2_zero: torch.Tensor | None,
    block_shape: list[int] | None,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    alpha: float | None,
    limit: float | None,
    is_vnni: bool,
) -> torch.Tensor:
    M, H = hidden_states.shape
    valid = topk_ids != -1
    local_ids = expert_map[topk_ids.clamp(min=0).long()]  # [M, topk]
    sel = valid & (local_ids != -1)
    if not bool(sel.any()):
        return hidden_states.new_zeros((M, H))

    token_idx, slot_idx = sel.nonzero(as_tuple=True)  # [S], [S]
    sel_hidden = hidden_states.index_select(0, token_idx).contiguous()
    sel_weights = topk_weights[token_idx, slot_idx].unsqueeze(1).contiguous()
    sel_ids = local_ids[token_idx, slot_idx].unsqueeze(1).to(torch.int32).contiguous()

    sel_out = fused_experts_cpu(
        sel_hidden,
        w1,
        w2,
        sel_weights,
        sel_ids,
        False,  # inplace
        CPUQuantMethod(moe_comp_method),
        w1_scale,
        w2_scale,
        w1_zero,
        w2_zero,
        block_shape,
        w1_bias,
        w2_bias,
        alpha,
        limit,
        is_vnni,
    )  # [S, H], already weighted

    out = hidden_states.new_zeros((M, sel_out.shape[1]))
    out.index_add_(0, token_idx, sel_out)
    return out


@torch.library.custom_op("vllm::fused_experts_cpu_local_skip", mutates_args=())
def fused_experts_cpu_local_skip(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    moe_comp_method: int,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    w1_zero: torch.Tensor | None,
    w2_zero: torch.Tensor | None,
    block_shape: list[int] | None,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    alpha: float | None,
    limit: float | None,
    is_vnni: bool,
) -> torch.Tensor:
    """Run only this rank's local expert selections; scatter-add back.

    Opaque to torch.compile (registered as a custom op) to avoid data-dependent
    control flow errors during AOT fullgraph compilation.
    """
    return _fused_experts_cpu_local_skip_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        expert_map,
        moe_comp_method,
        w1_scale,
        w2_scale,
        w1_zero,
        w2_zero,
        block_shape,
        w1_bias,
        w2_bias,
        alpha,
        limit,
        is_vnni,
    )


@fused_experts_cpu_local_skip.register_fake
def _(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    expert_map,
    moe_comp_method,
    w1_scale,
    w2_scale,
    w1_zero,
    w2_zero,
    block_shape,
    w1_bias,
    w2_bias,
    alpha,
    limit,
    is_vnni,
):
    return hidden_states.new_empty(hidden_states.shape)


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

        if expert_map is not None:
            # Expert parallelism: select_experts returns global expert ids, but
            # w1/w2 only hold this rank's local experts. Skip the non-local
            # selections entirely (instead of masking them to weight 0 and
            # still running their GEMMs), saving ~world_size x of expert
            # compute per rank.
            return fused_experts_cpu_local_skip(
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                expert_map,
                int(CPUQuantMethod.FP8_W8A16),  # moe_comp_method
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


class CPUExpertsInt4(mk.FusedMoEExpertsMonolithic):
    """CPU INT4 W4A16 group-quantized monolithic MoE experts.

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
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "CPUExpertsInt4 (W4A16) does not support "
                "apply_router_weight_on_input=True. "
            )

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

        return fused_experts_cpu(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            False,  # inplace
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


def prepare_int8_moe_layer_for_cpu(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VNNI-prepack INT8 MoE weights for CPU kernel."""
    packed_w13 = torch.ops._C.convert_weight_packed(w13)
    packed_w2 = torch.ops._C.convert_weight_packed(w2)
    return packed_w13, packed_w2


class CPUExpertsInt8(mk.FusedMoEExpertsMonolithic):
    """CPU INT8 W8A8 per-channel weight / dynamic per-token activation
    monolithic MoE experts."""

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
        from vllm.model_executor.utils import replace_parameter

        w13, w2 = prepare_int8_moe_layer_for_cpu(layer.w13_weight, layer.w2_weight)
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)

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

        return fused_experts_cpu(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            False,  # inplace
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
