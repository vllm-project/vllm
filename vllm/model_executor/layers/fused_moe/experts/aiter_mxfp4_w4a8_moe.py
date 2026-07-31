# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kMxfp4Static,
)

__all__ = [
    "AiterW4A8ExpertsMonolithic",
    "AiterW4A16ExpertsMonolithic",
    "aiter_triton_kernel_w4a8_moe_forward",
    "aiter_triton_kernel_w4a16_moe_forward",
]


def aiter_triton_kernel_w4a8_moe_forward(
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: MoEActivation = MoEActivation.SWIGLUOAI,
    quant_config: FusedMoEQuantConfig | None = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    unpadded_N_w1=None,
    unpadded_K_w1=None,
    unpadded_N_w2=None,
    unpadded_K_w2=None,
):
    assert (
        quant_config is not None
        and quant_config.use_mxfp4_w4a8
        and rocm_aiter_ops.is_enabled()
    )
    from vllm.platforms.rocm import on_gfx1250

    try:
        from aiter.ops.triton.moe.moe_routing import routing as _routing_mod
    except ImportError:
        from aiter.ops.triton.moe_routing import routing as _routing_mod

    if on_gfx1250():
        _routing_mod.is_tdm_avail = lambda: False
    aiter_routing = _routing_mod.routing

    routing_data, gather_idx, scatter_idx = aiter_routing(
        gating_output, topk, sm_first=not renormalize
    )

    # gfx1250: aiter's in-kernel gather is numerically broken
    if on_gfx1250():
        gather_src = gather_idx.to(torch.long) // topk
        hidden_states = hidden_states[gather_src]
        gather_idx = None

    return triton_kernel_fused_mxfp4_w4a8_experts(
        None,
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_idx,
        scatter_idx,
        activation=activation.value,
        quant_config=quant_config,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        unpadded_N_w1=unpadded_N_w1,
        unpadded_K_w1=unpadded_K_w1,
        unpadded_N_w2=unpadded_N_w2,
        unpadded_K_w2=unpadded_K_w2,
    )


def triton_kernel_fused_mxfp4_w4a8_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    routing_data,  # RoutingData
    gather_indx,  # GatherIndx
    scatter_indx,  # ScatterIndx
    activation: str = "silu",
    quant_config: FusedMoEQuantConfig | None = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    a1q_scale: torch.Tensor | None = None,
    unpadded_N_w1=None,
    unpadded_K_w1=None,
    unpadded_N_w2=None,
    unpadded_K_w2=None,
) -> torch.Tensor:
    assert quant_config is not None
    # type check, uint8 means mxfp4
    assert hidden_states.dtype == torch.bfloat16
    assert quant_config.w1_bias is None or quant_config.w1_bias.dtype == torch.float32
    assert quant_config.w2_bias is None or quant_config.w2_bias.dtype == torch.float32

    # Shape check: weights are padded (e.g. hidden_size padded for
    # GFX950 swizzle).
    assert hidden_states.shape[-1] == w1.shape[-2]
    assert w2.shape[-1] == w1.shape[1]

    E, _, N = w1.shape

    if global_num_experts == -1:
        global_num_experts = E

    gammas = routing_data.gate_scal if routing_data else None

    from aiter.ops.triton.moe_op_gemm_a8w4 import moe_gemm_a8w4
    from aiter.ops.triton.quant_moe import downcast_to_static_fp8

    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        should_use_cdna4_mx_scale_swizzle,
    )

    _swizzle_mx_scale = "CDNA4_SCALE" if should_use_cdna4_mx_scale_swizzle() else None

    assert quant_config.w1_precision is not None, (
        "w1_precision in quant config can't be None"
    )
    assert quant_config.w2_precision is not None, (
        "w2_precision in quant config can't be None"
    )

    hidden_states = downcast_to_static_fp8(
        hidden_states, quant_config.w1_precision.flex_ctx.lhs_data.scale
    )

    intermediate_cache1 = moe_gemm_a8w4(
        hidden_states,
        w1.storage.data,
        None,
        quant_config.w1_precision.weight_scale.storage.data,
        quant_config.w1_precision.flex_ctx.lhs_data.scale,
        quant_config.w2_precision.flex_ctx.lhs_data.scale,
        quant_config.w1_bias,
        routing_data,
        gather_indx=gather_indx,
        gammas=gammas if apply_router_weight_on_input else None,
        swizzle_mx_scale=_swizzle_mx_scale,
        out_dtype=torch.float8_e4m3fn,
        apply_swiglu=True,
        alpha=swiglu_alpha,
        limit=swiglu_limit,
        unpadded_N=unpadded_N_w1,
        unpadded_K=unpadded_K_w1,
    )

    intermediate_cache3 = moe_gemm_a8w4(
        intermediate_cache1,
        w2.storage.data,
        None,
        quant_config.w2_precision.weight_scale.storage.data,
        quant_config.w2_precision.flex_ctx.lhs_data.scale,
        None,
        quant_config.w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        gammas=None if apply_router_weight_on_input else gammas,
        swizzle_mx_scale=_swizzle_mx_scale,
        unpadded_N=unpadded_N_w2,
        unpadded_K=unpadded_K_w2,
    )

    return intermediate_cache3


class AiterW4A8ExpertsMonolithic(mk.FusedMoEExpertsMonolithic):
    """
    Monolithic MXFP4 W4A8 expert using AITER triton kernels.

    This backend uses:
    - aiter.ops.triton.moe_routing.routing for routing
    - aiter.ops.triton.moe_op_gemm_a8w4.moe_gemm_a8w4 for computation

    Weight format: MXFP4 weights with GFX950 swizzle
    Activation: Static FP8 quantization
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.topk = moe_config.experts_per_token
        self.renormalize = moe_config.routing_method in (
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        if not rocm_aiter_ops.is_enabled():
            return False
        from vllm.platforms.rocm import on_gfx950, on_gfx1250

        return on_gfx950() or on_gfx1250()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # W4A8: MXFP4 weights with static FP8 activations
        SUPPORTED_W_A = [
            (kMxfp4Static, kFp8StaticTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # Only SILU activation (swiglu) is supported
        return activation == MoEActivation.SWIGLUOAI

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
            and moe_parallel_config.dp_size <= 1
        )

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    @property
    def expects_unquantized_inputs(self) -> bool:
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
        assert self.moe_config.intermediate_size_per_partition_unpadded is not None
        assert self.moe_config.hidden_dim_unpadded is not None
        return aiter_triton_kernel_w4a8_moe_forward(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            gating_output=router_logits,
            topk=self.topk,
            renormalize=self.renormalize,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            quant_config=self.quant_config,
            apply_router_weight_on_input=apply_router_weight_on_input,
            unpadded_N_w1=self.moe_config.intermediate_size_per_partition_unpadded * 2,
            unpadded_K_w1=self.moe_config.hidden_dim_unpadded,
            unpadded_N_w2=self.moe_config.hidden_dim_unpadded,
            unpadded_K_w2=self.moe_config.intermediate_size_per_partition_unpadded,
        )


def _aiter_raw(t):
    if t is None or isinstance(t, torch.Tensor):
        return t
    return t.storage.data if hasattr(t, "storage") else t


def _aiter_w4a16_silu_via_a8w4(
    hidden_states: torch.Tensor,
    w1_data,
    w2_data,
    w1_wscale,
    w2_wscale,
    w1_bias,
    w2_bias,
    routing_data,
    gather_idx,
    scatter_idx,
    gammas,
    apply_router_weight_on_input: bool,
    swiglu_limit: float,
    unpadded_N_w1,
    unpadded_K_w1,
    unpadded_N_w2,
    unpadded_K_w2,
) -> torch.Tensor:
    """
    MXFP4 w4a16 MoE with a SILU (concatenated ``[gate | up]``) activation.
    """
    from aiter.ops.triton.fusions.fused_clamp_act_mul import fused_clamp_act_mul
    from aiter.ops.triton.moe_op_gemm_a8w4 import moe_gemm_a8w4
    from aiter.ops.triton.quant import dynamic_mxfp8_quant

    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        should_use_cdna4_mx_scale_swizzle,
    )

    swz = "CDNA4_SCALE" if should_use_cdna4_mx_scale_swizzle() else None
    quant_dtype = torch.float8_e4m3fn

    g1_gammas = gammas if apply_router_weight_on_input else None
    g2_gammas = None if apply_router_weight_on_input else gammas

    hidden_q, a1_scale = dynamic_mxfp8_quant(hidden_states, quant_dtype=quant_dtype)
    raw_gate_up = moe_gemm_a8w4(
        hidden_q,
        w1_data,
        a1_scale,
        w1_wscale,
        None,
        None,
        w1_bias,
        routing_data,
        gather_indx=gather_idx,
        gammas=g1_gammas,
        swizzle_mx_scale=swz,
        out_dtype=torch.bfloat16,
        apply_swiglu=False,
        unpadded_N=unpadded_N_w1,
        unpadded_K=unpadded_K_w1,
    )
    if unpadded_N_w1 is not None:
        raw_gate_up = raw_gate_up[:, :unpadded_N_w1]

    interim_fp8, a2_scale = fused_clamp_act_mul(
        raw_gate_up,
        swiglu_limit=swiglu_limit,
        activation="silu",
        dtype_quant=quant_dtype,
        scale_dtype_fmt="ue8m0",
        quant_block_size=32,
    )

    out = moe_gemm_a8w4(
        interim_fp8,
        w2_data,
        a2_scale,
        w2_wscale,
        None,
        None,
        w2_bias,
        routing_data,
        scatter_indx=scatter_idx,
        gammas=g2_gammas,
        swizzle_mx_scale=swz,
        unpadded_N=unpadded_N_w2,
        unpadded_K=unpadded_K_w2,
    )
    return out


def aiter_triton_kernel_w4a16_moe_forward(
    hidden_states: torch.Tensor,
    w1,
    w2,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: MoEActivation = MoEActivation.SWIGLUOAI,
    quant_config: FusedMoEQuantConfig | None = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    unpadded_N_w1=None,
    unpadded_K_w1=None,
    unpadded_N_w2=None,
    unpadded_K_w2=None,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    e_score_correction_bias: torch.Tensor | None = None,
    routed_scaling_factor: float | None = None,
    score_mode: str | None = None,
):
    assert quant_config is not None and rocm_aiter_ops.is_enabled()
    from vllm.platforms.rocm import on_gfx1250

    try:
        from aiter.ops.triton.moe.moe_op_gemm_a16w4 import moe_gemm_a16w4
        from aiter.ops.triton.moe.moe_routing import routing as _routing_mod
    except ImportError:
        from aiter.ops.triton.moe.moe_op_gemm_a16w4 import moe_gemm_a16w4
        from aiter.ops.triton.moe_routing import routing as _routing_mod

    if on_gfx1250():
        _routing_mod.is_tdm_avail = lambda: False
    aiter_routing = _routing_mod.routing

    if score_mode is not None:
        use_grouped_topk = num_expert_group is not None and num_expert_group > 1
        routing_data, gather_idx, scatter_idx = aiter_routing(
            gating_output,
            topk,
            score_mode=score_mode,
            bias=e_score_correction_bias,
            renorm=renormalize,
            routed_scaling_factor=(
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            ),
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
        )
    else:
        routing_data, gather_idx, scatter_idx = aiter_routing(
            gating_output, topk, sm_first=not renormalize
        )

    if on_gfx1250():
        gather_src = gather_idx.to(torch.long) // topk
        hidden_states = hidden_states[gather_src]
        gather_idx = None

    assert quant_config.w1_precision is not None
    assert quant_config.w2_precision is not None

    w1_data = _aiter_raw(w1)
    w2_data = _aiter_raw(w2)
    w1_wscale = _aiter_raw(quant_config.w1_precision.weight_scale)
    w2_wscale = _aiter_raw(quant_config.w2_precision.weight_scale)

    gammas = routing_data.gate_scal if routing_data else None

    swiglu_alpha = (
        quant_config.gemm1_alpha if quant_config.gemm1_alpha is not None else 1.0
    )
    swiglu_limit = (
        quant_config.gemm1_clamp_limit
        if quant_config.gemm1_clamp_limit is not None
        else 7.0
    )

    # SILU on gfx1250: use the verified a8w4 kernel (dynamic MXFP8); a16w4 faults.
    if activation == MoEActivation.SILU and on_gfx1250():
        return _aiter_w4a16_silu_via_a8w4(
            hidden_states,
            w1_data,
            w2_data,
            w1_wscale,
            w2_wscale,
            quant_config.w1_bias,
            quant_config.w2_bias,
            routing_data,
            gather_idx,
            scatter_idx,
            gammas,
            apply_router_weight_on_input,
            swiglu_limit,
            unpadded_N_w1,
            unpadded_K_w1,
            unpadded_N_w2,
            unpadded_K_w2,
        )

    # SILU: silu(gate) * up — same kernel, just no "+1" residual in swiglu.
    swiglu_add_residual = activation != MoEActivation.SILU

    intermediate = moe_gemm_a16w4(
        hidden_states,
        w1_data,
        None,
        w1_wscale,
        None,
        None,
        quant_config.w1_bias,
        routing_data,
        gather_indx=gather_idx,
        gammas=gammas if apply_router_weight_on_input else None,
        swizzle_mx_scale=None,
        apply_swiglu=True,
        alpha=swiglu_alpha,
        limit=swiglu_limit,
        swiglu_add_residual=swiglu_add_residual,
        unpadded_N=unpadded_N_w1,
        unpadded_K=unpadded_K_w1,
    )

    out = moe_gemm_a16w4(
        intermediate,
        w2_data,
        None,
        w2_wscale,
        None,
        None,
        quant_config.w2_bias,
        routing_data,
        scatter_indx=scatter_idx,
        gammas=None if apply_router_weight_on_input else gammas,
        swizzle_mx_scale=None,
        unpadded_N=unpadded_N_w2,
        unpadded_K=unpadded_K_w2,
    )

    return out


class AiterW4A16ExpertsMonolithic(mk.FusedMoEExpertsMonolithic):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.topk = moe_config.experts_per_token
        self.renormalize = moe_config.routing_method in (
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
            RoutingMethodType.DeepseekV4,
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        if not rocm_aiter_ops.is_enabled():
            return False
        from vllm.platforms.rocm import on_gfx950, on_gfx1250

        return on_gfx950() or on_gfx1250()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, None)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SWIGLUOAI, MoEActivation.SILU)

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
            and moe_parallel_config.dp_size <= 1
        )

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
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

    @property
    def expects_unquantized_inputs(self) -> bool:
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
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        assert self.moe_config.intermediate_size_per_partition_unpadded is not None
        assert self.moe_config.hidden_dim_unpadded is not None
        score_mode = (
            "sqrtsoftplus"
            if self.moe_config.routing_method == RoutingMethodType.DeepseekV4
            else None
        )
        return aiter_triton_kernel_w4a16_moe_forward(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            gating_output=router_logits,
            topk=self.topk,
            renormalize=self.renormalize,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            quant_config=self.quant_config,
            apply_router_weight_on_input=apply_router_weight_on_input,
            unpadded_N_w1=self.moe_config.intermediate_size_per_partition_unpadded * 2,
            unpadded_K_w1=self.moe_config.hidden_dim_unpadded,
            unpadded_N_w2=self.moe_config.hidden_dim_unpadded,
            unpadded_K_w2=self.moe_config.intermediate_size_per_partition_unpadded,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            score_mode=score_mode,
        )
