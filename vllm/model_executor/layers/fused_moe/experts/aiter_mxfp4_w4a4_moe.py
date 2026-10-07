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
    kMxfp4Dynamic,
    kMxfp4Static,
)

__all__ = [
    "AiterW4A4ExpertsMonolithic",
    "aiter_triton_kernel_w4a4_moe_forward",
]


def _aiter_raw(t):
    if t is None or isinstance(t, torch.Tensor):
        return t
    return t.storage.data if hasattr(t, "storage") else t


def aiter_triton_kernel_w4a4_moe_forward(
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
    input_ids: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
):
    assert quant_config is not None and rocm_aiter_ops.is_enabled()

    from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4, mxfp4_quant

    try:
        from aiter.ops.triton.moe.moe_routing import routing as _routing_mod
    except ImportError:
        from aiter.ops.triton.moe_routing import routing as _routing_mod

    from vllm.platforms.rocm import on_gfx1250

    if on_gfx1250():
        _routing_mod.is_tdm_avail = lambda: False
    aiter_routing = _routing_mod.routing

    gating_output = torch.nan_to_num(gating_output, nan=0.0, posinf=0.0, neginf=0.0)

    if hash_indices_table is not None:
        assert input_ids is not None, "hash routing requires input_ids"
        n_tokens, n_expts_tot = gating_output.shape
        tokens_per_expt = max(1, n_tokens * topk // n_expts_tot)
        block_m = max(16, min(1 << (tokens_per_expt - 1).bit_length(), 128))
        routing_data, gather_idx, scatter_idx = _routing_mod.routing_from_hash(
            gating_output,
            hash_indices_table,
            input_ids.to(hash_indices_table.dtype),
            topk,
            block_m,
            score_mode=score_mode or "sqrtsoftplus",
            renorm=renormalize,
            routed_scaling_factor=(
                routed_scaling_factor
                if routed_scaling_factor is not None
                else 1.0
            ),
        )
    elif score_mode is not None:
        use_grouped_topk = (
            num_expert_group is not None and num_expert_group > 1
        )
        routing_data, gather_idx, scatter_idx = aiter_routing(
            gating_output,
            topk,
            score_mode=score_mode,
            bias=(
                e_score_correction_bias.float()
                if e_score_correction_bias is not None
                else None
            ),
            renorm=renormalize,
            routed_scaling_factor=(
                routed_scaling_factor
                if routed_scaling_factor is not None
                else 1.0
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

    w1_data = _aiter_raw(w1)
    w2_data = _aiter_raw(w2)
    w1_scale = _aiter_raw(quant_config.w1_scale)
    w2_scale = _aiter_raw(quant_config.w2_scale)

    gammas = routing_data.gate_scal if routing_data else None

    swiglu_limit = (
        quant_config.gemm1_clamp_limit
        if quant_config.gemm1_clamp_limit is not None
        else 7.0
    )
    # SWIGLUOAI needs the alpha/residual swiglu fused into the GEMM1 epilogue,
    # which reads gate/up interleaved along N (see the oracle's weight prep).
    fused_swiglu = activation in (
        MoEActivation.SWIGLUOAI,
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
    )
    swiglu_kwargs: dict[str, object] = {}
    if fused_swiglu:
        assert quant_config.gemm1_beta in (None, 1.0), (
            "aiter's fused swiglu hardcodes the residual to (up + 1); "
            f"gemm1_beta={quant_config.gemm1_beta} cannot be expressed"
        )
        swiglu_kwargs = {
            "alpha": (
                quant_config.gemm1_alpha
                if quant_config.gemm1_alpha is not None
                else 1.0
            ),
            "limit": swiglu_limit,
            "swiglu_add_residual": quant_config.gemm1_beta == 1.0,
        }
    swizzle_mx_scale = (
        "GFX1250_SCALE" if on_gfx1250() else None
    )

    x_q, x_scale = mxfp4_quant(hidden_states.to(torch.bfloat16))

    # GEMM1: gate+up projection.
    raw_intermediate = moe_gemm_a4w4(
        x_q,
        w1_data,
        x_scale,
        w1_scale,
        bias=quant_config.w1_bias,
        routing_data=routing_data,
        gather_indx=gather_idx,
        gammas=gammas if apply_router_weight_on_input else None,
        swizzle_mx_scale=swizzle_mx_scale,
        apply_swiglu=fused_swiglu,
        **swiglu_kwargs,
    )

    if fused_swiglu:
        # The swiglu epilogue already halved N.
        intermediate = (
            raw_intermediate
            if unpadded_N_w1 is None
            else raw_intermediate[:, : unpadded_N_w1 // 2]
        )
    else:
        if unpadded_N_w1 is not None:
            raw_intermediate = raw_intermediate[:, :unpadded_N_w1]

        # SiLU(gate) * up on the concatenated [gate | up] halves
        from aiter.ops.triton.fusions.fused_clamp_act_mul import fused_clamp_act_mul

        half_n = raw_intermediate.shape[-1] // 2
        intermediate = torch.empty(
            raw_intermediate.shape[0], half_n,
            dtype=raw_intermediate.dtype,
            device=raw_intermediate.device,
        )
        fused_clamp_act_mul(
            raw_intermediate,
            out=intermediate,
            swiglu_limit=swiglu_limit,
            activation="silu",
            dtype_quant=None,
        )

    # GEMM2: down projection with scatter-reduce
    mid_q, mid_scale = mxfp4_quant(intermediate.to(torch.bfloat16))

    out = moe_gemm_a4w4(
        mid_q,
        w2_data,
        mid_scale,
        w2_scale,
        bias=quant_config.w2_bias,
        routing_data=routing_data,
        scatter_indx=scatter_idx,
        gammas=None if apply_router_weight_on_input else gammas,
        swizzle_mx_scale=swizzle_mx_scale,
        apply_swiglu=False,
    )

    return out


class AiterW4A4ExpertsMonolithic(mk.FusedMoEExpertsMonolithic):
    """Monolithic MXFP4 W4A4 expert using AITER Triton/Gluon kernels (gfx1250).

    Uses moe_gemm_a4w4 (auto-selects gluon backend on gfx1250) with dynamic
    MXFP4 activation quantization via mxfp4_quant.
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
            RoutingMethodType.DeepseekV4,
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.MiniMax2,
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_hash_routing() -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        if not rocm_aiter_ops.is_enabled():
            return False
        from vllm.platforms.rocm import on_gfx1250

        return on_gfx1250()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, kMxfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            MoEActivation.SILU,
        )

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
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.MiniMax2,
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
        routing_replay_out: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        hash_indices_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.moe_config.intermediate_size_per_partition_unpadded is not None
        assert self.moe_config.hidden_dim_unpadded is not None
        routing_method = self.moe_config.routing_method
        if routing_method == RoutingMethodType.DeepseekV4:
            score_mode = "sqrtsoftplus"
        elif routing_method in (
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.MiniMax2,
        ):
            # Sigmoid scores + routing bias; grouped for DeepSeek-R1/V3,
            # ungrouped for MiniMax-M3.
            score_mode = "sigmoid"
        else:
            score_mode = None
        return aiter_triton_kernel_w4a4_moe_forward(
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
            unpadded_N_w1=(
                self.moe_config.intermediate_size_per_partition_unpadded * 2
            ),
            unpadded_K_w1=self.moe_config.hidden_dim_unpadded,
            unpadded_N_w2=self.moe_config.hidden_dim_unpadded,
            unpadded_K_w2=(
                self.moe_config.intermediate_size_per_partition_unpadded
            ),
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            score_mode=score_mode,
            input_ids=input_ids,
            hash_indices_table=hash_indices_table,
        )
