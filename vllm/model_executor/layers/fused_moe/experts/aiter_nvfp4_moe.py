# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4-BF16 MoE experts through AITER's FlyDSL fused MoE."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    moe_kernel_quantize_input,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950


class AiterNvfp4Experts(mk.FusedMoEExpertsModular):
    """NVFP4-BF16 MoE experts using AITER's fused_moe implementation."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.w1_scale_val = quant_config.w1_scale
        self.w2_scale_val = quant_config.w2_scale
        self.w1_global_scale = quant_config.g1_alphas
        self.w2_global_scale = quant_config.g2_alphas

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        # AITER NVFP4-BF16 consumes BF16 activations and NVFP4 weights.
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_rocm()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if moe_config.is_lora_enabled:
            return False, "kernel does not support LoRA"
        if moe_config.has_bias:
            return False, "kernel does not support bias"

        is_supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )

        if not is_supported:
            return is_supported, reason

        if not current_platform.is_rocm() or not on_gfx950():
            return False, "kernel available only on AMD gfx950 devices for now"

        if not rocm_aiter_ops.is_enabled():
            return (
                False,
                "kernel requires aiter library (enable with VLLM_ROCM_USE_AITER=1)",
            )

        if not is_aiter_found_and_supported():
            return (
                False,
                "kernel requires aiter library (not found in user environment)",
            )

        return is_supported, reason

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
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
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        return (M, topk, activation_out_dim), (0,), (M, K)

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
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import (
            get_2stage_cfgs,
            get_padded_M,
            moe_sorting,
        )
        from aiter.ops.flydsl.moe_kernels import (
            flydsl_moe_stage1,
            flydsl_moe_stage2,
            get_flydsl_kernel_params,
        )

        assert activation == MoEActivation.SILU
        assert hidden_states.dtype == torch.bfloat16
        assert w1.dtype == torch.uint8
        assert w2.dtype == torch.uint8
        assert self.w1_scale_val is not None
        assert self.w2_scale_val is not None
        assert self.w1_global_scale is not None
        assert self.w2_global_scale is not None

        E, num_tokens, N, K, topk = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        inter_dim = self.adjust_N_for_activation(N, activation)
        is_g1u1 = inter_dim != w1.shape[1]

        if expert_tokens_meta is not None:
            num_local_tokens = expert_tokens_meta.expert_num_tokens
        else:
            num_local_tokens = None

        if expert_map is not None:
            local_mask = (expert_map >= 0).to(torch.int32)
            expert_mask = torch.cat([local_mask, local_mask.new_zeros(1)])
        else:
            expert_mask = None

        metadata = get_2stage_cfgs(
            get_padded_M(num_tokens),
            K,
            inter_dim,
            E,
            topk,
            output.dtype,
            torch.bfloat16,
            "nvfp4_bf16",
            QuantType.No,
            is_g1u1,
            ActivationType.Silu,
            apply_router_weight_on_input,
            0,
            0,
            True,
            is_ep=expert_mask is not None,
        )
        if metadata.run_1stage:
            raise NotImplementedError("AiterNvfp4Experts only supports 2-stage MoE")

        block_m = int(metadata.block_m)
        global_num_experts_for_sort = (
            expert_mask.numel() if expert_mask is not None else E
        )
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
            topk_ids.to(torch.int32),
            topk_weights.to(torch.float32),
            global_num_experts_for_sort,
            K,
            output.dtype,
            block_m,
            expert_mask,
            num_local_tokens,
        )

        hidden_states_qdq, _ = moe_kernel_quantize_input(
            A=hidden_states,
            A_scale=self.quant_config.a1_gscale,
            quant_dtype="nvfp4",
            per_act_token_quant=False,
            quantization_emulation=True,
        )

        stage1_func = metadata.stage1
        stage2_func = metadata.stage2
        stage1_kernel_name = getattr(stage1_func, "keywords", {}).get("kernelName", "")
        stage2_kernel_name = getattr(stage2_func, "keywords", {}).get("kernelName", "")
        stage1_params = get_flydsl_kernel_params(stage1_kernel_name)
        stage2_params = get_flydsl_kernel_params(stage2_kernel_name)

        intermediate = _resize_cache(workspace13, (num_tokens, topk, inter_dim))
        flydsl_moe_stage1(
            a=hidden_states_qdq,
            w1=w1,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            out=intermediate,
            topk=topk,
            tile_m=stage1_params["tile_m"],
            tile_n=stage1_params["tile_n"],
            tile_k=stage1_params["tile_k"],
            a_dtype=stage1_params["a_dtype"],
            b_dtype=stage1_params["b_dtype"],
            out_dtype=stage1_params["out_dtype"],
            act="silu",
            w1_scale=self.w1_scale_val,
            global_scale=self.w1_global_scale,
            sorted_weights=(sorted_weights if apply_router_weight_on_input else None),
            use_async_copy=True,
            k_batch=stage1_params.get("k_batch", 1),
            waves_per_eu=stage1_params.get("waves_per_eu", 3),
            b_nt=stage1_params.get("b_nt", 2),
            gate_mode=stage1_params.get("gate_mode", "separated"),
            a_scale_one=stage1_params.get("a_scale_one", False),
            xcd_swizzle=stage1_params.get("xcd_swizzle", 0),
        )

        intermediate_qdq, _ = moe_kernel_quantize_input(
            A=intermediate.view(-1, inter_dim),
            A_scale=self.quant_config.a2_gscale,
            quant_dtype="nvfp4",
            per_act_token_quant=False,
            quantization_emulation=True,
        )
        intermediate_qdq = intermediate_qdq.view(num_tokens, topk, inter_dim)

        if stage2_params.get("mode", "atomic") == "atomic":
            output.zero_()
        flydsl_moe_stage2(
            inter_states=intermediate_qdq,
            w2=w2,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            out=output,
            topk=topk,
            tile_m=stage2_params["tile_m"],
            tile_n=stage2_params["tile_n"],
            tile_k=stage2_params["tile_k"],
            a_dtype=stage2_params["a_dtype"],
            b_dtype=stage2_params["b_dtype"],
            out_dtype=stage2_params["out_dtype"],
            mode=stage2_params.get("mode", "atomic"),
            w2_scale=self.w2_scale_val,
            global_scale=self.w2_global_scale,
            sorted_weights=(
                sorted_weights if not apply_router_weight_on_input else None
            ),
            sort_block_m=stage2_params.get("sort_block_m", 0),
            waves_per_eu=stage2_params.get("waves_per_eu", None),
            use_async_copy=stage2_params.get("use_async_copy", False),
            cu_num_mul=stage2_params.get("cu_num_mul", 1),
            b_nt=stage2_params.get("b_nt", 0),
            persist=stage2_params.get("persist", None),
            xcd_swizzle=stage2_params.get("xcd_swizzle", 0),
            expert_mask=expert_mask,
            topk_ids=topk_ids,
        )
