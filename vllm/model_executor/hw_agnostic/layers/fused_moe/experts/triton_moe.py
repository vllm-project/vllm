# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-based MoE expert implementations."""

import torch

import vllm.model_executor.hw_agnostic.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.hw_agnostic.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.hw_agnostic.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment,
    invoke_fused_moe_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.utils import (
    _resize_cache,
    moe_kernel_quantize_input,
    swiglu_limit_func,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl


class TritonExperts(mk.FusedMoEExpertsModular):
    """Triton-based fused MoE expert implementation."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def activation(
        self,
        activation: MoEActivation,
        output: torch.Tensor,
        input: torch.Tensor,
        **kwargs,
    ) -> None:
        gemm1_clamp_limit = self.quant_config.gemm1_clamp_limit
        if activation == MoEActivation.SILU and gemm1_clamp_limit is not None:
            swiglu_limit_func(output, input, float(gemm1_clamp_limit))
            return

        super().activation(
            activation,
            output,
            input,
            clamp_limit=gemm1_clamp_limit,
        )

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
        workspace1 = (M, topk, max(activation_out_dim, K))
        workspace2 = (M, topk, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

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
        # Check constraints.
        assert hidden_states.size(-1) == w1.size(2), (
            f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
        )
        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert hidden_states.dim() == 2
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ]

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            self.quant_config.config_name(hidden_states.dtype),
            num_tokens,
            block_shape=self.block_shape,
        )

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        elif (
            hidden_states.dtype == torch.float8_e4m3fn
            or hidden_states.dtype == torch.float8_e4m3fnuz
        ):
            compute_type = tl.bfloat16
        else:
            raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

        # Note that the output tensor might be in workspace1
        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
        cache2_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, cache2_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _prepare_expert_assignment(
                topk_ids,
                config,
                num_tokens,
                top_k_num,
                global_num_experts,
                expert_map,
            )
        )

        invoke_fused_moe_triton_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            a1q_scale,
            self.w1_scale,
            None,  # topk_weights
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,  # mul_routed_weights
            top_k_num,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            use_int8_w8a8=self.quant_config.use_int8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            per_channel_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
            B_bias=self.w1_bias,
        )

        a2q_scale: torch.Tensor | None = None

        # Fuse SiLU+Mul + FP8 block quantize into a single kernel when
        # conditions permit (gated SiLU, fp8 block quant with group_size=128).
        if (
            activation == MoEActivation.SILU
            and self.quant_config.use_fp8_w8a8
            and self.block_shape == [128, 128]
        ):
            qintermediate_cache2, a2q_scale = ops.silu_and_mul_per_block_quant(
                intermediate_cache1.view(-1, N),
                group_size=128,
                quant_dtype=current_platform.fp8_dtype(),
            )
        else:
            self.activation(
                activation, intermediate_cache2, intermediate_cache1.view(-1, N)
            )

            qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
                intermediate_cache2,
                a2_scale,
                self.quant_dtype,
                self.per_act_token_quant,
                self.block_shape,
            )

        invoke_fused_moe_triton_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            a2q_scale,
            self.w2_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            use_int8_w8a8=self.quant_config.use_int8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            per_channel_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
            B_bias=self.w2_bias,
        )

        self.moe_sum(intermediate_cache3, output)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        torch.sum(input, dim=1, out=output)
