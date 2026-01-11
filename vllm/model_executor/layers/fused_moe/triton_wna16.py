# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    TritonExperts,
    _resize_cache,
    invoke_fused_moe_wna16_triton_kernel,
    moe_align_block_size,
    moe_kernel_quantize_input,
    try_get_optimal_moe_config,
)
from vllm.triton_utils import tl


class TritonWNA16Experts(TritonExperts):
    def __init__(
        self,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(quant_config)

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
        # Check constraints.
        if self.quant_config.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), "Hidden size mismatch"
        else:
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
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, N // 2)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], global_num_experts, expert_map
        )

        invoke_fused_moe_wna16_triton_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            self.w1_scale,
            self.quant_config.w1_zp,
            None,  # topk_weights
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,  # mul_routed_weights
            top_k_num,
            config,
            compute_type=compute_type,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            block_shape=self.block_shape,
        )

        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        a2q_scale: torch.Tensor | None = None

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            intermediate_cache2,
            a2_scale,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
        )

        invoke_fused_moe_wna16_triton_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            self.w2_scale,
            self.quant_config.w2_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            block_shape=self.block_shape,
        )

        # separate function is required for MoE + LoRA
        self.moe_sum(intermediate_cache3, output)
