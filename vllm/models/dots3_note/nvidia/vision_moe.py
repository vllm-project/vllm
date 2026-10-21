# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NOTE vision MoE execution matching the native encoder's FP8 semantics."""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment,
    dispatch_fused_moe_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.triton_utils import tl

_BLOCK_SHAPE = [128, 128]


def note_vision_fused_moe_fp8(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    """Run top-k SwiGLU MoE with the encoder's block-FP8 formula.

    Unlike vLLM's generic functional MoE path, the native NOTE encoder keeps
    dynamic activation scales as FP32 instead of rounding them to E8M0.  Weight
    scales are block 128x128 and activations are quantized per token/group-128.
    """
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            "NOTE vision FP8 MoE requires bfloat16 activations, got "
            f"{hidden_states.dtype}"
        )

    num_tokens = hidden_states.shape[0]
    num_experts, intermediate_size, _ = w13.shape
    output_size = w2.shape[1]
    topk = topk_ids.shape[1]
    config = try_get_optimal_moe_config(
        w13.shape,
        w2.shape,
        topk,
        "fp8_w8a8",
        num_tokens,
        block_shape=_BLOCK_SHAPE,
    )

    cache = torch.empty(
        num_tokens * topk * max(intermediate_size, output_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    first_gemm = cache[: num_tokens * topk * intermediate_size].view(
        num_tokens, topk, intermediate_size
    )
    second_gemm = cache[: num_tokens * topk * output_size].view(
        num_tokens, topk, output_size
    )
    activated_size = intermediate_size // 2
    activated = torch.empty(
        (num_tokens * topk, activated_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    quantized_input, input_scale = per_token_group_quant_fp8(
        hidden_states,
        _BLOCK_SHAPE[1],
        use_ue8m0=False,
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = _prepare_expert_assignment(
        topk_ids,
        config,
        num_tokens,
        topk,
        num_experts,
        None,
        block_shape=_BLOCK_SHAPE,
    )
    dispatch_fused_moe_kernel(
        quantized_input,
        w13,
        first_gemm,
        input_scale,
        w13_scale,
        None,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        topk,
        config,
        tl.bfloat16,
        True,
        False,
        False,
        False,
        False,
        _BLOCK_SHAPE,
    )

    apply_moe_activation(
        MoEActivation.SILU,
        activated,
        first_gemm.view(-1, intermediate_size),
    )
    quantized_activated, activated_scale = per_token_group_quant_fp8(
        activated,
        _BLOCK_SHAPE[1],
        use_ue8m0=False,
    )
    dispatch_fused_moe_kernel(
        quantized_activated,
        w2,
        second_gemm,
        activated_scale,
        w2_scale,
        None,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        True,
        1,
        config,
        tl.bfloat16,
        True,
        False,
        False,
        False,
        False,
        _BLOCK_SHAPE,
    )

    output = torch.empty_like(hidden_states)
    ops.moe_sum(second_gemm, output)
    return output
