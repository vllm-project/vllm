# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform


def _merge_attn_states_torch(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
    prefill_tokens_with_context: int | None = None,
    output_scale: torch.Tensor | None = None,
) -> None:
    assert output_scale is None, "CPU fallback does not support FP8 output"

    num_tokens = output.shape[0]
    if prefill_tokens_with_context is None:
        prefill_tokens_with_context = num_tokens

    prefix_lse = torch.where(
        torch.isinf(prefix_lse) & (prefix_lse > 0),
        torch.full_like(prefix_lse, float("-inf")),
        prefix_lse,
    )
    suffix_lse = torch.where(
        torch.isinf(suffix_lse) & (suffix_lse > 0),
        torch.full_like(suffix_lse, float("-inf")),
        suffix_lse,
    )

    if prefill_tokens_with_context < num_tokens:
        output[prefill_tokens_with_context:].copy_(
            suffix_output[prefill_tokens_with_context:]
        )
        if output_lse is not None:
            output_lse[:, prefill_tokens_with_context:].copy_(
                suffix_lse[:, prefill_tokens_with_context:]
            )

    if prefill_tokens_with_context == 0:
        return

    prefix_lse = prefix_lse[:, :prefill_tokens_with_context].float()
    suffix_lse = suffix_lse[:, :prefill_tokens_with_context].float()
    max_lse = torch.maximum(prefix_lse, suffix_lse)
    prefix_se = torch.exp(prefix_lse - max_lse)
    suffix_se = torch.exp(suffix_lse - max_lse)
    out_se = prefix_se + suffix_se
    both_empty = max_lse == float("-inf")

    prefix_scale = (prefix_se / out_se).transpose(0, 1).unsqueeze(-1)
    suffix_scale = (suffix_se / out_se).transpose(0, 1).unsqueeze(-1)
    merged = (
        prefix_output[:prefill_tokens_with_context].float() * prefix_scale
        + suffix_output[:prefill_tokens_with_context].float() * suffix_scale
    )
    merged.masked_fill_(both_empty.transpose(0, 1).unsqueeze(-1), 0.0)
    output[:prefill_tokens_with_context].copy_(merged.to(output.dtype))

    if output_lse is not None:
        merged_lse = torch.log(out_se) + max_lse
        merged_lse.masked_fill_(both_empty, float("-inf"))
        output_lse[:, :prefill_tokens_with_context].copy_(merged_lse)


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
    prefill_tokens_with_context: int | None = None,
    output_scale: torch.Tensor | None = None,
) -> None:
    """Merge partial attention outputs from prefix (KV cache) and suffix
    (new tokens) into a single output tensor using the log-sum-exp (LSE)
    rescaling method described in section 2.2 of
    https://www.arxiv.org/pdf/2501.01005.

    For tokens that have prefix context (token index < prefill_tokens_with_context),
    the prefix and suffix partial outputs are combined as a weighted sum.
    For tokens without prefix context, the suffix output is copied directly.

    Args:
        output: Output tensor of shape [NUM_TOKENS, NUM_HEADS, HEAD_SIZE].
        prefix_output: Partial attention output over the prefix (KV cache),
            shape [NUM_TOKENS, NUM_HEADS, HEAD_SIZE].
        prefix_lse: Log-sum-exp values for the prefix attention,
            shape [NUM_HEADS, NUM_TOKENS].
        suffix_output: Partial attention output over the suffix (new KV),
            shape [NUM_TOKENS, NUM_HEADS, HEAD_SIZE].
        suffix_lse: Log-sum-exp values for the suffix attention,
            shape [NUM_HEADS, NUM_TOKENS].
        output_lse: Optional tensor to store the merged LSE values,
            shape [NUM_HEADS, NUM_TOKENS]. If None, LSE is not written out.
        prefill_tokens_with_context: Number of prefill tokens that have
            prefix context and therefore require merging. Tokens at indices
            >= this value are decode or context-free prefill tokens whose
            output is taken directly from suffix_output. If None, all tokens
            are treated as having context.
        output_scale: Optional scalar tensor for FP8 static quantization.
            When provided, output must be FP8 dtype.
    """

    # Both the CUDA and Triton kernels derive the suffix head stride from
    # prefix_output, so suffix_output must share the same head stride.
    assert prefix_output.stride(1) == suffix_output.stride(1), (
        "merge_attn_states requires prefix_output and suffix_output to have "
        f"matching head strides, got {prefix_output.stride(1)} and "
        f"{suffix_output.stride(1)}"
    )

    # NOTE(DefTruth): Currently, custom merge_attn_states CUDA kernel
    # does not support FP8 dtype for inputs, fallback to use Triton kernel.
    # However, when output_scale is provided, the inputs are still BF16/FP16
    # and the output is FP8 — both CUDA and Triton support this.
    # FP8 output requires output_scale to be set.
    if output.dtype not in (torch.float32, torch.half, torch.bfloat16):
        assert output_scale is not None, (
            f"output_scale is required when output is {output.dtype}"
        )

    def supported_dtypes(prefix: torch.Tensor) -> bool:
        return prefix.dtype in [torch.float32, torch.half, torch.bfloat16]

    # NOTE(DefTruth): Currently, custom merge_attn_states CUDA
    # kernel load/store 128b(16 bytes) per memory issue within
    # thread. Namely, the headsize(headdim) must be multiple of
    # pack_size based on input dtype (float32 -> 4, half/bfloat16 -> 8).
    def supported_headdim(prefix: torch.Tensor) -> bool:
        headdim = prefix.shape[2]  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        if prefix.dtype == torch.float32:
            return headdim % 4 == 0
        return headdim % 8 == 0

    if (
        current_platform.is_cuda()
        and supported_dtypes(prefix_output)
        and supported_headdim(prefix_output)
    ):
        from vllm._custom_ops import merge_attn_states

        return merge_attn_states(
            output,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse,
            prefill_tokens_with_context,
            output_scale,
        )
    elif current_platform.is_cpu():
        return _merge_attn_states_torch(
            output,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse,
            prefill_tokens_with_context,
            output_scale,
        )
    else:
        from vllm.v1.attention.ops.triton_merge_attn_states import (
            merge_attn_states,
        )

        return merge_attn_states(
            output,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse,
            prefill_tokens_with_context,
            output_scale,
        )
