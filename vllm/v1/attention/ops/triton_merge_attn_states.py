# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

float8_info = torch.finfo(current_platform.fp8_dtype())


# Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
# can be used to combine partial attention results (in the split-KV case)
def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
    prefill_tokens_with_context: int | None = None,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    quant_group_size: int | None = None,
    quant_scale_ue8m0: bool = False,
) -> None:
    num_tokens = output.shape[0]
    num_query_heads = output.shape[1]
    head_size = output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)
    # We assume the output stride on num_head is not always as same as the
    # `suffix_output` and `prefix_output`, as them might be padded by the
    # attention backend.
    prefix_head_stride = prefix_output.stride(1)
    output_head_stride = output.stride(1)

    # If prefill_tokens_with_context is None, all tokens should use prefix context
    if prefill_tokens_with_context is None:
        prefill_tokens_with_context = num_tokens

    if output_block_scale is not None:
        assert quant_group_size is not None, (
            "quant_group_size is required for per-token-per-group FP8 merge"
        )
        assert output_scale is None, (
            "output_scale must be None for per-token-per-group FP8 merge"
        )
        assert head_size % quant_group_size == 0, (
            f"head_size ({head_size}) must be a multiple of "
            f"quant_group_size ({quant_group_size})"
        )
        groups_per_head = head_size // quant_group_size
        sf_token_stride, sf_group_stride = output_block_scale.stride()
        merge_attn_states_group_fp8_kernel[(num_tokens, num_query_heads)](
            output,
            output_lse,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_block_scale,
            sf_token_stride,
            sf_group_stride,
            prefix_head_stride,
            output_head_stride,
            head_size,
            quant_group_size,
            groups_per_head,
            triton.next_power_of_2(quant_group_size),
            output_lse is not None,
            prefill_tokens_with_context,
            quant_scale_ue8m0,
        )
        return

    # TODO(woosuk): Use CUDA kernel instead of Triton to minimize CPU overhead.
    merge_attn_states_kernel[(num_tokens, num_query_heads)](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        prefix_head_stride,
        output_head_stride,
        output_scale,
        head_size,
        padded_head_size,
        output_lse is not None,
        prefill_tokens_with_context,
        output_scale is not None,
    )


@triton.jit
def merge_attn_states_kernel(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_head_stride,
    output_head_stride,
    output_scale,  # scale tensor or None
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
    prefill_tokens_with_context: tl.constexpr,
    USE_FP8: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    prefix_mask = token_idx < prefill_tokens_with_context

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE

    # For tokens without context (token_idx >= prefill_tokens_with_context),
    # directly copy from suffix_output
    if not prefix_mask:
        s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)
        if OUTPUT_LSE:
            tl.store(output_lse + head_idx * num_tokens + token_idx, s_lse)

        s_out = tl.load(
            suffix_output
            + token_idx * num_heads * prefix_head_stride
            + head_idx * prefix_head_stride
            + head_arange,
            mask=head_mask,
        )

        if USE_FP8:
            s_out = s_out * (1.0 / tl.load(output_scale))
            s_out = tl.clamp(s_out, FP8_MIN, FP8_MAX)
            s_out = s_out.to(output.dtype.element_ty)

        tl.store(
            output
            + token_idx * num_heads * output_head_stride
            + head_idx * output_head_stride
            + head_arange,
            s_out,
            mask=head_mask,
        )
        return

    # For tokens with context (token_idx < prefill_tokens_with_context),
    # perform normal merge operation
    p_lse = tl.load(prefix_lse + head_idx * num_tokens + token_idx)
    s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)

    # FA2 and FA3 have different behavior for when the sum-exp is 0, this namely
    # arises with 0 len seqlens. FA3 returns -inf here while FA2 returns inf.
    # If we see an inf assume FA2 and convert inf to -inf for consistency
    # and correctness. Inf generally doesn't make sense in this context outside
    # of undefined-behavior/FA2-case, so I think this a safe assumption.
    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    # Will reuse precomputed Exp values for scale factor computation.
    p_se = tl.exp(p_lse)
    s_se = tl.exp(s_lse)
    out_se = p_se + s_se

    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        tl.store(output_lse + head_idx * num_tokens + token_idx, out_lse)

    p_out = tl.load(
        prefix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
        + head_arange,
        mask=head_mask,
    )
    s_out = tl.load(
        suffix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
        + head_arange,
        mask=head_mask,
    )

    # NOTE(woosuk): Be careful with the numerical stability.
    # We should compute the scale first, and then multiply it with the output.
    # Do not multiply the output with tl.exp(p_lse) or tl.exp(s_lse) directly.
    p_scale = p_se / out_se
    s_scale = s_se / out_se
    out = p_out * p_scale + s_out * s_scale

    if USE_FP8:
        out = out * (1.0 / tl.load(output_scale))
        out = tl.clamp(out, FP8_MIN, FP8_MAX)
        out = out.to(output.dtype.element_ty)

    tl.store(
        output
        + token_idx * num_heads * output_head_stride
        + head_idx * output_head_stride
        + head_arange,
        out,
        mask=head_mask,
    )


@triton.jit
def merge_attn_states_group_fp8_kernel(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE] fp8
    output_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,
    suffix_output,
    suffix_lse,
    output_block_scale,  # fp32 SF tensor; layout via strides
    sf_token_stride,
    sf_group_stride,
    prefix_head_stride,
    output_head_stride,
    HEAD_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_HEAD: tl.constexpr,
    PADDED_GROUP_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
    prefill_tokens_with_context: tl.constexpr,
    USE_UE8M0: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    EPS: tl.constexpr = 1e-10,
):
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    prefix_mask = token_idx < prefill_tokens_with_context

    if prefix_mask:
        p_lse = tl.load(prefix_lse + head_idx * num_tokens + token_idx)
        s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)
        p_lse = float("-inf") if p_lse == float("inf") else p_lse
        s_lse = float("-inf") if s_lse == float("inf") else s_lse
        max_lse = tl.maximum(p_lse, s_lse)
        # All-(-inf) edge case: chunked prefill can produce p_lse=s_lse=-inf
        # for a token with no prefix hit; the normal path would yield NaN
        # (-inf - (-inf)). Fall back to emitting prefix_output (expected to
        # be zeros), matching the CUDA kernel.
        if max_lse == float("-inf"):
            if OUTPUT_LSE:
                tl.store(output_lse + head_idx * num_tokens + token_idx, max_lse)
            p_scale = 1.0
            s_scale = 0.0
        else:
            p_lse_n = p_lse - max_lse
            s_lse_n = s_lse - max_lse
            p_se = tl.exp(p_lse_n)
            s_se = tl.exp(s_lse_n)
            out_se = p_se + s_se
            if OUTPUT_LSE:
                out_lse = tl.log(out_se) + max_lse
                tl.store(output_lse + head_idx * num_tokens + token_idx, out_lse)
            p_scale = p_se / out_se
            s_scale = s_se / out_se
    else:
        if OUTPUT_LSE:
            s_lse_only = tl.load(suffix_lse + head_idx * num_tokens + token_idx)
            tl.store(output_lse + head_idx * num_tokens + token_idx, s_lse_only)
        p_scale = 0.0
        s_scale = 1.0

    group_arange = tl.arange(0, PADDED_GROUP_SIZE)
    group_mask = group_arange < GROUP_SIZE

    for g in tl.static_range(0, GROUPS_PER_HEAD):
        offsets = g * GROUP_SIZE + group_arange

        if prefix_mask:
            p_out = tl.load(
                prefix_output
                + token_idx * num_heads * prefix_head_stride
                + head_idx * prefix_head_stride
                + offsets,
                mask=group_mask,
                other=0.0,
            )
            s_out = tl.load(
                suffix_output
                + token_idx * num_heads * prefix_head_stride
                + head_idx * prefix_head_stride
                + offsets,
                mask=group_mask,
                other=0.0,
            )
            merged = p_out.to(tl.float32) * p_scale + s_out.to(tl.float32) * s_scale
        else:
            s_out = tl.load(
                suffix_output
                + token_idx * num_heads * prefix_head_stride
                + head_idx * prefix_head_stride
                + offsets,
                mask=group_mask,
                other=0.0,
            )
            merged = s_out.to(tl.float32)

        absmax = tl.maximum(tl.max(tl.abs(merged)), EPS)
        scale_raw = absmax * (1.0 / FP8_MAX)
        scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if USE_UE8M0 else scale_raw

        global_group_idx = head_idx * GROUPS_PER_HEAD + g
        tl.store(
            output_block_scale
            + token_idx * sf_token_stride
            + global_group_idx * sf_group_stride,
            scale,
        )

        q = tl.clamp(merged * (1.0 / scale), FP8_MIN, FP8_MAX)
        q = q.to(output.dtype.element_ty)
        tl.store(
            output
            + token_idx * num_heads * output_head_stride
            + head_idx * output_head_stride
            + offsets,
            q,
            mask=group_mask,
        )
