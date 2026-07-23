# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

float8_info = torch.finfo(current_platform.fp8_dtype())


def mask_empty_context(
    lse: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    context_start_loc: torch.Tensor,
) -> None:
    """Neutralize context chunks that cover no keys before merging.

    A prefill query whose context chunk is empty attended to no keys, so its
    partial attention is undefined: the backend leaves the output rows as
    uninitialized scratch (which may hold NaN/Inf) even when it reports an LSE
    of -inf. Sanitize both here so ``merge_attn_states`` can stay generic:
    force the LSE to -inf (zero softmax weight) and zero the undefined output
    rows (so a zero weight cannot combine with NaN/Inf). Emptiness is derived
    from the context offsets, not from the -inf LSE, so no merge kernel has to
    reason about undefined partials.

    Args:
        lse: Chunk log-sum-exp, shape [num_heads, num_tokens].
        output: Chunk attention output, shape [num_tokens, num_heads, ...].
        query_start_loc: Prefill query cumulative offsets, shape [num_reqs + 1].
        context_start_loc: Chunk context cumulative offsets,
            shape [num_reqs + 1]; an empty chunk has a zero-length span.
    """
    num_heads, num_tokens = lse.shape
    num_reqs = query_start_loc.shape[0] - 1
    block_size = 128
    # Reserve the worst-case number of request-local blocks.
    num_query_blocks = num_tokens // block_size + num_reqs
    is_empty = torch.zeros(num_tokens, dtype=torch.bool, device=lse.device)
    mask_empty_context_kernel[(num_query_blocks,)](
        lse,
        is_empty,
        query_start_loc,
        context_start_loc,
        lse.stride(0),
        lse.stride(1),
        num_reqs,
        NUM_HEADS=num_heads,
        BLOCK_SIZE=block_size,
        BLOCK_HEADS=8,
        num_warps=8,
    )
    output.masked_fill_(is_empty[:, None, None], 0.0)


@triton.jit
def mask_empty_context_kernel(
    lse,
    is_empty,
    query_start_loc,
    context_start_loc,
    lse_head_stride,
    lse_token_stride,
    num_reqs,
    NUM_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    query_block_idx = tl.program_id(0)

    lanes = tl.arange(0, 32)
    chunk_start = 0
    req_idx = 0
    req_idx_found = False
    while (chunk_start < num_reqs) & (not req_idx_found):
        req_offsets = chunk_start + lanes
        req_mask = req_offsets < num_reqs
        query_starts = tl.load(query_start_loc + req_offsets, mask=req_mask)
        # Assume the worst-case number of blocks for each request.
        req_block_starts = query_starts // BLOCK_SIZE + req_offsets
        matched_idx = tl.sum(
            (req_mask & (req_block_starts <= query_block_idx)).to(tl.int32)
        )
        # matched_idx == 32 means the match is past this warp chunk.
        req_idx = chunk_start + matched_idx - 1
        req_idx_found = matched_idx < 32
        chunk_start += 32

    query_start = tl.load(query_start_loc + req_idx)
    query_end = tl.load(query_start_loc + req_idx + 1)
    query_len = query_end - query_start
    req_first_block = query_start // BLOCK_SIZE + req_idx
    block_in_req = query_block_idx - req_first_block
    token_offset = block_in_req * BLOCK_SIZE
    if token_offset >= query_len:
        return

    context_start = tl.load(context_start_loc + req_idx)
    context_end = tl.load(context_start_loc + req_idx + 1)
    if context_start != context_end:
        return

    token_offsets = token_offset + tl.arange(0, BLOCK_SIZE)
    token_indices = query_start + token_offsets
    token_lse_offsets = token_indices * lse_token_stride
    valid_tokens = token_offsets < query_len
    tl.store(is_empty + token_indices, True, mask=valid_tokens)
    head_offsets = tl.arange(0, BLOCK_HEADS)
    for head_start in range(0, NUM_HEADS, BLOCK_HEADS):
        head_indices = head_start + head_offsets
        lse_ptrs = (
            lse + head_indices[:, None] * lse_head_stride + token_lse_offsets[None, :]
        )
        valid_heads = head_indices < NUM_HEADS
        tl.store(
            lse_ptrs,
            float("-inf"),
            mask=valid_heads[:, None] & valid_tokens[None, :],
        )


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
        # Both sides empty (max_lse == -inf) => undefined merge; keep -inf so
        # downstream merges continue to treat the token as empty.
        out_lse = tl.where(max_lse == float("-inf"), float("-inf"), out_lse)
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
    # If both sides are empty (max_lse == -inf) the scales are 0/0 = NaN; emit
    # zeros rather than NaN. Callers with empty chunks (see mask_empty_context)
    # zero those inputs, so this only guards the fully-undefined corner.
    out = tl.where(max_lse == float("-inf"), 0.0, out)

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
