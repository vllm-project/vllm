# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

float8_info = torch.finfo(current_platform.fp8_dtype())

# waves_per_eu is only recognized by the Triton AMD backend; passing it on
# other backends raises KeyError at launch.
_ROCM_KERNEL_KWARGS = {"waves_per_eu": 0} if current_platform.is_rocm() else {}


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
    prefix_head_stride = prefix_output.stride(1)
    output_head_stride = output.stride(1)
    prefix_lse_head_stride = prefix_lse.stride(0)
    prefix_lse_token_stride = prefix_lse.stride(1)
    suffix_lse_head_stride = suffix_lse.stride(0)
    suffix_lse_token_stride = suffix_lse.stride(1)
    output_lse_head_stride = output_lse.stride(0) if output_lse is not None else 0
    output_lse_token_stride = output_lse.stride(1) if output_lse is not None else 0

    if prefill_tokens_with_context is None:
        prefill_tokens_with_context = num_tokens

    # Target a fixed number of elements per program: enough per lane to
    # vectorize the loads, few enough to keep the grid wide. Never tile wider
    # than the batch -- a padded tile costs address math and masked lanes that
    # move no data, which dominates at decode-sized token counts.
    BLOCK_TOKENS = max(1, min(128, 2048 // padded_head_size))
    BLOCK_TOKENS = min(BLOCK_TOKENS, triton.next_power_of_2(num_tokens))

    grid = (triton.cdiv(num_tokens, BLOCK_TOKENS), num_query_heads)

    # Stride along token dim for prefix/suffix and output tensors
    prefix_token_stride = num_query_heads * prefix_head_stride
    output_token_stride = num_query_heads * output_head_stride

    merge_attn_states_kernel[grid](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        prefix_head_stride,
        output_head_stride,
        prefix_token_stride,
        output_token_stride,
        prefix_lse_head_stride,
        prefix_lse_token_stride,
        suffix_lse_head_stride,
        suffix_lse_token_stride,
        output_lse_head_stride,
        output_lse_token_stride,
        output_scale,
        prefill_tokens_with_context,
        num_tokens,
        head_size,
        padded_head_size,
        output_lse is not None,
        output_scale is not None,
        BLOCK_TOKENS,
        num_warps=2,
        num_stages=1,
        **_ROCM_KERNEL_KWARGS,
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
    prefix_token_stride,
    output_token_stride,
    prefix_lse_head_stride,
    prefix_lse_token_stride,
    suffix_lse_head_stride,
    suffix_lse_token_stride,
    output_lse_head_stride,
    output_lse_token_stride,
    output_scale,  # scale tensor or None
    prefill_tokens_with_context,
    num_tokens,
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
    USE_FP8: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    pid_token = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Token offsets for this tile [BLOCK_TOKENS]
    token_offsets = pid_token * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    token_mask = token_offsets < num_tokens

    # Head dimension offsets [PADDED_HEAD_SIZE]
    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE

    # Which tokens need merge vs copy
    prefix_token_mask = token_offsets < prefill_tokens_with_context  # [BLOCK_TOKENS]
    prefix_load_mask = token_mask & prefix_token_mask

    # Load LSE values for all tokens in the tile [BLOCK_TOKENS]
    p_lse = tl.load(
        prefix_lse
        + head_idx * prefix_lse_head_stride
        + token_offsets * prefix_lse_token_stride,
        mask=prefix_load_mask,
        other=float("-inf"),
    )
    s_lse = tl.load(
        suffix_lse
        + head_idx * suffix_lse_head_stride
        + token_offsets * suffix_lse_token_stride,
        mask=token_mask,
        other=float("-inf"),
    )

    # Save original s_lse for the copy path (before inf->-inf conversion)
    s_lse_orig = s_lse

    # FA2 compatibility: convert +inf to -inf
    p_lse = tl.where(p_lse == float("inf"), float("-inf"), p_lse)
    s_lse = tl.where(s_lse == float("inf"), float("-inf"), s_lse)

    # Compute merge scales [BLOCK_TOKENS]
    max_lse = tl.maximum(p_lse, s_lse)
    p_lse_shifted = p_lse - max_lse
    s_lse_shifted = s_lse - max_lse
    p_se = tl.exp(p_lse_shifted)
    s_se = tl.exp(s_lse_shifted)
    out_se = p_se + s_se
    p_scale = p_se / out_se  # [BLOCK_TOKENS]
    s_scale = s_se / out_se  # [BLOCK_TOKENS]

    max_lse_is_neginf = max_lse == float("-inf")

    # Output LSE
    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        out_lse = tl.where(max_lse_is_neginf, float("-inf"), out_lse)
        # For non-context tokens, output_lse = original s_lse
        out_lse = tl.where(prefix_token_mask, out_lse, s_lse_orig)

        tl.store(
            output_lse
            + head_idx * output_lse_head_stride
            + token_offsets * output_lse_token_stride,
            out_lse,
            mask=token_mask,
        )

    # Load FP8 scale if needed
    if USE_FP8:
        inv_scale = 1.0 / tl.load(output_scale)

    # Build 2D pointer offsets [BLOCK_TOKENS, PADDED_HEAD_SIZE]
    p_ptrs = (
        prefix_output
        + token_offsets[:, None] * prefix_token_stride
        + head_idx * prefix_head_stride
        + head_arange[None, :]
    )
    s_ptrs = (
        suffix_output
        + token_offsets[:, None] * prefix_token_stride
        + head_idx * prefix_head_stride
        + head_arange[None, :]
    )
    o_ptrs = (
        output
        + token_offsets[:, None] * output_token_stride
        + head_idx * output_head_stride
        + head_arange[None, :]
    )

    # 2D mask [BLOCK_TOKENS, PADDED_HEAD_SIZE]
    mask_2d = token_mask[:, None] & head_mask[None, :]
    prefix_mask_2d = mask_2d & prefix_token_mask[:, None]

    # Load 2D tiles [BLOCK_TOKENS, PADDED_HEAD_SIZE]
    p_out = tl.load(p_ptrs, mask=prefix_mask_2d, other=0.0)
    s_out = tl.load(s_ptrs, mask=mask_2d, other=0.0)

    # Broadcast scales from [BLOCK_TOKENS] to [BLOCK_TOKENS, PADDED_HEAD_SIZE]
    p_scale_2d = p_scale[:, None]
    s_scale_2d = s_scale[:, None]

    # Merge: out = p_out * p_scale + s_out * s_scale
    merged = p_out * p_scale_2d + s_out * s_scale_2d
    # Handle both-empty case
    merged = tl.where(max_lse_is_neginf[:, None], 0.0, merged)

    # For non-context tokens, just copy suffix
    result = tl.where(prefix_token_mask[:, None], merged, s_out)

    if USE_FP8:
        result = result * inv_scale
        result = tl.clamp(result, FP8_MIN, FP8_MAX)
        result = result.to(output.dtype.element_ty)

    tl.store(o_ptrs, result, mask=mask_2d)
