# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

from vllm.platforms import current_platform


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: Optional[torch.Tensor] = None,
) -> None:

    # NOTE(DefTruth): Currently, custom merge_attn_states CUDA kernel
    # is not support for FP8 dtype, fallback to use Triton kernel.
    def supported_dtypes(o: torch.Tensor) -> bool:
        return o.dtype in [torch.float32, torch.half, torch.bfloat16]

    # NOTE(DefTruth): Currently, custom merge_attn_states CUDA
    # kernel load/store 128b(16 bytes) per memory issue within
    # thread. Namely, the headsize(headdim) must be multiple of
    # pack_size (float32 -> 4, half/bfloat16 -> 8).
    def supported_headdim(o: torch.Tensor) -> bool:
        headdim = o.shape[2]  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        if o.dtype == torch.float32:
            return headdim % 4 == 0
        return headdim % 8 == 0

    if (current_platform.is_cuda() and supported_dtypes(output)
            and supported_headdim(output)):
        from vllm._custom_ops import merge_attn_states
        return merge_attn_states(output, prefix_output, prefix_lse,
                                 suffix_output, suffix_lse, output_lse)
    else:
        from vllm.attention.ops.triton_merge_attn_states import (
            merge_attn_states)
        return merge_attn_states(output, prefix_output, prefix_lse,
                                 suffix_output, suffix_lse, output_lse)


def merge_multi_attn_states(partials: torch.Tensor,
                            lse: torch.Tensor) -> torch.Tensor:
    """Merge attention partials across a parallel dimension using LSE.

    Args:
        partials: [tp, B, H_owned, D]
        lse: [tp, B, H_owned]

    Returns:
        merged: [B, H_owned, D]
    """
    assert partials.dim() == 4 and lse.dim() == 3, (
        f"partials shape {partials.shape}, lse shape {lse.shape}")
    tp, batch_size, heads_owned, dim = partials.shape
    # [tp, B, H_owned] -> [B, H_owned]
    max_lse, _ = torch.max(lse, dim=0)
    # Avoid -inf producing NaNs
    max_lse = torch.where(torch.isfinite(max_lse), max_lse,
                          torch.zeros_like(max_lse))

    # Compute exp-corrected weights and normalize across tp
    # [tp, B, H_owned]
    weights = torch.exp(lse - max_lse.unsqueeze(0))
    denom = torch.clamp(weights.sum(dim=0, keepdim=False), min=1e-20)
    weights = weights / denom

    # Apply weights to partials: broadcast weights to dim
    # [tp, B, H_owned, D]
    weighted = partials * weights.unsqueeze(-1)
    merged = weighted.sum(dim=0)
    return merged


def reduce_lse_over_tp(lse: torch.Tensor) -> torch.Tensor:
    """Reduce per-rank LSE across TP via stable log-sum-exp.

    Args:
        lse: [tp, B, H_owned]

    Returns:
        reduced_lse: [B, H_owned]
    """
    assert lse.dim() == 3
    tp_max, _ = torch.max(lse, dim=0)
    tp_max = torch.where(torch.isfinite(tp_max), tp_max,
                         torch.zeros_like(tp_max))
    weights = torch.exp(lse - tp_max.unsqueeze(0))
    denom = torch.clamp(weights.sum(dim=0, keepdim=False), min=1e-20)
    return torch.log(denom) + tp_max


def merge_multi_attn_states_with_lse(
        partials: torch.Tensor,
        lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused helper that returns merged outputs and reduced LSE.

    Args:
        partials: [tp, B, H_owned, D]
        lse: [tp, B, H_owned]

    Returns:
        (merged, reduced_lse):
            merged: [B, H_owned, D]
            reduced_lse: [B, H_owned]
    """
    merged = merge_multi_attn_states(partials, lse)
    reduced = reduce_lse_over_tp(lse)
    return merged, reduced
