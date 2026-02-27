# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Early ViT token pruning for vision encoders.

Enables pruning redundant visual tokens at an intermediate transformer layer
(e.g., layer 8 out of 32), so that remaining layers process only the retained
tokens. This provides significant speedup for video inputs where many tokens
are temporally redundant.

Pipeline:
  pixels -> [Conv3d] -> [K layers x ALL tokens] -> EVS prune
         -> [remaining layers x retained tokens] -> [merger] -> LLM

Three components:
  1. compute_intermediate_evs()  -- EVS dissimilarity on intermediate features
  2. prune_and_reindex()         -- gather retained tokens & recompute metadata
  3. get_per_video_grid_thw()    -- helper for per-video grid dimensions

The pruning operates at spatial-merge-unit granularity: each decision to
retain/prune affects a group of spatial_merge_size^2 consecutive ViT tokens
that will later be merged by the patch merger. This ensures the merger
receives well-formed spatial groups.
"""

import torch

from vllm.multimodal.evs import compute_retained_tokens_count


def get_per_video_grid_thw(
    grid_thw: list[list[int]],
) -> list[tuple[int, int, int]]:
    """
    Extract per-video grid dimensions from grid_thw.

    Args:
        grid_thw: List of [T, H, W] for each video/image in the batch.

    Returns:
        List of (T, H, W) tuples, one per video/image.
    """
    return [(int(t), int(h), int(w)) for t, h, w in grid_thw]


def _compute_per_video_cu_seqlens_boundaries(
    cu_seqlens: torch.Tensor,
    grid_thw_list: list[tuple[int, int, int]],
) -> list[tuple[int, int]]:
    """
    Compute the start and end indices into cu_seqlens for each video.

    cu_seqlens has one entry per frame across all videos. For a video with
    T frames, it contributes T entries to cu_seqlens (after the initial 0).

    Args:
        cu_seqlens: (num_total_frames + 1,) cumulative sequence lengths.
        grid_thw_list: List of (T, H, W) per video.

    Returns:
        List of (cu_start_idx, cu_end_idx) where cu_seqlens[cu_start_idx]
        is the token offset of the first frame and cu_seqlens[cu_end_idx]
        is the token offset after the last frame of each video.
    """
    boundaries = []
    frame_offset = 0
    for t, _h, _w in grid_thw_list:
        boundaries.append((frame_offset, frame_offset + t))
        frame_offset += t
    return boundaries


def compute_intermediate_evs(
    hidden_states: torch.Tensor,
    grid_thw_list: list[list[int]],
    spatial_merge_size: int,
    prune_rate: float,
) -> torch.Tensor:
    """
    Compute EVS retention mask on intermediate ViT features.

    Operates on the hidden states from an intermediate transformer layer.
    For each video in the batch, computes cosine dissimilarity between
    consecutive frames at the spatial-merge-unit level, then selects
    the top-scoring tokens to retain.

    The retention mask is at the spatial-merge-unit granularity (H' x W'
    where H' = H / spatial_merge_size, W' = W / spatial_merge_size).
    It is then expanded to cover all spatial_merge_size^2 ViT tokens in
    each unit.

    For single-frame inputs (images), all tokens are retained.

    Args:
        hidden_states: (total_tokens, 1, hidden_dim) intermediate ViT
            features across all videos in the batch.
        grid_thw_list: List of [T, H, W] grid dimensions per video/image.
        spatial_merge_size: The spatial merge factor (typically 2).
        prune_rate: Fraction of tokens to prune [0, 1). 0 means no pruning.

    Returns:
        retention_mask: (total_tokens,) boolean tensor. True = keep token.
    """
    if prune_rate <= 0:
        return torch.ones(
            hidden_states.shape[0], dtype=torch.bool,
            device=hidden_states.device,
        )

    merge_unit = spatial_merge_size ** 2
    total_tokens = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[2]
    device = hidden_states.device

    # Build retention mask at full ViT token granularity
    retention_mask = torch.ones(total_tokens, dtype=torch.bool, device=device)

    token_offset = 0
    for thw in grid_thw_list:
        t, h, w = int(thw[0]), int(thw[1]), int(thw[2])
        tokens_per_frame = h * w  # ViT tokens per frame (pre-merge)
        total_video_tokens = t * tokens_per_frame

        if t <= 1:
            # Single-frame: keep all tokens (images, or single-frame video)
            token_offset += total_video_tokens
            continue

        # Spatial-merge-level dimensions
        h_prime = h // spatial_merge_size
        w_prime = w // spatial_merge_size
        tokens_per_frame_merged = h_prime * w_prime  # merged tokens per frame

        # Extract this video's hidden states: (T * H * W, 1, D)
        video_hidden = hidden_states[
            token_offset:token_offset + total_video_tokens, 0, :
        ]  # (T * H * W, D)

        # Reshape to (T, H/m, m, W/m, m, D) -> (T, H', W', m^2, D)
        # Token ordering in ViT is spatial-merge order:
        # (h_div, w_div, merge_h, merge_w) flattened
        # So for each frame: (h_prime * w_prime * merge_unit) tokens
        # arranged as merge groups
        video_hidden = video_hidden.reshape(
            t, h_prime, w_prime, merge_unit, hidden_dim
        )

        # Average pool across the merge unit to get per-merged-token features
        # Shape: (T, H', W', D)
        merged_features = video_hidden.mean(dim=3)

        # Compute cosine dissimilarity between consecutive frames
        # similarity shape: (T-1, H', W')
        similarity = torch.nn.functional.cosine_similarity(
            merged_features[1:], merged_features[:-1], dim=-1
        )
        dissimilarity = 1.0 - similarity

        # First frame always retained: set high dissimilarity sentinel
        first_frame_sentinel = 255.0 * torch.ones(
            1, h_prime, w_prime,
            dtype=dissimilarity.dtype, device=device,
        )
        dissimilarity = torch.cat(
            [first_frame_sentinel, dissimilarity], dim=0
        )  # (T, H', W')

        # Flatten and select top-k
        dissimilarity_flat = dissimilarity.view(-1)
        retain_num = compute_retained_tokens_count(
            tokens_per_frame=tokens_per_frame_merged,
            num_frames=t,
            q=prune_rate,
        )

        # Use topk for efficiency (avoid full argsort)
        _, topk_indices = torch.topk(
            dissimilarity_flat, retain_num, sorted=False
        )

        # Build merged-level retention mask
        merged_mask = torch.zeros(
            t * tokens_per_frame_merged,
            dtype=torch.bool, device=device,
        )
        merged_mask[topk_indices] = True

        # Expand to ViT-token-level mask: each merged position maps to
        # merge_unit consecutive tokens
        # merged_mask: (T * H' * W') -> expand to (T * H' * W' * merge_unit)
        expanded_mask = merged_mask.unsqueeze(-1).expand(
            -1, merge_unit
        ).contiguous().view(-1)

        # Write into global retention mask
        retention_mask[token_offset:token_offset + total_video_tokens] = (
            expanded_mask
        )

        token_offset += total_video_tokens

    return retention_mask


def prune_and_reindex(
    hidden_states: torch.Tensor,
    retention_mask: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb_cos: torch.Tensor,
    rotary_pos_emb_sin: torch.Tensor,
    grid_thw_list: list[list[int]],
    spatial_merge_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Remove pruned tokens and recompute attention metadata.

    After applying the retention mask, this function:
    1. Gathers retained hidden states
    2. Gathers retained rotary embeddings (cos, sin)
    3. Recomputes cu_seqlens for the pruned per-frame sequence lengths
    4. Computes the new max_seqlen

    The cu_seqlens in the ViT tracks per-frame boundaries (each frame is
    a separate attention window in flash attention). After pruning, each
    frame may have a different number of retained tokens.

    Args:
        hidden_states: (total_tokens, 1, hidden_dim) ViT hidden states.
        retention_mask: (total_tokens,) boolean mask from
            compute_intermediate_evs().
        cu_seqlens: (num_frames + 1,) cumulative sequence lengths.
        rotary_pos_emb_cos: (total_tokens, rotary_dim) cos embeddings.
        rotary_pos_emb_sin: (total_tokens, rotary_dim) sin embeddings.
        grid_thw_list: List of [T, H, W] per video/image.
        spatial_merge_size: The spatial merge factor.

    Returns:
        Tuple of:
        - hidden_states: (retained_tokens, 1, hidden_dim)
        - cu_seqlens: (num_frames + 1,) new cumulative sequence lengths
        - rotary_pos_emb_cos: (retained_tokens, rotary_dim)
        - rotary_pos_emb_sin: (retained_tokens, rotary_dim)
        - max_seqlen: scalar tensor with max sequence length after pruning
    """
    # Gather retained tokens
    retained_indices = retention_mask.nonzero(as_tuple=True)[0]
    hidden_states = hidden_states[retained_indices]
    rotary_pos_emb_cos = rotary_pos_emb_cos[retained_indices]
    rotary_pos_emb_sin = rotary_pos_emb_sin[retained_indices]

    # Recompute cu_seqlens: count retained tokens per frame
    # cu_seqlens defines frame boundaries: frame i has tokens from
    # cu_seqlens[i] to cu_seqlens[i+1].
    num_frames = cu_seqlens.shape[0] - 1
    new_frame_lengths = torch.zeros(
        num_frames, dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )

    for i in range(num_frames):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        # Count how many tokens in this frame are retained
        frame_mask = retention_mask[start:end]
        new_frame_lengths[i] = frame_mask.sum()

    new_cu_seqlens = torch.zeros(
        num_frames + 1, dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    torch.cumsum(new_frame_lengths, dim=0, out=new_cu_seqlens[1:])

    max_seqlen = new_frame_lengths.max()

    return (
        hidden_states,
        new_cu_seqlens,
        rotary_pos_emb_cos,
        rotary_pos_emb_sin,
        max_seqlen,
    )
