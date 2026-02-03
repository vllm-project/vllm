# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import typing

import torch


def compute_retained_tokens_count(
    tokens_per_frame: int, num_frames: int, q: float
) -> int:
    """
    Compute the number of retained tokens for a given video.
    Method ensures that we retain all the tokens from the first frame
    regardless of the pruning rate.

    Args:
        tokens_per_frame: The number of tokens per frame.
        num_frames: The total number of frames.
        q: The pruning rate.

    Returns:
        The number of retained tokens.
    """
    total_tokens = tokens_per_frame * num_frames
    evs_num_tokens = int(total_tokens * (1 - q))
    min_num_tokens = tokens_per_frame
    return max(min_num_tokens, evs_num_tokens)


def compute_retention_mask(
    video_embeds: torch.Tensor,
    video_size_thw: torch.LongTensor | tuple[int, int, int],
    spatial_merge_size: int,
    q: float,
) -> torch.Tensor:
    """
    Computes the retention mask for input video embeddings.

    Args:
        video_embeds (`torch.Tensor`): The input video embeddings
            of shape `(T * H * W // spatial_merge_size ^ 2, hidden_size)`
        video_size_thw (`torch.LongTensor` of shape `(3)`):
            The temporal, height and width of video.
        spatial_merge_size: Size reduction for rows & cols dimensions.
        q: (`float`): Pruning rate factor [0,1)

    Returns:
        `torch.Tensor`: The retention mask for the video embeddings of
            `(T * H * W // spatial_merge_size ^ 2)` shape.
    """
    T, H, W = map(int, video_size_thw)

    # Use reshape instead of einops to avoid graph breaks
    video_embeds = video_embeds.reshape(
        T,
        H // spatial_merge_size,
        W // spatial_merge_size,
        video_embeds.size(-1),
    )
    tokens_per_frame = (H // spatial_merge_size) * (W // spatial_merge_size)
    # Core EVS
    similarity = torch.nn.functional.cosine_similarity(
        video_embeds[1:, ...], video_embeds[:-1, ...], dim=-1
    )
    dissimilarity = 1 - similarity

    # Always ensure we include all tokens from the first frame
    dissimilarity = torch.cat(
        [255 * torch.ones_like(video_embeds[:1, :, :, 0]), dissimilarity], dim=0
    )

    dissimilarity_flat = dissimilarity.view(-1)
    order = torch.argsort(dissimilarity_flat, dim=-1, descending=True, stable=True)
    retain_num_tokens = compute_retained_tokens_count(
        tokens_per_frame=tokens_per_frame, num_frames=T, q=q
    )
    topk_indices = order[:retain_num_tokens]

    retention_mask = torch.zeros_like(dissimilarity_flat, dtype=torch.bool)
    retention_mask[topk_indices] = True
    retention_mask = retention_mask.reshape(dissimilarity.size())

    mask = retention_mask.view(-1)  # "T H W -> (T H W)"
    return mask


def compute_mrope_for_media(
    video_size_thw: torch.LongTensor,
    spatial_merge_size: int,
    tokens_per_second: float = 1.0,
    video_second_per_grid: float = 1.0,
) -> torch.Tensor:
    """
    Computes the mrope for video embeddings based on the grid dimensions.
    Computed mrope positions match original qwen 2.5 implementation,
    but positions are built for media being the first element in sequence.

    Args:
        video_size_thw: Media size (num frames, rows, cols)
        spatial_merge_size: Size reduction for rows & cols dimensions.
        tokens_per_second: Number of tokens per second.
        video_second_per_grid: Number of seconds per video.

    Returns:
        Tensor of shape `(T * H * W, 4)` where last dimension
        represents mrope positions [0:3), while the last channel
        contains value of llm_grid_w repeated for all positions.
    """
    llm_grid_t = video_size_thw[0]
    llm_grid_h = video_size_thw[1] // spatial_merge_size
    llm_grid_w = video_size_thw[2] // spatial_merge_size

    t_index = (
        (
            torch.arange(llm_grid_t)
            .view(-1, 1)
            .expand(-1, llm_grid_h * llm_grid_w)
            .mul(tokens_per_second * video_second_per_grid)
        )
        .long()
        .flatten()
    )
    h_index = (
        torch.arange(llm_grid_h)
        .view(1, -1, 1)
        .expand(llm_grid_t, -1, llm_grid_w)
        .flatten()
    )
    w_index = (
        torch.arange(llm_grid_w)
        .view(1, 1, -1)
        .expand(llm_grid_t, llm_grid_h, -1)
        .flatten()
    )
    llm_grid_w = (
        torch.tensor([llm_grid_w])
        .view(1, 1, 1)
        .expand(llm_grid_t, llm_grid_h, llm_grid_w)
        .flatten()
    )

    positions = torch.stack([t_index, h_index, w_index, llm_grid_w], dim=1)
    return positions


def recompute_mrope_positions(
    input_ids: torch.LongTensor,
    multimodal_positions: list[torch.Tensor],
    mrope_positions: torch.LongTensor,
    num_computed_tokens: int,
    vision_start_token_id: int,
    image_token_id: int,
    video_token_id: int,
) -> tuple[torch.LongTensor, int]:
    """
    Update part of input mrope positions.
    Original mrope_positions are computed incorrectly, so once we prune media
    tokens we should reflect this in the mrope positions for the LLM.

    This method supports chunked prefill approach where
    multimodal_embeddings are passed to LLM in chunks, so input
    multimodal_embeddings may contain zero, some or even some part of all
    multimodal_embeddings for a given prompt.

    Each multimodal_positions has 4 extra channels
    (First 3 channels corresponds to original 3 mrope positions, last channel
    is the maximum width of the media repeated). Provided multimodal_positions
    do not reflect location of media position in sequence - they are computed
    like the media is in the 0-th position in the sequence.

    Method works as follows: it recomputes mrope_positions starting from the
    `num_computed_tokens` for `total_len_of_multimodal_embeddings` and then
    shifts all text tokens that goes after total_len_of_multimodal_embeddings.

    It also handles case when multimodal_embeddings is partial
    (e.g. one media is split into two prefill stages)

    Args:
        input_ids: (N,) All input tokens of the prompt (entire sequence).
        multimodal_positions: List of mrope positions for each media.
        mrope_positions: Existing mrope positions (4, N) for entire sequence.
        num_computed_tokens: A number of computed tokens so far.
        vision_start_token_id: Token indicating start of vision media.
        image_token_id: Image token id
        video_token_id: Video token id

    Returns:
        Tuple of (mrope_positions, mrope_position_delta).
    """

    # Tensors
    positions: torch.LongTensor = typing.cast(
        torch.LongTensor, mrope_positions.clone()
    )  # (3, N)
    N = input_ids.numel()

    image_mask = input_ids.eq(image_token_id)
    video_mask = input_ids.eq(video_token_id)
    media_mask = image_mask | video_mask
    text_mask = ~media_mask

    # Early exit: no media in this chunk
    if len(multimodal_positions) == 0:
        delta = int((positions.max().item() + 1) - N) if positions.numel() else -N
        return positions, delta

    total_mm_tokens = torch.count_nonzero(media_mask)
    seen_mm_tokens = torch.count_nonzero(media_mask[:num_computed_tokens])

    # Early exit: we've updated positions for all media tokens
    # (and consequently - for all remaining text tokens)
    if seen_mm_tokens == total_mm_tokens:
        delta = int((positions.max().item() + 1) - N) if positions.numel() else -N
        return positions, delta

    vision_start_indices = (input_ids == vision_start_token_id).nonzero(as_tuple=True)[
        0
    ]

    for mm_pos in multimodal_positions:
        # Each mm_pos can be a complete embedding for single media
        # or it can be a part of a single media (due to chunked prefill)

        # Cases to cover
        # - Current prefill chunk has no vision start indexes at all
        # - Vision start token appeared in previous prefill round
        # - Regular case
        seen_vision_start_indices = vision_start_indices[
            vision_start_indices < num_computed_tokens
        ]

        if len(seen_vision_start_indices):
            # If we have encountered some vision start indexes,
            # then we should check the condition:
            # | --- prefill 1 ------| ---- prefill 2 ----- |
            # | TTTTTTTTTSVVVVVVVVVV|VVVVVVTTTTTTTTTTTTTTTT|
            last_vision_start_token = seen_vision_start_indices[-1]
            seem_mm_tokens_before_last_vision_start = torch.count_nonzero(
                media_mask[:last_vision_start_token]
            )
            in_the_middle_of_media = (
                seen_mm_tokens > seem_mm_tokens_before_last_vision_start
            )

            if in_the_middle_of_media:
                mm_embeddings_seen = (
                    seen_mm_tokens - seem_mm_tokens_before_last_vision_start
                )
                global_mm_start = last_vision_start_token
            else:
                # We have completed previous mm_embedding part and
                # ready to start a new one
                next_vision_start_token = vision_start_indices[
                    vision_start_indices >= num_computed_tokens
                ][0]
                mm_embeddings_seen = 0
                global_mm_start = next_vision_start_token

        else:
            # If there were no vision start indexes so far,
            # let's find first vision start index
            next_vision_start_token = vision_start_indices[
                vision_start_indices >= num_computed_tokens
            ][0]

            mm_embeddings_seen = 0
            global_mm_start = next_vision_start_token

        # Offset right after vision_start_token
        base = positions[-1, global_mm_start] + 1
        local_start = global_mm_start + 1 + mm_embeddings_seen
        local_end = local_start + mm_pos.shape[1]
        positions[:, local_start:local_end] = mm_pos[0:3] + base

        # mm_pos[3, 0] is the max width of the media
        offset = mm_pos[3, 0] + base

        text_pos_sum = torch.cumsum(text_mask[local_end:].long(), dim=0)

        positions[:, local_end:N] = text_pos_sum + offset - 1

        # Include distance to the next vision start token
        num_computed_tokens += mm_pos.shape[1]

    mrope_positions_delta = (positions.max() + 1 - N).item()
    return positions, mrope_positions_delta
