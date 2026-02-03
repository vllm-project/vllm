# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import cast

import torch
import torch.nn as nn

DEFAULT_MAX_AUDIO_LEN_S = 655
DEFAULT_MERGE_FACTOR = 4
# Default convolution parameters: (padding, kernel_size, stride)
# These correspond to the two conv layers in GlmAsrEncoder
DEFAULT_CONV_PARAMS = [(1, 3, 1), (1, 3, 2)]


def _calculate_conv_output_length(
    input_length: torch.Tensor, padding: int, kernel_size: int, stride: int
) -> torch.Tensor:
    """Calculate Conv1d output length using standard formula."""
    # Standard formula: floor((input + 2*padding - kernel_size) / stride) + 1
    return (input_length + 2 * padding - kernel_size) // stride + 1


def _as_list_chunk_counts(
    chunk_counts: torch.Tensor | list[int] | list[torch.Tensor],
) -> list[int]:
    if isinstance(chunk_counts, torch.Tensor):
        return chunk_counts.tolist()
    if chunk_counts and isinstance(chunk_counts[0], torch.Tensor):
        tensor_counts = cast(list[torch.Tensor], chunk_counts)
        return [int(c.item()) for c in tensor_counts]
    return [int(c) for c in chunk_counts]


def _normalize_chunk_counts(
    chunk_counts: torch.Tensor | list[int] | list[torch.Tensor] | None,
    num_chunks: int,
) -> list[int]:
    if chunk_counts is None:
        return [1] * num_chunks
    return _as_list_chunk_counts(chunk_counts)


def _get_audio_output_lengths_from_lengths(
    audio_lengths: torch.Tensor,
    merge_factor: int,
    conv_params: list[tuple[int, int, int]],
) -> torch.Tensor:
    for padding, kernel_size, stride in conv_params:
        audio_lengths = _calculate_conv_output_length(
            audio_lengths, padding, kernel_size, stride
        )
    return (audio_lengths - merge_factor) // merge_factor + 1


def _get_audio_output_lengths_from_mask(
    mask: torch.Tensor,
    merge_factor: int,
    conv_params: list[tuple[int, int, int]],
) -> torch.Tensor:
    audio_lengths = mask.sum(-1)
    return _get_audio_output_lengths_from_lengths(
        audio_lengths, merge_factor, conv_params
    )


def _get_audio_output_lengths_for_tower(
    audio_tower: nn.Module,
    audio_lengths: torch.Tensor,
    merge_factor: int,
    conv_params: list[tuple[int, int, int]],
) -> torch.Tensor:
    """
    Calculate the output lengths after audio processing.

    The output length accounts for:
    1. Convolution layers (downsampling)
    2. Merge factor (further downsampling during projection)

    Args:
        audio_tower: The audio encoder module
        audio_lengths: Input feature lengths [batch_size]
        merge_factor: Factor for merging adjacent features
        conv_params: List of (padding, kernel_size, stride) for each conv layer

    Returns:
        Output lengths after all processing [batch_size]
    """
    # First, calculate the output length after convolutions
    if hasattr(audio_tower, "_get_feat_extract_output_lengths"):
        _, conv_output_lengths = audio_tower._get_feat_extract_output_lengths(
            audio_lengths
        )
    else:
        conv_output_lengths = audio_lengths
        for padding, kernel_size, stride in conv_params:
            conv_output_lengths = _calculate_conv_output_length(
                conv_output_lengths, padding, kernel_size, stride
            )

    # Then, apply merge_factor to get final output length
    # Formula: (conv_output_lengths - merge_factor) // merge_factor + 1
    return (conv_output_lengths - merge_factor) // merge_factor + 1


def _flatten_audio_features_by_length(
    audio_features: torch.Tensor,
    audio_output_lengths: torch.Tensor,
) -> torch.Tensor:
    num_chunks, max_audio_tokens, embed_dim = audio_features.shape
    audio_output_lengths = audio_output_lengths.unsqueeze(1)
    audio_features_mask = (
        torch.arange(max_audio_tokens)
        .expand(num_chunks, max_audio_tokens)
        .to(audio_output_lengths.device)
        < audio_output_lengths
    )
    return audio_features[audio_features_mask].view(-1, embed_dim)


def _group_audio_embeddings(
    chunk_embeddings: Sequence[torch.Tensor],
    chunk_counts: Sequence[int],
) -> tuple[torch.Tensor, ...]:
    grouped_embeddings = []
    current_idx = 0
    for count in chunk_counts:
        audio_chunks = chunk_embeddings[current_idx : current_idx + count]
        grouped_embeddings.append(torch.cat(audio_chunks, dim=0))
        current_idx += count
    return tuple(grouped_embeddings)


def _normalize_to_tensor(mask: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """Convert mask to tensor, handling both list and tensor formats."""
    if isinstance(mask, list):
        return (
            torch.stack(mask)
            if mask and isinstance(mask[0], torch.Tensor)
            else torch.tensor(mask)
        )
    return mask


def _extract_mask_for_item(
    feature_attention_mask: torch.Tensor | list[torch.Tensor],
    chunk_counts: torch.Tensor | list[int] | None,
    item_idx: int,
) -> torch.Tensor:
    """Extract attention mask for a specific audio item."""
    if chunk_counts is None:
        # Single item per audio
        mask = feature_attention_mask[item_idx]
        if isinstance(feature_attention_mask, torch.Tensor):
            return mask.unsqueeze(0)
        return _normalize_to_tensor(mask)

    # Multiple chunks per audio: calculate slice indices
    counts = _as_list_chunk_counts(chunk_counts)
    start_idx = sum(counts[:item_idx])
    end_idx = start_idx + counts[item_idx]

    # Extract slice
    if isinstance(feature_attention_mask, torch.Tensor):
        return feature_attention_mask[start_idx:end_idx]
    mask_slice = feature_attention_mask[start_idx:end_idx]
    return _normalize_to_tensor(mask_slice)
