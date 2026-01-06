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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Follows transformers' apply_rotary_pos_emb exactly.
    Supports partial rotary where only the first rotary_dim of head_dim is rotated.

    Args:
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        cos: [batch, seq_len, rotary_dim]
        sin: [batch, seq_len, rotary_dim]
    """
    # unsqueeze_dim=1 to add head dimension: [batch, 1, seq_len, rotary_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Get the rotary dimension from cos/sin
    rotary_dim = cos.shape[-1]

    # Split into rotary and pass-through parts
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)

    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value tensors for Grouped Query Attention.

    Args:
        hidden_states: [batch, num_kv_heads, seq_len, head_dim]
        n_rep: Number of repetitions

    Returns:
        [batch, num_kv_heads * n_rep, seq_len, head_dim]
    """
    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


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


def _get_num_features_for_item(
    feature_attention_mask: torch.Tensor | None,
    chunk_counts: torch.Tensor | list[int] | None,
    item_idx: int,
    audio_embeds: list[torch.Tensor] | None,
    merge_factor: int,
    conv_params: list[tuple[int, int, int]],
) -> int:
    """Get number of features for a specific audio item."""
    if feature_attention_mask is not None:
        mask = _extract_mask_for_item(feature_attention_mask, chunk_counts, item_idx)
        audio_output_lengths = _get_audio_output_lengths_from_mask(
            mask, merge_factor, conv_params
        )
        return audio_output_lengths.sum().item()
    if audio_embeds is not None:
        return audio_embeds[item_idx].shape[0]
    raise ValueError("Either feature_attention_mask or audio_embeds must be provided")
