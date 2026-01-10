# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Lightweight utility functions for Gemma3n audio processing.

This module is separate from gemma3n_mm.py to avoid heavy CUDA dependencies,
making it testable without a full vLLM build.
"""

import torch


def adjust_audio_features_to_expected_length(
    audio_features: torch.Tensor,
    expected_tokens: int,
    audio_padding_embs: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Adjust audio features to expected token length via padding or truncation.

    The Gemma3nProcessor expects all audio will be ~30s in length and inserts
    a fixed number of audio soft tokens into the text. However, the audio
    preprocessing and encoder do not guarantee they will produce exactly that
    many soft tokens; they may produce fewer tokens (for shorter audio) or more
    tokens (for longer audio or due to BOA/EOA special tokens).

    This function handles both cases:
    - If fewer tokens: pad with the provided padding embeddings
    - If more tokens: truncate to the expected count

    Args:
        audio_features: Audio embeddings tensor of shape
            (batch_size, seq_len, embed_dim)
        expected_tokens: The expected number of audio tokens (e.g., 188)
        audio_padding_embs: Padding embeddings tensor of shape (1, 1, embed_dim)

    Returns:
        Tuple of:
        - adjusted_features: Audio features adjusted to expected_tokens length
        - tokens_truncated: Number of tokens truncated (0 if padding was applied)
    """
    audio_batch_size, audio_seq_len, audio_embed_dim = audio_features.shape
    tokens_truncated = 0

    if audio_seq_len < expected_tokens:
        # Pad to expected length with padding embeddings
        extra_padding_tokens = expected_tokens - audio_seq_len
        extra_padding_features = audio_padding_embs.expand(
            audio_batch_size, extra_padding_tokens, audio_embed_dim
        )
        audio_features = torch.cat((audio_features, extra_padding_features), dim=1)
    elif audio_seq_len > expected_tokens:
        # Truncate to expected length (audio encoder produced more tokens
        # than expected, e.g., due to longer audio or placeholder mismatch)
        tokens_truncated = audio_seq_len - expected_tokens
        audio_features = audio_features[:, :expected_tokens, :]

    return audio_features, tokens_truncated
