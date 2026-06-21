# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared bitmask utilities for structured output with beam search."""

import torch

_bitmask_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def bitmask_to_token_ids(bitmask_row: torch.Tensor, vocab_size: int) -> list[int]:
    """Convert a packed int32 bitmask row to a list of allowed token IDs.

    The bitmask is produced by structured output backends (xgrammar,
    guidance, etc.) and encodes which tokens the grammar allows at
    the current FSM state.  Each bit at position *i* in the packed
    int32 array indicates whether token *i* is allowed.

    Args:
        bitmask_row: A 1-D tensor of packed int32 values representing
            the allowed-token bitmask for a single sequence.
        vocab_size: The model vocabulary size used to interpret the
            bitmask length.

    Returns:
        A list of integer token IDs that are allowed by the grammar.
    """
    if vocab_size not in _bitmask_cache:
        indices = torch.arange(vocab_size)
        _bitmask_cache[vocab_size] = (
            indices,
            indices >> 5,  # i // 32
            indices & 31,  # i % 32
        )
    indices, word_indices, bit_indices = _bitmask_cache[vocab_size]
    mask = ((bitmask_row[word_indices] >> bit_indices) & 1).bool()
    return indices[mask].tolist()
