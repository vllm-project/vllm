# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selection and validation of the Qwen4Exp PLE embedding mode.

`ple_embedding_mode` is an optional field of the text `config.json`:

* `"ngram"` (default, also used when the field is absent): hashed n-gram
  embedding table, the original behaviour.
* `"per_token"`: a replicated `[vocab_size, ple_embed_dim]` table indexed by
  the current token id (no hashing, no context from earlier tokens).
"""

from enum import StrEnum
from typing import Any


class PleEmbeddingMode(StrEnum):
    NGRAM = "ngram"
    PER_TOKEN = "per_token"


def validate_ple_embedding_config(config: Any) -> PleEmbeddingMode:
    """Validate the PLE embedding fields of a text config.

    Args:
        config: Text config of the model.

    Returns:
        The validated mode.

    Raises:
        ValueError: If the mode or the fields it depends on are invalid.

    """
    value = getattr(config, "ple_embedding_mode", None)
    try:
        mode = PleEmbeddingMode.NGRAM if value is None else PleEmbeddingMode(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid ple_embedding_mode {value!r}; expected one of "
            f"{', '.join(PleEmbeddingMode)}"
        ) from exc
    if mode == PleEmbeddingMode.NGRAM or not config.ple_layer_ids:
        return mode
    embed_dim = int(config.ple_embed_dim)
    if embed_dim <= 0:
        raise ValueError(f"ple_embed_dim must be positive, got {embed_dim}")
    vocab_size = int(config.vocab_size)
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if len(set(config.ple_layer_ids)) != 1:
        raise ValueError(
            "ple_embedding_mode='per_token' supports exactly one PLE layer, "
            f"got ple_layer_ids={list(config.ple_layer_ids)}"
        )
    return mode
