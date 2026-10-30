# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selection and validation of the Qwen4Exp PLE embedding mode.

``ple_embedding_mode`` is an optional field of the text ``config.json``:

* ``"ngram"`` (default, also used when the field is absent): hashed n-gram
  embedding table, the original behaviour.
* ``"per_token"``: a replicated ``[vocab_size, ple_embed_dim]`` table indexed by
  the current token id (no hashing, no context from earlier tokens).
* ``"zero"``: the PLE embedding is all zeros; no table is allocated or loaded.
"""

from typing import Any

PLE_EMBEDDING_MODE_NGRAM = "ngram"
PLE_EMBEDDING_MODE_PER_TOKEN = "per_token"
PLE_EMBEDDING_MODE_ZERO = "zero"
PLE_EMBEDDING_MODES = (
    PLE_EMBEDDING_MODE_NGRAM,
    PLE_EMBEDDING_MODE_PER_TOKEN,
    PLE_EMBEDDING_MODE_ZERO,
)


def get_ple_embedding_mode(config: Any) -> str:
    """Return the validated PLE embedding mode of a text config."""
    mode = getattr(config, "ple_embedding_mode", None)
    if mode is None:
        mode = PLE_EMBEDDING_MODE_NGRAM
    if mode not in PLE_EMBEDDING_MODES:
        raise ValueError(
            f"Invalid ple_embedding_mode {mode!r}; expected one of "
            f"{PLE_EMBEDDING_MODES}"
        )
    return mode


def validate_ple_embedding_config(config: Any) -> str:
    """Validate the PLE embedding fields of a text config; return the mode."""
    mode = get_ple_embedding_mode(config)
    if mode == PLE_EMBEDDING_MODE_NGRAM or not config.ple_layer_ids:
        return mode
    embed_dim = int(config.ple_embed_dim)
    if embed_dim <= 0:
        raise ValueError(f"ple_embed_dim must be positive, got {embed_dim}")
    if mode == PLE_EMBEDDING_MODE_PER_TOKEN:
        vocab_size = int(config.vocab_size)
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        heads = (int(config.ngram_size) - 1) * int(config.heads_per_ngram)
        if heads <= 0 or embed_dim % heads:
            raise ValueError(
                "ple_embedding_mode='per_token' requires ple_embed_dim to equal "
                "heads * head_dim for the configured n-gram heads: "
                f"ple_embed_dim={embed_dim}, heads={heads}"
            )
        if len(set(config.ple_layer_ids)) != 1:
            raise ValueError(
                "ple_embedding_mode='per_token' supports exactly one PLE layer, "
                f"got ple_layer_ids={list(config.ple_layer_ids)}"
            )
    return mode
