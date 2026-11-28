# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import StrEnum
from typing import Any

from vllm.platforms import current_platform


class PleEmbeddingMode(StrEnum):
    NGRAM = "ngram"
    PER_TOKEN = "per_token"


def validate_ple_embedding_config(config: Any) -> PleEmbeddingMode:
    value = getattr(config, "ple_embedding_mode", None)
    try:
        mode = PleEmbeddingMode.NGRAM if value is None else PleEmbeddingMode(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid ple_embedding_mode {value!r}; expected one of "
            f"{', '.join(PleEmbeddingMode)}"
        ) from exc
    if mode == PleEmbeddingMode.NGRAM:
        return mode
    if config.ple_layer_ids:
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
    if not current_platform.is_rocm():
        raise NotImplementedError(
            f"ple_embedding_mode={mode.value!r} is only implemented for ROCm"
        )
    return mode
