# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from types import SimpleNamespace

import pytest

from vllm.model_executor.models.ultravox import UltravoxModel

# whisper-large-v3(-turbo), used by all released Ultravox checkpoints
MAX_SOURCE_POSITIONS = 1500
STACK_FACTOR = 8
# mel frames per full 30s chunk (conv downsampling factor of 2)
CHUNK_MEL_FRAMES = 2 * MAX_SOURCE_POSITIONS
ENCODER_DS_FACTOR = 2


def _make_ultravox_model(
    max_source_positions: int = MAX_SOURCE_POSITIONS,
    stack_factor: int = STACK_FACTOR,
) -> UltravoxModel:
    model = UltravoxModel.__new__(UltravoxModel)
    model.config = SimpleNamespace(
        audio_config=SimpleNamespace(max_source_positions=max_source_positions),
        stack_factor=stack_factor,
    )
    return model


@pytest.mark.parametrize("num_chunks", [1, 2, 5])
@pytest.mark.parametrize("last_chunk_tokens", [1, 100, 188])
def test_ultravox_mm_lora_token_counts(num_chunks: int, last_chunk_tokens: int) -> None:
    # With chunks padded to the tower's full 30s context (enforced when
    # tower/connector LoRA is enabled), the tower runs on
    # `max_source_positions` tokens per chunk and the connector on
    # ceil(max_source_positions / stack_factor) tokens per chunk. The tower
    # and connector LoRA mappings rely on these counts (see gpu_model_runner).
    model = _make_ultravox_model()
    tokens_per_full_chunk = math.ceil(MAX_SOURCE_POSITIONS / STACK_FACTOR)

    num_audio_tokens = tokens_per_full_chunk * (num_chunks - 1) + last_chunk_tokens
    num_encoder_tokens = num_chunks * MAX_SOURCE_POSITIONS
    num_connector_tokens = num_chunks * tokens_per_full_chunk

    assert model.get_num_mm_encoder_tokens(num_audio_tokens) == num_encoder_tokens
    assert model.get_num_mm_connector_tokens(num_encoder_tokens) == num_connector_tokens


@pytest.mark.parametrize(
    "chunk_mel_lens",
    [
        [10],
        [3000],
        [3000, 1],
        [3000, 2999],
        [3000, 3000, 1234],
    ],
)
def test_ultravox_mm_lora_token_counts_match_processor(
    chunk_mel_lens: list[int],
) -> None:
    # `UltravoxProcessor` emits ceil(mel_len / (encoder_ds_factor *
    # stack_factor)) placeholder tokens per chunk; check that the helpers
    # recover the exact padded per-chunk counts from the placeholder total.
    model = _make_ultravox_model()

    num_audio_tokens = sum(
        math.ceil(mel_len / (ENCODER_DS_FACTOR * STACK_FACTOR))
        for mel_len in chunk_mel_lens
    )
    num_chunks = len(chunk_mel_lens)

    # Each chunk is padded to CHUNK_MEL_FRAMES mel frames, so the tower always
    # runs on max_source_positions frames per chunk and the connector on the
    # frame-stacked length.
    num_encoder_tokens = num_chunks * MAX_SOURCE_POSITIONS
    stacked_len = math.ceil(MAX_SOURCE_POSITIONS / STACK_FACTOR)
    num_connector_tokens = num_chunks * stacked_len

    assert model.get_num_mm_encoder_tokens(num_audio_tokens) == num_encoder_tokens
    assert model.get_num_mm_connector_tokens(num_encoder_tokens) == num_connector_tokens
