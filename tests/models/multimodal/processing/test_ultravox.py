# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Ultravox tower/connector LoRA token-count helpers.

The tower/connector LoRA mappings must recover the padded per-chunk token
counts from the placeholder count alone (see `gpu_model_runner`), so these
helpers are checked both as pure functions and against the token counts the
actual multimodal processor emits at chunk boundaries.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.models.ultravox import StackAudioFrames, UltravoxModel
from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context

MODEL_ID = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
SAMPLE_RATE = 16000

# whisper-large-v3(-turbo), used by all released Ultravox checkpoints
MAX_SOURCE_POSITIONS = 1500
STACK_FACTOR = 8


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
    # ceil(max_source_positions / stack_factor) tokens per chunk.
    model = _make_ultravox_model()
    tokens_per_full_chunk = math.ceil(MAX_SOURCE_POSITIONS / STACK_FACTOR)

    num_audio_tokens = tokens_per_full_chunk * (num_chunks - 1) + last_chunk_tokens
    num_encoder_tokens = num_chunks * MAX_SOURCE_POSITIONS
    num_connector_tokens = num_chunks * tokens_per_full_chunk

    assert model.get_num_mm_encoder_tokens(num_audio_tokens) == num_encoder_tokens
    assert model.get_num_mm_connector_tokens(num_encoder_tokens) == num_connector_tokens


@pytest.mark.parametrize(
    "num_seconds",
    [
        0.1,  # single token
        29.99,  # just below the 30s chunk boundary
        30.0,  # exactly one full chunk
        30.02,  # minimal second chunk
        60.0,
        100.0,  # partial last chunk
    ],
)
def test_ultravox_mm_lora_token_counts_match_processor(num_seconds: float) -> None:
    ctx = build_model_context(MODEL_ID)
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    audio = np.zeros(int(num_seconds * SAMPLE_RATE), dtype=np.float32)
    processed = processor(
        "<|audio|>",
        mm_items=processor.info.parse_mm_data({"audio": [(audio, SAMPLE_RATE)]}),
        hf_processor_mm_kwargs={},
    )

    num_audio_tokens = sum(p.length for p in processed["mm_placeholders"]["audio"])
    out_mm_data = processed["mm_kwargs"].get_data()
    num_chunks = int(out_mm_data["audio_num_chunks"].sum())
    # A chunk with no placeholder tokens would break the mapping's
    # chunk-count recovery.
    assert torch.all(out_mm_data["audio_token_len"] > 0)

    hf_config = ctx.model_config.hf_config
    max_source_positions = hf_config.audio_config.max_source_positions
    stack_factor = hf_config.stack_factor
    model = _make_ultravox_model(max_source_positions, stack_factor)

    # With `pad_audio_to_max_context` (enforced when tower/connector LoRA is
    # enabled), every chunk runs through the tower at the full context length
    # and through the connector at the frame-stacked length.
    num_encoder_tokens = num_chunks * max_source_positions
    stacked_len = StackAudioFrames(stack_factor)(
        torch.zeros(1, max_source_positions, 1)
    ).size(1)

    assert model.get_num_mm_encoder_tokens(num_audio_tokens) == num_encoder_tokens
    assert (
        model.get_num_mm_connector_tokens(num_encoder_tokens)
        == num_chunks * stacked_len
    )
