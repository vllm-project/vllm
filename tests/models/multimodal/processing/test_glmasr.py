# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.models.glmasr import GlmAsrForConditionalGeneration
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
)

from ...utils import build_model_context

MODEL_ID = "zai-org/GLM-ASR-Nano-2512"
SAMPLE_RATE = 16000

# GLM-ASR-Nano-2512: 30s chunks -> 3000 mel frames -> conv (stride 2) ->
# 1500 tower frames -> 4x frame merge -> 375 projector rows / LM tokens.
TOWER_TOKENS_PER_CHUNK = 1500
CONNECTOR_TOKENS_PER_CHUNK = 375


class _StubModel:
    """Carries only `config`, which is all the token helpers read from
    `self`, so the real methods can be exercised without constructing the
    full `nn.Module` (audio tower, language model, etc.)."""

    _get_audio_merge_ratio = GlmAsrForConditionalGeneration._get_audio_merge_ratio
    _get_num_tower_tokens_per_chunk = (
        GlmAsrForConditionalGeneration._get_num_tower_tokens_per_chunk
    )
    _get_num_connector_tokens_per_chunk = (
        GlmAsrForConditionalGeneration._get_num_connector_tokens_per_chunk
    )
    get_num_mm_encoder_tokens = GlmAsrForConditionalGeneration.get_num_mm_encoder_tokens
    get_num_mm_connector_tokens = (
        GlmAsrForConditionalGeneration.get_num_mm_connector_tokens
    )
    get_mm_lora_token_counts = GlmAsrForConditionalGeneration.get_mm_lora_token_counts

    def __init__(self, config):
        self.config = config


def _make_stub(
    max_position_embeddings: int = 1500,
    hidden_size: int = 1280,
    intermediate_size: int = 5120,
) -> _StubModel:
    audio_config = SimpleNamespace(
        max_position_embeddings=max_position_embeddings,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    return _StubModel(SimpleNamespace(audio_config=audio_config))


def _make_input_features_item(
    num_chunks: int, chunk_length: int, num_mel_bins: int = 128
) -> MultiModalKwargsItem:
    elem = MultiModalFieldElem(
        data=torch.zeros(num_chunks, num_mel_bins, chunk_length),
        field=MultiModalSharedField(batch_size=1),
    )
    return MultiModalKwargsItem({"input_features": elem})


@pytest.mark.parametrize(
    ("num_audio_tokens", "expected_num_chunks"),
    [
        (0, 0),
        (1, 1),
        (CONNECTOR_TOKENS_PER_CHUNK, 1),
        (CONNECTOR_TOKENS_PER_CHUNK + 1, 2),
        (2 * CONNECTOR_TOKENS_PER_CHUNK, 2),
        (21 * CONNECTOR_TOKENS_PER_CHUNK, 21),
    ],
)
def test_num_mm_tokens_roundtrip(num_audio_tokens, expected_num_chunks):
    """Only the last chunk can yield fewer LM tokens, so the chunk count is the
    ceiling over the full-chunk token count; each chunk then contributes a
    fixed number of tower and connector rows."""
    stub = _make_stub()

    encoder_tokens = stub.get_num_mm_encoder_tokens(num_audio_tokens)
    assert encoder_tokens == expected_num_chunks * TOWER_TOKENS_PER_CHUNK

    connector_tokens = stub.get_num_mm_connector_tokens(encoder_tokens)
    assert connector_tokens == expected_num_chunks * CONNECTOR_TOKENS_PER_CHUNK

    assert stub.get_mm_lora_token_counts(
        modality="audio", mm_kwargs=None, num_mm_embeds=num_audio_tokens
    ) == (encoder_tokens, connector_tokens)


@pytest.mark.parametrize(
    ("max_position_embeddings", "hidden_size", "intermediate_size"),
    [
        (1500, 1280, 5120),  # zai-org/GLM-ASR-Nano-2512
        (750, 512, 1024),  # hypothetical 2x merge
    ],
)
def test_num_mm_tokens_follow_config(
    max_position_embeddings, hidden_size, intermediate_size
):
    stub = _make_stub(max_position_embeddings, hidden_size, intermediate_size)
    merge_ratio = intermediate_size // hidden_size
    tokens_per_chunk = max_position_embeddings // merge_ratio

    encoder_tokens = stub.get_num_mm_encoder_tokens(3 * tokens_per_chunk)
    assert encoder_tokens == 3 * max_position_embeddings
    assert stub.get_num_mm_connector_tokens(encoder_tokens) == 3 * tokens_per_chunk


@pytest.mark.parametrize(
    ("num_chunks", "chunk_length", "expected_tower", "expected_connector"),
    [
        (1, 3000, 1500, 375),
        (2, 3000, 3000, 750),
        (21, 3000, 31500, 7875),
        # Non-default padding: conv1 keeps the length, conv2 halves it.
        (1, 1000, 500, 125),
    ],
)
def test_lora_token_counts_from_mm_kwargs(
    num_chunks, chunk_length, expected_tower, expected_connector
):
    """With the processed item available, the counts are derived from the
    actual chunks so they stay exact even when `num_mm_embeds` alone would
    under-count (see `test_lora_token_counts_nearly_empty_tail_chunk`)."""
    stub = _make_stub()
    mm_kwargs = _make_input_features_item(num_chunks, chunk_length)

    assert stub.get_mm_lora_token_counts(
        modality="audio", mm_kwargs=mm_kwargs, num_mm_embeds=1
    ) == (expected_tower, expected_connector)


@pytest.mark.parametrize(
    "duration_s",
    [
        1.0,  # partial single chunk
        30.0,  # exactly one full chunk
        30.02,  # full chunk + nearly empty tail chunk (0 LM tokens)
        55.0,  # full chunk + partial chunk
        90.0,  # three full chunks
    ],
)
def test_lora_token_counts_match_processor(duration_s):
    """The counts must agree with what the real HF processor produces: one
    padded chunk per 30s window in the tower/connector, and the LM token
    count is bounded by the full-chunk count."""
    ctx = build_model_context(MODEL_ID, limit_mm_per_prompt={"audio": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor = processor.info.get_hf_processor()

    num_samples = int(duration_s * SAMPLE_RATE)
    audio = np.sin(np.linspace(0, 440 * 2 * np.pi * duration_s, num_samples))
    audio = audio.astype(np.float32)

    prompt = f"<|user|>\n{hf_processor.audio_token}<|assistant|>\n"
    result = processor(
        prompt,
        mm_items=processor.info.parse_mm_data({"audio": [(audio, SAMPLE_RATE)]}),
        hf_processor_mm_kwargs={},
    )

    num_mm_embeds = result["mm_placeholders"]["audio"][0].get_num_embeds()
    mm_item = result["mm_kwargs"]["audio"][0]
    num_chunks = int(mm_item["chunk_counts"].data.item())
    assert num_chunks == -(-num_samples // (30 * SAMPLE_RATE))
    assert mm_item["input_features"].data.shape[0] == num_chunks

    stub = _make_stub(**_audio_config_kwargs(ctx.model_config.hf_config))
    tower_tokens, connector_tokens = stub.get_mm_lora_token_counts(
        modality="audio", mm_kwargs=mm_item, num_mm_embeds=num_mm_embeds
    )
    assert tower_tokens == num_chunks * TOWER_TOKENS_PER_CHUNK
    assert connector_tokens == num_chunks * CONNECTOR_TOKENS_PER_CHUNK
    assert 0 < num_mm_embeds <= connector_tokens

    # Without the item, the counts fall back to the ceiling over LM tokens,
    # which is exact unless the tail chunk yields no LM tokens at all.
    encoder_tokens = stub.get_num_mm_encoder_tokens(num_mm_embeds)
    assert encoder_tokens <= tower_tokens
    if num_mm_embeds > (num_chunks - 1) * CONNECTOR_TOKENS_PER_CHUNK:
        assert encoder_tokens == tower_tokens


def test_lora_token_counts_nearly_empty_tail_chunk():
    stub = _make_stub()
    # 30.02s: the second chunk holds 2 mel frames -> 0 LM tokens, but the
    # tower and projector still run on a full padded chunk.
    num_mm_embeds = CONNECTOR_TOKENS_PER_CHUNK
    mm_kwargs = _make_input_features_item(num_chunks=2, chunk_length=3000)

    assert stub.get_num_mm_encoder_tokens(num_mm_embeds) == TOWER_TOKENS_PER_CHUNK
    assert stub.get_mm_lora_token_counts(
        modality="audio", mm_kwargs=mm_kwargs, num_mm_embeds=num_mm_embeds
    ) == (2 * TOWER_TOKENS_PER_CHUNK, 2 * CONNECTOR_TOKENS_PER_CHUNK)


def _audio_config_kwargs(hf_config) -> dict[str, int]:
    audio_config = hf_config.audio_config
    return dict(
        max_position_embeddings=audio_config.max_position_embeddings,
        hidden_size=audio_config.hidden_size,
        intermediate_size=audio_config.intermediate_size,
    )
