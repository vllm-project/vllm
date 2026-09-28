# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short Voxtral Realtime clips are rejected before the encoder."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.voxtral_realtime import (
    VoxtralRealtimeGeneration,
    VoxtralRealtimeMultiModalProcessor,
    minimum_realtime_audio_samples,
    realtime_pooled_rows,
    validate_realtime_audio_length,
)

# Shipping Voxtral Realtime geometry: 16 kHz, window 400, hop 160, pool 4.
_HOP = 160
_WINDOW = 400
_POOL = 4


def _geometry_kwargs() -> dict[str, int]:
    return {
        "hop_length": _HOP,
        "window_size": _WINDOW,
        "pool_size": _POOL,
    }


def test_reported_short_clip_has_no_pooled_rows():
    """50 ms (800 samples) is the clip that emptied the encoder tensor."""
    assert realtime_pooled_rows(800, **_geometry_kwargs()) == 0
    assert realtime_pooled_rows(160, **_geometry_kwargs()) == 0
    assert realtime_pooled_rows(201, **_geometry_kwargs()) == 0


def test_minimum_length_is_one_pooled_block():
    minimum = minimum_realtime_audio_samples(**_geometry_kwargs())
    assert minimum == 1280
    assert realtime_pooled_rows(minimum - 1, **_geometry_kwargs()) == 0
    assert realtime_pooled_rows(minimum, **_geometry_kwargs()) > 0


def test_validate_realtime_audio_length_rejects_short_clip():
    with pytest.raises(ValueError, match="at least 1280 samples"):
        validate_realtime_audio_length(800, **_geometry_kwargs())
    validate_realtime_audio_length(1280, **_geometry_kwargs())


def _tokenizer_must_not_run():
    raise AssertionError("tokenizer should not run")


def test_processor_rejects_short_audio_before_tokenization():
    processor = VoxtralRealtimeMultiModalProcessor.__new__(
        VoxtralRealtimeMultiModalProcessor
    )
    info = SimpleNamespace(
        get_hf_config=lambda: SimpleNamespace(
            audio_config=SimpleNamespace(
                hop_length=_HOP,
                window_size=_WINDOW,
                block_pool_size=_POOL,
            )
        ),
        get_tokenizer=_tokenizer_must_not_run,
    )
    processor.info = info
    audio = SimpleNamespace(data=torch.zeros(800))
    mm_res = SimpleNamespace(kwargs={"audio": [{"audio_arrays": audio}]})

    with pytest.raises(ValueError, match="too short to encode"):
        processor._maybe_apply_prompt_updates(None, mm_res)


def _melspec_must_not_run(audio):
    raise AssertionError("melspec should not run")


def test_embed_multimodal_returns_empty_for_short_audio():
    model = VoxtralRealtimeGeneration.__new__(VoxtralRealtimeGeneration)
    model.config = SimpleNamespace(
        audio_config=SimpleNamespace(
            hop_length=_HOP,
            window_size=_WINDOW,
            block_pool_size=_POOL,
        )
    )
    model.whisper_encoder = SimpleNamespace(
        whisper_encoder=SimpleNamespace(total_stride=2),
        compute_whisper_melspec=_melspec_must_not_run,
    )
    model._parse_and_validate_audio_arrays = lambda **kwargs: [torch.zeros(800)]

    assert model.embed_multimodal() == []
