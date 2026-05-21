# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache the timestamp anchor token id used by ``_get_verbose_segments``.

The base ``OpenAISpeechToText.__init__`` resolves ``"<|0.00|>"`` once via the
loaded tokenizer and stores it on the serving instance, so each verbose
transcription request avoids an otherwise-redundant tokenizer call per audio
chunk.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vllm.config import ModelConfig
from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
    TranscriptionSegment,
)
from vllm.entrypoints.speech_to_text.transcription.serving import (
    OpenAIServingTranscription,
)
from vllm.logprobs import Logprob

INIT_TOKEN_ID = 1000


class _StubVerboseModel:
    """Stand-in for a SupportsTranscription implementation with timestamps."""

    no_space_languages: set[str] = {"ja", "zh"}
    supports_segment_timestamp = True

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        return SpeechToTextConfig(
            sample_rate=16000.0,
            max_audio_clip_s=5.0,
        )

    @classmethod
    def post_process_output(cls, text: str) -> str:
        return text


def _build_tokenizer_mock() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 999

    def _encode(text: str, add_special_tokens: bool = True) -> list[int]:
        if text == "<|0.00|>":
            return [INIT_TOKEN_ID]
        raise AssertionError(f"unexpected encode() call: {text!r}")

    def _decode(token_ids) -> str:
        # Echo the token sequence so callers can assert the helper hands the
        # correct slice (and not, e.g., the timestamp anchors) to the decoder.
        return "/".join(str(t) for t in token_ids)

    tokenizer.encode.side_effect = _encode
    tokenizer.decode.side_effect = _decode
    return tokenizer


def _build_serving(tokenizer: MagicMock) -> OpenAIServingTranscription:
    engine_client = MagicMock()
    engine_client.model_config = MagicMock()
    engine_client.model_config.get_diff_sampling_param.return_value = {
        "max_tokens": 256,
        "temperature": 0.0,
    }
    engine_client.model_config.max_model_len = 8192
    engine_client.errored = False

    models = MagicMock(spec=OpenAIServingModels)
    models.lora_requests = {}
    models.is_base_model.return_value = True

    with (
        patch(
            "vllm.model_executor.model_loader.get_model_cls",
            return_value=_StubVerboseModel,
        ),
        patch(
            "vllm.entrypoints.speech_to_text.base.serving.get_tokenizer",
            return_value=tokenizer,
        ),
    ):
        return OpenAIServingTranscription(engine_client, models, request_logger=None)


def test_init_caches_segment_init_token():
    """``__init__`` resolves ``<|0.00|>`` once and stores it on the instance."""

    tokenizer = _build_tokenizer_mock()
    serving = _build_serving(tokenizer)

    assert serving._segment_init_token_id == INIT_TOKEN_ID
    assert tokenizer.encode.call_count == 1


def test_get_verbose_segments_does_not_re_encode_init_token():
    """Repeated verbose-segment builds reuse the cached id without re-encoding."""

    tokenizer = _build_tokenizer_mock()
    serving = _build_serving(tokenizer)

    init_call_count = tokenizer.encode.call_count
    assert init_call_count == 1

    # Two timestamp tokens bracket three content tokens; this matches the
    # Whisper-style layout the helper expects.
    tokens = (INIT_TOKEN_ID, 50, 51, 52, INIT_TOKEN_ID + 10)
    log_probs: list[dict[int, Logprob]] = [
        {tok: Logprob(logprob=-0.1)} for tok in tokens
    ]
    request = TranscriptionRequest.model_construct(
        file=MagicMock(),
        model="stub-model",
        language="en",
        stream=False,
        response_format="verbose_json",
        temperature=0.0,
    )

    for _ in range(3):
        segments = serving._get_verbose_segments(
            tokens=tokens,
            log_probs=log_probs,
            request=request,
            segment_class=TranscriptionSegment,
            start_time=0.0,
        )
        # Sanity: the helper uses the cached anchor and emits segments.
        assert segments, "expected at least one segment for timestamp-bracketed tokens"

    # No additional encode() calls should have happened after init.
    assert tokenizer.encode.call_count == init_call_count


def test_get_verbose_segments_segment_boundaries_use_cached_init_token():
    """The cached init token still drives correct segment timing and decode slice."""

    tokenizer = _build_tokenizer_mock()
    serving = _build_serving(tokenizer)

    # Two consecutive timestamp tokens at positions 0 and 4 → one zero-width
    # segment at the start, one segment spanning offsets [0, 10] * 0.02s.
    tokens = (INIT_TOKEN_ID, 50, 51, 52, INIT_TOKEN_ID + 10)
    log_probs = [{tok: Logprob(logprob=-0.1)} for tok in tokens]
    request = TranscriptionRequest.model_construct(
        file=MagicMock(),
        model="stub-model",
        language="en",
        stream=False,
        response_format="verbose_json",
        temperature=0.0,
    )

    segments = serving._get_verbose_segments(
        tokens=tokens,
        log_probs=log_probs,
        request=request,
        segment_class=TranscriptionSegment,
        start_time=0.0,
    )

    # Last segment spans the full content span (10 ticks of 0.02s = 0.2s).
    final_segment = segments[-1]
    assert final_segment.start == pytest.approx(0.0)
    assert final_segment.end == pytest.approx(0.2)
    # The helper must hand only the content tokens (between the two timestamp
    # anchors) to the decoder; assert via the echo-style mock decode.
    assert final_segment.text == "50/51/52"
    assert list(final_segment.tokens) == [50, 51, 52]
