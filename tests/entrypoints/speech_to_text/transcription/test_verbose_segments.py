# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verbose responses must preserve decoded text, even without timestamp tokens."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.entrypoints.speech_to_text.transcription.protocol import TranscriptionRequest
from vllm.entrypoints.speech_to_text.transcription.serving import (
    OpenAIServingTranscription,
)
from vllm.entrypoints.speech_to_text.translation.protocol import TranslationRequest
from vllm.entrypoints.speech_to_text.translation.serving import OpenAIServingTranslation
from vllm.logprobs import FlatLogprobs, Logprob
from vllm.outputs import CompletionOutput, RequestOutput

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

TIMESTAMP_BEGIN = 1000
EOS = 900


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("serving_class", "request_class", "task_type"),
    [
        (OpenAIServingTranscription, TranscriptionRequest, "transcribe"),
        (OpenAIServingTranslation, TranslationRequest, "translate"),
    ],
)
@pytest.mark.parametrize("flat_logprobs", [False, True])
@pytest.mark.parametrize(
    ("chunk_tokens", "expected_bounds"),
    [
        (((1, EOS), (2, EOS)), [(0.0, 29.5), (29.5, 32.87)]),
        (((1,), (2,)), [(0.0, 29.5), (29.5, 32.87)]),
        (((1, 1100, EOS), (2, EOS)), [(0.0, 2.0), (29.5, 32.87)]),
        (((1, 1100, EOS), (2, 1150, EOS)), [(0.0, 2.0), (29.5, 32.5)]),
    ],
    ids=["no-timestamps", "no-eos", "mixed-chunks", "timestamped"],
)
async def test_verbose_response_preserves_text_and_actual_chunk_bounds(
    serving_class,
    request_class,
    task_type,
    flat_logprobs,
    chunk_tokens,
    expected_bounds,
):
    """Use actual chunk ends only when timestamps are absent, preserving text."""
    words = {1: "hello", 2: "world"}
    tokenizer = MagicMock(eos_token_id=EOS)
    tokenizer.encode.return_value = [TIMESTAMP_BEGIN]
    tokenizer.decode.side_effect = lambda tokens, **kwargs: "".join(
        words.get(token, "") for token in tokens
    )

    async def generate(tokens):
        logprobs = [
            {token: Logprob(logprob=-999.0 if token == EOS else -0.3)}
            for token in tokens
        ]
        if flat_logprobs:
            flat = FlatLogprobs()
            flat.extend(logprobs)
            logprobs = flat
        yield RequestOutput(
            request_id="test",
            prompt=None,
            prompt_token_ids=None,
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text=tokenizer.decode(tokens),
                    token_ids=tokens,
                    cumulative_logprob=None,
                    logprobs=logprobs,
                    finish_reason="stop",
                )
            ],
            finished=True,
        )

    serving = serving_class.__new__(serving_class)
    serving.task_type = task_type
    serving.tokenizer = tokenizer
    serving.model_cls = SimpleNamespace(
        supports_segment_timestamp=True,
        no_space_languages={"ja", "zh"},
        post_process_output=lambda text: text,
    )
    serving.model_config = SimpleNamespace(max_model_len=448)
    serving.default_sampling_params = {"max_tokens": 448}
    serving.asr_config = SpeechToTextConfig(max_audio_clip_s=30)
    serving._check_model = AsyncMock(return_value=None)
    serving._preflight = MagicMock()
    serving._maybe_get_adapters = MagicMock(return_value=None)
    serving._log_inputs = MagicMock()
    serving._preprocess_speech_to_text = AsyncMock(
        return_value=([MagicMock(), MagicMock()], 32.87, [0.0, 29.5])
    )
    serving.engine_client = MagicMock()
    create = (
        serving.create_transcription
        if task_type == "transcribe"
        else serving.create_translation
    )
    for response_format in ("json", "verbose_json"):
        serving.engine_client.generate.side_effect = [
            generate(tokens) for tokens in chunk_tokens
        ]
        request = request_class.model_construct(
            file=MagicMock(),
            model="stub-whisper",
            language="en",
            response_format=response_format,
            temperature=0.7,
        )
        response = await create(b"audio", request)
        assert response.text == "hello world"
        if response_format == "verbose_json":
            assert response.duration == pytest.approx(32.87)
            assert [segment.text for segment in response.segments] == ["hello", "world"]
            for segment, tokens, bounds in zip(
                response.segments, chunk_tokens, expected_bounds
            ):
                assert (segment.start, segment.end) == pytest.approx(bounds)
                assert segment.tokens == [tokens[0]]
                assert segment.temperature == 0.7
                if all(token < TIMESTAMP_BEGIN for token in tokens):
                    assert segment.avg_logprob == pytest.approx(-0.3)
