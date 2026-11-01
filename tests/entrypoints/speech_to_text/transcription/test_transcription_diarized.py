# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``diarized_json`` responses from ``create_transcription`` (mocked engine)."""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm.config import ModelConfig
from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
    TranscriptionResponseDiarized,
)
from vllm.entrypoints.speech_to_text.transcription.serving import (
    OpenAIServingTranscription,
)
from vllm.model_executor.models.interfaces import StreamingTranscriptionPostProcessor
from vllm.model_executor.models.vibevoice_asr import (
    VibeVoiceAsrForConditionalGeneration,
)
from vllm.outputs import CompletionOutput, RequestOutput

TRANSCRIPT = (
    '[{"Start":0.0,"End":1.5,"Speaker":0,"Content":"Hello there."},'
    '{"Start":1.62,"End":3.0,"Speaker":1,"Content":"Hi!"}]'
)


class _StubVibeVoiceAsr:
    """VibeVoice-ASR output handling without loading its processor."""

    no_space_languages: set[str] = {"ja", "zh"}
    supports_segment_timestamp = False
    supports_diarized_transcription = True
    post_process_output = VibeVoiceAsrForConditionalGeneration.post_process_output
    parse_diarized_transcript = (
        VibeVoiceAsrForConditionalGeneration.parse_diarized_transcript
    )

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        return SpeechToTextConfig(
            sample_rate=24000.0,
            max_audio_clip_s=None,
            min_energy_split_window_size=None,
        )

    @classmethod
    def get_streaming_post_processor_cls(
        cls,
    ) -> type[StreamingTranscriptionPostProcessor]:
        return StreamingTranscriptionPostProcessor


async def _generate(*args, **kwargs) -> AsyncGenerator[RequestOutput, None]:
    yield RequestOutput(
        request_id="rid",
        prompt=None,
        prompt_token_ids=None,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=TRANSCRIPT,
                token_ids=(1, 2, 3),
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop",
            )
        ],
        finished=True,
    )


@pytest.fixture
def serving():
    engine_client = MagicMock()
    engine_client.model_config.get_diff_sampling_param.return_value = {}
    engine_client.model_config.max_model_len = 8192
    engine_client.errored = False
    engine_client.generate.side_effect = _generate

    models = MagicMock(spec=OpenAIServingModels)
    models.lora_requests = {}
    models.is_base_model.return_value = True

    with (
        patch(
            "vllm.model_executor.model_loader.get_model_cls",
            return_value=_StubVibeVoiceAsr,
        ),
        patch.object(
            SpeechToTextBaseServing,
            "_preprocess_speech_to_text",
            AsyncMock(return_value=([MagicMock()], 3.0, [0.0])),
        ),
    ):
        yield OpenAIServingTranscription(engine_client, models, request_logger=None)


@pytest.mark.asyncio
async def test_json_response_text_is_post_processed(serving):
    request = TranscriptionRequest.model_construct(
        file=MagicMock(),
        model="stub-model",
        language=None,
        stream=False,
        response_format="json",
    )

    response = await serving.create_transcription(b"", request, raw_request=None)

    assert not isinstance(response, ErrorResponse)
    assert response.text == "Hello there. Hi!"


@pytest.mark.asyncio
async def test_diarized_segments_are_parsed_from_raw_output(serving):
    """``parse_diarized_transcript`` must see the raw output, which the
    model's ``post_process_output`` reduces to plain text."""
    request = TranscriptionRequest.model_construct(
        file=MagicMock(),
        model="stub-model",
        language=None,
        stream=False,
        response_format="diarized_json",
    )

    response = await serving.create_transcription(b"", request, raw_request=None)

    assert isinstance(response, TranscriptionResponseDiarized)
    assert response.text == "Hello there. Hi!"
    assert [
        (segment.start, segment.end, segment.speaker, segment.text)
        for segment in response.segments
    ] == [
        (0.0, 1.5, "0", "Hello there."),
        (1.62, 3.0, "1", "Hi!"),
    ]
