# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`prompt_tokens` of streamed transcriptions: every audio chunk's prompt is
counted, and the duration based audio estimate is added only for
encoder-decoder models, whose decoder prompt does not hold the audio."""

import json
from collections.abc import AsyncGenerator
from types import SimpleNamespace

import pytest

from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.speech_to_text.transcription.serving import (
    OpenAIServingTranscription,
)
from vllm.model_executor.models.interfaces import StreamingTranscriptionPostProcessor
from vllm.outputs import CompletionOutput, RequestOutput

AUDIO_ESTIMATE = 500


class _StubTranscriptionModel:
    @classmethod
    def get_num_audio_tokens(cls, audio_duration_s, stt_config, model_config):
        return AUDIO_ESTIMATE


def _request_output(
    prompt_len: int | None, text: str, finish_reason: str | None
) -> RequestOutput:
    return RequestOutput(
        request_id="rid",
        prompt=None,
        prompt_token_ids=None if prompt_len is None else [7] * prompt_len,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=(1, 2),
                cumulative_logprob=None,
                logprobs=None,
                finish_reason=finish_reason,
            )
        ],
        finished=finish_reason is not None,
    )


async def _chunk(prompt_len: int) -> AsyncGenerator[RequestOutput, None]:
    # The prompt token ids come with the first output only.
    yield _request_output(prompt_len, "hello", None)
    yield _request_output(None, " world", "stop")


async def _stream_usages(
    is_encoder_decoder: bool, prompt_lens: list[int]
) -> list[dict]:
    serving = OpenAIServingTranscription.__new__(OpenAIServingTranscription)
    serving.enable_force_include_usage = False
    serving.model_cls = _StubTranscriptionModel
    serving.model_config = SimpleNamespace(is_encoder_decoder=is_encoder_decoder)
    serving.asr_config = None
    serving.streaming_post_processor_cls = StreamingTranscriptionPostProcessor
    serving.task_type = "transcribe"  # type: ignore[misc]
    request = SimpleNamespace(
        model="stub-model",
        stream_include_usage=True,
        stream_continuous_usage_stats=True,
    )
    usages = []
    async for frame in OpenAIServingTranscription.transcription_stream_generator(
        serving,
        request=request,
        result_generator=[_chunk(n) for n in prompt_lens],
        request_id="rid",
        request_metadata=RequestResponseMetadata(request_id="rid"),
        audio_duration_s=60.0,
        separator=" ",
    ):
        payload = frame.removeprefix("data: ").strip()
        if payload != "[DONE]":
            usages.append(json.loads(payload)["usage"])
    return usages


@pytest.mark.asyncio
async def test_decoder_only_prompt_already_holds_the_audio():
    # Two 30 s chunks whose prompts carry their own audio placeholders.
    usages = await _stream_usages(False, [380, 390])
    assert [u["prompt_tokens"] for u in usages] == [380, 380, 770, 770, 770]
    assert usages[-1] == {
        "prompt_tokens": 770,
        "completion_tokens": 8,
        "total_tokens": 778,
    }


@pytest.mark.asyncio
async def test_encoder_decoder_adds_the_audio_estimate_once():
    usages = await _stream_usages(True, [4, 4])
    assert [u["prompt_tokens"] for u in usages] == [504, 504, 508, 508, 508]
    assert usages[-1]["total_tokens"] == 516
