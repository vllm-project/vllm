# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from http import HTTPStatus
from typing import Literal, TypeAlias

import torch
from fastapi import HTTPException, UploadFile
from pydantic import (
    Field,
    model_validator,
)

from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    OpenAIBaseModel,
    UsageInfo,
)
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
)
from vllm.utils import random_uuid

logger = init_logger(__name__)
_LONG_INFO = torch.iinfo(torch.long)


class TranscriptionResponseStreamChoice(OpenAIBaseModel):
    delta: DeltaMessage
    finish_reason: str | None = None
    stop_reason: int | str | None = None


class TranscriptionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"trsc-{random_uuid()}")
    object: Literal["transcription.chunk"] = "transcription.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[TranscriptionResponseStreamChoice]
    usage: UsageInfo | None = Field(default=None)


## Protocols for Audio
AudioResponseFormat: TypeAlias = Literal["json", "text", "srt", "verbose_json", "vtt"]


class TranscriptionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/audio/createTranscription

    file: UploadFile
    """
    The audio file object (not file name) to transcribe, in one of these
    formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.
    """

    model: str | None = None
    """ID of the model to use.
    """

    language: str | None = None
    """The language of the input audio.

    Supplying the input language in
    [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) format
    will improve accuracy and latency.
    """

    prompt: str = Field(default="")
    """An optional text to guide the model's style or continue a previous audio
    segment.

    The [prompt](https://platform.openai.com/docs/guides/speech-to-text#prompting)
    should match the audio language.
    """

    response_format: AudioResponseFormat = Field(default="json")
    """
    The format of the output, in one of these options: `json`, `text`, `srt`,
    `verbose_json`, or `vtt`.
    """

    ## TODO (varun) : Support if set to 0, certain thresholds are met !!

    timestamp_granularities: list[Literal["word", "segment"]] = Field(
        alias="timestamp_granularities[]", default=[]
    )
    """The timestamp granularities to populate for this transcription.

    `response_format` must be set `verbose_json` to use timestamp granularities.
    Either or both of these options are supported: `word`, or `segment`. Note:
    There is no additional latency for segment timestamps, but generating word
    timestamps incurs additional latency.
    """

    stream: bool | None = False
    """When set, it will enable output to be streamed in a similar fashion
    as the Chat Completion endpoint.
    """
    # --8<-- [start:transcription-extra-params]
    # Flattened stream option to simplify form data.
    stream_include_usage: bool | None = False
    stream_continuous_usage_stats: bool | None = False

    vllm_xargs: dict[str, str | int | float] | None = Field(
        default=None,
        description=(
            "Additional request parameters with string or "
            "numeric values, used by custom extensions."
        ),
    )
    # --8<-- [end:transcription-extra-params]

    to_language: str | None = None
    """The language of the output audio we transcribe to.

    Please note that this is not currently used by supported models at this
    time, but it is a placeholder for future use, matching translation api.
    """

    # --8<-- [start:transcription-sampling-params]
    temperature: float = Field(default=0.0)
    """The sampling temperature, between 0 and 1.

    Higher values like 0.8 will make the output more random, while lower values
    like 0.2 will make it more focused / deterministic. If set to 0, the model
    will use [log probability](https://en.wikipedia.org/wiki/Log_probability)
    to automatically increase the temperature until certain thresholds are hit.
    """

    top_p: float | None = None
    """Enables nucleus (top-p) sampling, where tokens are selected from the
    smallest possible set whose cumulative probability exceeds `p`.
    """

    top_k: int | None = None
    """Limits sampling to the `k` most probable tokens at each step."""

    min_p: float | None = None
    """Filters out tokens with a probability lower than `min_p`, ensuring a
    minimum likelihood threshold during sampling.
    """

    seed: int | None = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    """The seed to use for sampling."""

    frequency_penalty: float | None = 0.0
    """The frequency penalty to use for sampling."""

    repetition_penalty: float | None = None
    """The repetition penalty to use for sampling."""

    presence_penalty: float | None = 0.0
    """The presence penalty to use for sampling."""

    max_completion_tokens: int | None = None
    """The maximum number of tokens to generate."""
    # --8<-- [end:transcription-sampling-params]

    # Default sampling parameters for transcription requests.
    _DEFAULT_SAMPLING_PARAMS: dict = {
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
    }

    def to_sampling_params(
        self, default_max_tokens: int, default_sampling_params: dict | None = None
    ) -> SamplingParams:
        max_tokens = default_max_tokens

        if default_sampling_params is None:
            default_sampling_params = {}

        # Default parameters
        if (temperature := self.temperature) is None:
            temperature = default_sampling_params.get(
                "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"]
            )
        if (top_p := self.top_p) is None:
            top_p = default_sampling_params.get(
                "top_p", self._DEFAULT_SAMPLING_PARAMS["top_p"]
            )
        if (top_k := self.top_k) is None:
            top_k = default_sampling_params.get(
                "top_k", self._DEFAULT_SAMPLING_PARAMS["top_k"]
            )
        if (min_p := self.min_p) is None:
            min_p = default_sampling_params.get(
                "min_p", self._DEFAULT_SAMPLING_PARAMS["min_p"]
            )

        if (repetition_penalty := self.repetition_penalty) is None:
            repetition_penalty = default_sampling_params.get(
                "repetition_penalty",
                self._DEFAULT_SAMPLING_PARAMS["repetition_penalty"],
            )

        return SamplingParams.from_optional(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=self.seed,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=repetition_penalty,
            presence_penalty=self.presence_penalty,
            output_kind=RequestOutputKind.DELTA
            if self.stream
            else RequestOutputKind.FINAL_ONLY,
            extra_args=self.vllm_xargs,
            skip_clone=True,  # Created fresh per request, safe to skip clone
        )

    @model_validator(mode="before")
    @classmethod
    def validate_transcription_request(cls, data):
        if isinstance(data.get("file"), str):
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail="Expected 'file' to be a file-like object, not 'str'.",
            )

        stream_opts = ["stream_include_usage", "stream_continuous_usage_stats"]
        stream = data.get("stream", False)
        if any(bool(data.get(so, False)) for so in stream_opts) and not stream:
            # Find which specific stream option was set
            invalid_param = next(
                (so for so in stream_opts if data.get(so, False)),
                "stream_include_usage",
            )
            raise VLLMValidationError(
                "Stream options can only be defined when `stream=True`.",
                parameter=invalid_param,
            )

        return data


# Transcription response objects
class TranscriptionUsageAudio(OpenAIBaseModel):
    type: Literal["duration"] = "duration"
    seconds: int


class TranscriptionResponse(OpenAIBaseModel):
    text: str
    """The transcribed text."""
    usage: TranscriptionUsageAudio


class TranscriptionWord(OpenAIBaseModel):
    end: float
    """End time of the word in seconds."""

    start: float
    """Start time of the word in seconds."""

    word: str
    """The text content of the word."""


class TranscriptionSegment(OpenAIBaseModel):
    id: int
    """Unique identifier of the segment."""

    avg_logprob: float | None = None
    """Average logprob of the segment.

    If the value is lower than -1, consider the logprobs failed.
    """

    compression_ratio: float | None = None
    """Compression ratio of the segment.

    If the value is greater than 2.4, consider the compression failed.
    """

    end: float
    """End time of the segment in seconds."""

    no_speech_prob: float | None = None
    """Probability of no speech in the segment.

    If the value is higher than 1.0 and the `avg_logprob` is below -1, consider
    this segment silent.
    """

    seek: int
    """Seek offset of the segment."""

    start: float
    """Start time of the segment in seconds."""

    temperature: float
    """Temperature parameter used for generating the segment."""

    text: str
    """Text content of the segment."""

    tokens: list[int]
    """Array of token IDs for the text content."""


class TranscriptionResponseVerbose(OpenAIBaseModel):
    duration: str
    """The duration of the input audio."""

    language: str
    """The language of the input audio."""

    text: str
    """The transcribed text."""

    segments: list[TranscriptionSegment] | None = None
    """Segments of the transcribed text and their corresponding details."""

    words: list[TranscriptionWord] | None = None
    """Extracted words and their corresponding timestamps."""


TranscriptionResponseVariant: TypeAlias = (
    TranscriptionResponse | TranscriptionResponseVerbose
)


class TranslationResponseStreamChoice(OpenAIBaseModel):
    delta: DeltaMessage
    finish_reason: str | None = None
    stop_reason: int | str | None = None


class TranslationStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"trsl-{random_uuid()}")
    object: Literal["translation.chunk"] = "translation.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[TranslationResponseStreamChoice]
    usage: UsageInfo | None = Field(default=None)


class TranslationRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/audio/createTranslation

    file: UploadFile
    """
    The audio file object (not file name) to translate, in one of these
    formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.
    """

    model: str | None = None
    """ID of the model to use.
    """

    prompt: str = Field(default="")
    """An optional text to guide the model's style or continue a previous audio
    segment.

    The [prompt](https://platform.openai.com/docs/guides/speech-to-text#prompting)
    should match the audio language.
    """

    response_format: AudioResponseFormat = Field(default="json")
    """
    The format of the output, in one of these options: `json`, `text`, `srt`,
    `verbose_json`, or `vtt`.
    """

    # TODO support additional sampling parameters
    # --8<-- [start:translation-sampling-params]
    seed: int | None = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    """The seed to use for sampling."""

    temperature: float = Field(default=0.0)
    """The sampling temperature, between 0 and 1.

    Higher values like 0.8 will make the output more random, while lower values
    like 0.2 will make it more focused / deterministic. If set to 0, the model
    will use [log probability](https://en.wikipedia.org/wiki/Log_probability)
    to automatically increase the temperature until certain thresholds are hit.
    """
    # --8<-- [end:translation-sampling-params]

    # --8<-- [start:translation-extra-params]
    language: str | None = None
    """The language of the input audio we translate from.

    Supplying the input language in
    [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) format
    will improve accuracy.
    """

    to_language: str | None = None
    """The language of the input audio we translate to.

    Please note that this is not supported by all models, refer to the specific
    model documentation for more details.
    For instance, Whisper only supports `to_language=en`.
    """

    stream: bool | None = False
    """Custom field not present in the original OpenAI definition. When set,
    it will enable output to be streamed in a similar fashion as the Chat
    Completion endpoint.
    """
    # Flattened stream option to simplify form data.
    stream_include_usage: bool | None = False
    stream_continuous_usage_stats: bool | None = False

    max_completion_tokens: int | None = None
    """The maximum number of tokens to generate."""
    # --8<-- [end:translation-extra-params]

    # Default sampling parameters for translation requests.
    _DEFAULT_SAMPLING_PARAMS: dict = {
        "temperature": 0,
    }

    def to_sampling_params(
        self, default_max_tokens: int, default_sampling_params: dict | None = None
    ) -> SamplingParams:
        max_tokens = default_max_tokens

        if default_sampling_params is None:
            default_sampling_params = {}
        # Default parameters
        if (temperature := self.temperature) is None:
            temperature = default_sampling_params.get(
                "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"]
            )

        return SamplingParams.from_optional(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=self.seed,
            output_kind=RequestOutputKind.DELTA
            if self.stream
            else RequestOutputKind.FINAL_ONLY,
            skip_clone=True,  # Created fresh per request, safe to skip clone
        )

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        stream_opts = ["stream_include_usage", "stream_continuous_usage_stats"]
        stream = data.get("stream", False)
        if any(bool(data.get(so, False)) for so in stream_opts) and not stream:
            # Find which specific stream option was set
            invalid_param = next(
                (so for so in stream_opts if data.get(so, False)),
                "stream_include_usage",
            )
            raise VLLMValidationError(
                "Stream options can only be defined when `stream=True`.",
                parameter=invalid_param,
            )

        return data


# Translation response objects
class TranslationResponse(OpenAIBaseModel):
    text: str
    """The translated text."""


class TranslationWord(OpenAIBaseModel):
    end: float
    """End time of the word in seconds."""

    start: float
    """Start time of the word in seconds."""

    word: str
    """The text content of the word."""


class TranslationSegment(OpenAIBaseModel):
    id: int
    """Unique identifier of the segment."""

    avg_logprob: float | None = None
    """Average logprob of the segment.

    If the value is lower than -1, consider the logprobs failed.
    """

    compression_ratio: float | None = None
    """Compression ratio of the segment.

    If the value is greater than 2.4, consider the compression failed.
    """

    end: float
    """End time of the segment in seconds."""

    no_speech_prob: float | None = None
    """Probability of no speech in the segment.

    If the value is higher than 1.0 and the `avg_logprob` is below -1, consider
    this segment silent.
    """

    seek: int
    """Seek offset of the segment."""

    start: float
    """Start time of the segment in seconds."""

    temperature: float
    """Temperature parameter used for generating the segment."""

    text: str
    """Text content of the segment."""

    tokens: list[int]
    """Array of token IDs for the text content."""


class TranslationResponseVerbose(OpenAIBaseModel):
    duration: str
    """The duration of the input audio."""

    language: str
    """The language of the input audio."""

    text: str
    """The translated text."""

    segments: list[TranslationSegment] | None = None
    """Segments of the translated text and their corresponding details."""

    words: list[TranslationWord] | None = None
    """Extracted words and their corresponding timestamps."""


TranslationResponseVariant: TypeAlias = TranslationResponse | TranslationResponseVerbose
