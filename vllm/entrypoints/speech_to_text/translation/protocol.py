# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import TYPE_CHECKING, Literal, TypeAlias

from fastapi import UploadFile
from pydantic import (
    Field,
    model_validator,
)

from vllm.config.speech_to_text import SpeechToTextParams
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    OpenAIBaseModel,
    UsageInfo,
)
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.sampling_params import (
    BeamSearchParams,
    RequestOutputKind,
    SamplingParams,
)
from vllm.utils import random_uuid

from ..base.protocol import _LONG_INFO, AudioResponseFormat

if TYPE_CHECKING:
    import numpy as np

    from vllm.config import ModelConfig, SpeechToTextConfig

logger = init_logger(__name__)


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
    use_beam_search: bool = False
    """Whether or not beam search should be used."""

    n: int = 1
    """The number of beams to be used in beam search."""

    length_penalty: float = 1.0
    """Length penalty to be used for beam search."""

    include_stop_str_in_output: bool = False
    """Whether to include the stop strings in output text."""

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

    hotwords: str | None = None
    """
    hotwords refers to a list of important words or phrases that the model
    should pay extra attention to during transcription.
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

    def build_stt_params(
        self,
        audio: "np.ndarray",
        stt_config: "SpeechToTextConfig",
        model_config: "ModelConfig",
        task_type: str,
    ) -> SpeechToTextParams:
        return SpeechToTextParams(
            audio=audio,
            stt_config=stt_config,
            model_config=model_config,
            language=self.language,
            task_type=task_type,
            request_prompt=self.prompt,
            to_language=self.to_language,
            hotwords=self.hotwords,
        )

    def to_beam_search_params(
        self,
        default_max_tokens: int,
        default_sampling_params: dict | None = None,
    ) -> BeamSearchParams:
        if default_sampling_params is None:
            default_sampling_params = {}

        max_tokens = default_max_tokens
        n = self.n if self.n is not None else 1

        # NOTE: Temp 0 is a different fallback than completions
        if (temperature := self.temperature) is None:
            temperature = default_sampling_params.get("temperature", 0)

        return BeamSearchParams(
            beam_width=n,
            max_tokens=max_tokens,
            temperature=temperature,
            length_penalty=self.length_penalty,
            include_stop_str_in_output=self.include_stop_str_in_output,
        )

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

    avg_logprob: float
    """Average logprob of the segment.

    If the value is lower than -1, consider the logprobs failed.
    """

    compression_ratio: float
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
