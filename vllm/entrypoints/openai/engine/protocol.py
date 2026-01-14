# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import json
import time
from http import HTTPStatus
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

import regex as re
import torch
from fastapi import HTTPException, UploadFile
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.sampling_params import (
    BeamSearchParams,
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.utils import random_uuid
from vllm.utils.import_utils import resolve_obj_by_qualname

logger = init_logger(__name__)

_LONG_INFO = torch.iinfo(torch.long)


class OpenAIBaseModel(BaseModel):
    # OpenAI API does allow extra fields
    model_config = ConfigDict(extra="allow")

    # Cache class field names
    field_names: ClassVar[set[str] | None] = None

    @model_validator(mode="wrap")
    @classmethod
    def __log_extra_fields__(cls, data, handler):
        result = handler(data)
        if not isinstance(data, dict):
            return result
        field_names = cls.field_names
        if field_names is None:
            # Get all class field names and their potential aliases
            field_names = set()
            for field_name, field in cls.model_fields.items():
                field_names.add(field_name)
                if alias := getattr(field, "alias", None):
                    field_names.add(alias)
            cls.field_names = field_names

        # Compare against both field names and aliases
        if any(k not in field_names for k in data):
            logger.warning(
                "The following fields were present in the request but ignored: %s",
                data.keys() - field_names,
            )
        return result


class ErrorInfo(OpenAIBaseModel):
    message: str
    type: str
    param: str | None = None
    code: int


class ErrorResponse(OpenAIBaseModel):
    error: ErrorInfo


class ModelPermission(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{random_uuid()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: str | None = None
    is_blocking: bool = False


class ModelCard(OpenAIBaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vllm"
    root: str | None = None
    parent: str | None = None
    max_model_len: int | None = None
    permission: list[ModelPermission] = Field(default_factory=list)


class ModelList(OpenAIBaseModel):
    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)


class PromptTokenUsageInfo(OpenAIBaseModel):
    cached_tokens: int | None = None


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0
    prompt_tokens_details: PromptTokenUsageInfo | None = None


class RequestResponseMetadata(BaseModel):
    request_id: str
    final_usage_info: UsageInfo | None = None


class JsonSchemaResponseFormat(OpenAIBaseModel):
    name: str
    description: str | None = None
    # schema is the field in openai but that causes conflicts with pydantic so
    # instead use json_schema with an alias
    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool | None = None


class LegacyStructuralTag(OpenAIBaseModel):
    begin: str
    # schema is the field, but that causes conflicts with pydantic so
    # instead use structural_tag_schema with an alias
    structural_tag_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    end: str


class LegacyStructuralTagResponseFormat(OpenAIBaseModel):
    type: Literal["structural_tag"]
    structures: list[LegacyStructuralTag]
    triggers: list[str]


class StructuralTagResponseFormat(OpenAIBaseModel):
    type: Literal["structural_tag"]
    format: Any


AnyStructuralTagResponseFormat: TypeAlias = (
    LegacyStructuralTagResponseFormat | StructuralTagResponseFormat
)


class ResponseFormat(OpenAIBaseModel):
    # type must be "json_schema", "json_object", or "text"
    type: Literal["text", "json_object", "json_schema"]
    json_schema: JsonSchemaResponseFormat | None = None


AnyResponseFormat: TypeAlias = (
    ResponseFormat | StructuralTagResponseFormat | LegacyStructuralTagResponseFormat
)


class StreamOptions(OpenAIBaseModel):
    include_usage: bool | None = True
    continuous_usage_stats: bool | None = False


class FunctionDefinition(OpenAIBaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


# extra="forbid" is a workaround to have kwargs as a field,
# see https://github.com/pydantic/pydantic/issues/3125
class LogitsProcessorConstructor(BaseModel):
    qualname: str
    args: list[Any] | None = None
    kwargs: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")


LogitsProcessors = list[str | LogitsProcessorConstructor]


def get_logits_processors(
    processors: LogitsProcessors | None, pattern: str | None
) -> list[Any] | None:
    if processors and pattern:
        logits_processors = []
        for processor in processors:
            qualname = processor if isinstance(processor, str) else processor.qualname
            if not re.match(pattern, qualname):
                raise ValueError(
                    f"Logits processor '{qualname}' is not allowed by this "
                    "server. See --logits-processor-pattern engine argument "
                    "for more information."
                )
            try:
                logits_processor = resolve_obj_by_qualname(qualname)
            except Exception as e:
                raise ValueError(
                    f"Logits processor '{qualname}' could not be resolved: {e}"
                ) from e
            if isinstance(processor, LogitsProcessorConstructor):
                logits_processor = logits_processor(
                    *processor.args or [], **processor.kwargs or {}
                )
            logits_processors.append(logits_processor)
        return logits_processors
    elif processors:
        raise ValueError(
            "The `logits_processors` argument is not supported by this "
            "server. See --logits-processor-pattern engine argument "
            "for more information."
        )
    return None


class CompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str | None = None
    prompt: list[int] | list[list[int]] | str | list[str] | None = None
    echo: bool | None = False
    frequency_penalty: float | None = 0.0
    logit_bias: dict[str, float] | None = None
    logprobs: int | None = None
    max_tokens: int | None = 16
    n: int = 1
    presence_penalty: float | None = 0.0
    seed: int | None = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    stop: str | list[str] | None = []
    stream: bool | None = False
    stream_options: StreamOptions | None = None
    suffix: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    user: str | None = None

    # --8<-- [start:completion-sampling-params]
    use_beam_search: bool = False
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    length_penalty: float = 1.0
    stop_token_ids: list[int] | None = []
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Annotated[int, Field(ge=-1, le=_LONG_INFO.max)] | None = (
        None
    )
    allowed_token_ids: list[int] | None = None
    prompt_logprobs: int | None = None
    # --8<-- [end:completion-sampling-params]

    # --8<-- [start:completion-extra-params]
    prompt_embeds: bytes | list[bytes] | None = None
    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."
        ),
    )
    response_format: AnyResponseFormat | None = Field(
        default=None,
        description=(
            "Similar to chat completion, this parameter specifies the format "
            "of output. Only {'type': 'json_object'}, {'type': 'json_schema'}"
            ", {'type': 'structural_tag'}, or {'type': 'text' } is supported."
        ),
    )
    structured_outputs: StructuredOutputsParams | None = Field(
        default=None,
        description="Additional kwargs for structured outputs",
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
        ),
    )
    request_id: str = Field(
        default_factory=random_uuid,
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    logits_processors: LogitsProcessors | None = Field(
        default=None,
        description=(
            "A list of either qualified names of logits processors, or "
            "constructor objects, to apply when sampling. A constructor is "
            "a JSON object with a required 'qualname' field specifying the "
            "qualified name of the processor class/factory, and optional "
            "'args' and 'kwargs' fields containing positional and keyword "
            "arguments. For example: {'qualname': "
            "'my_module.MyLogitsProcessor', 'args': [1, 2], 'kwargs': "
            "{'param': 'value'}}."
        ),
    )

    return_tokens_as_token_ids: bool | None = Field(
        default=None,
        description=(
            "If specified with 'logprobs', tokens are represented "
            " as strings of the form 'token_id:{token_id}' so that tokens "
            "that are not JSON-encodable can be identified."
        ),
    )
    return_token_ids: bool | None = Field(
        default=None,
        description=(
            "If specified, the result will include token IDs alongside the "
            "generated text. In streaming mode, prompt_token_ids is included "
            "only in the first chunk, and token_ids contains the delta tokens "
            "for each chunk. This is useful for debugging or when you "
            "need to map generated text back to input tokens."
        ),
    )

    cache_salt: str | None = Field(
        default=None,
        description=(
            "If specified, the prefix cache will be salted with the provided "
            "string to prevent an attacker to guess prompts in multi-user "
            "environments. The salt should be random, protected from "
            "access by 3rd parties, and long enough to be "
            "unpredictable (e.g., 43 characters base64-encoded, corresponding "
            "to 256 bit)."
        ),
    )

    kv_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description="KVTransfer parameters used for disaggregated serving.",
    )

    vllm_xargs: dict[str, str | int | float] | None = Field(
        default=None,
        description=(
            "Additional request parameters with string or "
            "numeric values, used by custom extensions."
        ),
    )

    # --8<-- [end:completion-extra-params]

    # Default sampling parameters for completion requests
    _DEFAULT_SAMPLING_PARAMS: dict = {
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
    }

    def to_beam_search_params(
        self,
        max_tokens: int,
        default_sampling_params: dict | None = None,
    ) -> BeamSearchParams:
        if default_sampling_params is None:
            default_sampling_params = {}
        n = self.n if self.n is not None else 1

        if (temperature := self.temperature) is None:
            temperature = default_sampling_params.get("temperature", 1.0)

        return BeamSearchParams(
            beam_width=n,
            max_tokens=max_tokens,
            ignore_eos=self.ignore_eos,
            temperature=temperature,
            length_penalty=self.length_penalty,
            include_stop_str_in_output=self.include_stop_str_in_output,
        )

    def to_sampling_params(
        self,
        max_tokens: int,
        logits_processor_pattern: str | None,
        default_sampling_params: dict | None = None,
    ) -> SamplingParams:
        if default_sampling_params is None:
            default_sampling_params = {}

        # Default parameters
        if (repetition_penalty := self.repetition_penalty) is None:
            repetition_penalty = default_sampling_params.get(
                "repetition_penalty",
                self._DEFAULT_SAMPLING_PARAMS["repetition_penalty"],
            )
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

        prompt_logprobs = self.prompt_logprobs
        if prompt_logprobs is None and self.echo:
            prompt_logprobs = self.logprobs

        echo_without_generation = self.echo and self.max_tokens == 0

        response_format = self.response_format
        if response_format is not None:
            # If structured outputs wasn't already enabled,
            # we must enable it for these features to work
            if self.structured_outputs is None:
                self.structured_outputs = StructuredOutputsParams()

            # Set structured output params for response format
            if response_format.type == "json_object":
                self.structured_outputs.json_object = True
            elif response_format.type == "json_schema":
                json_schema = response_format.json_schema
                assert json_schema is not None
                self.structured_outputs.json = json_schema.json_schema
            elif response_format.type == "structural_tag":
                structural_tag = response_format
                assert structural_tag is not None and isinstance(
                    structural_tag,
                    (
                        LegacyStructuralTagResponseFormat,
                        StructuralTagResponseFormat,
                    ),
                )
                s_tag_obj = structural_tag.model_dump(by_alias=True)
                self.structured_outputs.structural_tag = json.dumps(s_tag_obj)

        extra_args: dict[str, Any] = self.vllm_xargs if self.vllm_xargs else {}
        if self.kv_transfer_params:
            # Pass in kv_transfer_params via extra_args
            extra_args["kv_transfer_params"] = self.kv_transfer_params
        return SamplingParams.from_optional(
            n=self.n,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=self.seed,
            stop=self.stop,
            stop_token_ids=self.stop_token_ids,
            logprobs=self.logprobs,
            ignore_eos=self.ignore_eos,
            max_tokens=max_tokens if not echo_without_generation else 1,
            min_tokens=self.min_tokens,
            prompt_logprobs=prompt_logprobs,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            include_stop_str_in_output=self.include_stop_str_in_output,
            logits_processors=get_logits_processors(
                self.logits_processors, logits_processor_pattern
            ),
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            output_kind=RequestOutputKind.DELTA
            if self.stream
            else RequestOutputKind.FINAL_ONLY,
            structured_outputs=self.structured_outputs,
            logit_bias=self.logit_bias,
            allowed_token_ids=self.allowed_token_ids,
            extra_args=extra_args or None,
            skip_clone=True,  # Created fresh per request, safe to skip clone
        )

    @model_validator(mode="before")
    @classmethod
    def check_structured_outputs_count(cls, data):
        if data.get("structured_outputs", None) is None:
            return data

        structured_outputs_kwargs = data["structured_outputs"]
        count = sum(
            structured_outputs_kwargs.get(k) is not None
            for k in ("json", "regex", "choice")
        )
        if count > 1:
            raise VLLMValidationError(
                "You can only use one kind of constraints for structured "
                "outputs ('json', 'regex' or 'choice').",
                parameter="structured_outputs",
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if (prompt_logprobs := data.get("prompt_logprobs")) is not None:
            if data.get("stream") and (prompt_logprobs > 0 or prompt_logprobs == -1):
                raise VLLMValidationError(
                    "`prompt_logprobs` are not available when `stream=True`.",
                    parameter="prompt_logprobs",
                )

            if prompt_logprobs < 0 and prompt_logprobs != -1:
                raise VLLMValidationError(
                    "`prompt_logprobs` must be a positive value or -1.",
                    parameter="prompt_logprobs",
                    value=prompt_logprobs,
                )
        if (logprobs := data.get("logprobs")) is not None and logprobs < 0:
            raise VLLMValidationError(
                "`logprobs` must be a positive value.",
                parameter="logprobs",
                value=logprobs,
            )

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        if data.get("stream_options") and not data.get("stream"):
            raise VLLMValidationError(
                "Stream options can only be defined when `stream=True`.",
                parameter="stream_options",
            )

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_prompt_and_prompt_embeds(cls, data):
        prompt = data.get("prompt")
        prompt_embeds = data.get("prompt_embeds")

        prompt_is_empty = prompt is None or (isinstance(prompt, str) and prompt == "")
        embeds_is_empty = prompt_embeds is None or (
            isinstance(prompt_embeds, list) and len(prompt_embeds) == 0
        )

        if prompt_is_empty and embeds_is_empty:
            raise ValueError(
                "Either prompt or prompt_embeds must be provided and non-empty."
            )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_cache_salt_support(cls, data):
        if data.get("cache_salt") is not None and (
            not isinstance(data["cache_salt"], str) or not data["cache_salt"]
        ):
            raise ValueError(
                "Parameter 'cache_salt' must be a non-empty string if provided."
            )
        return data


class CompletionLogProbs(OpenAIBaseModel):
    text_offset: list[int] = Field(default_factory=list)
    token_logprobs: list[float | None] = Field(default_factory=list)
    tokens: list[str] = Field(default_factory=list)
    top_logprobs: list[dict[str, float] | None] = Field(default_factory=list)


class CompletionResponseChoice(OpenAIBaseModel):
    index: int
    text: str
    logprobs: CompletionLogProbs | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"
        ),
    )
    token_ids: list[int] | None = None  # For response
    prompt_logprobs: list[dict[int, Logprob] | None] | None = None
    prompt_token_ids: list[int] | None = None  # For prompt


class CompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None
    system_fingerprint: str | None = None
    usage: UsageInfo

    # vLLM-specific fields that are not in OpenAI spec
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None, description="KVTransfer parameters."
    )


class CompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    text: str
    logprobs: CompletionLogProbs | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"
        ),
    )
    # not part of the OpenAI spec but for tracing the tokens
    # prompt tokens is put into choice to align with CompletionResponseChoice
    prompt_token_ids: list[int] | None = None
    token_ids: list[int] | None = None


class CompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseStreamChoice]
    usage: UsageInfo | None = Field(default=None)


class FunctionCall(OpenAIBaseModel):
    name: str
    arguments: str


class ToolCall(OpenAIBaseModel):
    id: str = Field(default_factory=make_tool_call_id)
    type: Literal["function"] = "function"
    function: FunctionCall


class DeltaFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


# a tool call delta where everything is optional
class DeltaToolCall(OpenAIBaseModel):
    id: str | None = None
    type: Literal["function"] | None = None
    index: int
    function: DeltaFunctionCall | None = None


class ExtractedToolCallInformation(BaseModel):
    # indicate if tools were called
    tools_called: bool

    # extracted tool calls
    tool_calls: list[ToolCall]

    # content - per OpenAI spec, content AND tool calls can be returned rarely
    # But some models will do this intentionally
    content: str | None = None


class DeltaMessage(OpenAIBaseModel):
    role: str | None = None
    content: str | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    """Deprecated: use `reasoning` instead."""
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)

    @model_validator(mode="after")
    def handle_deprecated_reasoning_content(self):
        """Copy reasoning to reasoning_content for backward compatibility."""
        self.reasoning_content = self.reasoning
        return self


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


####### Tokens IN <> Tokens OUT #######
class GenerateRequest(BaseModel):
    request_id: str = Field(
        default_factory=random_uuid,
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    token_ids: list[int]
    """The token ids to generate text from."""

    # features: MultiModalFeatureSpec
    # TODO (NickLucche): implement once Renderer work is completed
    features: str | None = None
    """The processed MM inputs for the model."""

    sampling_params: SamplingParams
    """The sampling parameters for the model."""

    model: str | None = None

    stream: bool | None = False
    stream_options: StreamOptions | None = None
    cache_salt: str | None = Field(
        default=None,
        description=(
            "If specified, the prefix cache will be salted with the provided "
            "string to prevent an attacker to guess prompts in multi-user "
            "environments. The salt should be random, protected from "
            "access by 3rd parties, and long enough to be "
            "unpredictable (e.g., 43 characters base64-encoded, corresponding "
            "to 256 bit)."
        ),
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
        ),
    )
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description="KVTransfer parameters used for disaggregated serving.",
    )
