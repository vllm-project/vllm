# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from typing import Any, Literal, TypeAlias

from openai.types.responses import (
    ResponseCodeInterpreterCallCodeDeltaEvent,
    ResponseCodeInterpreterCallCodeDoneEvent,
    ResponseCodeInterpreterCallCompletedEvent,
    ResponseCodeInterpreterCallInProgressEvent,
    ResponseCodeInterpreterCallInterpretingEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseFunctionToolCall,
    ResponseInputItemParam,
    ResponseMcpCallArgumentsDeltaEvent,
    ResponseMcpCallArgumentsDoneEvent,
    ResponseMcpCallCompletedEvent,
    ResponseMcpCallInProgressEvent,
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponsePrompt,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseStatus,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
)
from openai.types.responses import (
    ResponseCompletedEvent as OpenAIResponseCompletedEvent,
)
from openai.types.responses import ResponseCreatedEvent as OpenAIResponseCreatedEvent
from openai.types.responses import (
    ResponseInProgressEvent as OpenAIResponseInProgressEvent,
)
from openai.types.responses.tool import Tool
from openai_harmony import Message as OpenAIHarmonyMessage

# Backward compatibility for OpenAI client versions
try:  # For older openai versions (< 1.100.0)
    from openai.types.responses import ResponseTextConfig
except ImportError:  # For newer openai versions (>= 1.100.0)
    from openai.types.responses import ResponseFormatTextConfig as ResponseTextConfig

from openai.types.responses.response import IncompleteDetails, ToolChoice
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai.types.shared import Metadata, Reasoning
from pydantic import (
    Field,
    ValidationError,
    field_serializer,
    model_validator,
)

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.entrypoints.openai.engine.protocol import (
    OpenAIBaseModel,
)
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.utils import random_uuid

logger = init_logger(__name__)


class InputTokensDetails(OpenAIBaseModel):
    cached_tokens: int
    input_tokens_per_turn: list[int] = Field(default_factory=list)
    cached_tokens_per_turn: list[int] = Field(default_factory=list)


class OutputTokensDetails(OpenAIBaseModel):
    reasoning_tokens: int = 0
    tool_output_tokens: int = 0
    output_tokens_per_turn: list[int] = Field(default_factory=list)
    tool_output_tokens_per_turn: list[int] = Field(default_factory=list)


class ResponseUsage(OpenAIBaseModel):
    input_tokens: int
    input_tokens_details: InputTokensDetails
    output_tokens: int
    output_tokens_details: OutputTokensDetails
    total_tokens: int


def serialize_message(msg):
    """
    Serializes a single message
    """
    if isinstance(msg, dict):
        return msg
    elif hasattr(msg, "to_dict"):
        return msg.to_dict()
    else:
        # fallback to pyandic dump
        return msg.model_dump_json()


def serialize_messages(msgs):
    """
    Serializes multiple messages
    """
    return [serialize_message(msg) for msg in msgs] if msgs else None


class ResponseRawMessageAndToken(OpenAIBaseModel):
    """Class to show the raw message.
    If message / tokens diverge, tokens is the source of truth"""

    message: str
    tokens: list[int]
    type: Literal["raw_message_tokens"] = "raw_message_tokens"


ResponseInputOutputMessage: TypeAlias = (
    list[ChatCompletionMessageParam] | list[ResponseRawMessageAndToken]
)
ResponseInputOutputItem: TypeAlias = ResponseInputItemParam | ResponseOutputItem


class ResponsesRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/responses/create
    background: bool | None = False
    include: (
        list[
            Literal[
                "code_interpreter_call.outputs",
                "computer_call_output.output.image_url",
                "file_search_call.results",
                "message.input_image.image_url",
                "message.output_text.logprobs",
                "reasoning.encrypted_content",
            ],
        ]
        | None
    ) = None
    input: str | list[ResponseInputOutputItem]
    instructions: str | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    metadata: Metadata | None = None
    model: str | None = None
    logit_bias: dict[str, float] | None = None
    parallel_tool_calls: bool | None = True
    previous_response_id: str | None = None
    prompt: ResponsePrompt | None = None
    reasoning: Reasoning | None = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] = "auto"
    store: bool | None = True
    stream: bool | None = False
    temperature: float | None = None
    text: ResponseTextConfig | None = None
    tool_choice: ToolChoice = "auto"
    tools: list[Tool] = Field(default_factory=list)
    top_logprobs: int | None = 0
    top_p: float | None = None
    top_k: int | None = None
    truncation: Literal["auto", "disabled"] | None = "disabled"
    user: str | None = None
    skip_special_tokens: bool = True

    # --8<-- [start:responses-extra-params]
    request_id: str = Field(
        default_factory=lambda: f"resp_{random_uuid()}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
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

    enable_response_messages: bool = Field(
        default=False,
        description=(
            "Dictates whether or not to return messages as part of the "
            "response object. Currently only supported for"
            "non-background and gpt-oss only. "
        ),
    )
    # similar to input_messages / output_messages in ResponsesResponse
    # we take in previous_input_messages (ie in harmony format)
    # this cannot be used in conjunction with previous_response_id
    # TODO: consider supporting non harmony messages as well
    previous_input_messages: list[OpenAIHarmonyMessage | dict] | None = None
    # --8<-- [end:responses-extra-params]

    _DEFAULT_SAMPLING_PARAMS = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
    }

    def to_sampling_params(
        self,
        default_max_tokens: int,
        default_sampling_params: dict | None = None,
    ) -> SamplingParams:
        if self.max_output_tokens is None:
            max_tokens = default_max_tokens
        else:
            max_tokens = min(self.max_output_tokens, default_max_tokens)

        default_sampling_params = default_sampling_params or {}
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
        stop_token_ids = default_sampling_params.get("stop_token_ids")

        # Structured output
        structured_outputs = None
        if self.text is not None and self.text.format is not None:
            response_format = self.text.format
            if (
                response_format.type == "json_schema"
                and response_format.schema_ is not None
            ):
                structured_outputs = StructuredOutputsParams(
                    json=response_format.schema_
                )
            elif response_format.type == "json_object":
                raise NotImplementedError("json_object is not supported")

        # TODO: add more parameters
        return SamplingParams.from_optional(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            logprobs=self.top_logprobs if self.is_include_output_logprobs() else None,
            stop_token_ids=stop_token_ids,
            output_kind=(
                RequestOutputKind.DELTA if self.stream else RequestOutputKind.FINAL_ONLY
            ),
            structured_outputs=structured_outputs,
            logit_bias=self.logit_bias,
            skip_clone=True,  # Created fresh per request, safe to skip clone
            skip_special_tokens=self.skip_special_tokens,
        )

    def is_include_output_logprobs(self) -> bool:
        """Check if the request includes output logprobs."""
        if self.include is None:
            return False
        return (
            isinstance(self.include, list)
            and "message.output_text.logprobs" in self.include
        )

    @model_validator(mode="before")
    def validate_background(cls, data):
        if not data.get("background"):
            return data
        if not data.get("store", True):
            raise ValueError("background can only be used when `store` is true")
        return data

    @model_validator(mode="before")
    def validate_prompt(cls, data):
        if data.get("prompt") is not None:
            raise VLLMValidationError(
                "prompt template is not supported", parameter="prompt"
            )
        return data

    @model_validator(mode="before")
    def check_cache_salt_support(cls, data):
        if data.get("cache_salt") is not None and (
            not isinstance(data["cache_salt"], str) or not data["cache_salt"]
        ):
            raise ValueError(
                "Parameter 'cache_salt' must be a non-empty string if provided."
            )
        return data

    @model_validator(mode="before")
    def function_call_parsing(cls, data):
        """Parse function_call dictionaries into ResponseFunctionToolCall objects.
        This ensures Pydantic can properly resolve union types in the input field.
        Function calls provided as dicts are converted to ResponseFunctionToolCall
        objects before validation, while invalid structures are left for Pydantic
        to reject with appropriate error messages.
        """

        input_data = data.get("input")

        # Early return for None, strings, or bytes
        # (strings are iterable but shouldn't be processed)
        if input_data is None or isinstance(input_data, (str, bytes)):
            return data

        # Convert iterators (like ValidatorIterator) to list
        if not isinstance(input_data, list):
            try:
                input_data = list(input_data)
            except TypeError:
                # Not iterable, leave as-is for Pydantic to handle
                return data

        processed_input = []
        for item in input_data:
            if isinstance(item, dict) and item.get("type") == "function_call":
                try:
                    processed_input.append(ResponseFunctionToolCall(**item))
                except ValidationError:
                    # Let Pydantic handle validation for malformed function calls
                    logger.debug(
                        "Failed to parse function_call to ResponseFunctionToolCall, "
                        "leaving for Pydantic validation"
                    )
                    processed_input.append(item)
            else:
                processed_input.append(item)

        data["input"] = processed_input
        return data


class ResponsesResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"resp_{random_uuid()}")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    # error: Optional[ResponseError] = None
    incomplete_details: IncompleteDetails | None = None
    instructions: str | None = None
    metadata: Metadata | None = None
    model: str
    object: Literal["response"] = "response"
    output: list[ResponseOutputItem]
    parallel_tool_calls: bool
    temperature: float
    tool_choice: ToolChoice
    tools: list[Tool]
    top_p: float
    background: bool
    max_output_tokens: int
    max_tool_calls: int | None = None
    previous_response_id: str | None = None
    prompt: ResponsePrompt | None = None
    reasoning: Reasoning | None = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"]
    status: ResponseStatus
    text: ResponseTextConfig | None = None
    top_logprobs: int | None = None
    truncation: Literal["auto", "disabled"]
    usage: ResponseUsage | None = None
    user: str | None = None

    # --8<-- [start:responses-response-extra-params]
    # These are populated when enable_response_messages is set to True
    # NOTE: custom serialization is needed
    # see serialize_input_messages and serialize_output_messages
    input_messages: ResponseInputOutputMessage | None = Field(
        default=None,
        description=(
            "If enable_response_messages, we can show raw token input to model."
        ),
    )
    output_messages: ResponseInputOutputMessage | None = Field(
        default=None,
        description=(
            "If enable_response_messages, we can show raw token output of model."
        ),
    )
    # --8<-- [end:responses-response-extra-params]

    # NOTE: openAI harmony doesn't serialize TextContent properly,
    # TODO: this fixes for TextContent, but need to verify for tools etc
    # https://github.com/openai/harmony/issues/78
    @field_serializer("output_messages", when_used="json")
    def serialize_output_messages(self, msgs, _info):
        return serialize_messages(msgs)

    # NOTE: openAI harmony doesn't serialize TextContent properly, this fixes it
    # https://github.com/openai/harmony/issues/78
    @field_serializer("input_messages", when_used="json")
    def serialize_input_messages(self, msgs, _info):
        return serialize_messages(msgs)

    @classmethod
    def from_request(
        cls,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        model_name: str,
        created_time: int,
        output: list[ResponseOutputItem],
        status: ResponseStatus,
        usage: ResponseUsage | None = None,
        input_messages: ResponseInputOutputMessage | None = None,
        output_messages: ResponseInputOutputMessage | None = None,
    ) -> "ResponsesResponse":
        incomplete_details: IncompleteDetails | None = None
        if status == "incomplete":
            incomplete_details = IncompleteDetails(reason="max_output_tokens")
        # TODO: implement the other reason for incomplete_details,
        # which is content_filter
        # incomplete_details = IncompleteDetails(reason='content_filter')
        return cls(
            id=request.request_id,
            created_at=created_time,
            incomplete_details=incomplete_details,
            instructions=request.instructions,
            metadata=request.metadata,
            model=model_name,
            output=output,
            input_messages=input_messages,
            output_messages=output_messages,
            parallel_tool_calls=request.parallel_tool_calls,
            temperature=sampling_params.temperature,
            tool_choice=request.tool_choice,
            tools=request.tools,
            top_p=sampling_params.top_p,
            background=request.background,
            max_output_tokens=sampling_params.max_tokens,
            max_tool_calls=request.max_tool_calls,
            previous_response_id=request.previous_response_id,
            prompt=request.prompt,
            reasoning=request.reasoning,
            service_tier=request.service_tier,
            status=status,
            text=request.text,
            top_logprobs=sampling_params.logprobs,
            truncation=request.truncation,
            user=request.user,
            usage=usage,
        )


# TODO: this code can be removed once
# https://github.com/openai/openai-python/issues/2634 has been resolved
class ResponseReasoningPartDoneEvent(OpenAIBaseModel):
    content_index: int
    """The index of the content part that is done."""

    item_id: str
    """The ID of the output item that the content part was added to."""

    output_index: int
    """The index of the output item that the content part was added to."""

    part: ResponseReasoningTextContent
    """The content part that is done."""

    sequence_number: int
    """The sequence number of this event."""

    type: Literal["response.reasoning_part.done"]
    """The type of the event. Always `response.reasoning_part.done`."""


# TODO: this code can be removed once
# https://github.com/openai/openai-python/issues/2634 has been resolved
class ResponseReasoningPartAddedEvent(OpenAIBaseModel):
    content_index: int
    """The index of the content part that is done."""

    item_id: str
    """The ID of the output item that the content part was added to."""

    output_index: int
    """The index of the output item that the content part was added to."""

    part: ResponseReasoningTextContent
    """The content part that is done."""

    sequence_number: int
    """The sequence number of this event."""

    type: Literal["response.reasoning_part.added"]
    """The type of the event. Always `response.reasoning_part.added`."""


# vLLM Streaming Events
# Note: we override the response type with the vLLM ResponsesResponse type
class ResponseCompletedEvent(OpenAIResponseCompletedEvent):
    response: ResponsesResponse  # type: ignore[override]


class ResponseCreatedEvent(OpenAIResponseCreatedEvent):
    response: ResponsesResponse  # type: ignore[override]


class ResponseInProgressEvent(OpenAIResponseInProgressEvent):
    response: ResponsesResponse  # type: ignore[override]


StreamingResponsesResponse: TypeAlias = (
    ResponseCreatedEvent
    | ResponseInProgressEvent
    | ResponseCompletedEvent
    | ResponseOutputItemAddedEvent
    | ResponseOutputItemDoneEvent
    | ResponseContentPartAddedEvent
    | ResponseContentPartDoneEvent
    | ResponseReasoningTextDeltaEvent
    | ResponseReasoningTextDoneEvent
    | ResponseReasoningPartAddedEvent
    | ResponseReasoningPartDoneEvent
    | ResponseCodeInterpreterCallInProgressEvent
    | ResponseCodeInterpreterCallCodeDeltaEvent
    | ResponseWebSearchCallInProgressEvent
    | ResponseWebSearchCallSearchingEvent
    | ResponseWebSearchCallCompletedEvent
    | ResponseCodeInterpreterCallCodeDoneEvent
    | ResponseCodeInterpreterCallInterpretingEvent
    | ResponseCodeInterpreterCallCompletedEvent
    | ResponseMcpCallArgumentsDeltaEvent
    | ResponseMcpCallArgumentsDoneEvent
    | ResponseMcpCallInProgressEvent
    | ResponseMcpCallCompletedEvent
)
