# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from typing import Any, ClassVar, Literal, TypeAlias

import regex as re
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.logger import init_logger
from vllm.sampling_params import (
    SamplingParams,
)
from vllm.utils import random_uuid
from vllm.utils.import_utils import resolve_obj_by_qualname

logger = init_logger(__name__)


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


class FunctionCall(OpenAIBaseModel):
    # Internal field to preserve native tool call ID from tool parser.
    # Excluded from serialization to maintain OpenAI API compatibility
    # (function object should only contain 'name' and 'arguments').
    id: str | None = Field(default=None, exclude=True)
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


class SlowDownRequest(OpenAIBaseModel):
    """Request to slow down forward passes for benchmarking P/D disaggregation."""

    forward_sleep_time: float | None = Field(
        default=None,
        description=(
            "Sleep time in seconds before each forward pass. "
            "Set to null or <= 0 to disable. Only for benchmarking."
        ),
    )
