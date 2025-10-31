# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import json
import time
from http import HTTPStatus
from typing import Annotated, Any, ClassVar, Generic, Literal, TypeAlias, TypeVar

import regex as re
import torch
from fastapi import HTTPException, UploadFile
from openai.types.chat.chat_completion_audio import (
    ChatCompletionAudio as OpenAIChatCompletionAudio,
)
from openai.types.chat.chat_completion_message import Annotation as OpenAIAnnotation
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
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponsePrompt,
    ResponseReasoningItem,
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
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai_harmony import Message as OpenAIHarmonyMessage

from vllm.config.pooler import get_use_activation
from vllm.tasks import PoolingTask
from vllm.utils.serial_utils import (
    EmbedDType,
    EncodingFormat,
    Endianness,
)

# Backward compatibility for OpenAI client versions
try:  # For older openai versions (< 1.100.0)
    from openai.types.responses import ResponseTextConfig
except ImportError:  # For newer openai versions (>= 1.100.0)
    from openai.types.responses import ResponseFormatTextConfig as ResponseTextConfig


from openai.types.responses.response import IncompleteDetails, ToolChoice
from openai.types.responses.tool import Tool
from openai.types.shared import Metadata, Reasoning
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from vllm import envs
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam, make_tool_call_id
from vllm.entrypoints.score_utils import ScoreContentPartParam, ScoreMultiModalParam
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.pooling_params import PoolingParams
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


class ChatCompletionToolsParam(OpenAIBaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionNamedFunction(OpenAIBaseModel):
    name: str


class ChatCompletionNamedToolChoiceParam(OpenAIBaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"


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


ResponseInputOutputItem: TypeAlias = (
    ResponseInputItemParam | ResponseReasoningItem | ResponseFunctionToolCall
)


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
    truncation: Literal["auto", "disabled"] | None = "disabled"
    user: str | None = None

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
            "to 256 bit). Not supported by vLLM engine V0."
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
            max_tokens=max_tokens,
            logprobs=self.top_logprobs if self.is_include_output_logprobs() else None,
            stop_token_ids=stop_token_ids,
            output_kind=(
                RequestOutputKind.DELTA if self.stream else RequestOutputKind.FINAL_ONLY
            ),
            structured_outputs=structured_outputs,
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
            raise ValueError("prompt template is not supported")
        return data

    @model_validator(mode="before")
    def check_cache_salt_support(cls, data):
        if data.get("cache_salt") is not None:
            if not envs.VLLM_USE_V1:
                raise ValueError(
                    "Parameter 'cache_salt' is not supported with "
                    "this instance of vLLM, which uses engine V0."
                )
            if not isinstance(data["cache_salt"], str) or not data["cache_salt"]:
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


class ChatCompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: list[ChatCompletionMessageParam]
    model: str | None = None
    frequency_penalty: float | None = 0.0
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = False
    top_logprobs: int | None = 0
    max_tokens: int | None = Field(
        default=None,
        deprecated="max_tokens is deprecated in favor of "
        "the max_completion_tokens field",
    )
    max_completion_tokens: int | None = None
    n: int | None = 1
    presence_penalty: float | None = 0.0
    response_format: AnyResponseFormat | None = None
    seed: int | None = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    stop: str | list[str] | None = []
    stream: bool | None = False
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[ChatCompletionToolsParam] | None = None
    tool_choice: (
        Literal["none"]
        | Literal["auto"]
        | Literal["required"]
        | ChatCompletionNamedToolChoiceParam
        | None
    ) = "none"
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    include_reasoning: bool = True

    # NOTE this will be ignored by vLLM -- the model determines the behavior
    parallel_tool_calls: bool | None = False
    user: str | None = None

    # --8<-- [start:chat-completion-sampling-params]
    best_of: int | None = None
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
    truncate_prompt_tokens: Annotated[int, Field(ge=-1)] | None = None
    prompt_logprobs: int | None = None
    allowed_token_ids: list[int] | None = None
    bad_words: list[str] = Field(default_factory=list)
    # --8<-- [end:chat-completion-sampling-params]

    # --8<-- [start:chat-completion-extra-params]
    echo: bool = Field(
        default=False,
        description=(
            "If true, the new message will be prepended with the last message "
            "if they belong to the same role."
        ),
    )
    add_generation_prompt: bool = Field(
        default=True,
        description=(
            "If true, the generation prompt will be added to the chat template. "
            "This is a parameter used by chat template in tokenizer config of the "
            "model."
        ),
    )
    continue_final_message: bool = Field(
        default=False,
        description=(
            "If this is set, the chat will be formatted so that the final "
            "message in the chat is open-ended, without any EOS tokens. The "
            "model will continue this message rather than starting a new one. "
            'This allows you to "prefill" part of the model\'s response for it. '
            "Cannot be used at the same time as `add_generation_prompt`."
        ),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."
        ),
    )
    documents: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "A list of dicts representing documents that will be accessible to "
            "the model if it is performing RAG (retrieval-augmented generation)."
            " If the template does not support RAG, this argument will have no "
            "effect. We recommend that each document should be a dict containing "
            '"title" and "text" keys.'
        ),
    )
    chat_template: str | None = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."
        ),
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the template renderer. "
            "Will be accessible by the chat template."
        ),
    )
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    structured_outputs: StructuredOutputsParams | None = Field(
        default=None,
        description="Additional kwargs for structured outputs",
    )
    guided_json: str | dict | BaseModel | None = Field(
        default=None,
        description=(
            "`guided_json` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please pass `json` to `structured_outputs` instead."
        ),
    )
    guided_regex: str | None = Field(
        default=None,
        description=(
            "`guided_regex` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please pass `regex` to `structured_outputs` instead."
        ),
    )
    guided_choice: list[str] | None = Field(
        default=None,
        description=(
            "`guided_choice` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please pass `choice` to `structured_outputs` instead."
        ),
    )
    guided_grammar: str | None = Field(
        default=None,
        description=(
            "`guided_grammar` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please pass `grammar` to `structured_outputs` instead."
        ),
    )
    structural_tag: str | None = Field(
        default=None,
        description=(
            "`structural_tag` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please pass `structural_tag` to `structured_outputs` instead."
        ),
    )
    guided_decoding_backend: str | None = Field(
        default=None,
        description=(
            "`guided_decoding_backend` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please remove it from your request."
        ),
    )
    guided_whitespace_pattern: str | None = Field(
        default=None,
        description=(
            "`guided_whitespace_pattern` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please pass `whitespace_pattern` to `structured_outputs` instead."
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
    request_id: str = Field(
        default_factory=lambda: f"{random_uuid()}",
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
            "to 256 bit). Not supported by vLLM engine V0."
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

    # --8<-- [end:chat-completion-extra-params]

    # Default sampling parameters for chat completion requests
    _DEFAULT_SAMPLING_PARAMS: dict = {
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
    }

    def to_beam_search_params(
        self, max_tokens: int, default_sampling_params: dict
    ) -> BeamSearchParams:
        n = self.n if self.n is not None else 1
        if (temperature := self.temperature) is None:
            temperature = default_sampling_params.get(
                "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"]
            )

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
        default_sampling_params: dict,
    ) -> SamplingParams:
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
            prompt_logprobs = self.top_logprobs

        # Forward deprecated guided_* parameters to structured_outputs
        if self.structured_outputs is None:
            kwargs = dict[str, Any](
                json=self.guided_json,
                regex=self.guided_regex,
                choice=self.guided_choice,
                grammar=self.guided_grammar,
                whitespace_pattern=self.guided_whitespace_pattern,
                structural_tag=self.structural_tag,
            )
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            if len(kwargs) > 0:
                self.structured_outputs = StructuredOutputsParams(**kwargs)

        response_format = self.response_format
        if response_format is not None:
            # If structured outputs wasn't already enabled,
            # we must enable it for these features to work
            if self.structured_outputs is None:
                self.structured_outputs = StructuredOutputsParams()

            # Set structured output params for response format
            if response_format is not None:
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
            best_of=self.best_of,
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
            logprobs=self.top_logprobs if self.logprobs else None,
            prompt_logprobs=prompt_logprobs,
            ignore_eos=self.ignore_eos,
            max_tokens=max_tokens,
            min_tokens=self.min_tokens,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            logits_processors=get_logits_processors(
                self.logits_processors, logits_processor_pattern
            ),
            include_stop_str_in_output=self.include_stop_str_in_output,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            output_kind=RequestOutputKind.DELTA
            if self.stream
            else RequestOutputKind.FINAL_ONLY,
            structured_outputs=self.structured_outputs,
            logit_bias=self.logit_bias,
            bad_words=self.bad_words,
            allowed_token_ids=self.allowed_token_ids,
            extra_args=extra_args or None,
        )

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError("Stream options can only be defined when `stream=True`.")

        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if (prompt_logprobs := data.get("prompt_logprobs")) is not None:
            if data.get("stream") and (prompt_logprobs > 0 or prompt_logprobs == -1):
                raise ValueError(
                    "`prompt_logprobs` are not available when `stream=True`."
                )

            if prompt_logprobs < 0 and prompt_logprobs != -1:
                raise ValueError("`prompt_logprobs` must be a positive value or -1.")
            if prompt_logprobs == -1 and not envs.VLLM_USE_V1:
                raise ValueError(
                    "`prompt_logprobs=-1` is only supported with vLLM engine V1."
                )
        if (top_logprobs := data.get("top_logprobs")) is not None:
            if top_logprobs < 0 and top_logprobs != -1:
                raise ValueError("`top_logprobs` must be a positive value or -1.")

            if (top_logprobs == -1 or top_logprobs > 0) and not data.get("logprobs"):
                raise ValueError(
                    "when using `top_logprobs`, `logprobs` must be set to true."
                )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_structured_outputs_count(cls, data):
        if isinstance(data, ValueError):
            raise data

        if data.get("structured_outputs", None) is None:
            return data

        structured_outputs_kwargs = data["structured_outputs"]
        count = sum(
            structured_outputs_kwargs.get(k) is not None
            for k in ("json", "regex", "choice")
        )
        # you can only use one kind of constraints for structured outputs
        if count > 1:
            raise ValueError(
                "You can only use one kind of constraints for structured "
                "outputs ('json', 'regex' or 'choice')."
            )
        # you can only either use structured outputs or tools, not both
        if count > 1 and data.get("tool_choice", "none") not in (
            "none",
            "auto",
            "required",
        ):
            raise ValueError(
                "You can only either use constraints for structured outputs "
                "or tools, not both."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_tool_usage(cls, data):
        # if "tool_choice" is not specified but tools are provided,
        # default to "auto" tool_choice
        if "tool_choice" not in data and data.get("tools"):
            data["tool_choice"] = "auto"

        # if "tool_choice" is "none" -- no validation is needed for tools
        if "tool_choice" in data and data["tool_choice"] == "none":
            return data

        # if "tool_choice" is specified -- validation
        if "tool_choice" in data and data["tool_choice"] is not None:
            # ensure that if "tool choice" is specified, tools are present
            if "tools" not in data or data["tools"] is None:
                raise ValueError("When using `tool_choice`, `tools` must be set.")

            # make sure that tool choice is either a named tool
            # OR that it's set to "auto" or "required"
            if data["tool_choice"] not in ["auto", "required"] and not isinstance(
                data["tool_choice"], dict
            ):
                raise ValueError(
                    f"Invalid value for `tool_choice`: {data['tool_choice']}! "
                    'Only named tools, "none", "auto" or "required" '
                    "are supported."
                )

            # if tool_choice is "required" but the "tools" list is empty,
            # override the data to behave like "none" to align with
            # OpenAI’s behavior.
            if (
                data["tool_choice"] == "required"
                and isinstance(data["tools"], list)
                and len(data["tools"]) == 0
            ):
                data["tool_choice"] = "none"
                del data["tools"]
                return data

            # ensure that if "tool_choice" is specified as an object,
            # it matches a valid tool
            correct_usage_message = (
                'Correct usage: `{"type": "function",'
                ' "function": {"name": "my_function"}}`'
            )
            if isinstance(data["tool_choice"], dict):
                valid_tool = False
                function = data["tool_choice"].get("function")
                if not isinstance(function, dict):
                    raise ValueError(
                        f"Invalid value for `function`: `{function}` in "
                        f"`tool_choice`! {correct_usage_message}"
                    )
                if "name" not in function:
                    raise ValueError(
                        f"Expected field `name` in `function` in "
                        f"`tool_choice`! {correct_usage_message}"
                    )
                function_name = function["name"]
                if not isinstance(function_name, str) or len(function_name) == 0:
                    raise ValueError(
                        f"Invalid `name` in `function`: `{function_name}`"
                        f" in `tool_choice`! {correct_usage_message}"
                    )
                for tool in data["tools"]:
                    if tool["function"]["name"] == function_name:
                        valid_tool = True
                        break
                if not valid_tool:
                    raise ValueError(
                        "The tool specified in `tool_choice` does not match any"
                        " of the specified `tools`"
                    )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_generation_prompt(cls, data):
        if data.get("continue_final_message") and data.get("add_generation_prompt"):
            raise ValueError(
                "Cannot set both `continue_final_message` and "
                "`add_generation_prompt` to True."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_cache_salt_support(cls, data):
        if data.get("cache_salt") is not None:
            if not envs.VLLM_USE_V1:
                raise ValueError(
                    "Parameter 'cache_salt' is not supported with "
                    "this instance of vLLM, which uses engine V0."
                )
            if not isinstance(data["cache_salt"], str) or not data["cache_salt"]:
                raise ValueError(
                    "Parameter 'cache_salt' must be a non-empty string if provided."
                )
        return data


class CompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str | None = None
    prompt: list[int] | list[list[int]] | str | list[str] | None = None
    best_of: int | None = None
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
    truncate_prompt_tokens: Annotated[int, Field(ge=-1)] | None = None
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
    guided_json: str | dict | BaseModel | None = Field(
        default=None,
        description=(
            "`guided_json` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please pass `json` to `structured_outputs` instead."
        ),
    )
    guided_regex: str | None = Field(
        default=None,
        description=(
            "`guided_regex` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please pass `regex` to `structured_outputs` instead."
        ),
    )
    guided_choice: list[str] | None = Field(
        default=None,
        description=(
            "`guided_choice` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please pass `choice` to `structured_outputs` instead."
        ),
    )
    guided_grammar: str | None = Field(
        default=None,
        description=(
            "`guided_grammar` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please pass `grammar` to `structured_outputs` instead."
        ),
    )
    structural_tag: str | None = Field(
        default=None,
        description=("If specified, the output will follow the structural tag schema."),
    )
    guided_decoding_backend: str | None = Field(
        default=None,
        description=(
            "`guided_decoding_backend` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please remove it from your request."
        ),
    )
    guided_whitespace_pattern: str | None = Field(
        default=None,
        description=(
            "`guided_whitespace_pattern` is deprecated. "
            "This will be removed in v0.12.0 or v1.0.0, whichever is soonest. "
            "Please pass `whitespace_pattern` to `structured_outputs` instead."
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
    request_id: str = Field(
        default_factory=lambda: f"{random_uuid()}",
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
            "to 256 bit). Not supported by vLLM engine V0."
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

        guided_json_object = None
        if self.response_format is not None:
            if self.response_format.type == "json_object":
                guided_json_object = True
            elif self.response_format.type == "json_schema":
                json_schema = self.response_format.json_schema
                assert json_schema is not None
                self.guided_json = json_schema.json_schema
            elif self.response_format.type == "structural_tag":
                structural_tag = self.response_format
                assert structural_tag is not None and isinstance(
                    structural_tag, StructuralTagResponseFormat
                )
                s_tag_obj = structural_tag.model_dump(by_alias=True)
                self.structural_tag = json.dumps(s_tag_obj)

        # Forward deprecated guided_* parameters to structured_outputs
        if self.structured_outputs is None:
            kwargs = dict[str, Any](
                json=self.guided_json,
                json_object=guided_json_object,
                regex=self.guided_regex,
                choice=self.guided_choice,
                grammar=self.guided_grammar,
                whitespace_pattern=self.guided_whitespace_pattern,
            )
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            if len(kwargs) > 0:
                self.structured_outputs = StructuredOutputsParams(**kwargs)

        extra_args: dict[str, Any] = self.vllm_xargs if self.vllm_xargs else {}
        if self.kv_transfer_params:
            # Pass in kv_transfer_params via extra_args
            extra_args["kv_transfer_params"] = self.kv_transfer_params
        return SamplingParams.from_optional(
            n=self.n,
            best_of=self.best_of,
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
            raise ValueError(
                "You can only use one kind of constraints for structured "
                "outputs ('json', 'regex' or 'choice')."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if (prompt_logprobs := data.get("prompt_logprobs")) is not None:
            if data.get("stream") and (prompt_logprobs > 0 or prompt_logprobs == -1):
                raise ValueError(
                    "`prompt_logprobs` are not available when `stream=True`."
                )

            if prompt_logprobs < 0 and prompt_logprobs != -1:
                raise ValueError("`prompt_logprobs` must be a positive value or -1.")
            if prompt_logprobs == -1 and not envs.VLLM_USE_V1:
                raise ValueError(
                    "`prompt_logprobs=-1` is only supported with vLLM engine V1."
                )
        if (logprobs := data.get("logprobs")) is not None and logprobs < 0:
            raise ValueError("`logprobs` must be a positive value.")

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError("Stream options can only be defined when `stream=True`.")

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
        if data.get("cache_salt") is not None:
            if not envs.VLLM_USE_V1:
                raise ValueError(
                    "Parameter 'cache_salt' is not supported with "
                    "this instance of vLLM, which uses engine V0."
                )
            if not isinstance(data["cache_salt"], str) or not data["cache_salt"]:
                raise ValueError(
                    "Parameter 'cache_salt' must be a non-empty string if provided."
                )
        return data


class EmbeddingCompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/embeddings
    model: str | None = None
    input: list[int] | list[list[int]] | str | list[str]
    encoding_format: EncodingFormat = "float"
    dimensions: int | None = None
    user: str | None = None
    truncate_prompt_tokens: Annotated[int, Field(ge=-1)] | None = None

    # --8<-- [start:embedding-extra-params]
    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."
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
    request_id: str = Field(
        default_factory=lambda: f"{random_uuid()}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    normalize: bool | None = Field(
        default=None,
        description="Whether to normalize the embeddings outputs. Default is True.",
    )
    embed_dtype: EmbedDType = Field(
        default="float32",
        description=(
            "What dtype to use for encoding. Default to using float32 for base64 "
            "encoding to match the OpenAI python client behavior. "
            "This parameter will affect base64 and binary_response."
        ),
    )
    endianness: Endianness = Field(
        default="native",
        description=(
            "What endianness to use for encoding. Default to using native for "
            "base64 encoding to match the OpenAI python client behavior."
            "This parameter will affect base64 and binary_response."
        ),
    )
    # --8<-- [end:embedding-extra-params]

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            dimensions=self.dimensions,
            normalize=self.normalize,
        )


class EmbeddingChatRequest(OpenAIBaseModel):
    model: str | None = None
    messages: list[ChatCompletionMessageParam]

    encoding_format: EncodingFormat = "float"
    dimensions: int | None = None
    user: str | None = None
    truncate_prompt_tokens: Annotated[int, Field(ge=-1)] | None = None

    # --8<-- [start:chat-embedding-extra-params]
    add_generation_prompt: bool = Field(
        default=False,
        description=(
            "If true, the generation prompt will be added to the chat template. "
            "This is a parameter used by chat template in tokenizer config of the "
            "model."
        ),
    )

    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."
        ),
    )
    chat_template: str | None = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."
        ),
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the template renderer. "
            "Will be accessible by the chat template."
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
    request_id: str = Field(
        default_factory=lambda: f"{random_uuid()}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    normalize: bool | None = Field(
        default=None,
        description="Whether to normalize the embeddings outputs. Default is True.",
    )
    embed_dtype: EmbedDType = Field(
        default="float32",
        description=(
            "What dtype to use for encoding. Default to using float32 for base64 "
            "encoding to match the OpenAI python client behavior. "
            "This parameter will affect base64 and binary_response."
        ),
    )
    endianness: Endianness = Field(
        default="native",
        description=(
            "What endianness to use for encoding. Default to using native for "
            "base64 encoding to match the OpenAI python client behavior."
            "This parameter will affect base64 and binary_response."
        ),
    )
    # --8<-- [end:chat-embedding-extra-params]

    @model_validator(mode="before")
    @classmethod
    def check_generation_prompt(cls, data):
        if data.get("continue_final_message") and data.get("add_generation_prompt"):
            raise ValueError(
                "Cannot set both `continue_final_message` and "
                "`add_generation_prompt` to True."
            )
        return data

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            dimensions=self.dimensions,
            normalize=self.normalize,
        )


EmbeddingRequest: TypeAlias = EmbeddingCompletionRequest | EmbeddingChatRequest


class PoolingCompletionRequest(EmbeddingCompletionRequest):
    task: PoolingTask | None = None
    softmax: bool | None = Field(
        default=None,
        description="softmax will be deprecated, please use use_activation instead.",
    )
    activation: bool | None = Field(
        default=None,
        description="activation will be deprecated, please use use_activation instead.",
    )
    use_activation: bool | None = Field(
        default=None,
        description="Whether to use activation for classification outputs. "
        "If it is a classify or token_classify task, the default is True; "
        "for other tasks, this value should be None.",
    )

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            dimensions=self.dimensions,
            normalize=self.normalize,
            use_activation=get_use_activation(self),
        )


class PoolingChatRequest(EmbeddingChatRequest):
    task: PoolingTask | None = None
    softmax: bool | None = Field(
        default=None,
        description="softmax will be deprecated, please use use_activation instead.",
    )
    activation: bool | None = Field(
        default=None,
        description="activation will be deprecated, please use use_activation instead.",
    )
    use_activation: bool | None = Field(
        default=None,
        description="Whether to use activation for classification outputs. "
        "If it is a classify or token_classify task, the default is True; "
        "for other tasks, this value should be None.",
    )

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            dimensions=self.dimensions,
            normalize=self.normalize,
            use_activation=get_use_activation(self),
        )


T = TypeVar("T")


class IOProcessorRequest(OpenAIBaseModel, Generic[T]):
    model: str | None = None

    priority: int = Field(default=0)
    """
    The priority of the request (lower means earlier handling;
    default: 0). Any priority other than 0 will raise an error
    if the served model does not use priority scheduling.
    """
    data: T

    task: PoolingTask = "plugin"
    encoding_format: EncodingFormat = "float"
    embed_dtype: EmbedDType = Field(
        default="float32",
        description=(
            "What dtype to use for encoding. Default to using float32 for base64 "
            "encoding to match the OpenAI python client behavior. "
            "This parameter will affect base64 and binary_response."
        ),
    )
    endianness: Endianness = Field(
        default="native",
        description=(
            "What endianness to use for encoding. Default to using native for "
            "base64 encoding to match the OpenAI python client behavior."
            "This parameter will affect base64 and binary_response."
        ),
    )

    def to_pooling_params(self):
        return PoolingParams()


class IOProcessorResponse(OpenAIBaseModel, Generic[T]):
    request_id: str | None = None
    """
    The request_id associated with this response
    """
    created_at: int = Field(default_factory=lambda: int(time.time()))

    data: T
    """
    When using plugins IOProcessor plugins, the actual output is generated
    by the plugin itself. Hence, we use a generic type for the response data
    """


PoolingRequest: TypeAlias = (
    PoolingCompletionRequest | PoolingChatRequest | IOProcessorRequest
)


class ScoreRequest(OpenAIBaseModel):
    model: str | None = None
    text_1: list[str] | str | ScoreMultiModalParam
    text_2: list[str] | str | ScoreMultiModalParam
    truncate_prompt_tokens: Annotated[int, Field(ge=-1)] | None = None

    # --8<-- [start:score-extra-params]

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

    softmax: bool | None = Field(
        default=None,
        description="softmax will be deprecated, please use use_activation instead.",
    )

    activation: bool | None = Field(
        default=None,
        description="activation will be deprecated, please use use_activation instead.",
    )

    use_activation: bool | None = Field(
        default=None,
        description="Whether to use activation for classification outputs. "
        "Default is True.",
    )
    # --8<-- [end:score-extra-params]

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            use_activation=get_use_activation(self),
        )


class RerankRequest(OpenAIBaseModel):
    model: str | None = None
    query: str | ScoreMultiModalParam
    documents: list[str] | ScoreMultiModalParam
    top_n: int = Field(default_factory=lambda: 0)
    truncate_prompt_tokens: Annotated[int, Field(ge=-1)] | None = None

    # --8<-- [start:rerank-extra-params]

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

    softmax: bool | None = Field(
        default=None,
        description="softmax will be deprecated, please use use_activation instead.",
    )

    activation: bool | None = Field(
        default=None,
        description="activation will be deprecated, please use use_activation instead.",
    )

    use_activation: bool | None = Field(
        default=None,
        description="Whether to use activation for classification outputs. "
        "Default is True.",
    )
    # --8<-- [end:rerank-extra-params]

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            use_activation=get_use_activation(self),
        )


class RerankDocument(BaseModel):
    text: str | None = None
    multi_modal: ScoreContentPartParam | None = None


class RerankResult(BaseModel):
    index: int
    document: RerankDocument
    relevance_score: float


class RerankUsage(BaseModel):
    total_tokens: int


class RerankResponse(OpenAIBaseModel):
    id: str
    model: str
    usage: RerankUsage
    results: list[RerankResult]


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


class EmbeddingResponseData(OpenAIBaseModel):
    index: int
    object: str = "embedding"
    embedding: list[float] | str


class EmbeddingResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"embd-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[EmbeddingResponseData]
    usage: UsageInfo


class EmbeddingBytesResponse(OpenAIBaseModel):
    body: list[bytes]
    metadata: str
    media_type: str = "application/octet-stream"


class PoolingResponseData(OpenAIBaseModel):
    index: int
    object: str = "pooling"
    data: list[list[float]] | list[float] | str


class PoolingResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"pool-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[PoolingResponseData]
    usage: UsageInfo


class PoolingBytesResponse(OpenAIBaseModel):
    body: list[bytes]
    metadata: str
    media_type: str = "application/octet-stream"


class ScoreResponseData(OpenAIBaseModel):
    index: int
    object: str = "score"
    score: float


class ScoreResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"embd-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[ScoreResponseData]
    usage: UsageInfo


class ClassificationRequest(OpenAIBaseModel):
    model: str | None = None
    input: list[str] | str
    truncate_prompt_tokens: int | None = None
    user: str | None = None

    # --8<-- [start:classification-extra-params]
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
        ),
    )

    softmax: bool | None = Field(
        default=None,
        description="softmax will be deprecated, please use use_activation instead.",
    )

    activation: bool | None = Field(
        default=None,
        description="activation will be deprecated, please use use_activation instead.",
    )

    use_activation: bool | None = Field(
        default=None,
        description="Whether to use activation for classification outputs. "
        "Default is True.",
    )
    # --8<-- [end:classification-extra-params]

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            use_activation=get_use_activation(self),
        )


class ClassificationData(OpenAIBaseModel):
    index: int
    label: str | None
    probs: list[float]
    num_classes: int


class ClassificationResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"classify-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[ClassificationData]
    usage: UsageInfo


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


class ChatMessage(OpenAIBaseModel):
    role: str
    content: str | None = None
    refusal: str | None = None
    annotations: OpenAIAnnotation | None = None
    audio: OpenAIChatCompletionAudio | None = None
    function_call: FunctionCall | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    # vLLM-specific fields that are not in OpenAI spec
    reasoning_content: str | None = None


class ChatCompletionLogProb(OpenAIBaseModel):
    token: str
    logprob: float = -9999.0
    bytes: list[int] | None = None


class ChatCompletionLogProbsContent(ChatCompletionLogProb):
    # Workaround: redefine fields name cache so that it's not
    # shared with the super class.
    field_names: ClassVar[set[str] | None] = None
    top_logprobs: list[ChatCompletionLogProb] = Field(default_factory=list)


class ChatCompletionLogProbs(OpenAIBaseModel):
    content: list[ChatCompletionLogProbsContent] | None = None


class ChatCompletionResponseChoice(OpenAIBaseModel):
    index: int
    message: ChatMessage
    logprobs: ChatCompletionLogProbs | None = None
    # per OpenAI spec this is the default
    finish_reason: str | None = "stop"
    # not part of the OpenAI spec but included in vLLM for legacy reasons
    stop_reason: int | str | None = None
    # not part of the OpenAI spec but is useful for tracing the tokens
    # in agent scenarios
    token_ids: list[int] | None = None


class ChatCompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None
    system_fingerprint: str | None = None
    usage: UsageInfo

    # vLLM-specific fields that are not in OpenAI spec
    prompt_logprobs: list[dict[int, Logprob] | None] | None = None
    prompt_token_ids: list[int] | None = None
    kv_transfer_params: dict[str, Any] | None = Field(
        default=None, description="KVTransfer parameters."
    )


class DeltaMessage(OpenAIBaseModel):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)


class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    delta: DeltaMessage
    logprobs: ChatCompletionLogProbs | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    # not part of the OpenAI spec but for tracing the tokens
    token_ids: list[int] | None = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
    usage: UsageInfo | None = Field(default=None)
    # not part of the OpenAI spec but for tracing the tokens
    prompt_token_ids: list[int] | None = None


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

    # --8<-- [start:responses-extra-params]
    # These are populated when enable_response_messages is set to True
    # NOTE: custom serialization is needed
    # see serialize_input_messages and serialize_output_messages
    input_messages: list[ChatCompletionMessageParam] | None = None
    output_messages: list[ChatCompletionMessageParam] | None = None
    # --8<-- [end:responses-extra-params]

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
        input_messages: list[ChatCompletionMessageParam] | None = None,
        output_messages: list[ChatCompletionMessageParam] | None = None,
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
)

BatchRequestInputBody: TypeAlias = (
    ChatCompletionRequest | EmbeddingRequest | ScoreRequest | RerankRequest
)


class BatchRequestInput(OpenAIBaseModel):
    """
    The per-line object of the batch input file.

    NOTE: Currently only the `/v1/chat/completions` endpoint is supported.
    """

    # A developer-provided per-request id that will be used to match outputs to
    # inputs. Must be unique for each request in a batch.
    custom_id: str

    # The HTTP method to be used for the request. Currently only POST is
    # supported.
    method: str

    # The OpenAI API relative URL to be used for the request. Currently
    # /v1/chat/completions is supported.
    url: str

    # The parameters of the request.
    body: BatchRequestInputBody

    @field_validator("body", mode="plain")
    @classmethod
    def check_type_for_url(cls, value: Any, info: ValidationInfo):
        # Use url to disambiguate models
        url: str = info.data["url"]
        if url == "/v1/chat/completions":
            return ChatCompletionRequest.model_validate(value)
        if url == "/v1/embeddings":
            return TypeAdapter(EmbeddingRequest).validate_python(value)
        if url.endswith("/score"):
            return ScoreRequest.model_validate(value)
        if url.endswith("/rerank"):
            return RerankRequest.model_validate(value)
        return TypeAdapter(BatchRequestInputBody).validate_python(value)


class BatchResponseData(OpenAIBaseModel):
    # HTTP status code of the response.
    status_code: int = 200

    # An unique identifier for the API request.
    request_id: str

    # The body of the response.
    body: (
        ChatCompletionResponse
        | EmbeddingResponse
        | ScoreResponse
        | RerankResponse
        | None
    ) = None


class BatchRequestOutput(OpenAIBaseModel):
    """
    The per-line object of the batch output and error files
    """

    id: str

    # A developer-provided per-request id that will be used to match outputs to
    # inputs.
    custom_id: str

    response: BatchResponseData | None

    # For requests that failed with a non-HTTP error, this will contain more
    # information on the cause of the failure.
    error: Any | None


class TokenizeCompletionRequest(OpenAIBaseModel):
    model: str | None = None
    prompt: str

    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."
        ),
    )
    return_token_strs: bool | None = Field(
        default=False,
        description=(
            "If true, also return the token strings corresponding to the token ids."
        ),
    )


class TokenizeChatRequest(OpenAIBaseModel):
    model: str | None = None
    messages: list[ChatCompletionMessageParam]

    add_generation_prompt: bool = Field(
        default=True,
        description=(
            "If true, the generation prompt will be added to the chat template. "
            "This is a parameter used by chat template in tokenizer config of the "
            "model."
        ),
    )
    return_token_strs: bool | None = Field(
        default=False,
        description=(
            "If true, also return the token strings corresponding to the token ids."
        ),
    )
    continue_final_message: bool = Field(
        default=False,
        description=(
            "If this is set, the chat will be formatted so that the final "
            "message in the chat is open-ended, without any EOS tokens. The "
            "model will continue this message rather than starting a new one. "
            'This allows you to "prefill" part of the model\'s response for it. '
            "Cannot be used at the same time as `add_generation_prompt`."
        ),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."
        ),
    )
    chat_template: str | None = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."
        ),
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the template renderer. "
            "Will be accessible by the chat template."
        ),
    )
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    tools: list[ChatCompletionToolsParam] | None = Field(
        default=None,
        description=("A list of tools the model may call."),
    )

    @model_validator(mode="before")
    @classmethod
    def check_generation_prompt(cls, data):
        if data.get("continue_final_message") and data.get("add_generation_prompt"):
            raise ValueError(
                "Cannot set both `continue_final_message` and "
                "`add_generation_prompt` to True."
            )
        return data


TokenizeRequest: TypeAlias = TokenizeCompletionRequest | TokenizeChatRequest


class TokenizeResponse(OpenAIBaseModel):
    count: int
    max_model_len: int
    tokens: list[int]
    token_strs: list[str] | None = None


class DetokenizeRequest(OpenAIBaseModel):
    model: str | None = None
    tokens: list[int]


class DetokenizeResponse(OpenAIBaseModel):
    prompt: str


class TokenizerInfoResponse(OpenAIBaseModel):
    """
    Response containing tokenizer configuration
    equivalent to tokenizer_config.json
    """

    model_config = ConfigDict(extra="allow")
    tokenizer_class: str


class LoadLoRAAdapterRequest(BaseModel):
    lora_name: str
    lora_path: str


class UnloadLoRAAdapterRequest(BaseModel):
    lora_name: str
    lora_int_id: int | None = Field(default=None)


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
            raise ValueError("Stream options can only be defined when `stream=True`.")

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

    no_speech_prob: float
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
        )

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        stream_opts = ["stream_include_usage", "stream_continuous_usage_stats"]
        stream = data.get("stream", False)
        if any(bool(data.get(so, False)) for so in stream_opts) and not stream:
            raise ValueError("Stream options can only be defined when `stream=True`.")

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

    no_speech_prob: float
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
