# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from argparse import Namespace
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from openai.types.chat import ChatCompletionContentPartParam
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Annotated, Required, TypedDict

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.entrypoints.openai.logits_processors import get_logits_processors
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import (LogitsProcessor, RequestOutputKind,
                                  SamplingParams)
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid

# torch is mocked during docs generation,
# so we have to provide the values as literals
_MOCK_LONG_INFO = Namespace(min=-9223372036854775808, max=9223372036854775807)
_LONG_INFO: Union["torch.iinfo", Namespace]

try:
    from sphinx.ext.autodoc.mock import _MockModule

    if isinstance(torch, _MockModule):
        _LONG_INFO = _MOCK_LONG_INFO
    else:
        _LONG_INFO = torch.iinfo(torch.long)
except ModuleNotFoundError:
    _LONG_INFO = torch.iinfo(torch.long)

assert _LONG_INFO.min == _MOCK_LONG_INFO.min
assert _LONG_INFO.max == _MOCK_LONG_INFO.max


class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""
    role: Required[str]
    """The role of the message's author."""

    content: Union[str, List[ChatCompletionContentPartParam]]
    """The contents of the message."""

    name: str
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the
    same role.
    """

    tool_call_id: Optional[str]

    tool_calls: Optional[List[dict]]


class OpenAIBaseModel(BaseModel):
    # OpenAI API does not allow extra fields
    model_config = ConfigDict(extra="forbid")


class ErrorResponse(OpenAIBaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


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
    group: Optional[str] = None
    is_blocking: bool = False


class ModelCard(OpenAIBaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vllm"
    root: Optional[str] = None
    parent: Optional[str] = None
    max_model_len: Optional[int] = None
    permission: List[ModelPermission] = Field(default_factory=list)


class ModelList(OpenAIBaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class RequestResponseMetadata(BaseModel):
    request_id: str
    final_usage_info: Optional[UsageInfo] = None


class JsonSchemaResponseFormat(OpenAIBaseModel):
    name: str
    description: Optional[str] = None
    # schema is the field in openai but that causes conflicts with pydantic so
    # instead use json_schema with an alias
    json_schema: Optional[Dict[str, Any]] = Field(default=None, alias='schema')
    strict: Optional[bool] = None


class ResponseFormat(OpenAIBaseModel):
    # type must be "json_schema", "json_object" or "text"
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None


class StreamOptions(OpenAIBaseModel):
    include_usage: Optional[bool] = True
    continuous_usage_stats: Optional[bool] = True


class FunctionDefinition(OpenAIBaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ChatCompletionToolsParam(OpenAIBaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionNamedFunction(OpenAIBaseModel):
    name: str


class ChatCompletionNamedToolChoiceParam(OpenAIBaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"


class ChatCompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: List[ChatCompletionMessageParam]
    model: str
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 0
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    tools: Optional[List[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[Literal["none"], Literal["auto"],
                                ChatCompletionNamedToolChoiceParam]] = "none"

    # NOTE this will be ignored by VLLM -- the model determines the behavior
    parallel_tool_calls: Optional[bool] = False
    user: Optional[str] = None

    # doc: begin-chat-completion-sampling-params
    best_of: Optional[int] = None
    use_beam_search: bool = False
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = False
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    prompt_logprobs: Optional[int] = None
    # doc: end-chat-completion-sampling-params

    # doc: begin-chat-completion-extra-params
    echo: bool = Field(
        default=False,
        description=(
            "If true, the new message will be prepended with the last message "
            "if they belong to the same role."),
    )
    add_generation_prompt: bool = Field(
        default=True,
        description=
        ("If true, the generation prompt will be added to the chat template. "
         "This is a parameter used by chat template in tokenizer config of the "
         "model."),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."),
    )
    documents: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=
        ("A list of dicts representing documents that will be accessible to "
         "the model if it is performing RAG (retrieval-augmented generation)."
         " If the template does not support RAG, this argument will have no "
         "effect. We recommend that each document should be a dict containing "
         "\"title\" and \"text\" keys."),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."),
    )
    chat_template_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. "
                     "Will be accessible by the chat template."),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description=("If specified, the output will follow the JSON schema."),
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[List[str]] = Field(
        default=None,
        description=(
            "If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the context free grammar."),
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be either "
            "'outlines' / 'lm-format-enforcer'"))
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."))

    # doc: end-chat-completion-extra-params

    def to_sampling_params(
            self, tokenizer: AnyTokenizer,
            guided_decode_logits_processor: Optional[LogitsProcessor],
            default_max_tokens: int) -> SamplingParams:
        max_tokens = self.max_tokens
        if max_tokens is None:
            max_tokens = default_max_tokens

        prompt_logprobs = self.prompt_logprobs
        if prompt_logprobs is None and self.echo:
            prompt_logprobs = self.top_logprobs

        # We now allow logprobs being true without top_logrobs.
        logits_processors = get_logits_processors(
            logit_bias=self.logit_bias,
            allowed_token_ids=None,
            tokenizer=tokenizer,
        )
        if guided_decode_logits_processor:
            logits_processors.append(guided_decode_logits_processor)

        return SamplingParams.from_optional(
            n=self.n,
            best_of=self.best_of,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            seed=self.seed,
            stop=self.stop,
            stop_token_ids=self.stop_token_ids,
            logprobs=self.top_logprobs if self.logprobs else None,
            prompt_logprobs=prompt_logprobs,
            ignore_eos=self.ignore_eos,
            max_tokens=max_tokens,
            min_tokens=self.min_tokens,
            use_beam_search=self.use_beam_search,
            early_stopping=self.early_stopping,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            include_stop_str_in_output=self.include_stop_str_in_output,
            length_penalty=self.length_penalty,
            logits_processors=logits_processors,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            output_kind=RequestOutputKind.DELTA if self.stream \
                else RequestOutputKind.FINAL_ONLY,
        )

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError(
                "Stream options can only be defined when `stream=True`.")

        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if (prompt_logprobs := data.get("prompt_logprobs")) is not None:
            if data.get("stream") and prompt_logprobs > 0:
                raise ValueError(
                    "`prompt_logprobs` are not available when `stream=True`.")

            if prompt_logprobs < 0:
                raise ValueError("`prompt_logprobs` must be a positive value.")

        if (top_logprobs := data.get("top_logprobs")) is not None:
            if top_logprobs < 0:
                raise ValueError("`top_logprobs` must be a positive value.")

            if not data.get("logprobs"):
                raise ValueError(
                    "when using `top_logprobs`, `logprobs` must be set to true."
                )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_guided_decoding_count(cls, data):
        if isinstance(data, ValueError):
            raise data

        guide_count = sum([
            "guided_json" in data and data["guided_json"] is not None,
            "guided_regex" in data and data["guided_regex"] is not None,
            "guided_choice" in data and data["guided_choice"] is not None
        ])
        # you can only use one kind of guided decoding
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "('guided_json', 'guided_regex' or 'guided_choice').")
        # you can only either use guided decoding or tools, not both
        if guide_count > 1 and data.get("tool_choice",
                                        "none") not in ("none", "auto"):
            raise ValueError(
                "You can only either use guided decoding or tools, not both.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_tool_usage(cls, data):

        # if "tool_choice" is not specified but tools are provided,
        # default to "auto" tool_choice
        if "tool_choice" not in data and data.get("tools"):
            data["tool_choice"] = "auto"

        # if "tool_choice" is specified -- validation
        if "tool_choice" in data:

            # ensure that if "tool choice" is specified, tools are present
            if "tools" not in data or data["tools"] is None:
                raise ValueError(
                    "When using `tool_choice`, `tools` must be set.")

            # make sure that tool choice is either a named tool
            # OR that it's set to "auto"
            if data["tool_choice"] != "auto" and not isinstance(
                    data["tool_choice"], dict):
                raise ValueError(
                    "`tool_choice` must either be a named tool or \"auto\". "
                    "`tool_choice=\"none\" is not supported.")

            # ensure that if "tool_choice" is specified as an object,
            # it matches a valid tool
            if isinstance(data["tool_choice"], dict):
                valid_tool = False
                specified_function = data["tool_choice"]["function"]
                if not specified_function:
                    raise ValueError(
                        "Incorrectly formatted `tool_choice`. Should be like "
                        "`{\"type\": \"function\","
                        " \"function\": {\"name\": \"my_function\"}}`")
                specified_function_name = specified_function["name"]
                if not specified_function_name:
                    raise ValueError(
                        "Incorrectly formatted `tool_choice`. Should be like "
                        "`{\"type\": \"function\", "
                        "\"function\": {\"name\": \"my_function\"}}`")
                for tool in data["tools"]:
                    if tool["function"]["name"] == specified_function_name:
                        valid_tool = True
                        break
                if not valid_tool:
                    raise ValueError(
                        "The tool specified in `tool_choice` does not match any"
                        " of the specified `tools`")
        return data


class CompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str
    prompt: Union[List[int], List[List[int]], str, List[str]]
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 16
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    user: Optional[str] = None

    # doc: begin-completion-sampling-params
    use_beam_search: bool = False
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = False
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    allowed_token_ids: Optional[List[int]] = None
    prompt_logprobs: Optional[int] = None
    # doc: end-completion-sampling-params

    # doc: begin-completion-extra-params
    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."),
    )
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description=
        ("Similar to chat completion, this parameter specifies the format of "
         "output. Only {'type': 'json_object'} or {'type': 'text' } is "
         "supported."),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description="If specified, the output will follow the JSON schema.",
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[List[str]] = Field(
        default=None,
        description=(
            "If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the context free grammar."),
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be one of "
            "'outlines' / 'lm-format-enforcer'"))
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."))

    # doc: end-completion-extra-params

    def to_sampling_params(
            self, tokenizer: AnyTokenizer,
            guided_decode_logits_processor: Optional[LogitsProcessor],
            default_max_tokens: int) -> SamplingParams:
        max_tokens = self.max_tokens
        if max_tokens is None:
            max_tokens = default_max_tokens

        prompt_logprobs = self.prompt_logprobs
        if prompt_logprobs is None and self.echo:
            prompt_logprobs = self.logprobs

        echo_without_generation = self.echo and self.max_tokens == 0

        logits_processors = get_logits_processors(
            logit_bias=self.logit_bias,
            allowed_token_ids=self.allowed_token_ids,
            tokenizer=tokenizer,
        )
        if guided_decode_logits_processor:
            logits_processors.append(guided_decode_logits_processor)

        return SamplingParams.from_optional(
            n=self.n,
            best_of=self.best_of,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            seed=self.seed,
            stop=self.stop,
            stop_token_ids=self.stop_token_ids,
            logprobs=self.logprobs,
            ignore_eos=self.ignore_eos,
            max_tokens=max_tokens if not echo_without_generation else 1,
            min_tokens=self.min_tokens,
            use_beam_search=self.use_beam_search,
            early_stopping=self.early_stopping,
            prompt_logprobs=prompt_logprobs,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            include_stop_str_in_output=self.include_stop_str_in_output,
            length_penalty=self.length_penalty,
            logits_processors=logits_processors,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            output_kind=RequestOutputKind.DELTA if self.stream \
                else RequestOutputKind.FINAL_ONLY,
        )

    @model_validator(mode="before")
    @classmethod
    def check_guided_decoding_count(cls, data):
        guide_count = sum([
            "guided_json" in data and data["guided_json"] is not None,
            "guided_regex" in data and data["guided_regex"] is not None,
            "guided_choice" in data and data["guided_choice"] is not None
        ])
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "('guided_json', 'guided_regex' or 'guided_choice').")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if (prompt_logprobs := data.get("prompt_logprobs")) is not None:
            if data.get("stream") and prompt_logprobs > 0:
                raise ValueError(
                    "`prompt_logprobs` are not available when `stream=True`.")

            if prompt_logprobs < 0:
                raise ValueError("`prompt_logprobs` must be a positive value.")

        if (logprobs := data.get("logprobs")) is not None and logprobs < 0:
            raise ValueError("`logprobs` must be a positive value.")

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError(
                "Stream options can only be defined when `stream=True`.")

        return data


class EmbeddingRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/embeddings
    model: str
    input: Union[List[int], List[List[int]], str, List[str]]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None

    # doc: begin-embedding-pooling-params
    additional_data: Optional[Any] = None

    # doc: end-embedding-pooling-params

    def to_pooling_params(self):
        return PoolingParams(additional_data=self.additional_data)


class CompletionLogProbs(OpenAIBaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str,
                                     float]]] = Field(default_factory=list)


class CompletionResponseChoice(OpenAIBaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"),
    )
    prompt_logprobs: Optional[List[Optional[Dict[int, Logprob]]]] = None


class CompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"),
    )


class CompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class EmbeddingResponseData(OpenAIBaseModel):
    index: int
    object: str = "embedding"
    embedding: Union[List[float], str]


class EmbeddingResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: List[EmbeddingResponseData]
    usage: UsageInfo


class FunctionCall(OpenAIBaseModel):
    name: str
    arguments: str


class ToolCall(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-tool-{random_uuid()}")
    type: Literal["function"] = "function"
    function: FunctionCall


class DeltaFunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


# a tool call delta where everything is optional
class DeltaToolCall(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-tool-{random_uuid()}")
    type: Literal["function"] = "function"
    index: int
    function: Optional[DeltaFunctionCall] = None


class ExtractedToolCallInformation(BaseModel):
    # indicate if tools were called
    tools_called: bool

    # extracted tool calls
    tool_calls: List[ToolCall]

    # content - per OpenAI spec, content AND tool calls can be returned rarely
    # But some models will do this intentionally
    content: Optional[str] = None


class ChatMessage(OpenAIBaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)


class ChatCompletionLogProb(OpenAIBaseModel):
    token: str
    logprob: float = -9999.0
    bytes: Optional[List[int]] = None


class ChatCompletionLogProbsContent(ChatCompletionLogProb):
    top_logprobs: List[ChatCompletionLogProb] = Field(default_factory=list)


class ChatCompletionLogProbs(OpenAIBaseModel):
    content: Optional[List[ChatCompletionLogProbsContent]] = None


class ChatCompletionResponseChoice(OpenAIBaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    # per OpenAI spec this is the default
    finish_reason: Optional[str] = "stop"
    # not part of the OpenAI spec but included in vLLM for legacy reasons
    stop_reason: Optional[Union[int, str]] = None


class ChatCompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    prompt_logprobs: Optional[List[Optional[Dict[int, Logprob]]]] = None


class DeltaMessage(OpenAIBaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: List[DeltaToolCall] = Field(default_factory=list)


class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


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
    body: Union[ChatCompletionRequest, EmbeddingRequest]


class BatchResponseData(OpenAIBaseModel):
    # HTTP status code of the response.
    status_code: int = 200

    # An unique identifier for the API request.
    request_id: str

    # The body of the response.
    body: Optional[Union[ChatCompletionResponse, EmbeddingResponse]] = None


class BatchRequestOutput(OpenAIBaseModel):
    """
    The per-line object of the batch output and error files
    """

    id: str

    # A developer-provided per-request id that will be used to match outputs to
    # inputs.
    custom_id: str

    response: Optional[BatchResponseData]

    # For requests that failed with a non-HTTP error, this will contain more
    # information on the cause of the failure.
    error: Optional[Any]


class TokenizeCompletionRequest(OpenAIBaseModel):
    model: str
    prompt: str

    add_special_tokens: bool = Field(default=True)


class TokenizeChatRequest(OpenAIBaseModel):
    model: str
    messages: List[ChatCompletionMessageParam]

    add_generation_prompt: bool = Field(default=True)
    add_special_tokens: bool = Field(default=False)


TokenizeRequest = Union[TokenizeCompletionRequest, TokenizeChatRequest]


class TokenizeResponse(OpenAIBaseModel):
    count: int
    max_model_len: int
    tokens: List[int]


class DetokenizeRequest(OpenAIBaseModel):
    model: str
    tokens: List[int]


class DetokenizeResponse(OpenAIBaseModel):
    prompt: str


class LoadLoraAdapterRequest(BaseModel):
    lora_name: str
    lora_path: str


class UnloadLoraAdapterRequest(BaseModel):
    lora_name: str
    lora_int_id: Optional[int] = Field(default=None)
