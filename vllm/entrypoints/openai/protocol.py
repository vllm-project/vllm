# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from typing import Any, Dict, List, Literal, Optional, Union

import openai.types.chat
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
# pydantic needs the TypedDict from typing_extensions
from typing_extensions import Annotated, Required, TypedDict

from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid


class CustomChatCompletionContentPartParam(TypedDict, total=False):
    __pydantic_config__ = ConfigDict(extra="allow")  # type: ignore

    type: Required[str]
    """The type of the content part."""


ChatCompletionContentPartParam = Union[
    openai.types.chat.ChatCompletionContentPartParam,
    CustomChatCompletionContentPartParam]


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


ChatCompletionMessageParam = Union[
    openai.types.chat.ChatCompletionMessageParam,
    CustomChatCompletionMessageParam]


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


class ResponseFormat(OpenAIBaseModel):
    # type must be "json_object" or "text"
    type: Literal["text", "json_object"]


class StreamOptions(OpenAIBaseModel):
    include_usage: Optional[bool]


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
    seed: Optional[int] = Field(None,
                                ge=torch.iinfo(torch.long).min,
                                le=torch.iinfo(torch.long).max)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    tools: Optional[List[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[Literal["none"],
                                ChatCompletionNamedToolChoiceParam]] = "none"
    user: Optional[str] = None

    # doc: begin-chat-completion-sampling-params
    best_of: Optional[int] = None
    use_beam_search: Optional[bool] = False
    top_k: Optional[int] = -1
    min_p: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    length_penalty: Optional[float] = 1.0
    early_stopping: Optional[bool] = False
    ignore_eos: Optional[bool] = False
    min_tokens: Optional[int] = 0
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    # doc: end-chat-completion-sampling-params

    # doc: begin-chat-completion-extra-params
    echo: Optional[bool] = Field(
        default=False,
        description=(
            "If true, the new message will be prepended with the last message "
            "if they belong to the same role."),
    )
    add_generation_prompt: Optional[bool] = Field(
        default=True,
        description=
        ("If true, the generation prompt will be added to the chat template. "
         "This is a parameter used by chat template in tokenizer config of the "
         "model."),
    )
    add_special_tokens: Optional[bool] = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to False (as is the "
            "default)."),
    )
    include_stop_str_in_output: Optional[bool] = Field(
        default=False,
        description=(
            "Whether to include the stop string in the output. "
            "This is only applied when the stop or stop_token_ids is set."),
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

    def to_sampling_params(self) -> SamplingParams:
        # We now allow logprobs being true without top_logrobs.

        logits_processors = None
        if self.logit_bias:

            def logit_bias_logits_processor(
                    token_ids: List[int],
                    logits: torch.Tensor) -> torch.Tensor:
                assert self.logit_bias is not None
                for token_id, bias in self.logit_bias.items():
                    # Clamp the bias between -100 and 100 per OpenAI API spec
                    bias = min(100, max(-100, bias))
                    logits[int(token_id)] += bias
                return logits

            logits_processors = [logit_bias_logits_processor]

        return SamplingParams(
            n=self.n,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            min_p=self.min_p,
            seed=self.seed,
            stop=self.stop,
            stop_token_ids=self.stop_token_ids,
            max_tokens=self.max_tokens,
            min_tokens=self.min_tokens,
            logprobs=self.top_logprobs if self.logprobs else None,
            prompt_logprobs=self.top_logprobs if self.echo else None,
            best_of=self.best_of,
            top_k=self.top_k,
            ignore_eos=self.ignore_eos,
            use_beam_search=self.use_beam_search,
            early_stopping=self.early_stopping,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            include_stop_str_in_output=self.include_stop_str_in_output,
            length_penalty=self.length_penalty,
            logits_processors=logits_processors,
        )

    @model_validator(mode='before')
    @classmethod
    def validate_stream_options(cls, values):
        if (values.get('stream_options') is not None
                and not values.get('stream')):
            raise ValueError(
                "stream_options can only be set if stream is true")
        return values

    @model_validator(mode="before")
    @classmethod
    def check_guided_decoding_count(cls, data):
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
        if guide_count > 1 and "tool_choice" in data and data[
                "tool_choice"] != "none":
            raise ValueError(
                "You can only either use guided decoding or tools, not both.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_tool_choice(cls, data):
        if "tool_choice" in data and data["tool_choice"] != "none":
            if not isinstance(data["tool_choice"], dict):
                raise ValueError("Currently only named tools are supported.")
            if "tools" not in data or data["tools"] is None:
                raise ValueError(
                    "When using `tool_choice`, `tools` must be set.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if "top_logprobs" in data and data["top_logprobs"] is not None:
            if "logprobs" not in data or data["logprobs"] is False:
                raise ValueError(
                    "when using `top_logprobs`, `logprobs` must be set to true."
                )
            elif data["top_logprobs"] < 0:
                raise ValueError(
                    "`top_logprobs` must be a value a positive value.")
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
    seed: Optional[int] = Field(None,
                                ge=torch.iinfo(torch.long).min,
                                le=torch.iinfo(torch.long).max)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    user: Optional[str] = None

    # doc: begin-completion-sampling-params
    use_beam_search: Optional[bool] = False
    top_k: Optional[int] = -1
    min_p: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    length_penalty: Optional[float] = 1.0
    early_stopping: Optional[bool] = False
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    ignore_eos: Optional[bool] = False
    min_tokens: Optional[int] = 0
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    # doc: end-completion-sampling-params

    # doc: begin-completion-extra-params
    include_stop_str_in_output: Optional[bool] = Field(
        default=False,
        description=(
            "Whether to include the stop string in the output. "
            "This is only applied when the stop or stop_token_ids is set."),
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
            "of the server for this specific request. If set, must be one of "
            "'outlines' / 'lm-format-enforcer'"))
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."))

    # doc: end-completion-extra-params

    def to_sampling_params(self):
        echo_without_generation = self.echo and self.max_tokens == 0

        logits_processors = None
        if self.logit_bias:

            def logit_bias_logits_processor(
                    token_ids: List[int],
                    logits: torch.Tensor) -> torch.Tensor:
                assert self.logit_bias is not None
                for token_id, bias in self.logit_bias.items():
                    # Clamp the bias between -100 and 100 per OpenAI API spec
                    bias = min(100, max(-100, bias))
                    logits[int(token_id)] += bias
                return logits

            logits_processors = [logit_bias_logits_processor]

        return SamplingParams(
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
            ignore_eos=self.ignore_eos,
            max_tokens=self.max_tokens if not echo_without_generation else 1,
            min_tokens=self.min_tokens,
            logprobs=self.logprobs,
            use_beam_search=self.use_beam_search,
            early_stopping=self.early_stopping,
            prompt_logprobs=self.logprobs if self.echo else None,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=(self.spaces_between_special_tokens),
            include_stop_str_in_output=self.include_stop_str_in_output,
            length_penalty=self.length_penalty,
            logits_processors=logits_processors,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
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
        if "logprobs" in data and data[
                "logprobs"] is not None and not data["logprobs"] >= 0:
            raise ValueError("if passed, `logprobs` must be a positive value.")
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError(
                "Stream options can only be defined when stream is True.")
        return data


class EmbeddingRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/embeddings
    model: str
    input: Union[List[int], List[List[int]], str, List[str]]
    encoding_format: Optional[str] = Field('float', pattern='^(float|base64)$')
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


class EmbeddingResponseData(BaseModel):
    index: int
    object: str = "embedding"
    embedding: List[float]


class EmbeddingResponse(BaseModel):
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


class ChatMessage(OpenAIBaseModel):
    role: str
    content: str
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
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None


class ChatCompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(OpenAIBaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)


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

    # The parameteters of the request.
    body: Union[ChatCompletionRequest, ]


class BatchRequestOutput(OpenAIBaseModel):
    """
    The per-line object of the batch output and error files
    """

    id: str

    # A developer-provided per-request id that will be used to match outputs to
    # inputs.
    custom_id: str

    response: Optional[ChatCompletionResponse]

    # For requests that failed with a non-HTTP error, this will contain more
    # information on the cause of the failure.
    error: Optional[Any]
