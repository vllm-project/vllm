# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SamplingParams(_message.Message):
    __slots__ = ("temperature", "top_p", "top_k", "min_p", "frequency_penalty", "presence_penalty", "repetition_penalty", "max_tokens", "min_tokens", "stop", "stop_token_ids", "skip_special_tokens", "spaces_between_special_tokens", "ignore_eos", "n", "logprobs", "prompt_logprobs", "seed", "include_stop_str_in_output", "logit_bias", "truncate_prompt_tokens", "json_schema", "regex", "grammar", "structural_tag", "json_object", "choice")
    class LogitBiasEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: float
        def __init__(self, key: _Optional[int] = ..., value: _Optional[float] = ...) -> None: ...
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    MIN_P_FIELD_NUMBER: _ClassVar[int]
    FREQUENCY_PENALTY_FIELD_NUMBER: _ClassVar[int]
    PRESENCE_PENALTY_FIELD_NUMBER: _ClassVar[int]
    REPETITION_PENALTY_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MIN_TOKENS_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    STOP_TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    SKIP_SPECIAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    SPACES_BETWEEN_SPECIAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    IGNORE_EOS_FIELD_NUMBER: _ClassVar[int]
    N_FIELD_NUMBER: _ClassVar[int]
    LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    PROMPT_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_STOP_STR_IN_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    LOGIT_BIAS_FIELD_NUMBER: _ClassVar[int]
    TRUNCATE_PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    JSON_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    REGEX_FIELD_NUMBER: _ClassVar[int]
    GRAMMAR_FIELD_NUMBER: _ClassVar[int]
    STRUCTURAL_TAG_FIELD_NUMBER: _ClassVar[int]
    JSON_OBJECT_FIELD_NUMBER: _ClassVar[int]
    CHOICE_FIELD_NUMBER: _ClassVar[int]
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    frequency_penalty: float
    presence_penalty: float
    repetition_penalty: float
    max_tokens: int
    min_tokens: int
    stop: _containers.RepeatedScalarFieldContainer[str]
    stop_token_ids: _containers.RepeatedScalarFieldContainer[int]
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    ignore_eos: bool
    n: int
    logprobs: int
    prompt_logprobs: int
    seed: int
    include_stop_str_in_output: bool
    logit_bias: _containers.ScalarMap[int, float]
    truncate_prompt_tokens: int
    json_schema: str
    regex: str
    grammar: str
    structural_tag: str
    json_object: bool
    choice: ChoiceConstraint
    def __init__(self, temperature: _Optional[float] = ..., top_p: _Optional[float] = ..., top_k: _Optional[int] = ..., min_p: _Optional[float] = ..., frequency_penalty: _Optional[float] = ..., presence_penalty: _Optional[float] = ..., repetition_penalty: _Optional[float] = ..., max_tokens: _Optional[int] = ..., min_tokens: _Optional[int] = ..., stop: _Optional[_Iterable[str]] = ..., stop_token_ids: _Optional[_Iterable[int]] = ..., skip_special_tokens: bool = ..., spaces_between_special_tokens: bool = ..., ignore_eos: bool = ..., n: _Optional[int] = ..., logprobs: _Optional[int] = ..., prompt_logprobs: _Optional[int] = ..., seed: _Optional[int] = ..., include_stop_str_in_output: bool = ..., logit_bias: _Optional[_Mapping[int, float]] = ..., truncate_prompt_tokens: _Optional[int] = ..., json_schema: _Optional[str] = ..., regex: _Optional[str] = ..., grammar: _Optional[str] = ..., structural_tag: _Optional[str] = ..., json_object: bool = ..., choice: _Optional[_Union[ChoiceConstraint, _Mapping]] = ...) -> None: ...

class ChoiceConstraint(_message.Message):
    __slots__ = ("choices",)
    CHOICES_FIELD_NUMBER: _ClassVar[int]
    choices: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, choices: _Optional[_Iterable[str]] = ...) -> None: ...

class TokenizedInput(_message.Message):
    __slots__ = ("original_text", "input_ids")
    ORIGINAL_TEXT_FIELD_NUMBER: _ClassVar[int]
    INPUT_IDS_FIELD_NUMBER: _ClassVar[int]
    original_text: str
    input_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, original_text: _Optional[str] = ..., input_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class GenerateRequest(_message.Message):
    __slots__ = ("request_id", "tokenized", "sampling_params", "stream")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TOKENIZED_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_PARAMS_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    tokenized: TokenizedInput
    sampling_params: SamplingParams
    stream: bool
    def __init__(self, request_id: _Optional[str] = ..., tokenized: _Optional[_Union[TokenizedInput, _Mapping]] = ..., sampling_params: _Optional[_Union[SamplingParams, _Mapping]] = ..., stream: bool = ...) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("request_id", "chunk", "complete", "error")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    chunk: GenerateStreamChunk
    complete: GenerateComplete
    error: GenerateError
    def __init__(self, request_id: _Optional[str] = ..., chunk: _Optional[_Union[GenerateStreamChunk, _Mapping]] = ..., complete: _Optional[_Union[GenerateComplete, _Mapping]] = ..., error: _Optional[_Union[GenerateError, _Mapping]] = ...) -> None: ...

class GenerateStreamChunk(_message.Message):
    __slots__ = ("token_ids", "prompt_tokens", "completion_tokens", "cached_tokens")
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    def __init__(self, token_ids: _Optional[_Iterable[int]] = ..., prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., cached_tokens: _Optional[int] = ...) -> None: ...

class GenerateComplete(_message.Message):
    __slots__ = ("output_ids", "finish_reason", "prompt_tokens", "completion_tokens", "cached_tokens")
    OUTPUT_IDS_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    output_ids: _containers.RepeatedScalarFieldContainer[int]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    def __init__(self, output_ids: _Optional[_Iterable[int]] = ..., finish_reason: _Optional[str] = ..., prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., cached_tokens: _Optional[int] = ...) -> None: ...

class GenerateError(_message.Message):
    __slots__ = ("message", "http_status_code", "details")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    HTTP_STATUS_CODE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    message: str
    http_status_code: str
    details: str
    def __init__(self, message: _Optional[str] = ..., http_status_code: _Optional[str] = ..., details: _Optional[str] = ...) -> None: ...

class EmbedRequest(_message.Message):
    __slots__ = ("request_id", "tokenized")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TOKENIZED_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    tokenized: TokenizedInput
    def __init__(self, request_id: _Optional[str] = ..., tokenized: _Optional[_Union[TokenizedInput, _Mapping]] = ...) -> None: ...

class EmbedResponse(_message.Message):
    __slots__ = ("request_id", "complete", "error")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    complete: EmbedComplete
    error: EmbedError
    def __init__(self, request_id: _Optional[str] = ..., complete: _Optional[_Union[EmbedComplete, _Mapping]] = ..., error: _Optional[_Union[EmbedError, _Mapping]] = ...) -> None: ...

class EmbedComplete(_message.Message):
    __slots__ = ("embedding", "prompt_tokens", "embedding_dim")
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_DIM_FIELD_NUMBER: _ClassVar[int]
    embedding: _containers.RepeatedScalarFieldContainer[float]
    prompt_tokens: int
    embedding_dim: int
    def __init__(self, embedding: _Optional[_Iterable[float]] = ..., prompt_tokens: _Optional[int] = ..., embedding_dim: _Optional[int] = ...) -> None: ...

class EmbedError(_message.Message):
    __slots__ = ("message", "code")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    message: str
    code: str
    def __init__(self, message: _Optional[str] = ..., code: _Optional[str] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("healthy", "message")
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    message: str
    def __init__(self, healthy: bool = ..., message: _Optional[str] = ...) -> None: ...

class AbortRequest(_message.Message):
    __slots__ = ("request_id", "reason")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    reason: str
    def __init__(self, request_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class AbortResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class GetModelInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetModelInfoResponse(_message.Message):
    __slots__ = ("model_path", "is_generation", "max_context_length", "vocab_size", "supports_vision")
    MODEL_PATH_FIELD_NUMBER: _ClassVar[int]
    IS_GENERATION_FIELD_NUMBER: _ClassVar[int]
    MAX_CONTEXT_LENGTH_FIELD_NUMBER: _ClassVar[int]
    VOCAB_SIZE_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_VISION_FIELD_NUMBER: _ClassVar[int]
    model_path: str
    is_generation: bool
    max_context_length: int
    vocab_size: int
    supports_vision: bool
    def __init__(self, model_path: _Optional[str] = ..., is_generation: bool = ..., max_context_length: _Optional[int] = ..., vocab_size: _Optional[int] = ..., supports_vision: bool = ...) -> None: ...

class GetServerInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetServerInfoResponse(_message.Message):
    __slots__ = ("active_requests", "is_paused", "last_receive_timestamp", "uptime_seconds", "server_type")
    ACTIVE_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    IS_PAUSED_FIELD_NUMBER: _ClassVar[int]
    LAST_RECEIVE_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SERVER_TYPE_FIELD_NUMBER: _ClassVar[int]
    active_requests: int
    is_paused: bool
    last_receive_timestamp: float
    uptime_seconds: float
    server_type: str
    def __init__(self, active_requests: _Optional[int] = ..., is_paused: bool = ..., last_receive_timestamp: _Optional[float] = ..., uptime_seconds: _Optional[float] = ..., server_type: _Optional[str] = ...) -> None: ...
