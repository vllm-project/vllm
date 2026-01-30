# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class SamplingParams(_message.Message):
    __slots__ = (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "max_tokens",
        "min_tokens",
        "stop",
        "stop_token_ids",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "ignore_eos",
        "n",
        "logprobs",
        "prompt_logprobs",
        "seed",
        "include_stop_str_in_output",
        "logit_bias",
        "truncate_prompt_tokens",
        "json_schema",
        "regex",
        "grammar",
        "structural_tag",
        "json_object",
        "choice",
    )
    class LogitBiasEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: float
        def __init__(
            self, key: int | None = ..., value: float | None = ...
        ) -> None: ...

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
    def __init__(
        self,
        temperature: float | None = ...,
        top_p: float | None = ...,
        top_k: int | None = ...,
        min_p: float | None = ...,
        frequency_penalty: float | None = ...,
        presence_penalty: float | None = ...,
        repetition_penalty: float | None = ...,
        max_tokens: int | None = ...,
        min_tokens: int | None = ...,
        stop: _Iterable[str] | None = ...,
        stop_token_ids: _Iterable[int] | None = ...,
        skip_special_tokens: bool = ...,
        spaces_between_special_tokens: bool = ...,
        ignore_eos: bool = ...,
        n: int | None = ...,
        logprobs: int | None = ...,
        prompt_logprobs: int | None = ...,
        seed: int | None = ...,
        include_stop_str_in_output: bool = ...,
        logit_bias: _Mapping[int, float] | None = ...,
        truncate_prompt_tokens: int | None = ...,
        json_schema: str | None = ...,
        regex: str | None = ...,
        grammar: str | None = ...,
        structural_tag: str | None = ...,
        json_object: bool = ...,
        choice: ChoiceConstraint | _Mapping | None = ...,
    ) -> None: ...

class ChoiceConstraint(_message.Message):
    __slots__ = ("choices",)
    CHOICES_FIELD_NUMBER: _ClassVar[int]
    choices: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, choices: _Iterable[str] | None = ...) -> None: ...

class TokenizedInput(_message.Message):
    __slots__ = ("original_text", "input_ids")
    ORIGINAL_TEXT_FIELD_NUMBER: _ClassVar[int]
    INPUT_IDS_FIELD_NUMBER: _ClassVar[int]
    original_text: str
    input_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(
        self, original_text: str | None = ..., input_ids: _Iterable[int] | None = ...
    ) -> None: ...

class GenerateRequest(_message.Message):
    __slots__ = ("request_id", "tokenized", "text", "sampling_params", "stream")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TOKENIZED_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_PARAMS_FIELD_NUMBER: _ClassVar[int]
    STREAM_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    tokenized: TokenizedInput
    text: str
    sampling_params: SamplingParams
    stream: bool
    def __init__(
        self,
        request_id: str | None = ...,
        tokenized: TokenizedInput | _Mapping | None = ...,
        text: str | None = ...,
        sampling_params: SamplingParams | _Mapping | None = ...,
        stream: bool = ...,
    ) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("chunk", "complete")
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_FIELD_NUMBER: _ClassVar[int]
    chunk: GenerateStreamChunk
    complete: GenerateComplete
    def __init__(
        self,
        chunk: GenerateStreamChunk | _Mapping | None = ...,
        complete: GenerateComplete | _Mapping | None = ...,
    ) -> None: ...

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
    def __init__(
        self,
        token_ids: _Iterable[int] | None = ...,
        prompt_tokens: int | None = ...,
        completion_tokens: int | None = ...,
        cached_tokens: int | None = ...,
    ) -> None: ...

class GenerateComplete(_message.Message):
    __slots__ = (
        "output_ids",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
    )
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
    def __init__(
        self,
        output_ids: _Iterable[int] | None = ...,
        finish_reason: str | None = ...,
        prompt_tokens: int | None = ...,
        completion_tokens: int | None = ...,
        cached_tokens: int | None = ...,
    ) -> None: ...

class EmbedRequest(_message.Message):
    __slots__ = ("request_id", "tokenized")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TOKENIZED_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    tokenized: TokenizedInput
    def __init__(
        self,
        request_id: str | None = ...,
        tokenized: TokenizedInput | _Mapping | None = ...,
    ) -> None: ...

class EmbedResponse(_message.Message):
    __slots__ = ("embedding", "prompt_tokens", "embedding_dim")
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_DIM_FIELD_NUMBER: _ClassVar[int]
    embedding: _containers.RepeatedScalarFieldContainer[float]
    prompt_tokens: int
    embedding_dim: int
    def __init__(
        self,
        embedding: _Iterable[float] | None = ...,
        prompt_tokens: int | None = ...,
        embedding_dim: int | None = ...,
    ) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("healthy", "message")
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    message: str
    def __init__(self, healthy: bool = ..., message: str | None = ...) -> None: ...

class AbortRequest(_message.Message):
    __slots__ = ("request_ids",)
    REQUEST_IDS_FIELD_NUMBER: _ClassVar[int]
    request_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, request_ids: _Iterable[str] | None = ...) -> None: ...

class AbortResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetModelInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetModelInfoResponse(_message.Message):
    __slots__ = (
        "model_path",
        "is_generation",
        "max_context_length",
        "vocab_size",
        "supports_vision",
    )
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
    def __init__(
        self,
        model_path: str | None = ...,
        is_generation: bool = ...,
        max_context_length: int | None = ...,
        vocab_size: int | None = ...,
        supports_vision: bool = ...,
    ) -> None: ...

class GetServerInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetServerInfoResponse(_message.Message):
    __slots__ = (
        "active_requests",
        "is_paused",
        "last_receive_timestamp",
        "uptime_seconds",
        "server_type",
    )
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
    def __init__(
        self,
        active_requests: int | None = ...,
        is_paused: bool = ...,
        last_receive_timestamp: float | None = ...,
        uptime_seconds: float | None = ...,
        server_type: str | None = ...,
    ) -> None: ...
