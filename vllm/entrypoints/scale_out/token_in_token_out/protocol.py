# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from vllm.config import ModelConfig
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProbs,
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionStreamResponse,
)
from vllm.entrypoints.openai.engine.protocol import StreamOptions, UsageInfo
from vllm.logprobs import Logprob
from vllm.renderers import TokenizeParams
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

####### Tokens IN <> Tokens OUT #######


class PlaceholderRangeInfo(BaseModel):
    """Serializable placeholder location for a single multi-modal item."""

    offset: int
    """Start index of the placeholder tokens in the prompt."""

    length: int
    """Number of placeholder tokens."""

    # TODO: add ``is_embed: list[bool] | None`` once the /generate side
    # consumes features — some models (e.g. Qwen-VL) use sparse
    # placeholder masks that cannot be recomputed from offset+length alone.


class MultiModalFeatures(BaseModel):
    """Lightweight multimodal metadata produced by the render step.

    Carries hashes (for cache lookup / identification) and placeholder
    positions so the downstream `/generate` service knows *where* in
    the token sequence each multimodal item lives.
    """

    mm_hashes: dict[str, list[str]]
    """Per-modality item hashes, e.g. `{"image": ["abc", "def"]}`."""

    mm_placeholders: dict[str, list[PlaceholderRangeInfo]]
    """Per-modality placeholder ranges in the token sequence."""

    kwargs_data: dict[str, list[str | None]] | None = None
    """Per-modality serialized tensor data.

    Each value is a list parallel to ``mm_hashes[modality]``.  A ``str``
    entry is a base64-encoded ``MultiModalKwargsItem``; ``None`` means
    the item should be resolved from cache.  The entire field is
    ``None`` for metadata-only (cache-hit) responses.
    """


class GenerateRequest(BaseModel):
    request_id: str = Field(
        default_factory=lambda: f"{random_uuid()}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    token_ids: list[int] = Field(min_length=1)
    """The token ids to generate text from."""

    assistant_tokens_mask: list[int] | None = None
    """Per-token mask (1 = assistant-generated, 0 = not).

    Only populated when the render request sets ``return_assistant_tokens_mask=True``
    and the chat template supports ``{% generation %}``.
    ``None`` when the mask was not requested or could not be computed.
    """

    @field_validator("token_ids")
    @classmethod
    def validate_token_ids(cls, v: list[int]) -> list[int]:
        if any(t < 0 for t in v):
            raise ValueError("token_ids must not contain negative values")
        return v

    token_offsets: list[tuple[int, int]] | None = None
    """Char-level (start, end) offsets per token, relative to the
    tokenized source string. Present only when the request set
    `return_token_offsets=True` and the renderer was able to compute
    them (Fast tokenizer, text input, no multimodal data). List length
    equals `token_ids` length when present. None otherwise."""

    features: MultiModalFeatures | None = None
    """Multimodal hashes and placeholder positions (populated for MM inputs)."""

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
        ge=-(2**63),
        le=2**63 - 1,
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
    ec_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description=(
            "ECTransfer parameters used for encoder-cache disaggregated serving."
        ),
    )

    # Tracks which keys the caller explicitly set inside ``sampling_params``
    # when the request was parsed from a JSON body. Lets the server tell
    # "client said max_tokens=16" from "client said nothing → dataclass
    # default 16" so it can apply server-side defaulting only in the latter
    # case. ``None`` means the request was constructed with a pre-built
    # ``SamplingParams`` instance (e.g. from internal callers that have
    # already resolved values), in which case all fields are considered set.
    _sampling_params_provided_keys: set[str] | None = PrivateAttr(default=None)

    @model_validator(mode="wrap")
    @classmethod
    def _capture_sampling_params_provided_keys(cls, data: Any, handler):
        provided: set[str] | None = None
        if isinstance(data, dict):
            sp = data.get("sampling_params")
            if isinstance(sp, dict):
                provided = set(sp.keys())
        instance = handler(data)
        instance._sampling_params_provided_keys = provided
        return instance

    def is_sampling_param_provided(self, name: str) -> bool:
        """Whether the caller explicitly set ``sampling_params.<name>``.

        For requests parsed from a JSON body, this reflects the raw input
        dict. For requests constructed with a pre-built ``SamplingParams``
        instance, all fields are considered provided so server-side defaults
        do not clobber values already resolved upstream.
        """
        if self._sampling_params_provided_keys is None:
            return True
        return name in self._sampling_params_provided_keys

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        return TokenizeParams(
            max_total_tokens=None,
            max_output_tokens=0,
        )


class GenerateResponseChoice(BaseModel):
    index: int
    logprobs: ChatCompletionLogProbs | None = None
    # per OpenAI spec this is the default
    finish_reason: str | None = "stop"
    token_ids: list[int] | None = None
    # Per-token expert routing decisions, base64-encoded ``.npy`` bytes
    # (numpy serialization). Shape after decode:
    #   (num_tokens - 1, num_layers, num_experts_per_tok)  dtype uint8/uint16
    # ``num_tokens - 1`` because the last sampled token has not been
    # forwarded yet and therefore has no routing data.
    # Decode:
    #   np.load(io.BytesIO(base64.b64decode(s)))
    # ``None`` if (a) the request was aborted before any forward pass,
    # or (b) ``enable_return_routed_experts`` is off server-side.
    routed_experts: str | None = None

    @field_validator("token_ids")
    @classmethod
    def validate_token_ids(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and any(t < 0 for t in v):
            raise ValueError("token_ids must not contain negative values")
        return v


class GenerateResponseStreamChoice(BaseModel):
    index: int
    logprobs: ChatCompletionLogProbs | None = None
    finish_reason: str | None = None
    token_ids: list[int] | None = None
    routed_experts: str | None = None


class GenerateStreamResponse(BaseModel):
    request_id: str = Field(
        default_factory=lambda: f"{random_uuid()}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    choices: list[GenerateResponseStreamChoice]
    usage: UsageInfo | None = Field(default=None)


class GenerateResponse(BaseModel):
    request_id: str = Field(
        default_factory=lambda: f"{random_uuid()}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    model: str | None = None
    created: int | None = None
    choices: list[GenerateResponseChoice]
    usage: UsageInfo | None = Field(default=None)
    prompt_logprobs: list[dict[int, Logprob] | None] | None = None

    kv_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description="KVTransfer parameters used for disaggregated serving.",
    )
    ec_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description=(
            "ECTransfer parameters used for encoder-cache disaggregated serving."
        ),
    )


class DerenderChatRequest(BaseModel):
    """Request for the /v1/chat/completions/derender endpoint (non-streaming).

    Wraps a complete GenerateResponse and caller supplied metadata needed to
    produce a fully formed ChatCompletionResponse without a GPU.
    """

    # --8<-- [start:derender-chat-request]
    stream: Literal[False] = False

    model: str
    """Served model name."""

    generate_response: GenerateResponse
    """The complete token-in / token-out engine response to derender."""

    prompt_tokens: int | None = None
    """Prompt token count for usage; defaults to 0 if omitted.

    GenerateResponse carries only output tokens; the caller already has
    len(GenerateRequest.token_ids) from the render step.
    """

    chat_request: ChatCompletionRequest | None = None
    """The original (post-adjust_request) ChatCompletionRequest from /render.

    Required by the parsing so that tool/reasoning parsers can receive the full
    request context they expect (request.tools, request.tool_choice,
    request._grammar_from_tool_parser, etc.).
    """
    # --8<-- [end:derender-chat-request]


class DerenderCompletionRequest(BaseModel):
    """Request for the /v1/completions/derender endpoint (non-streaming).

    Parallel to DerenderChatRequest but handles the multi-prompt completions
    case: one GenerateResponse per prompt, mirroring the list[GenerateRequest]
    returned by /v1/completions/render.
    """

    # --8<-- [start:derender-completion-request]
    stream: Literal[False] = False

    model: str
    """Served model name."""

    generate_responses: list[GenerateResponse]
    """One response per prompt, parallel to the list[GenerateRequest]
    returned by /v1/completions/render."""

    prompt_tokens: list[int] | None = None
    """One prompt token count per response; each defaults to 0 if omitted.

    If provided, len(prompt_tokens) must equal len(generate_responses).
    """

    completion_request: CompletionRequest | None = None
    """The original (post-adjust_request) CompletionRequest from /render.

    Mirrors chat_request on DerenderChatRequest. Required by the parsing
    so parsers receive the full request context.
    """
    # --8<-- [end:derender-completion-request]

    @model_validator(mode="after")
    def _validate_prompt_tokens_length(self) -> "DerenderCompletionRequest":
        if self.prompt_tokens is not None and len(self.prompt_tokens) != len(
            self.generate_responses
        ):
            raise ValueError(
                f"prompt_tokens length ({len(self.prompt_tokens)}) must equal "
                f"generate_responses length ({len(self.generate_responses)})"
            )
        return self


class DerenderStreamState(BaseModel):
    """Per sequence state for stateless streaming derender.

    The client carries this between successive per chunk HTTP calls to the
    streaming derender endpoint. All fields are plain JSON serializable data.
    No opaque tokenizer or parser internals are stored here.

    The detokenization strategy carries the incremental decode offsets
    directly rather than re-sending the whole token history each chunk.
    ``detokenize_incrementally`` only ever reads the trailing token window
    ``prev_tokens[prefix_offset:]``, so we carry just that tail plus the two
    offsets. Each chunk resumes exactly where the last one stopped, including
    any partially processed multi-byte character (tracked by ``read_offset``),
    then trims and rebases the window so it never grows with generation length.

    Performance:
    - Compute per chunk is O(delta). One ``detokenize_incrementally`` call per
      new token, independent of how many tokens preceded it.
    - Transport per chunk is O(window). The carried tail is bounded by the
      incremental detokenization offset, so cumulative bytes over the wire are
      O(n) rather than the O(n^2) a full history round trip would incur.
    """

    prev_tokens: list[str] = Field(default_factory=list)
    """Trailing decode window. Token strings from ``prefix_offset`` onward.

    Bounded, trimmed and rebased each chunk to the tail
    ``detokenize_incrementally`` still reads, so it does not grow with the
    number of chunks.
    """

    prefix_offset: int = Field(default=0, ge=0)
    """Prefix offset into ``prev_tokens`` for incremental detokenization."""

    read_offset: int = Field(default=0, ge=0)
    """Read offset into ``prev_tokens`` for incremental detokenization."""

    @field_validator("prev_tokens")
    @classmethod
    def _bound_prev_tokens(cls, v: list[str]) -> list[str]:
        # INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET is small (5) and the trimmed
        # window is O(offset). A generous limit rejects unusually large or malformed
        # payloads without restricting legitimate multi-byte sequences.
        limit = 1024
        if len(v) > limit:
            raise ValueError(f"prev_tokens length ({len(v)}) exceeds maximum ({limit})")
        return v

    role_sent: bool = False
    """True once the initial ``role: "assistant"`` delta has been emitted.

    Prevents re-emitting the role on subsequent chunks even when the detok
    window is transiently empty (e.g. usage only final chunk).
    """

    # TODO: Properties used in follow on PR for tool call parsing
    last_content: str | None = None
    """Last emitted cumulative assistant content text."""

    last_reasoning: str | None = None
    """Last emitted cumulative reasoning text."""

    last_tool_call_ids: list[str] = Field(default_factory=list)
    """Stable tool-call IDs, assigned once when each call first appears.

    Prevents ID regeneration across re-parsing.
    """


class DerenderChatStreamRequest(BaseModel):
    """One chunk streaming derender request for /v1/chat/completions/derender.

    The client sends one request per SSE chunk received from
    ``/inference/v1/generate``.  Each request carries the generate chunk
    plus the ``stream_state`` returned by the previous call (``None`` on the
    first call).  The response contains the derendered chunk and the updated
    state to be passed to the next call.

    This implements stateless no server side session. All mutable state lives in
    the client carried ``stream_state``.
    """

    stream: Literal[True]

    model: str
    generate_chunk: GenerateStreamResponse
    """One SSE chunk from ``/inference/v1/generate`` (``stream=True``)."""

    stream_state: DerenderStreamState | None = None
    """Client carried detok state from the previous call. ``None`` on first."""

    prompt_tokens: int | None = None
    """Prompt token count for usage. Forwarded from the render step."""

    chat_request: ChatCompletionRequest | None = None
    """The original (post adjust_request) ChatCompletionRequest from /render."""


class DerenderCompletionStreamRequest(BaseModel):
    """One chunk streaming derender request for /v1/completions/derender.

    Parallel to ``DerenderChatStreamRequest`` for the completions endpoint.
    Each call processes one SSE chunk (one output sequence's delta) and
    returns the derendered chunk plus updated state.
    """

    stream: Literal[True]

    model: str
    generate_chunk: GenerateStreamResponse
    """One SSE chunk from ``/inference/v1/generate``."""

    stream_state: DerenderStreamState | None = None
    """Client-carried detok state. ``None`` on the first call."""

    prompt_tokens: int | None = None
    """Prompt token count for usage."""

    completion_request: CompletionRequest | None = None
    """The original (post adjust_request) CompletionRequest from /render."""


class DerenderChatStreamResponse(BaseModel):
    """Response for one streaming chat derender chunk.

    Pairs the derendered SSE chunk with the updated client carried state to
    pass to the next call.
    """

    chunk: ChatCompletionStreamResponse
    stream_state: DerenderStreamState


class DerenderCompletionStreamResponse(BaseModel):
    """Response for one streaming completions derender chunk.

    Parallel to ``DerenderChatStreamResponse`` for the completions endpoint.
    """

    chunk: CompletionStreamResponse
    stream_state: DerenderStreamState


# Determines the type by checking the ``stream`` field's literal value. A body without
# ``stream`` validates as the non-streaming member
# (``stream`` defaults to ``False`` there), so FastAPI can validate and dispatch both
# shapes on a single path.
DerenderChatRequestUnion: TypeAlias = DerenderChatRequest | DerenderChatStreamRequest
DerenderCompletionRequestUnion: TypeAlias = (
    DerenderCompletionRequest | DerenderCompletionStreamRequest
)
