# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from vllm.config import ModelConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionLogProbs
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
    token_ids: list[int]
    """The token ids to generate text from."""

    @field_validator("token_ids")
    @classmethod
    def validate_token_ids(cls, v: list[int]) -> list[int]:
        if any(t < 0 for t in v):
            raise ValueError("token_ids must not contain negative values")
        return v

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


class GenerateResponseStreamChoice(BaseModel):
    index: int
    logprobs: ChatCompletionLogProbs | None = None
    finish_reason: str | None = None
    token_ids: list[int] | None = None


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
    choices: list[GenerateResponseChoice]

    prompt_logprobs: list[dict[int, Logprob] | None] | None = None

    kv_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description="KVTransfer parameters used for disaggregated serving.",
    )
