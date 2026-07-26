# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.entrypoints.openai.engine.protocol import (
    AnyResponseFormat,
    OpenAIBaseModel,
    PerRequestTimingMetrics,
    StreamOptions,
    UsageInfo,
    structured_outputs_from_response_format,
    validate_structural_tag_response_format,
    validate_structured_outputs_structural_tag,
)
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.renderers import TokenizeParams
from vllm.sampling_params import (
    BeamSearchParams,
    RepetitionDetectionParams,
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
    ThinkingTokenBudget,
)
from vllm.utils import random_uuid
from vllm.utils.collection_utils import is_list_of

logger = init_logger(__name__)


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class CompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str | None = None
    prompt: (
        list[Annotated[int, Field(ge=0)]]
        | list[list[Annotated[int, Field(ge=0)]]]
        | str
        | list[str]
        | None
    ) = None
    echo: bool | None = False
    frequency_penalty: float | None = 0.0
    logit_bias: dict[str, float] | None = None
    logprobs: int | None = None
    max_tokens: int | None = 16
    n: int = 1
    presence_penalty: float | None = 0.0
    seed: int | None = Field(None, ge=_INT64_MIN, le=_INT64_MAX)
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
    truncate_prompt_tokens: Annotated[int, Field(ge=-1, le=_INT64_MAX)] | None = None
    truncation_side: Literal["left", "right"] | None = Field(
        default=None,
        description=(
            "Which side to truncate from when truncate_prompt_tokens is active. "
            "'right' keeps the first N tokens. "
            "'left' keeps the last N tokens."
        ),
    )
    allowed_token_ids: list[int] | None = None
    prompt_logprobs: int | None = None
    logprob_token_ids: list[int] | None = Field(
        default=None,
        description=(
            "Specific vocab token IDs to return logprobs for at each generated "
            "position, in addition to the sampled token. More efficient than "
            "requesting the full vocab when only a small fixed label set is "
            "needed (e.g. multilabel "
            "scoring where each label corresponds to a known vocab id). When "
            "set, this explicit token selection takes precedence over the "
            "natural top-k selected by `logprobs`. Requires `logprobs` to be "
            "set."
        ),
    )
    bad_words: list[str] = Field(default_factory=list)
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
        ge=_INT64_MIN,
        le=_INT64_MAX,
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
    return_token_offsets: bool | None = Field(
        default=False,
        description=(
            "If true, return char-level (start, end) offsets for each "
            "token relative to the tokenized source string in the "
            "`token_offsets` field of the rendered response. Only "
            "supported on the `/v1/completions/render` and "
            "`/v1/chat/completions/render` endpoints; ignored on regular "
            "generation endpoints. Honored only for Fast (Rust-backed) "
            "tokenizers; otherwise `token_offsets` is null. For chat "
            "requests, offsets are relative to the templated prompt "
            "string (after applying the chat template). Multimodal "
            "inputs and pre-tokenized inputs always yield null."
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

    ec_transfer_params: dict[str, Any] | None = Field(
        default=None,
        description=(
            "ECTransfer parameters used for encoder-cache disaggregated serving."
        ),
    )

    vllm_xargs: dict[str, str | int | float | list[str | int | float]] | None = Field(
        default=None,
        description=(
            "Additional request parameters with (list of) string or "
            "numeric values, used by custom extensions."
        ),
    )

    repetition_detection: RepetitionDetectionParams | None = Field(
        default=None,
        description="Parameters for detecting repetitive N-gram patterns "
        "in output tokens. If such repetition is detected, generation will "
        "be ended early. LLMs can sometimes generate repetitive, unhelpful "
        "token patterns, stopping only when they hit the maximum output length "
        "(e.g. 'abcdabcdabcd...' or '\\emoji \\emoji \\emoji ...'). This feature "
        "can detect such behavior and terminate early, saving time and tokens.",
    )

    thinking_token_budget: ThinkingTokenBudget = Field(
        default=None,
        description=(
            "Maximum number of tokens allowed for thinking operations "
            "(reasoning models). Non-negative integer sets the limit; "
            "-1 means unlimited (treated as unset)."
        ),
    )

    # --8<-- [end:completion-extra-params]

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=self.max_tokens or 0,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            truncation_side=self.truncation_side,
            add_special_tokens=self.add_special_tokens,
            needs_detokenization=bool(self.echo and not self.return_token_ids),
            max_total_tokens_param="max_model_len",
            max_output_tokens_param="max_tokens",
            return_token_offsets=bool(self.return_token_offsets),
        )

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

    def extract_structured_outputs(self) -> StructuredOutputsParams | None:
        """Normalize request constraints into ``StructuredOutputsParams``."""
        return structured_outputs_from_response_format(
            self.structured_outputs,
            self.response_format,
        )

    def to_sampling_params(
        self,
        max_tokens: int,
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

        # Merge server-default stop_token_ids (e.g., model-specific tokens
        # like </call> for gpt-oss) with any request-specified ones
        stop_token_ids = self.stop_token_ids
        default_stop_ids = default_sampling_params.get("stop_token_ids")
        if default_stop_ids:
            if not stop_token_ids:
                stop_token_ids = list(default_stop_ids)
            else:
                stop_token_ids = list(
                    dict.fromkeys([*stop_token_ids, *default_stop_ids])
                )

        prompt_logprobs = self.prompt_logprobs
        if prompt_logprobs is None and self.echo:
            prompt_logprobs = self.logprobs

        echo_without_generation = self.echo and self.max_tokens == 0

        extra_args: dict[str, Any] = self.vllm_xargs if self.vllm_xargs else {}
        if self.kv_transfer_params:
            # Pass in kv_transfer_params via extra_args
            extra_args["kv_transfer_params"] = self.kv_transfer_params
        if self.ec_transfer_params:
            # Pass in ec_transfer_params via extra_args
            extra_args["ec_transfer_params"] = self.ec_transfer_params
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
            stop_token_ids=stop_token_ids,
            logprobs=None if self.logprob_token_ids else self.logprobs,
            ignore_eos=self.ignore_eos,
            max_tokens=max_tokens if not echo_without_generation else 1,
            min_tokens=self.min_tokens,
            prompt_logprobs=prompt_logprobs,
            logprob_token_ids=self.logprob_token_ids or None,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            include_stop_str_in_output=self.include_stop_str_in_output,
            output_kind=RequestOutputKind.DELTA
            if self.stream
            else RequestOutputKind.FINAL_ONLY,
            structured_outputs=self.extract_structured_outputs(),
            logit_bias=self.logit_bias,
            allowed_token_ids=self.allowed_token_ids,
            bad_words=self.bad_words,
            extra_args=extra_args or None,
            skip_clone=True,  # Created fresh per request, safe to skip clone
            repetition_detection=self.repetition_detection,
            thinking_token_budget=self.thinking_token_budget,
        )

    @model_validator(mode="before")
    @classmethod
    def normalize_null_max_tokens(cls, data):
        if isinstance(data, dict) and data.get("max_tokens") is None:
            data = data.copy()
            data["max_tokens"] = cls.model_fields["max_tokens"].default
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_response_format(cls, data):
        response_format = data.get("response_format")
        if response_format is None:
            return data

        rf_type = (
            response_format.get("type")
            if isinstance(response_format, dict)
            else getattr(response_format, "type", None)
        )

        if rf_type == "json_schema":
            json_schema = (
                response_format.get("json_schema")
                if isinstance(response_format, dict)
                else getattr(response_format, "json_schema", None)
            )
            if json_schema is None:
                raise VLLMValidationError(
                    "When response_format type is 'json_schema', the "
                    "'json_schema' field must be provided.",
                    parameter="response_format",
                )

        if rf_type == "structural_tag":
            validate_structural_tag_response_format(response_format)

        return data

    @model_validator(mode="before")
    @classmethod
    def check_structured_outputs_count(cls, data):
        if data.get("structured_outputs", None) is None:
            return data

        structured_outputs_kwargs = data["structured_outputs"]
        # structured_outputs may arrive as a dict (from JSON/raw kwargs) or
        # as a StructuredOutputsParams dataclass instance.
        is_dataclass = isinstance(structured_outputs_kwargs, StructuredOutputsParams)
        count = sum(
            (
                getattr(structured_outputs_kwargs, k, None)
                if is_dataclass
                else structured_outputs_kwargs.get(k)
            )
            is not None
            for k in ("json", "regex", "choice")
        )
        if count > 1:
            raise VLLMValidationError(
                "You can only use one kind of constraints for structured "
                "outputs ('json', 'regex' or 'choice').",
                parameter="structured_outputs",
            )
        validate_structured_outputs_structural_tag(structured_outputs_kwargs)
        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if data.get("logprob_token_ids") and data.get("use_beam_search"):
            raise VLLMValidationError(
                "`logprob_token_ids` is not supported with beam search.",
                parameter="logprob_token_ids",
            )

        if (
            data.get("logprob_token_ids")
            and data.get("echo")
            and data.get("max_tokens") == 0
        ):
            raise VLLMValidationError(
                "`logprob_token_ids` is not supported when `echo=True` and "
                "`max_tokens=0` because no output tokens are generated.",
                parameter="logprob_token_ids",
            )

        if data.get("logprob_token_ids") and data.get("logprobs") is None:
            raise VLLMValidationError(
                "when using `logprob_token_ids`, `logprobs` must be set.",
                parameter="logprob_token_ids",
            )

        # These fields are integers, but `mode="before"` runs on the raw
        # request data, so a non-numeric value (e.g. a JSON string) would
        # reach the comparisons below and raise TypeError -> HTTP 500. Reject
        # it here so the client gets a clean 400 instead.
        for field_name in ("prompt_logprobs", "logprobs"):
            field_value = data.get(field_name)
            if field_value is not None and not isinstance(field_value, (int, float)):
                raise VLLMValidationError(
                    f"`{field_name}` must be an integer.",
                    parameter=field_name,
                    value=field_value,
                )
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
            raise VLLMValidationError(
                "Either prompt or prompt_embeds must be provided and non-empty.",
                parameter="prompt",
            )

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_prompt_list_length(cls, data):
        max_prompts = envs.VLLM_MAX_COMPLETION_PROMPTS

        prompt = data.get("prompt")
        if (
            isinstance(prompt, list)
            and len(prompt) > 0
            and not is_list_of(prompt, int)
            and len(prompt) > max_prompts
        ):
            raise VLLMValidationError(
                f"prompt list length {len(prompt)} exceeds the maximum "
                f"allowed count of {max_prompts}. To increase this "
                "limit, set the VLLM_MAX_COMPLETION_PROMPTS "
                "environment variable.",
                parameter="prompt",
            )

        prompt_embeds = data.get("prompt_embeds")
        if isinstance(prompt_embeds, list) and len(prompt_embeds) > max_prompts:
            raise VLLMValidationError(
                f"prompt_embeds list length {len(prompt_embeds)} exceeds "
                f"the maximum allowed count of {max_prompts}. To increase "
                "this limit, set the VLLM_MAX_COMPLETION_PROMPTS "
                "environment variable.",
                parameter="prompt_embeds",
            )

        return data

    @model_validator(mode="before")
    @classmethod
    def check_cache_salt_support(cls, data):
        if data.get("cache_salt") is not None and (
            not isinstance(data["cache_salt"], str) or not data["cache_salt"]
        ):
            raise VLLMValidationError(
                "Parameter 'cache_salt' must be a non-empty string if provided.",
                parameter="cache_salt",
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
    ec_transfer_params: dict[str, Any] | None = Field(
        default=None, description="ECTransfer parameters."
    )
    metrics: PerRequestTimingMetrics | None = None


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
    # Set only on the final chunk of a stream to mirror non-streaming responses
    # without the per-chunk serialization overhead.
    system_fingerprint: str | None = None
    metrics: PerRequestTimingMetrics | None = None
