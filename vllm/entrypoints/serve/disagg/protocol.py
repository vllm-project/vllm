# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from pydantic import BaseModel, Field

from vllm.config import ModelConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionLogProbs
from vllm.entrypoints.openai.engine.protocol import (
    SamplingParams,
    StreamOptions,
)
from vllm.logprobs import Logprob
from vllm.renderers import TokenizeParams
from vllm.utils import random_uuid


####### Tokens IN <> Tokens OUT #######
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

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=0,
            max_total_tokens_param="max_model_len",
        )


class GenerateResponseChoice(BaseModel):
    index: int
    logprobs: ChatCompletionLogProbs | None = None
    # per OpenAI spec this is the default
    finish_reason: str | None = "stop"
    token_ids: list[int] | None = None


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
