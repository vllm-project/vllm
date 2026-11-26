# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field, NonNegativeInt, PositiveInt, Tag

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.sampling_params import SamplingParams


class HashTokensRequest(GenerateRequest):
    sampling_params: SamplingParams = Field(
        default_factory=lambda: SamplingParams(max_tokens=1)
    )
    # Raw media needs placeholder expansion through chat rendering.
    content_parts: None = None
    prompt_embeds: None = None


def _request_kind(value: Any) -> str | None:
    if isinstance(value, ChatCompletionRequest):
        return "chat"
    if isinstance(value, CompletionRequest):
        return "completion"
    if isinstance(value, HashTokensRequest):
        return "tokens"
    if isinstance(value, dict):
        kinds = {"messages": "chat", "prompt": "completion", "token_ids": "tokens"}
        present = [kind for key, kind in kinds.items() if key in value]
        if len(present) == 1:
            return present[0]
    return None


HashRequest = Annotated[
    Annotated[ChatCompletionRequest, Tag("chat")]
    | Annotated[CompletionRequest, Tag("completion")]
    | Annotated[HashTokensRequest, Tag("tokens")],
    Discriminator(_request_kind),
]


class HashPrompt(BaseModel):
    num_tokens: NonNegativeInt
    block_hashes: list[str]


class HashGroup(BaseModel):
    group_idx: NonNegativeInt
    kind: str
    block_size: PositiveInt
    sliding_window: int | None = None


class HashResponse(BaseModel):
    schema_version: Literal[1] = 1
    model: str
    hash_algorithm: str
    hash_encoding: Literal["hex"] = "hex"
    hash_block_size: PositiveInt
    num_hash_lookahead_tokens: NonNegativeInt
    groups: list[HashGroup]
    prompts: list[HashPrompt]
