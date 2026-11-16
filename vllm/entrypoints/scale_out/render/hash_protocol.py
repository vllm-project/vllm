# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt

from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.sampling_params import SamplingParams


class HashTokensRequest(GenerateRequest):
    sampling_params: SamplingParams = Field(
        default_factory=lambda: SamplingParams(max_tokens=1)
    )
    # Raw media needs placeholder expansion through chat rendering.
    content_parts: None = None
    prompt_embeds: None = None


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
