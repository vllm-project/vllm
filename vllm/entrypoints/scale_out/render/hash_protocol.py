# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from vllm.entrypoints.generate.base.protocol import validate_cache_salt
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    MultiModalFeatures,
)
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid


class HashTokensRequest(BaseModel):
    model: str | None = None
    token_ids: list[NonNegativeInt] = Field(min_length=1)
    features: MultiModalFeatures | None = None
    cache_salt: str | None = None
    # Raw media needs placeholder expansion through chat rendering.
    content_parts: None = None
    prompt_embeds: None = None

    @field_validator("cache_salt", mode="before")
    @classmethod
    def check_cache_salt(cls, value: object) -> object:
        validate_cache_salt(value)
        return value

    @model_validator(mode="after")
    def check_placeholder_bounds(self) -> "HashTokensRequest":
        if self.features:
            for ranges in self.features.mm_placeholders.values():
                if any(p.offset + p.length > len(self.token_ids) for p in ranges):
                    raise ValueError("mm_placeholders must remain within token_ids")
        return self

    def to_generate_request(self) -> GenerateRequest:
        # Hashing accepts metadata-only features without an encoder transfer.
        return GenerateRequest.model_construct(
            request_id=f"hash-{random_uuid()}",
            model=self.model,
            token_ids=self.token_ids,
            features=self.features,
            cache_salt=self.cache_salt,
            sampling_params=SamplingParams(max_tokens=1),
        )


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
