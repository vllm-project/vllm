# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import Any, TypeAlias

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from vllm import PoolingParams
from vllm.config.pooler import get_use_activation
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.entrypoints.pooling.base.protocol import (
    ClassifyRequestMixin,
    PoolingBasicRequestMixin,
)
from vllm.entrypoints.pooling.score.utils import (
    ScoreContentPartParam,
    ScoreMultiModalParam,
)
from vllm.utils import random_uuid

# Exact number of token IDs required in label_token_ids for generative scoring
REQUIRED_LABEL_TOKEN_IDS = 2


class ScoreRequestMixin(PoolingBasicRequestMixin, ClassifyRequestMixin):
    # --8<-- [start:score-extra-params]
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    label_token_ids: list[int] | None = Field(
        default=None,
        description=(
            "List of token IDs to compute probabilities for when using "
            f"CausalLM models. Required for generative scoring. "
            f"Must contain exactly {REQUIRED_LABEL_TOKEN_IDS} token IDs."
        ),
    )
    # --8<-- [end:score-extra-params]

    @field_validator('label_token_ids')
    @classmethod
    def validate_label_token_ids(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and len(v) != REQUIRED_LABEL_TOKEN_IDS:
            raise ValueError(
                f"label_token_ids must contain exactly {REQUIRED_LABEL_TOKEN_IDS} "
                f"token IDs for generative scoring, but got {len(v)}"
            )
        return v

    def to_pooling_params(self):
        return PoolingParams(
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            use_activation=get_use_activation(self),
        )


class ScoreDataRequest(ScoreRequestMixin):
    data_1: list[str] | str | ScoreMultiModalParam
    data_2: list[str] | str | ScoreMultiModalParam


class ScoreQueriesDocumentsRequest(ScoreRequestMixin):
    queries: list[str] | str | ScoreMultiModalParam
    documents: list[str] | str | ScoreMultiModalParam

    @property
    def data_1(self):
        return self.queries

    @property
    def data_2(self):
        return self.documents


class ScoreTextRequest(ScoreRequestMixin):
    text_1: list[str] | str | ScoreMultiModalParam
    text_2: list[str] | str | ScoreMultiModalParam

    @property
    def data_1(self):
        return self.text_1

    @property
    def data_2(self):
        return self.text_2


ScoreRequest: TypeAlias = (
    ScoreQueriesDocumentsRequest | ScoreDataRequest | ScoreTextRequest
)


class RerankRequest(PoolingBasicRequestMixin, ClassifyRequestMixin):
    query: str | ScoreMultiModalParam
    documents: list[str] | ScoreMultiModalParam
    top_n: int = Field(default_factory=lambda: 0)

    # --8<-- [start:rerank-extra-params]
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    # --8<-- [end:rerank-extra-params]


class RerankDocument(BaseModel):
    text: str | None = None
    multi_modal: ScoreContentPartParam | None = None


class RerankResult(BaseModel):
    index: int
    document: RerankDocument
    relevance_score: float


class RerankUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class RerankResponse(OpenAIBaseModel):
    id: str
    model: str
    usage: RerankUsage
    results: list[RerankResult]


class ScoreResponseData(OpenAIBaseModel):
    index: int
    object: str = "score"
    score: float


class ScoreResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"embd-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[ScoreResponseData]
    usage: UsageInfo
