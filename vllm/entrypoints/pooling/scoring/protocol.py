# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import TypeAlias

from pydantic import BaseModel, Field

from vllm import PoolingParams
from vllm.config import ModelConfig
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.renderers import TokenizeParams
from vllm.tasks import PoolingTask
from vllm.utils import random_uuid

from ..base.protocol import ClassifyRequestMixin, PoolingBasicRequestMixin
from .typing import ScoreContentPartParam, ScoreInput


class ScoringRequestMixin(PoolingBasicRequestMixin, ClassifyRequestMixin):
    # --8<-- [start:scoring-common-params]
    max_tokens_per_query: int = Field(
        default=0,
        description=(
            "Maximum number of tokens per query. Queries longer than "
            "this will be truncated to this length. 0 means no "
            "query-level truncation is applied."
        ),
    )
    max_tokens_per_doc: int = Field(
        default=0,
        description=(
            "Maximum number of tokens per document. Documents longer than "
            "this will be truncated to this length. 0 means no "
            "document-level truncation is applied (only truncate_prompt_tokens "
            "applies to the combined query+document)."
        ),
    )
    # --8<-- [end:scoring-common-params]

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=0,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            truncation_side=self.truncation_side,
            do_lower_case=encoder_config.get("do_lower_case", False),
            max_total_tokens_param="max_model_len",
        )

    def to_pooling_params(self, task: PoolingTask = "classify"):
        return PoolingParams(
            task=task,
            use_activation=self.use_activation,
        )


class ScoreDataRequest(ScoringRequestMixin):
    data_1: ScoreInput | list[ScoreInput]
    data_2: ScoreInput | list[ScoreInput]


class ScoreQueriesDocumentsRequest(ScoringRequestMixin):
    # --8<-- [start:score-request-params]
    queries: ScoreInput | list[ScoreInput]
    documents: ScoreInput | list[ScoreInput]
    # --8<-- [end:score-request-params]

    @property
    def data_1(self):
        return self.queries

    @property
    def data_2(self):
        return self.documents


class ScoreQueriesItemsRequest(ScoringRequestMixin):
    queries: ScoreInput | list[ScoreInput]
    items: ScoreInput | list[ScoreInput]

    @property
    def data_1(self):
        return self.queries

    @property
    def data_2(self):
        return self.items


class ScoreTextRequest(ScoringRequestMixin):
    text_1: ScoreInput | list[ScoreInput]
    text_2: ScoreInput | list[ScoreInput]

    @property
    def data_1(self):
        return self.text_1

    @property
    def data_2(self):
        return self.text_2


ScoreRequest: TypeAlias = (
    ScoreQueriesDocumentsRequest
    | ScoreQueriesItemsRequest
    | ScoreDataRequest
    | ScoreTextRequest
)


class RerankRequest(ScoringRequestMixin):
    # --8<-- [start:rerank-request-params]
    query: ScoreInput
    documents: ScoreInput | list[ScoreInput]
    top_n: int = Field(default_factory=lambda: 0)
    # --8<-- [end:rerank-request-params]


ScoringRequest: TypeAlias = ScoreRequest | RerankRequest


class RerankDocument(BaseModel):
    text: str | None = None
    multi_modal: list[ScoreContentPartParam] | None = None


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


ScoringResponse: TypeAlias = RerankResponse | ScoreResponse
