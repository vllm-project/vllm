# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import Any, TypeAlias

from pydantic import BaseModel, Field

from vllm import PoolingParams
from vllm.config import ModelConfig
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.entrypoints.pooling.base.protocol import (
    ClassifyRequestMixin,
    PoolingBasicRequestMixin,
)
from vllm.entrypoints.pooling.score.utils import (
    ScoreContentPartParam,
    ScoreInput,
    ScoreInputs,
)
from vllm.renderers import TokenizeParams
from vllm.utils import random_uuid


class ScoreRequestMixin(PoolingBasicRequestMixin, ClassifyRequestMixin):
    # --8<-- [start:score-extra-params]
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    # --8<-- [end:score-extra-params]

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=0,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            do_lower_case=encoder_config.get("do_lower_case", False),
            max_total_tokens_param="max_model_len",
        )

    def to_pooling_params(self):
        return PoolingParams(
            task="score",
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            use_activation=self.use_activation,
        )


class ScoreDataRequest(ScoreRequestMixin):
    data_1: ScoreInputs
    data_2: ScoreInputs


class ScoreQueriesDocumentsRequest(ScoreRequestMixin):
    queries: ScoreInputs
    documents: ScoreInputs

    @property
    def data_1(self):
        return self.queries

    @property
    def data_2(self):
        return self.documents


class ScoreQueriesItemsRequest(ScoreRequestMixin):
    queries: ScoreInputs
    items: ScoreInputs

    @property
    def data_1(self):
        return self.queries

    @property
    def data_2(self):
        return self.items


class ScoreTextRequest(ScoreRequestMixin):
    text_1: ScoreInputs
    text_2: ScoreInputs

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


class RerankRequest(PoolingBasicRequestMixin, ClassifyRequestMixin):
    query: ScoreInput
    documents: ScoreInputs
    top_n: int = Field(default_factory=lambda: 0)

    # --8<-- [start:rerank-extra-params]
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    # --8<-- [end:rerank-extra-params]

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=0,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            do_lower_case=encoder_config.get("do_lower_case", False),
            max_total_tokens_param="max_model_len",
        )


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
