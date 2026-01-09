# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Set
from typing import TypeAlias

import torch
import torch.nn as nn

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.pooler import ClassifierFn
from vllm.model_executor.layers.pooler.activations import (
    PoolerActivation,
    PoolerNormalize,
    resolve_classifier_act_fn,
)
from vllm.model_executor.models.adapters import _load_st_projector
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from .methods import TokenPoolingMethodOutputItem

TokenPoolerHeadOutputItem: TypeAlias = torch.Tensor | None


class TokenPoolerHead(nn.Module, ABC):
    @abstractmethod
    def get_supported_tasks(self) -> Set[PoolingTask]:
        raise NotImplementedError

    @abstractmethod
    def forward_chunk(
        self,
        pooled_data: TokenPoolingMethodOutputItem,
        pooling_param: PoolingParams,
    ) -> TokenPoolerHeadOutputItem:
        raise NotImplementedError

    def forward(
        self,
        pooled_data: list[TokenPoolingMethodOutputItem],
        pooling_metadata: PoolingMetadata,
    ) -> list[TokenPoolerHeadOutputItem]:
        pooling_params = pooling_metadata.pooling_params
        assert len(pooled_data) == len(pooling_params)

        return [self.forward_chunk(d, p) for d, p in zip(pooled_data, pooling_params)]


class TokenEmbeddingPoolerHead(TokenPoolerHead):
    def __init__(self) -> None:
        super().__init__()

        # Load ST projector if available
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config

        self.projector = _load_st_projector(model_config)
        self.head_dtype = model_config.head_dtype

        self.activation = PoolerNormalize()

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed"}

    def forward_chunk(
        self,
        pooled_data: TokenPoolingMethodOutputItem,
        pooling_param: PoolingParams,
    ) -> TokenPoolerHeadOutputItem:
        # for unfinished chunked prefill
        if pooled_data is None:
            return None

        pooled_data = pooled_data.to(self.head_dtype)
        # pooled_data shape: [n_tokens, hidden_dimension]

        # Apply ST projector
        if self.projector is not None:
            pooled_data = self.projector(pooled_data)
        # pooled_data shape: [n_tokens, embedding_dimension]

        # for matryoshka representation
        pooled_data = pooled_data[..., : pooling_param.dimensions]

        # for normalize
        if pooling_param.normalize:
            pooled_data = self.activation(pooled_data)

        # pooled_data shape: [n_tokens, embedding_dimension]
        return pooled_data


class TokenClassifierPoolerHead(TokenPoolerHead):
    def __init__(
        self,
        classifier: ClassifierFn | None = None,
        act_fn: PoolerActivation | str | None = None,
    ) -> None:
        super().__init__()

        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config

        self.classifier = classifier
        self.logit_bias: float | None = model_config.pooler_config.logit_bias
        self.head_dtype = model_config.head_dtype

        self.act_fn = resolve_classifier_act_fn(
            model_config, static_num_labels=False, act_fn=act_fn
        )

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_classify"}

    def forward_chunk(
        self,
        pooled_data: TokenPoolingMethodOutputItem,
        pooling_param: PoolingParams,
    ) -> TokenPoolerHeadOutputItem:
        # for unfinished chunked prefill
        if pooled_data is None:
            return None

        pooled_data = pooled_data.to(self.head_dtype)
        # hidden_states shape: [n_token, hidden_size]

        if self.classifier is not None:
            scores = self.classifier(pooled_data)
        else:
            scores = pooled_data
        # scores shape: [n_token, num_labels]

        if self.logit_bias is not None:
            scores -= self.logit_bias

        if pooling_param.use_activation:
            scores = self.act_fn(scores)

        # scores shape: [n_token, num_labels]
        return scores
