# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Set
from typing import TypeAlias

import torch

from vllm.config import PoolerConfig
from vllm.model_executor.layers.pooler.activations import PoolerActivation
from vllm.model_executor.layers.pooler.common import ClassifierFn, PoolingParamsUpdate
from vllm.model_executor.layers.pooler.tokwise.heads import TokenPoolerHead
from vllm.model_executor.layers.pooler.tokwise.methods import (
    TokenPoolingMethod,
    get_tok_pooling_method,
)
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from ..abstract import Pooler
from .heads import TokenClassifierPoolerHead, TokenEmbeddingPoolerHead

TokenPoolerOutput: TypeAlias = list[torch.Tensor | None]


class TokenPooler(Pooler):
    """
    A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Postprocesses the output based on pooling head.
    3. Returns structured results as `PoolerOutput`.
    """

    def __init__(
        self,
        pooling: TokenPoolingMethod,
        head: TokenPoolerHead,
    ) -> None:
        super().__init__()

        self.pooling = pooling
        self.head = head

    def get_supported_tasks(self) -> Set[PoolingTask]:
        tasks = set[PoolingTask]()

        tasks &= self.pooling.get_supported_tasks()
        tasks &= self.head.get_supported_tasks()

        return tasks

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        updates = PoolingParamsUpdate()

        if isinstance(self.pooling, TokenPoolingMethod):
            updates |= self.pooling.get_pooling_updates(task)

        return updates

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenPoolerOutput:
        pooled_data = self.pooling(hidden_states, pooling_metadata)
        pooling_params = pooling_metadata.pooling_params
        assert len(pooled_data) == len(pooling_params)

        return [self.head(d, p) for d, p in zip(pooled_data, pooling_params)]


def pooler_for_token_embed(pooler_config: PoolerConfig):
    pooling = get_tok_pooling_method(pooler_config.get_pooling_type())
    head = TokenEmbeddingPoolerHead()

    return TokenPooler(pooling=pooling, head=head)


def pooler_for_token_classify(
    pooler_config: PoolerConfig,
    *,
    classifier: ClassifierFn | None = None,
    act_fn: PoolerActivation | str | None = None,
):
    pooling = get_tok_pooling_method(pooler_config.get_pooling_type())
    head = TokenClassifierPoolerHead(classifier=classifier, act_fn=act_fn)

    return TokenPooler(pooling=pooling, head=head)
