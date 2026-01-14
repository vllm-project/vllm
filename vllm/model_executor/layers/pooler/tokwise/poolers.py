# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Set
from typing import TypeAlias

import torch

from vllm.config import PoolerConfig, get_current_vllm_config
from vllm.model_executor.layers.pooler import ClassifierFn, PoolingParamsUpdate
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.layers.pooler.activations import (
    PoolerActivation,
    PoolerNormalize,
    resolve_classifier_act_fn,
)
from vllm.model_executor.models.adapters import _load_st_projector
from vllm.tasks import POOLING_TASKS, PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from .heads import (
    TokenClassifierPoolerHead,
    TokenEmbeddingPoolerHead,
    TokenPoolerHead,
    TokenPoolerHeadOutputItem,
)
from .methods import (
    TokenPoolingMethod,
    TokenPoolingMethodOutputItem,
    get_tok_pooling_method,
)

TokenPoolingFn: TypeAlias = Callable[
    [torch.Tensor, PoolingMetadata],
    list[TokenPoolingMethodOutputItem],
]
TokenPoolingHeadFn: TypeAlias = Callable[
    [list[TokenPoolingMethodOutputItem], PoolingMetadata],
    list[TokenPoolerHeadOutputItem],
]

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
        pooling: TokenPoolingMethod | TokenPoolingFn,
        head: TokenPoolerHead | TokenPoolingHeadFn,
    ) -> None:
        super().__init__()

        self.pooling = pooling
        self.head = head

    def get_supported_tasks(self) -> Set[PoolingTask]:
        tasks = set(POOLING_TASKS)

        if isinstance(self.pooling, TokenPoolingMethod):
            tasks &= self.pooling.get_supported_tasks()
        if isinstance(self.head, TokenPoolerHead):
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
        pooled_data = self.head(pooled_data, pooling_metadata)
        return pooled_data


def pooler_for_token_embed(pooler_config: PoolerConfig):
    pooling = get_tok_pooling_method(pooler_config.get_tok_pooling_type())

    vllm_config = get_current_vllm_config()
    model_config = vllm_config.model_config
    head = TokenEmbeddingPoolerHead(
        head_dtype=model_config.head_dtype,
        projector=_load_st_projector(model_config),
        activation=PoolerNormalize(),
    )

    return TokenPooler(pooling=pooling, head=head)


def pooler_for_token_classify(
    pooler_config: PoolerConfig,
    *,
    pooling: TokenPoolingMethod | TokenPoolingFn | None = None,
    classifier: ClassifierFn | None = None,
    act_fn: PoolerActivation | str | None = None,
):
    if pooling is None:
        pooling = get_tok_pooling_method(pooler_config.get_tok_pooling_type())

    vllm_config = get_current_vllm_config()
    model_config = vllm_config.model_config
    head = TokenClassifierPoolerHead(
        head_dtype=model_config.head_dtype,
        classifier=classifier,
        logit_bias=model_config.pooler_config.logit_bias,
        activation=resolve_classifier_act_fn(
            model_config, static_num_labels=False, act_fn=act_fn
        ),
    )

    return TokenPooler(pooling=pooling, head=head)
