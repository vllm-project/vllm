# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Set
from typing import TypeAlias

import torch

from vllm.config import PoolerConfig
from vllm.model_executor.layers.pooler import ClassifierFn, PoolingParamsUpdate
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.layers.pooler.activations import PoolerActivation
from vllm.tasks import POOLING_TASKS, PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from .heads import (
    ClassifierPoolerHead,
    EmbeddingPoolerHead,
    SequencePoolerHead,
    SequencePoolerHeadOutput,
)
from .methods import (
    SequencePoolingMethod,
    SequencePoolingMethodOutput,
    get_seq_pooling_method,
)

SequencePoolingFn: TypeAlias = Callable[
    [torch.Tensor, PoolingMetadata],
    SequencePoolingMethodOutput,
]
SequencePoolingHeadFn: TypeAlias = Callable[
    [SequencePoolingMethodOutput, PoolingMetadata],
    SequencePoolerHeadOutput,
]

SequencePoolerOutput: TypeAlias = torch.Tensor | list[torch.Tensor]


class SequencePooler(Pooler):
    """
    A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Postprocesses the output based on pooling head.
    3. Returns structured results as `PoolerOutput`.
    """

    def __init__(
        self,
        pooling: SequencePoolingMethod | SequencePoolingFn,
        head: SequencePoolerHead | SequencePoolingHeadFn,
    ) -> None:
        super().__init__()

        self.pooling = pooling
        self.head = head

    def get_supported_tasks(self) -> Set[PoolingTask]:
        tasks = set(POOLING_TASKS)

        if isinstance(self.pooling, SequencePoolingMethod):
            tasks &= self.pooling.get_supported_tasks()
        if isinstance(self.head, SequencePoolerHead):
            tasks &= self.head.get_supported_tasks()

        return tasks

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        updates = PoolingParamsUpdate()

        if isinstance(self.pooling, SequencePoolingMethod):
            updates |= self.pooling.get_pooling_updates(task)

        return updates

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolerOutput:
        pooled_data = self.pooling(hidden_states, pooling_metadata)
        pooled_data = self.head(pooled_data, pooling_metadata)
        return pooled_data


def pooler_for_embed(pooler_config: PoolerConfig):
    pooling = get_seq_pooling_method(pooler_config.get_seq_pooling_type())
    head = EmbeddingPoolerHead()

    return SequencePooler(pooling=pooling, head=head)


def pooler_for_classify(
    pooler_config: PoolerConfig,
    *,
    pooling: SequencePoolingMethod | SequencePoolingFn | None = None,
    classifier: ClassifierFn | None = None,
    act_fn: PoolerActivation | str | None = None,
):
    if pooling is None:
        pooling = get_seq_pooling_method(pooler_config.get_seq_pooling_type())

    head = ClassifierPoolerHead(classifier=classifier, act_fn=act_fn)

    return SequencePooler(pooling=pooling, head=head)
