# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Callable, Set
from typing import TypeAlias

import torch

from vllm.config import PoolerConfig, get_current_vllm_config
from vllm.model_executor.layers.pooler.activations import (
    PoolerActivation,
    resolve_classifier_act_fn,
)
from vllm.model_executor.layers.pooler.common import ClassifierFn, PoolingParamsUpdate
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from ..abstract import Pooler
from .heads import BatchedPoolerHeadOutput, EmbeddingPoolerHead
from .methods import (
    BatchedPoolingMethod,
    BatchedPoolingMethodOutput,
    get_token_pooling_method,
)

BatchedPoolingFn: TypeAlias = Callable[
    [torch.Tensor, PoolingMetadata],
    BatchedPoolerHeadOutput,
]
BatchedPoolingHeadFn: TypeAlias = Callable[
    [BatchedPoolingMethodOutput, PoolingMetadata],
    BatchedPoolerHeadOutput,
]

BatchedPoolerOutput: TypeAlias = torch.Tensor | list[torch.Tensor]


class BatchedPooler(Pooler):
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> BatchedPoolerOutput:
        raise NotImplementedError


class SimplePooler(BatchedPooler):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.
    """

    def __init__(
        self,
        pooling: BatchedPoolingMethod,
        head: BatchedPoolingHeadFn,
    ) -> None:
        super().__init__()

        self.pooling = pooling
        self.head = head

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return self.pooling.get_supported_tasks()

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return self.pooling.get_pooling_updates(task)

    # Returns subset of BatchedPoolerOutput to satisfy BatchedPoolingFn
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> BatchedPoolerHeadOutput:
        pooled_data = self.pooling(hidden_states, pooling_metadata)
        pooled_data = self.head(pooled_data, pooling_metadata)
        return pooled_data


class ClassifierPooler(BatchedPooler):
    """A pooling layer for classification tasks.

    This layer does the following:
    1. Applies a classification layer to the hidden states.
    2. Optionally applies a pooler layer.
    3. Applies an activation function to the output.
    """

    def __init__(
        self,
        pooling: BatchedPoolingFn,
        classifier: ClassifierFn | None = None,
        act_fn: PoolerActivation | str | None = None,
    ) -> None:
        super().__init__()

        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config

        self.pooling = pooling
        self.classifier = classifier
        self.act_fn = resolve_classifier_act_fn(
            model_config, static_num_labels=True, act_fn=act_fn
        )
        self.logit_bias: float | None = model_config.pooler_config.logit_bias
        self.head_dtype = model_config.head_dtype

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"classify", "score"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> BatchedPoolerOutput:
        pooled_data = self.pooling(hidden_states, pooling_metadata)
        if isinstance(pooled_data, list):
            pooled_data = torch.stack(pooled_data)
        # pooled_data shape: [batchsize, hidden_size]

        pooled_data = pooled_data.to(self.head_dtype)

        if self.classifier is not None:
            pooled_data = self.classifier(pooled_data)
        # pooled_data shape: [batchsize, num_labels]

        if self.logit_bias is not None:
            pooled_data -= self.logit_bias

        pooling_params = pooling_metadata.pooling_params
        flags = [p.use_activation for p in pooling_params]

        if len(set(flags)) == 1:
            scores = self.act_fn(pooled_data) if flags[0] else pooled_data
        else:
            scores = [
                self.act_fn(vecs) if f else vecs for vecs, f in zip(pooled_data, flags)
            ]

        # scores shape: [batchsize, num_labels]
        return scores


def pooler_for_embed(pooler_config: PoolerConfig):
    pooling = get_token_pooling_method(pooler_config.get_pooling_type())
    head = EmbeddingPoolerHead()

    return SimplePooler(pooling=pooling, head=head)


def pooler_for_classify(
    pooler_config: PoolerConfig,
    *,
    classifier: ClassifierFn | None = None,
    act_fn: PoolerActivation | str | None = None,
):
    pooling = get_token_pooling_method(pooler_config.get_pooling_type())

    return ClassifierPooler(pooling=pooling, classifier=classifier, act_fn=act_fn)
