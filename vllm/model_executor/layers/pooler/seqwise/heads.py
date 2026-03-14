# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Set
from typing import TypeAlias

import torch
import torch.nn as nn

from vllm.model_executor.layers.pooler import ActivationFn, ClassifierFn, ProjectorFn
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from .methods import SequencePoolingMethodOutput

SequencePoolerHeadOutput: TypeAlias = torch.Tensor | list[torch.Tensor]


class SequencePoolerHead(nn.Module, ABC):
    @abstractmethod
    def get_supported_tasks(self) -> Set[PoolingTask]:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        pooled_data: SequencePoolingMethodOutput,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolerHeadOutput:
        raise NotImplementedError


class EmbeddingPoolerHead(SequencePoolerHead):
    def __init__(
        self,
        projector: ProjectorFn | None = None,
        head_dtype: torch.dtype | str | None = None,
        activation: ActivationFn | None = None,
    ) -> None:
        super().__init__()

        self.projector = projector
        self.head_dtype = head_dtype
        self.activation = activation

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"embed"}

    def forward(
        self,
        pooled_data: SequencePoolingMethodOutput,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolerHeadOutput:
        pooling_params = pooling_metadata.pooling_params
        assert len(pooled_data) == len(pooling_params)

        if isinstance(pooled_data, list):
            pooled_data = torch.stack(pooled_data)
        # pooled_data shape: [batchsize, hidden_dimension]

        if self.head_dtype is not None:
            pooled_data = pooled_data.to(self.head_dtype)

        # Apply ST projector
        if self.projector is not None:
            pooled_data = self.projector(pooled_data)
        # pooled_data shape: [batchsize, embedding_dimension]

        # for matryoshka representation
        dimensions_list = [pooling_param.dimensions for pooling_param in pooling_params]
        if any(d is not None for d in dimensions_list):
            # change the output dimension
            assert len(pooled_data) == len(dimensions_list)
            if len(set(dimensions_list)) == 1 and not isinstance(pooled_data, list):
                # if all dimensions are the same
                d = dimensions_list[0]
                pooled_data = pooled_data[..., :d]
            else:
                pooled_data = [
                    vecs if d is None else vecs[..., :d]
                    for vecs, d in zip(pooled_data, dimensions_list)
                ]

        # for normalize
        if self.activation is not None:
            flags = [p.use_activation for p in pooling_params]
            if len(set(flags)) == 1:
                if flags[0]:
                    pooled_data = self.activation(pooled_data)
            else:
                pooled_data = [
                    self.activation(vecs) if f else vecs
                    for vecs, f in zip(pooled_data, flags)
                ]

        # pooled_data shape: [batchsize, embedding_dimension]
        return pooled_data


class ClassifierPoolerHead(SequencePoolerHead):
    def __init__(
        self,
        classifier: ClassifierFn | None = None,
        logit_bias: float | None = None,
        head_dtype: torch.dtype | str | None = None,
        activation: ActivationFn | None = None,
    ) -> None:
        super().__init__()

        self.classifier = classifier
        self.logit_bias = logit_bias
        self.head_dtype = head_dtype
        self.activation = activation

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"classify", "score"}

    def forward(
        self,
        pooled_data: SequencePoolingMethodOutput,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolerHeadOutput:
        pooling_params = pooling_metadata.pooling_params
        assert len(pooled_data) == len(pooling_params)

        if isinstance(pooled_data, list):
            pooled_data = torch.stack(pooled_data)
        # pooled_data shape: [batchsize, hidden_size]

        if self.head_dtype is not None:
            pooled_data = pooled_data.to(self.head_dtype)

        if self.classifier is not None:
            pooled_data = self.classifier(pooled_data)
        # pooled_data shape: [batchsize, num_labels]

        if self.logit_bias is not None:
            pooled_data -= self.logit_bias

        if self.activation is not None:
            flags = [p.use_activation for p in pooling_params]
            if len(set(flags)) == 1:
                pooled_data = self.activation(pooled_data) if flags[0] else pooled_data
            else:
                pooled_data = [
                    self.activation(vecs) if f else vecs
                    for vecs, f in zip(pooled_data, flags)
                ]

        # pooled_data shape: [batchsize, num_labels]
        return pooled_data
