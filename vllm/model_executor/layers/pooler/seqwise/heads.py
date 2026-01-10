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
    def __init__(self) -> None:
        super().__init__()

        # Load ST projector if available
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config

        self.projector = _load_st_projector(model_config)
        self.head_dtype = model_config.head_dtype

        self.activation = PoolerNormalize()

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
        flags = [p.normalize for p in pooling_params]
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
        act_fn: PoolerActivation | str | None = None,
    ) -> None:
        super().__init__()

        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config

        self.classifier = classifier
        self.logit_bias: float | None = model_config.pooler_config.logit_bias
        self.head_dtype = model_config.head_dtype

        self.act_fn = resolve_classifier_act_fn(
            model_config, static_num_labels=True, act_fn=act_fn
        )

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

        pooled_data = pooled_data.to(self.head_dtype)

        if self.classifier is not None:
            pooled_data = self.classifier(pooled_data)
        # pooled_data shape: [batchsize, num_labels]

        if self.logit_bias is not None:
            pooled_data -= self.logit_bias

        flags = [p.use_activation for p in pooling_params]
        if len(set(flags)) == 1:
            scores = self.act_fn(pooled_data) if flags[0] else pooled_data
        else:
            scores = [
                self.act_fn(vecs) if f else vecs for vecs, f in zip(pooled_data, flags)
            ]

        # scores shape: [batchsize, num_labels]
        return scores
