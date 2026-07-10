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

    def extra_repr(self) -> str:
        attrs = []
        if self.head_dtype is not None:
            attrs.append(f"head_dtype={self.head_dtype}")
        if self.projector is not None:
            attrs.append("projector=True")
        if self.activation is not None:
            attrs.append(f"activation={self.activation.__class__.__name__}")
        return ", ".join(attrs)

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"embed"}

    def forward(
        self,
        pooled_data: SequencePoolingMethodOutput,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolerHeadOutput:
        pooling_params = pooling_metadata.pooling_params
        if len(pooled_data) != len(pooling_params):
            raise ValueError(
                f"pooled_data length ({len(pooled_data)}) does not match "
                f"pooling_params length ({len(pooling_params)})"
            )

        if isinstance(pooled_data, list):
            pooled_data = torch.stack(pooled_data)
        # pooled_data shape: [batchsize, hidden_size]

        if self.head_dtype is not None:
            pooled_data = pooled_data.to(self.head_dtype)

        # Apply ST projector
        if self.projector is not None:
            embeddings = self.projector(pooled_data)
        else:
            embeddings = pooled_data
        # embeddings shape: [batchsize, embedding_size]

        # for matryoshka representation
        dimensions_list = [pooling_param.dimensions for pooling_param in pooling_params]
        if any(d is not None for d in dimensions_list):
            # change the output dimension
            if len(embeddings) != len(dimensions_list):
                raise ValueError(
                    f"embeddings length ({len(embeddings)}) does not match "
                    f"dimensions_list length ({len(dimensions_list)})"
                )
            if len(set(dimensions_list)) == 1 and not isinstance(embeddings, list):
                # if all dimensions are the same
                d = dimensions_list[0]
                embeddings = embeddings[..., :d]
            else:
                embeddings = [
                    vecs if d is None else vecs[..., :d]
                    for vecs, d in zip(embeddings, dimensions_list)
                ]

        # for normalize
        if self.activation is not None:
            flags = [p.use_activation for p in pooling_params]
            if len(set(flags)) == 1:
                if flags[0]:
                    embeddings = self.activation(embeddings)
            else:
                embeddings = [
                    self.activation(vecs) if f else vecs
                    for vecs, f in zip(embeddings, flags)
                ]

        # embeddings shape: [batchsize, embedding_size]
        return embeddings


class ClassifierPoolerHead(SequencePoolerHead):
    def __init__(
        self,
        classifier: ClassifierFn | None = None,
        logit_mean: float | None = None,
        logit_sigma: float | None = None,
        head_dtype: torch.dtype | str | None = None,
        activation: ActivationFn | None = None,
    ) -> None:
        super().__init__()

        self.classifier = classifier
        self.logit_mean = logit_mean
        self.logit_sigma = logit_sigma
        self.head_dtype = head_dtype
        self.activation = activation

    def extra_repr(self) -> str:
        attrs = []
        if self.head_dtype is not None:
            attrs.append(f"head_dtype={self.head_dtype}")
        if self.classifier is not None:
            attrs.append("classifier=True")
        if self.logit_mean is not None:
            attrs.append(f"logit_mean={self.logit_mean}")
        if self.logit_sigma is not None:
            attrs.append(f"logit_sigma={self.logit_sigma}")
        if self.activation is not None:
            attrs.append(f"activation={self.activation.__class__.__name__}")
        return ", ".join(attrs)

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"classify"}

    def forward(
        self,
        pooled_data: SequencePoolingMethodOutput,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolerHeadOutput:
        pooling_params = pooling_metadata.pooling_params
        if len(pooled_data) != len(pooling_params):
            raise ValueError(
                f"pooled_data length ({len(pooled_data)}) does not match "
                f"pooling_params length ({len(pooling_params)})"
            )

        if isinstance(pooled_data, list):
            pooled_data = torch.stack(pooled_data)
        # pooled_data shape: [batchsize, hidden_size]

        if self.head_dtype is not None:
            pooled_data = pooled_data.to(self.head_dtype)

        if self.classifier is not None:
            logits = self.classifier(pooled_data)
        else:
            logits = pooled_data

        # logits shape: [batchsize, num_labels]
        # Affine score calibration: activation((logit - mean) / sigma)
        if self.logit_mean is not None:
            logits = logits - self.logit_mean
        if self.logit_sigma is not None:
            logits = logits / self.logit_sigma

        if self.activation is not None:
            flags = [p.use_activation for p in pooling_params]
            if len(set(flags)) == 1:
                logits = self.activation(logits) if flags[0] else logits
            else:
                logits = [
                    self.activation(vecs) if f else vecs
                    for vecs, f in zip(logits, flags)
                ]

        # logits shape: [batchsize, num_labels]
        return logits
