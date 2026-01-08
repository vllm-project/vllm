# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import TypeAlias

import torch
import torch.nn as nn

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.pooler.activations import PoolerNormalize
from vllm.model_executor.models.adapters import _load_st_projector
from vllm.v1.pool.metadata import PoolingMetadata

from .methods import BatchedPoolingMethodOutput

BatchedPoolerHeadOutput: TypeAlias = torch.Tensor | list[torch.Tensor]


class BatchedPoolerHead(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        pooled_data: BatchedPoolingMethodOutput,
        pooling_metadata: PoolingMetadata,
    ) -> BatchedPoolerHeadOutput:
        raise NotImplementedError


class EmbeddingPoolerHead(BatchedPoolerHead):
    def __init__(self) -> None:
        super().__init__()

        # Load ST projector if available
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config

        self.projector = _load_st_projector(model_config)
        self.head_dtype = model_config.head_dtype

        self.activation = PoolerNormalize()

    def forward(
        self,
        pooled_data: BatchedPoolingMethodOutput,
        pooling_metadata: PoolingMetadata,
    ) -> BatchedPoolerHeadOutput:
        if isinstance(pooled_data, list):
            pooled_data = torch.stack(pooled_data)
        # pooled_data shape: [batchsize, hidden_dimension]

        pooled_data = pooled_data.to(self.head_dtype)

        # Apply ST projector
        if self.projector is not None:
            pooled_data = self.projector(pooled_data)
        # pooled_data shape: [batchsize, embedding_dimension]

        pooling_params = pooling_metadata.pooling_params

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
