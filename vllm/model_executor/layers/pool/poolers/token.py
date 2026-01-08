# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Set
from typing import TypeAlias

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.pool.activations import (
    PoolerActivation,
    resolve_classifier_act_fn,
)
from vllm.model_executor.layers.pool.common import ClassifierFn
from vllm.model_executor.layers.pool.methods import TokenPoolingMethodOutput
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from .base import Pooler, TokenPoolerOutput

TokenPoolingMethod: TypeAlias = Callable[
    [torch.Tensor, PoolingMetadata],
    TokenPoolingMethodOutput,
]


class ClassifierPooler(Pooler):
    """A pooling layer for classification tasks.

    This layer does the following:
    1. Applies a classification layer to the hidden states.
    2. Optionally applies a pooler layer.
    3. Applies an activation function to the output.
    """

    def __init__(
        self,
        pooling: TokenPoolingMethod,
        classifier: ClassifierFn | None,
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
    ) -> TokenPoolerOutput:
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
