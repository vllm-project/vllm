# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drex pointer pooler.

Drex reads no generated token. Each request row carries the packed state and
one question branch; the caller passes the row positions of every ``</opt>``
and of ``<decide>`` through ``PoolingParams.extra_kwargs`` and the pooler
scores the options from the row's hidden states.
"""

import torch

from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from .drex_head import PointerHead, scores_for_pooling


class DrexPointerPooler(Pooler):
    """Score each row's options with the trained pointer head."""

    def __init__(self, head: PointerHead) -> None:
        super().__init__()
        self.pooling = AllPool()
        self.head = head

    def get_supported_tasks(self) -> set[PoolingTask]:
        return {"classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        # Marker positions arrive through `extra_kwargs`, so the pooler does
        # not need the prompt token ids.
        return PoolingParamsUpdate()

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        pooled = self.pooling(hidden_states, pooling_metadata)
        outputs: list[torch.Tensor | None] = []
        for hidden, params in zip(pooled, pooling_metadata.pooling_params):
            if hidden is None:
                outputs.append(None)
                continue
            extra = params.extra_kwargs or {}
            outputs.append(
                scores_for_pooling(
                    hidden,
                    extra.get("opt_positions"),
                    extra.get("decide_position"),
                    params.use_activation,
                    self.head,
                )
            )
        return outputs
