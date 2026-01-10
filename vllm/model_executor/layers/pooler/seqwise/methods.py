# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Set
from typing import TypeAlias

import torch
import torch.nn as nn

from vllm.config.pooler import SequencePoolingType
from vllm.model_executor.layers.pooler import PoolingParamsUpdate
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

SequencePoolingMethodOutput: TypeAlias = torch.Tensor | list[torch.Tensor]


class SequencePoolingMethod(nn.Module, ABC):
    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify", "embed", "classify", "score"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolingMethodOutput:
        raise NotImplementedError


class CLSPool(SequencePoolingMethod):
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolingMethodOutput:
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        assert not pooling_cursor.is_partial_prefill(), (
            "partial prefill not supported with CLS pooling"
        )

        return hidden_states[pooling_cursor.first_token_indices_gpu]


class LastPool(SequencePoolingMethod):
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolingMethodOutput:
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        return hidden_states[pooling_cursor.last_token_indices_gpu]


class MeanPool(SequencePoolingMethod):
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolingMethodOutput:
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        assert not pooling_cursor.is_partial_prefill(), (
            "partial prefill not supported with MEAN pooling"
        )

        prompt_lens = pooling_cursor.prompt_lens_cpu.to(
            hidden_states.device, non_blocking=True
        )

        # Use float32 for torch.cumsum in MeanPool,
        # otherwise precision will be lost significantly.
        cumsum = torch.cumsum(hidden_states, dim=0, dtype=torch.float32)

        start_indices = pooling_cursor.first_token_indices_gpu
        end_indices = pooling_cursor.last_token_indices_gpu

        return (
            cumsum[end_indices] - cumsum[start_indices] + hidden_states[start_indices]
        ) / prompt_lens.unsqueeze(1)


def get_seq_pooling_method(pooling_type: SequencePoolingType | str):
    if pooling_type == "CLS":
        return CLSPool()
    if pooling_type == "LAST":
        return LastPool()
    if pooling_type == "MEAN":
        return MeanPool()

    raise NotImplementedError(f"Unknown sequence pooling type: {pooling_type!r}")
