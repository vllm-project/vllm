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

_MEAN_POOL_ACCUMULATION_CHUNK_BYTES = 16 * 1024 * 1024  # 16MB


class SequencePoolingMethod(nn.Module, ABC):
    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify", "embed", "classify"}

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
            hidden_states.device, dtype=torch.int64, non_blocking=True
        )

        num_seqs = prompt_lens.numel()
        hidden_size = hidden_states.shape[-1]

        if num_seqs == 0:
            # early return for empty batch
            return hidden_states.new_empty((0, hidden_size), dtype=torch.float32)

        # eg. [2, 1, 3] -> [0, 0, 1, 2, 2, 2]
        segment_ids = torch.repeat_interleave(
            torch.arange(num_seqs, device=hidden_states.device, dtype=torch.long),
            prompt_lens,
        )
        segment_sums = torch.zeros(
            (num_seqs, hidden_size),
            dtype=torch.float32,
            device=hidden_states.device,
        )

        bytes_per_token = hidden_size * torch.finfo(torch.float32).bits // 8
        chunk_size = max(1, _MEAN_POOL_ACCUMULATION_CHUNK_BYTES // bytes_per_token)

        # iterate over the batch in chunks
        for start in range(0, hidden_states.shape[0], chunk_size):
            end = min(start + chunk_size, hidden_states.shape[0])
            # using index_add_ to accumulate for each segment
            segment_sums.index_add_(
                0,
                segment_ids[start:end],
                hidden_states[start:end].to(dtype=torch.float32),
            )

        return segment_sums / prompt_lens.unsqueeze(1)


def get_seq_pooling_method(pooling_type: SequencePoolingType | str):
    if pooling_type == "CLS":
        return CLSPool()
    if pooling_type == "LAST":
        return LastPool()
    if pooling_type == "MEAN":
        return MeanPool()

    raise NotImplementedError(f"Unknown sequence pooling type: {pooling_type!r}")
