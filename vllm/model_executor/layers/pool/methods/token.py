# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Set

import torch

from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from .base import PoolingMethod, TokenPoolingMethodOutput


class CLSPool(PoolingMethod):
    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify", "embed", "classify", "score"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenPoolingMethodOutput:
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        assert not pooling_cursor.is_partial_prefill(), (
            "partial prefill not supported with CLS pooling"
        )

        return hidden_states[pooling_cursor.first_token_indices_gpu]


class LastPool(PoolingMethod):
    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify", "embed", "classify", "score"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenPoolingMethodOutput:
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        return hidden_states[pooling_cursor.last_token_indices_gpu]


class MeanPool(PoolingMethod):
    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify", "embed", "classify", "score"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenPoolingMethodOutput:
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
