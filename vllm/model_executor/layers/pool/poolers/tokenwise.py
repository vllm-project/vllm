# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Set

import torch

from vllm.model_executor.layers.pool.common import PoolingParamsUpdate
from vllm.model_executor.layers.pool.heads.tokenwise import TokenwisePoolerHead
from vllm.model_executor.layers.pool.methods.tokenwise import AllPool
from vllm.tasks import PoolingTask
from vllm.v1.outputs import TokenwisePoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from .base import Pooler


class AllPooler(Pooler):
    def __init__(self, head: TokenwisePoolerHead) -> None:
        super().__init__()

        self.pooling = AllPool()
        self.head = head

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenwisePoolerOutput:
        pooled_data = self.pooling(hidden_states, pooling_metadata)
        pooling_params = pooling_metadata.pooling_params
        assert len(pooled_data) == len(pooling_params)

        return [self.head(d, p) for d, p in zip(pooled_data, pooling_params)]


class StepPooler(Pooler):
    def __init__(self, head: TokenwisePoolerHead) -> None:
        super().__init__()

        self.pooling = AllPool()
        self.head = head

    def extract_states(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> list[torch.Tensor | None]:
        pooled_data_lst = self.pooling(hidden_states, pooling_metadata)
        prompt_token_ids = pooling_metadata.get_prompt_token_ids()
        pooling_params = pooling_metadata.pooling_params

        pooled_data = list[torch.Tensor | None]()
        for data, token_id, pooling_param in zip(
            pooled_data_lst, prompt_token_ids, pooling_params
        ):
            # for unfinished chunked prefill
            if data is None:
                pooled_data.append(data)
                continue

            step_tag_id = pooling_param.step_tag_id
            returned_token_ids = pooling_param.returned_token_ids

            if returned_token_ids is not None and len(returned_token_ids) > 0:
                data = data[:, returned_token_ids]

            if step_tag_id is not None:
                data = data[token_id == step_tag_id]
            pooled_data.append(data)

        return pooled_data

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenwisePoolerOutput:
        pooled_data = self.extract_states(hidden_states, pooling_metadata)
        pooling_params = pooling_metadata.pooling_params
        assert len(pooled_data) == len(pooling_params)

        return [self.head(d, p) for d, p in zip(pooled_data, pooling_params)]
