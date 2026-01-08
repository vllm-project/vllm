# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Set
from typing import TypeAlias

import torch
import torch.nn as nn

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

RequestPoolingMethodOutputItem: TypeAlias = torch.Tensor | None


class RequestPoolingMethod(nn.Module, ABC):
    @abstractmethod
    def get_supported_tasks(self) -> Set[PoolingTask]:
        raise NotImplementedError

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> list[RequestPoolingMethodOutputItem]:
        raise NotImplementedError


class AllPool(RequestPoolingMethod):
    def __init__(self):
        super().__init__()

        vllm_config = get_current_vllm_config()
        scheduler_config = vllm_config.scheduler_config

        self.enable_chunked_prefill = scheduler_config.enable_chunked_prefill

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> list[RequestPoolingMethodOutputItem]:
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        hidden_states_all = hidden_states.split(
            pooling_cursor.num_scheduled_tokens_cpu.tolist()
        )
        hidden_states_lst = [hidden_states_all[i] for i in pooling_cursor.index]

        if not self.enable_chunked_prefill:
            return hidden_states_lst

        pooling_states = pooling_metadata.pooling_states

        # If chunked_prefill is enabled
        # 1. first store the chunked hidden_states in pooling_states.hidden_states_cache
        for p, hs_chunk in zip(pooling_states, hidden_states_lst):
            p.hidden_states_cache.append(hs_chunk)

        # 2. Once prefill is finished, send hidden_states_cache to PoolerHead
        output_list = list[RequestPoolingMethodOutputItem]()
        for p, finished in zip(pooling_states, pooling_cursor.is_finished()):
            if finished:
                hidden_states_cache = p.hidden_states_cache
                if len(hidden_states_cache) == 1:
                    output_list.append(hidden_states_cache[0])
                else:
                    output_list.append(torch.concat(hidden_states_cache, dim=0))
                p.clean()
            else:
                output_list.append(None)

        return output_list
