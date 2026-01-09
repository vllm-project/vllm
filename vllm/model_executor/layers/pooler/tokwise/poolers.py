# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Set
from typing import TypeAlias

import torch

from vllm.config import PoolerConfig
from vllm.model_executor.layers.pooler.activations import PoolerActivation
from vllm.model_executor.layers.pooler.common import ClassifierFn, PoolingParamsUpdate
from vllm.model_executor.layers.pooler.tokwise.heads import TokenPoolerHead
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from ..abstract import Pooler
from .heads import TokenClassifierPoolerHead, TokenEmbeddingPoolerHead

TokenPoolerOutput: TypeAlias = list[torch.Tensor | None]


class TokenPooler(Pooler):
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenPoolerOutput:
        raise NotImplementedError


class AllPooler(TokenPooler):
    def __init__(self, head: TokenPoolerHead) -> None:
        super().__init__()

        self.pooling = AllPool()
        self.head = head

    def get_supported_tasks(self) -> Set[PoolingTask]:
        tasks = self.pooling.get_supported_tasks()
        tasks &= self.head.get_supported_tasks()
        return tasks

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenPoolerOutput:
        pooled_data = self.pooling(hidden_states, pooling_metadata)
        pooling_params = pooling_metadata.pooling_params
        assert len(pooled_data) == len(pooling_params)

        return [self.head(d, p) for d, p in zip(pooled_data, pooling_params)]


class StepPooler(TokenPooler):
    def __init__(self, head: TokenPoolerHead) -> None:
        super().__init__()

        self.pooling = AllPool()
        self.head = head

    def get_supported_tasks(self) -> Set[PoolingTask]:
        tasks = self.pooling.get_supported_tasks()
        tasks &= self.head.get_supported_tasks()
        return tasks

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

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
                pass
            else:
                step_tag_id = pooling_param.step_tag_id
                returned_token_ids = pooling_param.returned_token_ids

                if returned_token_ids is not None and len(returned_token_ids) > 0:
                    data = data[:, returned_token_ids]

                if step_tag_id is not None:
                    data = data[token_id == step_tag_id]

            pooled_data.append(data)

        return pooled_data

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> TokenPoolerOutput:
        pooled_data = self.extract_states(hidden_states, pooling_metadata)
        pooling_params = pooling_metadata.pooling_params
        assert len(pooled_data) == len(pooling_params)

        return [self.head(d, p) for d, p in zip(pooled_data, pooling_params)]


def pooler_for_token_embed(pooler_config: PoolerConfig):
    pooling_type = pooler_config.get_pooling_type()
    head = TokenEmbeddingPoolerHead()

    if pooling_type == "ALL":
        return AllPooler(head=head)
    if pooling_type == "STEP":
        return StepPooler(head=head)

    # TODO: Have separate pooling types for tokwise and seqwise poolers
    return AllPooler(head=head)


def pooler_for_token_classify(
    pooler_config: PoolerConfig,
    *,
    classifier: ClassifierFn | None = None,
    act_fn: PoolerActivation | str | None = None,
):
    pooling_type = pooler_config.get_pooling_type()
    head = TokenClassifierPoolerHead(classifier=classifier, act_fn=act_fn)

    if pooling_type == "ALL":
        return AllPooler(head=head)
    if pooling_type == "STEP":
        return StepPooler(head=head)

    # TODO: Have separate pooling types for tokwise and seqwise poolers
    return AllPooler(head=head)
