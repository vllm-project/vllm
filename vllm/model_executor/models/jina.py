# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://huggingface.co/jinaai/jina-reranker-v3/blob/main/modeling.py
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from ..layers.pooler import DispatchPooler
from ..layers.pooler.tokwise import (
    StepPool,
    TokenPooler,
    TokenPoolingMethodOutputItem,
)
from .interfaces import SupportsLateInteraction
from .qwen3 import Qwen3Model
from .utils import AutoWeightsLoader, maybe_prefix


class JinaForRanking(nn.Module, SupportsLateInteraction):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.projector_dim: int = config.embedding_size

        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2, bias=False),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, self.projector_dim, bias=False),
        )

        self.pooler = DispatchPooler(
            {
                "token_embed": TokenPooler(
                    pooling=JinaForRankingPool(self.projector),
                )
            }
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=(["lm_head."]))
        return loader.load_weights(weights)


class JinaForRankingPool(StepPool):
    def __init__(self, projector: nn.Sequential):
        super().__init__()

        self.doc_token_id = 151670
        self.query_token_id = 151671
        self.projector = projector

    def get_supported_tasks(self) -> set[PoolingTask]:
        return {"token_embed"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> list[TokenPoolingMethodOutputItem]:
        pooled_data_lst = super().forward(hidden_states, pooling_metadata)
        prompt_token_ids = pooling_metadata.get_prompt_token_ids()

        embeds_list = list[torch.Tensor | None]()
        for data, token_ids in zip(pooled_data_lst, prompt_token_ids):
            # for unfinished chunked prefill
            if data is None:
                embeds_list.append(None)
            else:
                docs_indexes = torch.where(torch.eq(token_ids, self.doc_token_id))[0]
                query_indexes = torch.where(torch.eq(token_ids, self.query_token_id))[0]

                # The JinaForRanking model concatenates docs first, then query.
                # Let's stay consistent with this novel design.
                indexes = torch.cat([docs_indexes, query_indexes])
                embeds = self.projector(data[indexes])
                embeds_list.append(embeds)

        return embeds_list
