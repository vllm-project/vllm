# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig

from ...sequence import IntermediateTensors
from ...tasks import PoolingTask
from ...v1.pool.metadata import PoolingMetadata
from ..layers.pooler import DispatchPooler
from ..layers.pooler.tokwise import (
    AllPool,
    TokenPooler,
    TokenPoolerHead,
    TokenPoolerHeadOutputItem,
    TokenPoolingMethodOutputItem,
)
from .qwen3 import Qwen3Model
from .utils import maybe_prefix


class JinaForRanking(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.projector_dim = 512
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config

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
                    pooling=AllPool(requires_token_ids=True),
                    head=JinaForRankingPoolerHead(projector=self.projector),
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


class JinaForRankingPoolerHead(TokenPoolerHead):
    def __init__(self, projector: nn.Sequential):
        super().__init__()

        self.doc_embed_token_id = 151670
        self.query_embed_token_id = 151671

        self.projector = projector

    def get_supported_tasks(self) -> set[PoolingTask]:
        return {"token_embed"}

    def forward(
        self,
        pooled_data: list[TokenPoolingMethodOutputItem],
        pooling_metadata: PoolingMetadata,
    ) -> list[TokenPoolerHeadOutputItem]:
        pooling_params = pooling_metadata.pooling_params
        assert len(pooled_data) == len(pooling_params)

        scores = []
        for b in range(len(pooled_data)):
            hidden_states = pooled_data[b]
            prompt_token_ids_cpu = pooling_metadata.prompt_token_ids_cpu[b]

            query_embed_token_indexes = torch.where(
                torch.eq(prompt_token_ids_cpu, self.query_embed_token_id)
            )[0]
            doc_embed_token_indexes = torch.where(
                torch.eq(prompt_token_ids_cpu, self.doc_embed_token_id)
            )[0]

            indexes = torch.cat([query_embed_token_indexes, doc_embed_token_indexes])
            if len(indexes) == 0:
                # profiling run
                scores.append(torch.empty(0))
                continue

            embeds = self.projector(hidden_states[indexes]).float()
            scores.append(F.cosine_similarity(embeds[0], embeds[1:], dim=-1))

        return scores
