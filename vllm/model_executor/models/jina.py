# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://huggingface.co/jinaai/jina-reranker-v3/blob/main/modeling.py
# ruff: noqa: E501


import torch
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
from .interfaces import SupportsLateInteraction
from .qwen3 import Qwen3Model
from .utils import maybe_prefix


class JinaForRanking(nn.Module, SupportsLateInteraction):
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

        self.doc_token_id = 151670
        self.query_token_id = 151671

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

        embeds_list = []
        for b in range(len(pooled_data)):
            hidden_states = pooled_data[b]
            input_ids = pooling_metadata.prompt_token_ids_cpu[b]

            docs_indexes = torch.where(torch.eq(input_ids, self.doc_token_id))[0]

            query_indexes = torch.where(torch.eq(input_ids, self.query_token_id))[0]

            # The JinaForRanking model concatenates docs first, then query.
            # Let's stay consistent with this novel design.
            indexes = torch.cat([docs_indexes, query_indexes])
            embeds = self.projector(hidden_states[indexes])
            embeds_list.append(embeds)

        return embeds_list


def sanitize_input(text: str, special_tokens: dict[str, str]) -> str:
    for token in special_tokens.values():
        text = text.replace(token, "")
    return text


def format_docs_prompts_func(
    query: str,
    docs: list[str],
    special_tokens: dict[str, str] | None = None,
    instruction: str | None = None,
    no_thinking: bool = True,
) -> str:
    # TODO: Try converting the code below into a chat template.

    default_special_tokens = {
        "query_embed_token": "<|rerank_token|>",
        "doc_embed_token": "<|embed_token|>",
    }
    if special_tokens is None:
        special_tokens = default_special_tokens

    query = sanitize_input(query, special_tokens)
    docs = [sanitize_input(doc, special_tokens) for doc in docs]

    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how well it answers the question. "
        "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    if no_thinking:
        suffix += "<think>\n\n</think>\n\n"

    doc_emb_token = special_tokens["doc_embed_token"]
    query_emb_token = special_tokens["query_embed_token"]

    prompt = (
        f"I will provide you with {len(docs)} passages, each indicated by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
    )

    if instruction:
        prompt += f"<instruct>\n{instruction}\n</instruct>\n"

    doc_prompts = [
        f'<passage id="{i}">\n{doc}{doc_emb_token}\n</passage>'
        for i, doc in enumerate(docs)
    ]
    prompt += "\n".join(doc_prompts) + "\n"
    prompt += f"<query>\n{query}{query_emb_token}\n</query>"

    return prefix + prompt + suffix
