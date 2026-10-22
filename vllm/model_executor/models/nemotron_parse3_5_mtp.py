# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-step auxiliary MTP proposer for Nemotron Parse 3.5."""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper


class _ParseAuxMTPInner(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.extra_proj = nn.Linear(hidden_size, hidden_size)
        self.extra_head = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor):
        token_embeds = self.embed_tokens(input_ids)
        return self.extra_head(hidden_states + self.extra_proj(token_embeds))


class ParseAuxMTPModel(nn.Module):
    """Draft one token using Parse 3.5's trained auxiliary projections."""

    has_own_embed_tokens = False
    has_own_lm_head = False

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "decoder.embed_tokens.": "model.embed_tokens.",
            "decoder.extra_heads.0.": "model.extra_head.",
            "decoder.extra_proj.0.": "model.extra_proj.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.config = vllm_config.model_config.hf_config
        decoder_config = self.config.decoder
        vocab_size = int(decoder_config.vocab_size)
        hidden_size = int(decoder_config.d_model)

        self.model = _ParseAuxMTPInner(vocab_size, hidden_size)
        self.lm_head = ParallelLMHead(vocab_size, hidden_size)
        self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(vocab_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del positions, inputs_embeds, kwargs
        return self.model(input_ids, hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def mtp_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                if name == "decoder.embed_tokens.weight" or name.startswith(
                    ("decoder.extra_heads.0.", "decoder.extra_proj.0.")
                ):
                    yield name, weight

        loader = AutoWeightsLoader(self)
        return loader.load_weights(mtp_weights(), mapper=self.hf_to_vllm_mapper)
