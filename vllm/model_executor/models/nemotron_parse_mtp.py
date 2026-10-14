# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-step auxiliary MTP proposer for Nemotron Parse 3.5."""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


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
        auxiliary_params = {
            "decoder.extra_heads.0.weight": self.model.extra_head.weight,
            "decoder.extra_heads.0.bias": self.model.extra_head.bias,
            "decoder.extra_proj.0.weight": self.model.extra_proj.weight,
            "decoder.extra_proj.0.bias": self.model.extra_proj.bias,
        }
        loaded: set[str] = set()
        loaded_auxiliary: set[str] = set()
        for name, weight in weights:
            if name == "decoder.embed_tokens.weight":
                default_weight_loader(self.model.embed_tokens.weight, weight)
                loaded.add("model.embed_tokens.weight")
            elif name in auxiliary_params:
                if name in loaded_auxiliary:
                    raise RuntimeError(f"Duplicate Parse 3.5 MTP tensor: {name}")
                parameter = auxiliary_params[name]
                if weight.shape != parameter.shape:
                    raise ValueError(
                        f"Parse 3.5 MTP tensor {name} has shape {tuple(weight.shape)}; "
                        f"expected {tuple(parameter.shape)}"
                    )
                default_weight_loader(parameter, weight)
                loaded_auxiliary.add(name)
                suffix = name.rsplit(".", 1)[-1]
                module_name = "extra_head" if ".extra_heads." in name else "extra_proj"
                loaded.add(f"model.{module_name}.{suffix}")
            elif name.startswith(("decoder.extra_heads.", "decoder.extra_proj.")):
                raise RuntimeError(f"Unsupported Parse 3.5 MTP tensor: {name}")

        missing = auxiliary_params.keys() - loaded_auxiliary
        if missing:
            raise RuntimeError(f"Missing Parse 3.5 MTP tensors: {sorted(missing)}")
        if "model.embed_tokens.weight" not in loaded:
            raise RuntimeError("Missing Parse 3.5 MTP embedding")
        return loaded
