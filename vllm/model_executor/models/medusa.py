# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .utils import maybe_prefix


class ResidualBlock(nn.Module):

    def __init__(self, config: VllmConfig, hidden_size: int,
                 num_layers: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(hidden_size,
                      hidden_size,
                      bias=getattr(config, "medusa_fc_bias", False))
            for _ in range(num_layers)
        ])
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + self.act(layer(x))
        return x


class Medusa(nn.Module):
    """This class implements the Medusa draft model from the paper: https://arxiv.org/abs/2401.10774
    Reference implementation: https://github.com/FasterDecoding/Medusa
    
    Differences from reference implementation:
    1. Currently this only supports generating proposals from top-1 tokens.
    2. We have an optional token_map which reduces draft vocab to most 
       frequently used tokens to give some additional speed-up by reducing 
       sampling overhead. This is disabled unless the checkpoint file has 
       explicit token_map tensor and config has an optional attribute 
       truncated_vocab_size < vocab_size. To use this technique, one has to find
       the top-k most frequent tokens in target dataset and add that as a tensor
       in the draft checkpoint (using key token_map). Also, the draft config
       needs to have truncated_vocab_size (=k) as an attribute."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.speculative_config.draft_model_config.hf_config
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([
            ResidualBlock(config=config,
                          hidden_size=self.config.hidden_size,
                          num_layers=self.config.num_hidden_layers)
            for _ in range(self.config.num_heads)
        ])
        self.orig_vocab_size = config.vocab_size
        self.truncated_vocab_size = config.truncated_vocab_size
        self.unpadded_vocab_size = self.truncated_vocab_size

        if getattr(config, "original_lm_head", False):
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=self.truncated_vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.lm_heads = [
                self.lm_head for _ in range(self.config.num_heads)
            ]
        else:
            self.lm_heads = nn.ModuleList([
                ParallelLMHead(
                    self.unpadded_vocab_size,
                    config.hidden_size,
                    org_num_embeddings=self.truncated_vocab_size,
                    padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                ) for _ in range(self.config.num_heads)
            ])

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                self.truncated_vocab_size,
                                                logit_scale)

        # Token map is a idx to token mapping to reduce the vocab size for
        # the draft model. Using smaller vocab size for draft, containing
        # only most frequent tokens reduces the speculation overhead. This
        # doesn't affect the acceptance rate much and thus gives more speed
        # -up. By default, this is disabled and is only used if the EAGLE
        # checkpoint file has token_map tensor.
        self.token_map = None

    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        return [block(hidden_states) for block in self.blocks]

    def compute_logits(
        self,
        hidden_states: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        logits_lst: list[torch.Tensor] = []

        for hs, lm_head in zip(hidden_states, self.lm_heads):
            _logits = self.logits_processor(lm_head, hs)

            if _logits is None:
                # _logits should only be None on rank > 0, in which case
                # it should remain true for every lm_head
                assert len(logits_lst) == 0
                continue

            if self.token_map is None:
                logits_lst.append(_logits)
            else:
                logits_lst.append(-torch.inf * torch.ones(
                    size=(*_logits.shape[:-1], self.orig_vocab_size),
                    device=_logits.device,
                    dtype=_logits.dtype))

                logits_lst[-1][..., self.token_map] = _logits

        return logits_lst

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        weights_map = {}

        for name, loaded_weight in weights:
            name = name.replace("medusa_heads.", "")

            if name == "token_map":
                if self.truncated_vocab_size < self.orig_vocab_size:
                    self.token_map = nn.Parameter(loaded_weight,
                                                  requires_grad=False)
            elif name in params_dict:
                weights_map[name] = loaded_weight
            elif (getattr(self.config, "original_lm_head", False)
                  and name == "lm_heads.0.weight"):
                weights_map["lm_head.weight"] = loaded_weight

        for name, loaded_weight in weights_map.items():
            if "lm_head" in name and self.token_map is not None and\
                loaded_weight.shape[0] > self.token_map.shape[0]:

                loaded_weight = loaded_weight[self.token_map]

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        if self.token_map is not None:
            self.token_map.to(device=self.lm_heads[0].weight.device)

        assert (self.truncated_vocab_size
                == self.orig_vocab_size) or (self.token_map is not None)

        return loaded_params
