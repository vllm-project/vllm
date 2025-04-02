# SPDX-License-Identifier: Apache-2.0

from typing import Iterable, Tuple

import torch
import torch.nn as nn
from transformers import LlamaConfig

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.llama import (LlamaDecoderLayer,
                                              LlamaForCausalLM)

from .utils import AutoWeightsLoader, maybe_prefix

logger = init_logger(__name__)


class LlamaDecoderLayer(LlamaDecoderLayer):

    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(config, prefix=prefix)

        # Skip the input_layernorm
        # https://github.com/SafeAILab/EAGLE/blob/35c78f6cdc19a73e05cf5c330b4c358dad970c6a/eagle/model/cnets.py#L427
        if layer_id == 0:
            del self.input_layernorm
            self.input_layernorm = lambda x: x


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                config,
                i,
                prefix=maybe_prefix(f"layers.{i}", prefix),
            ) for i in range(config.num_hidden_layers)
        ])
        self.fc = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        input_embeds = self.embed_tokens(input_ids)
        hidden_states = self.fc(
            torch.cat((input_embeds, hidden_states), dim=-1))
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        return hidden_states + residual


class LlamaForCausalLMEagle(LlamaForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = LlamaModel(config=config,
                                prefix=maybe_prefix(prefix, "eagle"))

        self.truncated_vocab_size = config.truncated_vocab_size
        self.unpadded_vocab_size = self.truncated_vocab_size

        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=self.truncated_vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            )

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                self.truncated_vocab_size,
                                                logit_scale)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        print("===========")
        print_module_names(self.model)
        print("===========")
        loader = AutoWeightsLoader(self)

        for name, loaded_weight in weights:
            if "lm_head" not in name:
                name = "eagle." + name
            loader.load_weights(name, loaded_weight)


def print_module_names(module, prefix=""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        print(full_name)
        print_module_names(child, full_name)
