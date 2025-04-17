# SPDX-License-Identifier: Apache-2.0

from typing import Iterable, Optional, Set, Tuple

import torch
import torch.nn as nn
from transformers import LlamaConfig

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import (LlamaDecoderLayer,
                                              LlamaForCausalLM)
from vllm.v1.sample.metadata import SamplingMetadata

from .utils import AutoWeightsLoader, maybe_prefix

logger = init_logger(__name__)


class LlamaDecoderLayer(LlamaDecoderLayer):

    def __init__(
        self,
        config: LlamaConfig,
        disable_input_layernorm: bool,
        prefix: str = "",
    ) -> None:
        super().__init__(config, prefix=prefix)

        # Skip the input_layernorm
        # https://github.com/SafeAILab/EAGLE/blob/35c78f6cdc19a73e05cf5c330b4c358dad970c6a/eagle/model/cnets.py#L427
        if disable_input_layernorm:
            del self.input_layernorm
            self.input_layernorm = nn.Identity()


class LlamaModel(nn.Module):

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.config.input_hidden_size = 2 * self.config.hidden_size
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                self.config,
                disable_input_layernorm=True,
                prefix=maybe_prefix(prefix, f"layers.{start_layer_id}"),
            )
        ])
        self.fc = torch.nn.Linear(self.config.hidden_size * 3,
                                  self.config.hidden_size,
                                  bias=False)
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

        # d2t=torch.zeros((self.config.draft_vocab_size),dtype=torch.long)
        # t2d=torch.zeros((self.config.vocab_size),dtype=torch.bool)
        # self.register_buffer("d2t", d2t)
        # self.register_buffer("t2d", t2d)

        # self.t2d = nn.Parameter(
        #     torch.zeros((self.config.vocab_size), dtype=torch.bool),
        #     requires_grad=False,
        # )

        self.input_layernorm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.hidden_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        input_embeds = self.embed_tokens(input_ids)
        input_embeds = self.input_layernorm(input_embeds)
        hidden_states = self.hidden_norm(hidden_states)
        if (hidden_states.shape != input_embeds.shape):
            hidden_states = self.fc(hidden_states)
        hidden_states = torch.cat((input_embeds, hidden_states), dim=-1)

        hidden_states, residual = self.layers[0](
            positions,
            hidden_states,
            None,
        )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if 'midlayer.input_layernorm' in name:
                name = name.replace('midlayer.', '')
            if 'midlayer.hidden_norm' in name:
                name = name.replace('midlayer.', '')
            if 'midlayer.' in name:
                name = name.replace('midlayer.', 'layers.0.')
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class EagleLlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, *, model_config: ModelConfig, start_layer_id: int = 0):
        nn.Module.__init__(self)
        self.config = model_config.hf_config
        self.model = LlamaModel(model_config=model_config,
                                start_layer_id=start_layer_id,
                                prefix="model")

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            org_num_embeddings=self.config.draft_vocab_size,
            padding_size=(DEFAULT_VOCAB_PADDING_SIZE),
            prefix="")
        self.logits_processor = LogitsProcessor(self.config.draft_vocab_size,
                                                scale=logit_scale)
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros((self.config.draft_vocab_size),
                        dtype=torch.long).type(torch.LongTensor),
            requires_grad=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, hidden_states)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        # pad logits from draft vocab to target vocab
        # and convert indices accordingly
        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full((
            logits.shape[0],
            self.config.vocab_size,
        ), float('-inf'))
        logits_new[:, targets] = logits
        return logits_new

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            # skip_prefixes=(["lm_head."]
            #                if self.config.tie_word_embeddings else None),
        )

        model_weights = {}
        for name, loaded_weight in weights:
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
            elif "lm_head" not in name:
                name = "model." + name
            model_weights[name] = loaded_weight

        loader.load_weights(model_weights.items())
