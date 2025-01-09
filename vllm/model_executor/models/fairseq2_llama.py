# Copyright 2024 The vLLM team.
# Copyright 2024 Meta Platforms, Inc. and affiliates. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Llama model for fairseq2 weights."""

import types
from typing import Iterable, Set, Tuple

import torch

from vllm.config import VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.logger import init_logger
from vllm.model_executor.models.llama import LlamaForCausalLM

from .utils import AutoWeightsLoader, WeightsMapper

logger = init_logger(__name__)


class Fairseq2LlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        # For the model loader to read only the relevant checkpoint files
        self.allow_patterns_overrides = [
            # either the full checkpoint
            "model.pt",
            # or the tp-sharded checkpoint of the current rank
            f"model.{self.tp_rank}.pt",
        ]

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        # fairseq2's serialization adds a wrapper to usual .pt state_dict's:
        # { "model_key": my_model_name, "my_model_name": state_dict }
        # which we first need to unpack
        weights_wrapped = dict(weights)
        weights = weights_wrapped[
            weights_wrapped["model_key"]].items()  # type: ignore

        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "decoder_frontend.embed.": "model.embed_tokens.",
                "decoder.": "model.",
                "final_proj.": "lm_head.",
            },
            orig_to_new_substr={
                ".self_attn_layer_norm.": ".input_layernorm.",
                ".ffn_layer_norm.": ".post_attention_layernorm.",
                ".self_attn.output_proj.": ".self_attn.o_proj.",
                ".ffn.gate_proj.": ".mlp.gate_proj.",
                ".ffn.inner_proj.": ".mlp.up_proj.",
                ".ffn.output_proj.": ".mlp.down_proj.",
                ".layer_norm.": ".norm.",
            },
        )
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(
            (self.reshape_fairseq2_weights(name, loaded_weight)
             for name, loaded_weight in weights),
            mapper=hf_to_vllm_mapper,
        )

    def reshape_fairseq2_weights(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:
        """Reshape fairseq2's weights."""

        def permute(w: torch.Tensor, n_heads: int) -> torch.Tensor:
            attn_in = self.config.head_dim * n_heads
            # check for a sharded weight on dim 0
            if attn_in // self.tp_size == w.size()[0]:
                attn_in //= self.tp_size
                n_heads //= self.tp_size
            attn_out = self.config.hidden_size
            return (w.view(n_heads, attn_in // n_heads // 2, 2,
                           attn_out).transpose(1,
                                               2).reshape(attn_in, attn_out))

        modules = name.split(".")

        # rotary embeds should be sliced
        if "k_proj" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_key_value_heads)

        elif "q_proj" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_attention_heads)

        # We make the loaded weights compatible with both
        # full checkpoints and tp sharded checkpoints.
        # Embeddings are repeated to fit the vocab size.
        # Other weights have their 'narrow' method monkey-patched.
        if any(emb in modules for emb in ["embed", "final_proj"]):
            # Embeddings are sharded on dim 0
            dim = 0
            # In fairseq2, vocab size has to be divisible by tp_size
            # so we don't worry about padding
            if self.tp_size > 1 and loaded_weight.shape[
                    dim] * self.tp_size == self.config.vocab_size:
                repeats = [1] * len(loaded_weight.size())
                repeats[dim] = self.tp_size
                # repeat to match vocab size and to be easily 'narrow'able
                loaded_weight = loaded_weight.repeat(repeats)
        else:
            # Monkey-patch the 'narrow' method to be conditional on tp_size:
            # if the checkpoint is already tp-sharded, we don't need to
            # narrow weights in weight_loader calls
            def maybe_narrow(self, dim: int, start: int, length: int):
                tp_size = get_tensor_model_parallel_world_size()
                if tp_size > 1 and self.shape[dim] // tp_size == length:
                    # weight is full and has to be narrowed
                    return torch.narrow(self, dim, start, length)
                else:
                    return self

            loaded_weight.narrow = types.MethodType(maybe_narrow,
                                                    loaded_weight)

        return name, loaded_weight
