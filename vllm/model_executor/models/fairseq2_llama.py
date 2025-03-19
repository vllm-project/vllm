# SPDX-License-Identifier: Apache-2.0

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

from typing import Iterable, Set, Tuple

import torch
from torch.nn import Parameter

from vllm.config import VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.linear import set_weight_attrs
from vllm.model_executor.models.llama import LlamaForCausalLM

from .utils import AutoWeightsLoader, WeightsMapper


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

        # remap keys
        fs2_to_vllm_mapper = WeightsMapper(
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
        weights = fs2_to_vllm_mapper.apply(weights)

        params = dict(self.named_parameters())

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(
            (self.reshape_fairseq2_weights(name, loaded_weight, params)
             for name, loaded_weight in weights))

    def flag_sharded_weights(self, params: dict[str, Parameter]):
        """Sets the `is_sharded_weight` flag to True for all sharded weights"""
        for name, param in params.items():
            modules = name.split(".")
            if "norm" in name and len(param.size()) < 2:
                # layer norms are not sharded
                continue
            elif any(emb in modules for emb in ["embed_tokens", "lm_head"]):
                # for now we repeat embedding layers for compatibility
                continue
            else:
                # all other layers are sharded
                set_weight_attrs(param, {"is_sharded_weight": True})

    def reshape_fairseq2_weights(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params: dict[str, Parameter],
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
        # Other weights are flagged for the weight_loader calls.
        if any(emb in modules for emb in ["embed_tokens", "lm_head"]):
            # Embeddings are sharded on dim 0
            dim = 0
            # In fairseq2, vocab size has to be divisible by tp_size
            # so we don't worry about padding
            if self.tp_size > 1 and loaded_weight.shape[
                    dim] < self.config.vocab_size:
                assert loaded_weight.shape[
                    dim] * self.tp_size == self.config.vocab_size, \
                        "vocab_size should be divisible by tp_size."
                repeats = [1] * len(loaded_weight.size())
                repeats[dim] = self.tp_size
                # repeat to match vocab size and to be easily 'narrow'able
                loaded_weight = loaded_weight.repeat(repeats)
                set_weight_attrs(params[name], {"is_sharded_weight": False})
                # if embeddings are sharded, the rest is too
                if "embed_tokens" in modules:
                    self.flag_sharded_weights(params)

        return name, loaded_weight
