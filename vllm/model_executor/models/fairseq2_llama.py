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

from vllm.config import VllmConfig
from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaModel

from .utils import AutoWeightsLoader, WeightsMapper


class Fairseq2LlamaForCausalLM(LlamaForCausalLM):
    def _init_model(self, vllm_config: VllmConfig, prefix: str = ""):
        return LlamaModel(vllm_config=vllm_config, prefix=prefix)

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Set[str]:
        # fairseq2's serialization adds a wrapper to usual .pt state_dict's:
        # { "model_key": my_model_name, "my_model_name": state_dict }
        # which we first need to unpack
        weights_wrapped = dict(weights)
        weights = weights_wrapped[weights_wrapped["model_key"]].items()  # type: ignore

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
            skip_prefixes=(
                ["lm_head."] if self.config.tie_word_embeddings else None
            ),
        )
        return loader.load_weights(
            (
                self.reshape_fairseq2_weights(name, loaded_weight)
                for name, loaded_weight in weights
            ),
            mapper=hf_to_vllm_mapper,
        )

    # This function is used to reshape the fairseq2 weight format
    def reshape_fairseq2_weights(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:
        def permute(w: torch.Tensor, n_heads: int):
            attn_in = self.config.head_dim * n_heads
            attn_out = self.config.hidden_size
            return (
                w.view(n_heads, attn_in // n_heads // 2, 2, attn_out)
                .transpose(1, 2)
                .reshape(attn_in, attn_out)
            )

        modules = name.split(".")

        # rotary embeds should be sliced
        if "k_proj" in modules:
            loaded_weight = permute(
                loaded_weight, self.config.num_key_value_heads
            )
        elif "q_proj" in modules:
            loaded_weight = permute(
                loaded_weight, self.config.num_attention_heads
            )

        return name, loaded_weight
