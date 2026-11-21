# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025-2026 SGLang Team
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
# ==============================================================================
"""Native config for the MiniCPM5 MoE model (model_type="minicpm5_moe")."""

from transformers import PreTrainedConfig


class MiniCPM5MoEConfig(PreTrainedConfig):
    model_type = "minicpm5_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | list[int] | None = 2,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        rope_parameters: dict | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        first_k_dense_replace: int = 0,
        n_routed_experts: int = 64,
        n_shared_experts: int = 2,
        routed_scaling_factor: float = 1.0,
        topk_method: str = "greedy",
        num_experts_per_tok: int | None = None,
        moe_intermediate_size: int = 1407,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        use_gated_attention: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        # Accept the checkpoint's Transformers 4 fields and the Transformers 5
        # schema. Never overwrite an explicit rope_parameters dictionary.
        rope_parameters = dict(
            rope_parameters if rope_parameters is not None else (rope_scaling or {})
        )
        legacy_type = rope_parameters.pop("type", "default")
        rope_parameters.setdefault("rope_type", legacy_type)
        rope_parameters.setdefault("rope_theta", rope_theta)
        self.rope_parameters = rope_parameters
        self.rope_theta = rope_parameters["rope_theta"]
        # Transformers 5 exposes rope_scaling as an alias of rope_parameters.
        # Reassigning the legacy value here would discard the normalized dict.
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.first_k_dense_replace = first_k_dense_replace
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.use_gated_attention = use_gated_attention

        super().__init__(
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
