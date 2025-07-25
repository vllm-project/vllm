# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2023 Stability AI, EleutherAI, and The HuggingFace Inc. team.
# All rights reserved.
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
"""StableLM-Alpha model configuration"""
# ruff: noqa: E501
from transformers import PretrainedConfig


class StableLMAlphaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`StableLMAlphaModel`]. It is used to instantiate
    a StableLM-Alpha model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the StableLM-Alpha
    [stabilityai/stablelm-base-alpha-3b](https://huggingface.co/stabilityai/stablelm-base-alpha-3b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the StableLM-Alpha model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`StableLMAlphaModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the normalization layers.
        rotary_pct (`float`, *optional*, defaults to 1.0):
            The percentage of the hidden size to use for rotary position embeddings.
        rotary_emb_base (`int`, *optional*, defaults to 10000):
            The base period of the RoPE embeddings.
        use_qkv_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the query, key, and value projection layers.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.

    Example:

    ```python
    >>> from transformers import StableLMAlphaModel, StableLMAlphaConfig

    >>> # Initializing a StableLM-Alpha stablelm-base-alpha-3b style configuration
    >>> configuration = StableLMAlphaConfig()

    >>> # Initializing a model from the stablelm-base-alpha-3b style configuration
    >>> model = StableLMAlphaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "stablelm_alpha"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50432,
        hidden_size=2560,
        num_hidden_layers=32,
        num_heads=32,
        max_position_embeddings=4096,
        initializer_range=0.02,
        norm_eps=1e-5,
        hidden_act="silu",
        rotary_pct=0.25,
        rotary_emb_base=10000,
        rotary_scaling_factor=1.0,
        tie_word_embeddings=False,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.hidden_act = hidden_act
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.rotary_scaling_factor = rotary_scaling_factor
        self.use_cache = use_cache

        # For compatibility with vLLM, provide these attributes
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_heads  # StableLM-Alpha uses MHA

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


