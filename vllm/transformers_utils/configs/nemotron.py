# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Nemotron model configuration"""

from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class NemotronConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`NemotronModel`]. It is used to instantiate a Nemotron model
    according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar
    configuration to that of the Nemotron-8B.

    Configuration objects inherit from [`PretrainedConfig`] and can be
    used to control the model outputs. Read the documentation from
    [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Nemotron model. Defines the number of
            different tokens that can be represented by the
            `inputs_ids` passed when calling [`NemotronModel`]
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 24576):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 48):
            Number of attention heads for each attention layer in the
            Transformer decoder.
        head_dim (`int`, *optional*):
            Projection weights dimension in multi-head attention. Set to
            hidden_size // num_attention_heads if None
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to
            implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use
            Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention
            (MQA) otherwise GQA is used. When converting a multi-head
            checkpoint to a GQA checkpoint, each group key and value
            head should be constructed by meanpooling all the original
            heads within that group. For more details checkout
            [this paper](https://arxiv.org/pdf/2305.13245.pdf). If it
            is not specified, will default to `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu2"`):
            The non-linear activation function (function or string) in the
            decoder.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used
            with.
        initializer_range (`float`, *optional*, defaults to 0.0134):
            The standard deviation of the truncated_normal_initializer for
            initializing all weight matrices.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values
            attentions (not used by all models). Only relevant if
            `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 3):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_parameters (`dict`, *optional*):
            The parameters of the RoPE embeddings. Expected contents:
                `rope_theta` (`float`): The base period of the RoPE embeddings.
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear',
                    'dynamic', 'yarn', 'longrope', 'llama3'], with 'default' being the
                    original RoPE implementation.
                `partial_rotary_factor` (`float`, *optional*, defaults to 0.5):
                    Percentage of the query and keys which will have rotary embedding.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output
            projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj and down_proj layers in the MLP
            layers.

    ```python
    >>> from transformers import NemotronModel, NemotronConfig
    >>> # Initializing a Nemotron nemotron-15b style configuration
    >>> configuration = NemotronConfig()
    >>> # Initializing a model from the nemotron-15b style configuration
    >>> model = NemotronModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "nemotron"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=256000,
        hidden_size=6144,
        intermediate_size=24576,
        num_hidden_layers=32,
        num_attention_heads=48,
        head_dim=None,
        num_key_value_heads=None,
        hidden_act="relu2",
        max_position_embeddings=4096,
        initializer_range=0.0134,
        norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=2,
        eos_token_id=3,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        head_dim = head_dim or kwargs.get("kv_channels")
        self.head_dim = (
            head_dim if head_dim is not None else (hidden_size // num_attention_heads)
        )

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`
        rope_scaling = kwargs.pop("rope_scaling", None)
        rope_parameters = rope_scaling or rope_parameters or {"rope_type": "default"}
        rope_theta = kwargs.pop("rope_theta", 10000.0)
        if "rope_theta" not in rope_parameters:
            rope_parameters["rope_theta"] = rope_theta
        # for backward compatibility
        partial_rotary_factor = (
            kwargs.get("rope_percent")
            or kwargs.get("rope_percentage")
            or kwargs.get("partial_rotary_factor")
            or 0.5
        )
        if "partial_rotary_factor" not in rope_parameters:
            rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        self.rope_parameters = rope_parameters
        self._rope_parameters_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_parameters_validation(self):
        """
        Validate the `rope_parameters` configuration.
        """
        if self.rope_parameters is None:
            return

        rope_type: str | None = self.rope_parameters.get("rope_type", None)
        factor: float | None = self.rope_parameters.get("factor", None)

        if rope_type not in {"default", "linear", "dynamic"}:
            raise ValueError(
                "`rope_type` must be one of ['default', 'linear', 'dynamic'], "
                f"got {rope_type}"
            )
        if rope_type != "default":
            if factor is None:
                raise ValueError(
                    "If `rope_type` is not 'default', `rope_parameters` "
                    "must include a `factor` field. Got `None`."
                )
            if not isinstance(factor, float) or factor <= 1.0:
                raise ValueError(
                    "`rope_parameters`'s factor field must be a float > 1, got "
                    f"{factor}"
                )
