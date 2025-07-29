# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copied from
# https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct/blob/main/configuration_exaone.py
# Copyright 2021 The LG AI Research EXAONE Lab. All rights reserved.
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
"""Exaone model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

EXAONE_PRETRAINED_CONFIG_ARCHIVE_MAP: dict[str, str] = {}


class ExaoneConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:
    `~transformers.ExaoneModel`. It is used to instantiate a GPT Lingvo model
    according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar
    configuration to that of the Exaone

    Configuration objects inherit from {class}`~transformers.PretrainedConfig`
    and can be used to control the model outputs. Read the documentation from :
    class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size ({obj}`int`, `optional`, defaults to 50257):
            Vocabulary size of the GPT Lingvo model. Defines the number of
            different tokens that can be represented by the {obj}`inputs_ids`
            passed when calling {class}`~transformers.ExaoneModel`. Vocabulary
            size of the model.
            Defines the different tokens that can be represented by the
            `inputs_ids` passed to the forward method of :class:
            `~transformers.EXAONEModel`.
        hidden_size ({obj}`int`, `optional`, defaults to 2048):
            Dimensionality of the encoder layers and the pooler layer.
        num_layers ({obj}`int`, `optional`, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the
            Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to
            implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi
            Head Attention (MHA), if `num_key_value_heads=1 the model will use
            Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint,
            each group key and value head should be constructed by meanpooling
            all the original heads within that group. For more details checkout
            [this paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not
            specified, will default to `num_attention_heads`.
        rotary_pct (`float`, *optional*, defaults to 0.25):
            percentage of hidden dimensions to allocate to rotary embeddings
        intermediate_size ({obj}`int`, `optional`, defaults to 8192):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in
            the Transformer encoder.
        activation_function ({obj}`str` or {obj}`function`, `optional`,
        defaults to {obj}`"gelu_new"`):
            The non-linear activation function (function or string) in the
            encoder and pooler. If string, {obj}`"gelu"`, {obj}`"relu"`,
            {obj}`"selu"` and {obj}`"gelu_new"` are supported.
        embed_dropout ({obj}`float`, `optional`, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the
            embeddings, encoder, and pooler.
        attention_dropout ({obj}`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        max_position_embeddings ({obj}`int`, `optional`, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
        type_vocab_size ({obj}`int`, `optional`, defaults to 2):
            The vocabulary size of the {obj}`token_type_ids` passed when calling
            {class}`~transformers.EXAONEModel`.
        initializer_range ({obj}`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for
            initializing all weight matrices.
        layer_norm_epsilon ({obj}`float`, `optional`, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        use_cache ({obj}`bool`, `optional`, defaults to {obj}`True`):
            Whether or not the model should return the last key/values
            attentions (not used by all models).
            Only relevant if ``config.is_decoder=True``.
        gradient_checkpointing ({obj}`bool`, `optional`,
        defaults to {obj}`False`):
            If True, use gradient checkpointing to save memory at the expense
            of slower backward pass.
        Example::

            >>> from transformers import ExoneModel, ExaoneConfig

            >>> # Initializing a EXAONE configuration
            >>> configuration = ExaoneConfig()

            >>> # Initializing a model from configuration
            >>> model = ExoneModel(configuration)

            >>> # Accessing the model configuration
            >>> configuration = model.config
    """

    model_type = "exaone"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=102400,
        max_position_embeddings=2048,
        hidden_size=2048,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        intermediate_size=None,
        activation_function="silu",
        rotary_pct=0.25,
        resid_dropout=0.0,
        embed_dropout=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-6,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_layers
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        if intermediate_size:
            self.intermediate_size = intermediate_size
        else:
            self.intermediate_size = hidden_size * 4
        self.activation_function = activation_function
        self.resid_dropout = resid_dropout
        self.embed_dropout = embed_dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rotary_pct = rotary_pct

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.use_logit_cap = kwargs.pop("use_logit_cap", False)
        self.ln_no_scale = kwargs.pop("ln_no_scale", False)
        self.use_gated = kwargs.pop("use_gated", False)
        self.use_emb_norm = kwargs.pop("use_emb_norm", False)
        self.use_rotary_pos = kwargs.pop("use_rotary_pos", False)
        self.rotary_type = kwargs.pop("rotary_type", None)
        self.scaling_factor = kwargs.pop("scaling_factor", 1)
        self.use_absolute_pos = kwargs.pop("use_absolute_pos", True)
        self.use_extra_logit = kwargs.pop("use_extra_logit", True)
        self.rotary_expand_length = kwargs.pop("rotary_expand_length", None)
        self.rotary_base = kwargs.pop("rotary_base", 10000.0)
        self.use_qkv_fuse = kwargs.pop("use_qkv_fuse", False)
        self.rescale_before_lm_head = kwargs.pop("rescale_before_lm_head",
                                                 (rotary_pct == 0.25))
        if self.use_rotary_pos:
            self.use_absolute_pos = False
