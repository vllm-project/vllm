# coding=utf-8
# Copyright 2022 the Big Science Workshop and HuggingFace Inc. team.  All rights reserved.
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

""" Telechat configuration"""

from packaging import version
from collections import OrderedDict
from transformers.utils import is_torch_available, logging
from transformers.configuration_utils import PretrainedConfig
from typing import TYPE_CHECKING, Any, List, Mapping, Optional

logger = logging.get_logger(__name__)

class Telechat2Config(PretrainedConfig):
    """
    Args:
        vocab_size (`int`, *optional*, defaults to 160256): Vocabulary size of the Telechat model.
        hidden_size (`int`, *optional*, defaults to 4096): Dimensionality of the embeddings and hidden states.
        ffn_hidden_size (`int`, *optional*, defaults to 12288): Dimensionality of the feed-forward hidden states.
        n_layer (`int`, *optional*, defaults to 30): Number of hidden layers in the Transformer
        n_head (`int`, *optional*, defaults to 32): Number of attention heads for each attention layer.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5): The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02): The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        apply_residual_connection_post_layernorm (`bool`, *optional*, defaults to `False`): If enabled, use the layer norm of the hidden states as the residual in the transformer blocks
        hidden_dropout (`float`, *optional*, defaults to 0.0): Dropout rate of the dropout function on the bias dropout.
        attention_dropout (`float`, *optional*, defaults to 0.0): Dropout rate applied to the attention probs
        use_cache (`bool`, *optional*, defaults to `True`): Whether or not the model should return the last key/values attentions.
        training_seqlen (`int`, *optional*, defaults to 8192): Sequence length during last finetuning.
        logn (`bool`, *optional*, defaults to `True`): Whether or not to use logN during extrapolation.
        embed_layernorm (`bool`, *optional*, defaults to `True`): Whether or not to use embedding layernorm.
    """

    model_type = "telechat"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
    }

    def __init__(
        self,
        vocab_size=160256,
        hidden_size=4096,
        n_layer=30,
        n_head=32,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        ffn_hidden_size=12288,
        training_seqlen = 8192,
        logn = True,
        embed_layernorm = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.logn = logn
        self.ffn_hidden_size = ffn_hidden_size
        self.training_seqlen = training_seqlen
        self.embed_layernorm = embed_layernorm
        self.num_key_value_heads= kwargs.pop("num_key_value_heads", None)


        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
