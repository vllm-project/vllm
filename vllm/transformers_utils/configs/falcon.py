# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://huggingface.co/tiiuae/falcon-7b/blob/main/configuration_RW.py
# Copyright 2023 The vLLM team.
# Copyright 2022 the Big Science Workshop and HuggingFace Inc. team.
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
"""Falcon configuration"""

from transformers.configuration_utils import PretrainedConfig


class RWConfig(PretrainedConfig):
    model_type = "falcon"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
        "num_kv_heads": "n_head_kv",
    }

    def __init__(
        self,
        vocab_size=250880,
        hidden_size=64,
        n_layer=2,
        n_head=8,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        multi_query=True,
        n_head_kv=None,
        alibi=False,
        bias=False,
        parallel_attn=False,
        new_decoder_architecture=False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        # Backward compatibility with n_embed kwarg
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.multi_query = multi_query
        self.n_head_kv = 1 if n_head_kv is None else n_head_kv
        self.alibi = alibi
        self.bias = bias
        self.parallel_attn = parallel_attn
        self.new_decoder_architecture = new_decoder_architecture

        if self.hidden_size == 8192:
            # Hack for falcon-40b
            self.new_decoder_architecture = True

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @property
    def head_dim(self):
        return self.hidden_size // self.n_head

    @property
    def rotary(self):
        return not self.alibi
