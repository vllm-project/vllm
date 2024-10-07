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
useLinear = False
import os

class RWKV5Config(PretrainedConfig):
    model_type = "rwkv5"
   

    def __init__(
        self,
        **kwargs,
    ) -> None:
        global useLinear
        useLinear = True

        print("RWKV5Config", kwargs)
        # exit()
        if(kwargs.get("num_attention_heads", False)):
            print(kwargs)
            kwargs["num_attention_heads"] = kwargs["attention_hidden_size"] // kwargs["head_size"]
            kwargs["num_kv_heads"] = kwargs["num_attention_heads"]
            kwargs["max_seq_len"] = -1
        
        super().__init__(**kwargs)

    @property
    def head_dim(self):
        return self.hidden_size // self.n_head

    @property
    def rotary(self):
        return False
    
    
