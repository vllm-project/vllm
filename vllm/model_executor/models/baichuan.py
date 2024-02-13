# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only BaiChuan model compatible with HuggingFace weights."""
from typing import Optional

import torch
from transformers import PretrainedConfig
from vllm.config import LoRAConfig

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)


class BaiChuanBaseForCausalLM(LlamaForCausalLM):

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        param_weight_map = [
            ("qkv_proj", "W_pack"),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            for (param_name, weight_name) in param_weight_map:
                name = name.replace(weight_name, param_name)

            if "rotary_emb.inv_freq" in name:
                continue
            if name == "lm_head.weight":
                # Unlike Baichuan, Baichuan2 normalizes the head weights. Refer to:
                # https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/blob/84603cde5ebffb6084e476cfaeceaf0b8b91fe54/modeling_baichuan.py#L508
                # Distinguish between Baichuan and Baichuan2 by checking the
                # vocab size. This is suggested by
                # https://github.com/vllm-project/vllm/pull/1022#discussion_r1325652704
                is_baichuan2 = self.config.vocab_size == 125696
                if is_baichuan2:
                    loaded_weight = torch.nn.functional.normalize(
                        loaded_weight)

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)


class BaichuanForCausalLM(BaiChuanBaseForCausalLM):
    """Baichuan 13B and Baichuan2 7B/13B."""

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        if config.hidden_size != 4096:  # baichuan 13b, baichuan2 13b
            config.postion_embedding = "ALIBI"
        super().__init__(config=config,
                         linear_method=linear_method,
                         lora_config=lora_config)


class BaiChuanForCausalLM(BaiChuanBaseForCausalLM):
    """Baichuan 7B."""

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__(config=config,
                         linear_method=linear_method,
                         lora_config=lora_config)
