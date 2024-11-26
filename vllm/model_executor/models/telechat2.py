# Copyright 2023 The vLLM team.
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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import (LlamaAttention, 
                                              LlamaDecoderLayer, 
                                              LlamaForCausalLM, LlamaMLP,
                                              LlamaModel)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .utils import AutoWeightsLoader, WeightsMapper, make_layers, maybe_prefix


class TeleChat2MLP(LlamaMLP):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__(hidden_size, intermediate_size, hidden_act,
                         quant_config, bias, prefix)
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=True,
            quant_config=quant_config,
        )


class TeleChat2Attention(LlamaAttention):

    def __init__(self,
                 config,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 rope_theta: float = 10000,
                 rope_scaling: Optional[Dict[str, Any]] = None,
                 max_position_embeddings: int = 8192,
                 quant_config: Optional[QuantizationConfig] = None,
                 bias: bool = False,
                 cache_config: Optional[CacheConfig] = None,
                 prefix: str = "") -> None:
        super().__init__(config, hidden_size, num_heads, num_kv_heads,
                         rope_theta, rope_scaling, max_position_embeddings,
                         quant_config, bias, cache_config, prefix)
        self.o_proj = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            bias=True,
            quant_config=quant_config,
            input_is_parallel=True,
            prefix=f"{prefix}.dense_proj",
        )


class TeleChat2DecoderLayer(LlamaDecoderLayer):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, cache_config, quant_config, prefix)
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = TeleChat2Attention(
            config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
        )
        self.mlp = TeleChat2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
        )


class TeleChat2Model(LlamaModel):

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: TeleChat2DecoderLayer(config=config,
                                                 cache_config=cache_config,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.layers"),
            prefix=f"{prefix}.layers",
        )

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            ('gate_up_proj', 'gate_proj', 0),
            ('gate_up_proj', 'up_proj', 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        total_num_heads = self.config.n_head
        head_dim = self.config.hidden_size // total_num_heads
        for name, loaded_weight in weights:
            if "self_attn.key_value" in name:
                k_weight = []
                v_weight = []
                for i in range(total_num_heads):
                    start = i * head_dim * 2
                    k_weight.append(loaded_weight[start:start + head_dim, :])
                    v_weight.append(loaded_weight[start + head_dim:start +
                                                  2 * head_dim:])
                k_weight = torch.cat(k_weight, dim=0)
                v_weight = torch.cat(v_weight, dim=0)
                name = name.replace("key_value", "qkv_proj")
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, k_weight, "k")
                weight_loader(param, v_weight, "v")
                loaded_params.add(name)
            elif "query" in name:
                name = name.replace("query", "qkv_proj")
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, "q")
                loaded_params.add(name)
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
        return loaded_params


class TeleChat2ForCausalLM(LlamaForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(LlamaForCausalLM, self).__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        config.intermediate_size = config.ffn_hidden_size
        config.hidden_act = "silu"
        config.rms_norm_eps = config.layer_norm_epsilon
        config.tie_word_embeddings = False
        self.config = config
        self.model = TeleChat2Model(vllm_config=vllm_config,
                                    prefix=maybe_prefix(prefix, "transformer"))

        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      bias=False,
                                      quant_config=quant_config,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:

        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "transformer.": "model.",
            },
            orig_to_new_substr={
                ".h.": ".layers.",
                ".self_attention.": ".self_attn.",
                ".word_embeddings.": ".embed_tokens.",
                ".dense.": ".o_proj.",
                ".ln_f.": ".norm.",
            },
        )
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=hf_to_vllm_mapper)
