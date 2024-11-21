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

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.transformers_utils.configs import Telechat2Config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from .utils import  make_layers

def maybe_prefix(prefix: str, name: str) -> str:
    """Add a prefix to a name if the prefix is non-empty.

    Args:
        prefix: The prefix to add. If empty, no prefix will be added.
        name: The name to potentially prefix.

    Returns:
        The string "prefix.name" if prefix was non-empty, otherwise just "name".
    """
    return name if not prefix else f"{prefix}.{name}"

class TeleChat2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
    
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=True,
            quant_config=quant_config,
            input_is_parallel=True,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
    

class TeleChat2Attention(nn.Module):

    def __init__(
        self,
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
        super().__init__()
        self.config = config 
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= self.tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.dense = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            bias=True,
            quant_config=quant_config,
            input_is_parallel=True,
            prefix=f"{prefix}.dense_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.self_attention")
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.dense(attn_output)
        return output


class TeleChat2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attention = TeleChat2Attention(
            config,
            hidden_size=self.hidden_size,
            num_heads=config.n_head,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.n_head),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attention"
        )
        self.mlp = TeleChat2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.apply_residual_connection_post_layernorm = \
            config.apply_residual_connection_post_layernorm

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        layernorm_output = self.input_layernorm(hidden_states)
        attn_outputs = self.self_attention(
            positions=positions,
            hidden_states=layernorm_output,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        attn_outputs = residual + attn_outputs
        residual = attn_outputs
        layernorm_output = self.post_attention_layernorm(attn_outputs)
        output = residual + self.mlp(layernorm_output)
        return output


class TeleChat2Model(nn.Module):
    
    def __init__(
        self,
        config: Telechat2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,     
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.word_embeddings = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=f"{prefix}.word_embeddings",
        )

        self.start_layer, self.end_layer, self.h = make_layers(
            config.num_hidden_layers,
            lambda prefix: TeleChat2DecoderLayer(config=config,
                                             cache_config=cache_config,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.layers"),
            prefix=f"{prefix}.h",
        )
        
        self.ln_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        
        hidden_states = self.get_input_embeddings(input_ids)
       
        for i in range(self.start_layer, self.end_layer):
            layer = self.h[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
            )

        hidden_states = self.ln_f(hidden_states)
        return hidden_states
    
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
            if "self_attention.key_value" in name:
                k_weight = []
                v_weight = []
                for i in range(total_num_heads):
                    start =i * head_dim * 2
                    k_weight.append(loaded_weight[start:start+head_dim,:])
                    v_weight.append(loaded_weight[start+head_dim:start+2*head_dim:])
                k_weight = torch.cat(k_weight,dim=0)
                v_weight = torch.cat(v_weight,dim=0)
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
    

class TeleChat2ForCausalLM(nn.Module):
    
    def __init__(
        self,
        config: Telechat2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        config.intermediate_size = config.ffn_hidden_size
        config.hidden_act = "silu"
        config.rms_norm_eps = config.layer_norm_epsilon
        config.tie_word_embeddings = False
        self.config = config
        self.transformer = TeleChat2Model(config, cache_config, quant_config,
                                prefix=maybe_prefix(prefix, "transformer"))
  
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(
                prefix, "lm_head")
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.transformer(input_ids, positions, kv_caches,
                                  attn_metadata)
        return model_output

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
            "residual":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
        })

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
    
import itertools
from vllm.logger import init_logger
logger = init_logger(__name__)
from .utils import PPMissingLayer
from dataclasses import dataclass, field
from typing import Mapping
WeightsMapping = Mapping[str, Optional[str]]


@dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns."""

    orig_to_new_substr: WeightsMapping = field(default_factory=dict)
    orig_to_new_prefix: WeightsMapping = field(default_factory=dict)
    orig_to_new_suffix: WeightsMapping = field(default_factory=dict)

    def _map_name(self, key: str) -> Optional[str]:
        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None

                key = key.replace(substr, new_key, 1)

        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None

                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None

                key = new_key.join(key.rsplit(suffix, 1))

        return key

    def apply(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        return ((out_name, data) for name, data in weights
                if (out_name := self._map_name(name)) is not None)

class AutoWeightsLoader:
    """
    Helper class to load weights into a :class:`torch.nn.Module`. It is able
    to automatically detect child modules and parameters while iterating over
    the weights only once.

    The weight loading logic for individual modules can be overridden
    by defining a ``load_weights`` method.

    Similarly, the weight loading logic for individual parameters can be
    overridden by defining a ``weight_loader`` method.

    Detailed weight loading information can be viewed by setting the
    environment variable ``VLLM_LOGGING_LEVEL=DEBUG``.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        skip_prefixes: Optional[List[str]] = None,
        ignore_unexpected_prefixes: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.module = module
        self.skip_prefixes = skip_prefixes or []
        self.ignore_unexpected_prefixes = ignore_unexpected_prefixes or []

    def _groupby_prefix(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Iterable[Tuple[str, Iterable[Tuple[str, torch.Tensor]]]]:
        weights_by_parts = ((weight_name.split(".", 1), weight_data)
                            for weight_name, weight_data in weights)

        for prefix, group in itertools.groupby(weights_by_parts,
                                               key=lambda x: x[0][0]):
            yield (
                prefix,
                # Because maxsplit=1 in weight_name.split(...),
                # the length of `parts` must either be 1 or 2
                (("" if len(parts) == 1 else parts[1], weights_data)
                 for parts, weights_data in group),
            )

    def _get_qualname(self, prefix: str, rest: str) -> str:
        if prefix == "":
            return rest
        if rest == "":
            return prefix

        return ".".join((prefix, rest))

    def _can_skip(self, qualname: str) -> bool:
        return any(qualname.startswith(p) for p in self.skip_prefixes)

    def _can_ignore_unexpected(self, qualname: str) -> bool:
        return any(
            qualname.startswith(p) for p in self.ignore_unexpected_prefixes)

    def _load_param(
        self,
        base_prefix: str,
        param: nn.Parameter,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        for weight_name, weight_data in weights:
            weight_qualname = self._get_qualname(base_prefix, weight_name)

            if self._can_skip(weight_qualname):
                logger.debug("Skipping weight %s", weight_qualname)

                continue

            if weight_name != "":
                if self._can_ignore_unexpected(weight_qualname):
                    logger.debug("Ignoring weight %s", weight_qualname)

                    continue

                raise ValueError(
                    f"Attempted to load nested weight '{weight_qualname}' "
                    f"into a single parameter '{base_prefix}'")

            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, weight_data)

            logger.debug("Loaded weight %s with shape %s", weight_qualname,
                         param.shape)

            yield weight_qualname

    def _load_module(
        self,
        base_prefix: str,
        module: nn.Module,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        if isinstance(module, PPMissingLayer):
            return

        # Avoid infinite recursion since this function is typically
        # called inside load_weights of the module itself
        if module != self.module:
            module_load_weights = getattr(module, "load_weights", None)
            if callable(module_load_weights):
                loaded_params = module_load_weights(weights)
                yield from map(lambda x: self._get_qualname(base_prefix, x),
                               loaded_params)

        child_modules = dict(module.named_children())
        child_params = dict(module.named_parameters(recurse=False))

        for child_prefix, child_weights in self._groupby_prefix(weights):
            prefix = self._get_qualname(base_prefix, child_prefix)

            if child_prefix in child_modules:
                if self._can_skip(prefix + "."):
                    logger.debug("Skipping module %s", prefix)

                    continue

                yield from self._load_module(prefix,
                                             child_modules[child_prefix],
                                             child_weights)
            elif child_prefix in child_params:
                if self._can_skip(prefix):
                    logger.debug("Skipping param %s", prefix)

                    continue

                yield from self._load_param(prefix, child_params[child_prefix],
                                            child_weights)
            else:
                can_skip_module = self._can_skip(prefix + ".")
                can_skip_param = self._can_skip(prefix)
                if can_skip_module or can_skip_param:
                    logger.debug("Skipping missing %s", prefix)

                    continue

                can_ignore_module = self._can_ignore_unexpected(prefix + ".")
                can_ignore_param = self._can_ignore_unexpected(prefix)
                if can_ignore_module or can_ignore_param:
                    logger.debug("Ignoring missing %s", prefix)

                    continue

                msg = (f"There is no module or parameter named '{prefix}' "
                       f"in {type(self.module).__name__}")
                raise ValueError(msg)

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        *,
        mapper: Optional[WeightsMapper] = None,
    ) -> Set[str]:
        if mapper is not None:
            weights = mapper.apply(weights)

        autoloaded_weights = set(self._load_module("", self.module, weights))
        return autoloaded_weights