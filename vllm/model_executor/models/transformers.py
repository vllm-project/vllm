# Copyright 2024 The vLLM team.
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
"""Wrapper around `transformers` models"""
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple, TypedDict, Union

import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from vllm.attention import Attention, AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.utils import divide
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .utils import maybe_prefix

logger = init_logger(__name__)


class VllmKwargsForCausalLM(TypedDict, total=False):
    """
    Keyword arguments for Flash Attention with Compile.
    Attributes:
        kv_cache
        maxattn_metadata_length
    """
    kv_cache: torch.Tensor
    attn_metadata: AttentionMetadata


def vllm_flash_attention_forward(_module,
                                 query: torch.Tensor,
                                 key: torch.Tensor,
                                 value: torch.Tensor,
                                 attention_mask: torch.Tensor,
                                 query_length: int = None,
                                 kv_caches: torch.Tensor = None,
                                 attn_metadata: AttentionMetadata = None,
                                 attention_instances=None,
                                 **kwargs):
    layer_idx = _module.layer_idx
    hidden = query.shape[-2]
    query, key, value = (x.transpose(1, 2) for x in (query, key, value))
    query, key, value = (x.reshape(hidden, -1) for x in (query, key, value))
    return attention_instances[layer_idx].forward(
        query,
        key,
        value,
        _kv_cache=kv_caches[layer_idx],
        _attn_metadata=attn_metadata), None


ALL_ATTENTION_FUNCTIONS["vllm"] = vllm_flash_attention_forward


# Linear Layer that is compatible with transformers internal forward
# TODO: This is a temporary solution, we should find a better way to integrate
class HFColumnParallelLinear(ColumnParallelLinear):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)[0]


class HFRowParallelLinear(RowParallelLinear):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)[0]


def replace_tp_linear_class(orig_module: nn.Linear, style: str):
    """
    In model configurations, we use a neutral type (string) to specify parallel
    styles, here we use it to translate nn.Linear into vllm-style tp Linear.
    """

    if not isinstance(style, str):
        raise ValueError(
            f"Unsupported parallel style type {type(style)}, expected str")

    input_size = orig_module.in_features
    output_size = orig_module.out_features
    bias = orig_module.bias is not None

    if style == "colwise":
        return HFColumnParallelLinear(input_size, output_size, bias)
    elif style == "rowwise":
        return HFRowParallelLinear(input_size, output_size, bias)
    # We don't consider colwise_rep since it's used in lm_head
    else:
        raise ValueError(f"Unsupported parallel style value: {style}")


class TransformersModel(nn.Module):
    embedding_padding_modules = ["lm_head"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        logger.info("Using Transformers backend.")

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.unpadded_vocab_size = config.vocab_size

        # MLP modifications
        self.tp_plan = self.config.base_model_tp_plan
        self.model: PreTrainedModel = AutoModel.from_config(
            self.config, torch_dtype=vllm_config.model_config.dtype)
        self.tensor_parallelize(self.model)

        # Attention modifications (assumes 1 attention op per hidden layer)
        tp_size = get_tensor_model_parallel_world_size()
        self.attention_instances = [
            Attention(
                divide(config.num_attention_heads, tp_size),
                config.head_dim,
                config.head_dim**
                -0.5,  # ish, the scaling is different for every attn layer
                num_kv_heads=divide(config.num_key_value_heads, tp_size),
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, f"{i}.attn"))
            for i in range(config.num_hidden_layers)
        ]
        self.config._attn_implementation_internal = "vllm"
            
        # Model modifications
        self.replace_vocab_embed_class(self.model)
        self.replace_rms_norm_class(self.model)

        # ForCausalLM modifications
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.get_input_embeddings().weight

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = get_sampler()

    def log_replacement(
            self, name: str, old_module: nn.Module, new_module: nn.Module):
        logger.debug("%s: %s -> %s", name,
                     old_module.__class__.__name__,
                     new_module.__class__.__name__)

    def tensor_parallelize(self, module: nn.Module, prefix: str = ""):
        if self.tp_plan is None:
            raise ValueError(
                "Trying to run tensor parallelization but the model does not "
                "support it yet!")

        for child_name, child_module in module.named_children():
            qual_name = prefix + child_name
            for pattern, style in self.tp_plan.items():
                if re.match(pattern, qual_name) and isinstance(
                        child_module, nn.Linear):
                    new_module = replace_tp_linear_class(child_module, style)
                    setattr(module, child_name, new_module)
                    self.log_replacement(qual_name, child_module, new_module)
            else:
                self.tensor_parallelize(child_module, prefix=f"{qual_name}.")

    def replace_vocab_embed_class(self, module: nn.Module):
        # Sorted by most frequently use (most frequent first)
        vocab_embed_names = (
            "embed_tokens", "word_embeddings", "wte", "embed_in")
        for vocab_embed_name in vocab_embed_names:
            if hasattr(module, vocab_embed_name):
                old_module = getattr(module, vocab_embed_name)
                new_module = VocabParallelEmbedding(
                    self.vocab_size,
                    self.config.hidden_size,
                    org_num_embeddings=self.config.vocab_size,
                    quant_config=None,
                )
                setattr(module, vocab_embed_name, new_module)
                self.log_replacement(vocab_embed_name, old_module, new_module)
                break

    def replace_rms_norm_class(self, module: nn.Module, prefix: str = ""):
        for child_name, child_module in module.named_children():
            qual_name = prefix + child_name
            if "RMSNorm" in child_module.__class__.__name__:
                rms_norm = RMSNorm(
                    self.config.hidden_size, eps=self.config.rms_norm_eps)
                setattr(module, child_name, rms_norm)
                self.log_replacement(qual_name, child_module, rms_norm)
            self.replace_rms_norm_class(child_module, prefix=f"{qual_name}.")


    def _autoset_attn_implementation(
        self,
        config,
        use_flash_attention_2: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
    ):
        config._attn_implementation = "vllm"
        config._attn_implementation_autoset = True
        return config

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(
            input_ids[None, ...],
            use_cache=False,
            position_ids=positions[None, ...],
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            attention_instances=self.attention_instances,
            return_dict=False)[0][0, ...]  # we remove batch dimension for now
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:

        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
