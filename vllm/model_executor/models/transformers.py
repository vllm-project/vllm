# SPDX-License-Identifier: Apache-2.0

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
from typing import Iterable, Literal, Optional, Union

import torch
from torch import nn
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.utils import divide, get_pp_indices
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, maybe_prefix)

logger = init_logger(__name__)


def vllm_flash_attention_forward(
        # Transformers args
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        # Transformers kwargs
        scaling: float = None,
        # vLLM kwargs
        attn_metadata: AttentionMetadata = None,
        attention_instances: dict[Attention] = None,
        **kwargs):
    self_attn = attention_instances[module.layer_idx]
    if scaling is not None:
        self_attn.impl.scale = float(scaling)
    hidden = query.shape[-2]
    query, key, value = (x.transpose(1, 2) for x in (query, key, value))
    query, key, value = (x.reshape(hidden, -1) for x in (query, key, value))
    return self_attn.forward(
        query,
        key,
        value,
        kv_cache=None,  # argument not used
        attn_metadata=attn_metadata), None


ALL_ATTENTION_FUNCTIONS["vllm"] = vllm_flash_attention_forward


def log_replacement(name: str, old_module: nn.Module, new_module: nn.Module):
    logger.debug("%s: %s -> %s", name, old_module, new_module)


def replace_linear_class(
        linear: nn.Linear,
        style: Literal["colwise", "rowwise"],
        quant_config=None) -> Union[ColumnParallelLinear, RowParallelLinear]:
    """
    Replace nn.Linear with one of vLLM's tensor parallel linear classes.
    
    `quant_config` is not yet supported.
    Args:
        linear (nn.Linear): `nn.Linear` to be replaced.
        style (str): Tensor parallel style of the new linear, e.g. "colwise".
        quant_config (QuantConfig): Quantization config for the new linear.
    Returns:
        Union[ColumnParallelLinear, RowParallelLinear]: The new linear.
    """

    if not isinstance(style, str):
        raise ValueError(
            f"Unsupported parallel style type {type(style)}, expected str")

    vllm_linear_cls = {
        "colwise": ColumnParallelLinear,
        "rowwise": RowParallelLinear,
    }.get(style)

    if vllm_linear_cls is None:
        logger.warning(
            "Unsupported parallel style value: %s. "
            "This layer will not be tensor parallelized.", style)
        return linear

    class HFCompatibleLinear(vllm_linear_cls):
        """
        Wrapper class that removes `output_bias` from returned output.
        """

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return super().forward(input)[0]

    return HFCompatibleLinear(
        input_size=linear.in_features,
        output_size=linear.out_features,
        bias=linear.bias is not None,
    )


class HFCompatiblePPMissingLayer(PPMissingLayer):
    """
    A version of `PPMissingLayer` that can replace Transformers
    transformer layers.
    """

    def forward(self, *args, **kwargs):
        input = args[0] if args else kwargs.get("input")
        return (super().forward(input), )


class TransformersModel(nn.Module, SupportsPP):
    embedding_padding_modules = ["lm_head"]
    embedding_modules = ["embed_tokens"
                         ]  # TODO transformers will have a util to get it

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        logger.info("Using Transformers backend.")

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.model: PreTrainedModel = AutoModel.from_config(
            self.config,
            attn_implementation="vllm",
            trust_remote_code=vllm_config.model_config.trust_remote_code,
        )
        prefix = self.model.base_model_prefix

        # Input embeddings
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            new_module = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=None,
            )
            self.model.set_input_embeddings(new_module)
        else:
            self.model.set_input_embeddings(PPMissingLayer())

        # Transformer layers
        self.attention_instances = self.create_attention_instances(
            config, cache_config, quant_config=None)
        self.apply_base_model_tp_plan(self.model)

        # Pipeline parallelise the transformer layers
        start_layer, end_layer = get_pp_indices(config.num_hidden_layers,
                                                get_pp_group().rank_in_group,
                                                get_pp_group().world_size)
        layers_index = float("inf")
        for i, (name, module) in enumerate(self.model.named_children()):
            if isinstance(module, nn.ModuleList):
                # Remove transformer layers that are't
                # part of the current pipeline stage
                for j in range(len(module)):
                    if j < start_layer or end_layer <= j:
                        module[j] = HFCompatiblePPMissingLayer()
                layers_index = i
                continue
            # Remove any layer norms that appear after the transformer
            # layers if this isn't the last pipeline stage
            if (not get_pp_group().is_last_rank  # not last pipeline stage
                    and i > layers_index  # after transformer layers
                    and "norm" in name.lower()  # is norm layer
                ):
                setattr(self.model, name, PPMissingLayer())

        # Output embeddings
        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=None,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.get_input_embeddings())

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
        else:
            self.lm_head = PPMissingLayer()

        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.hidden_size))

    def create_attention_instances(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig = None,
        quant_config: QuantizationConfig = None,
    ) -> dict[int, Attention]:
        """
        Create `Attention` instances to inform KV cache allocation.
        """
        tp_size = get_tensor_model_parallel_world_size()
        pp_size = get_pp_group().world_size
        pp_rank = get_pp_group().rank_in_group
        layers_per_rank = divide(config.num_hidden_layers, pp_size)
        offset = layers_per_rank * pp_rank
        return {
            i + offset:
            Attention(
                num_heads=divide(config.num_attention_heads, tp_size),
                head_size=config.head_dim,
                # NOTE: We use Llama scale as default, if it's set by
                # Transformers, it's updated in vllm_flash_attention_forward
                scale=config.head_dim**-0.5,
                num_kv_heads=divide(config.num_key_value_heads, tp_size),
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{i + offset}.attn")
            for i in range(layers_per_rank)
        }

    def apply_base_model_tp_plan(self, module: nn.Module, prefix: str = ""):
        """
        Apply the base model tensor parallelization plan to a module.
        Currently only supports linear layers.
        """
        if (self.config.base_model_tp_plan is None
                and get_tensor_model_parallel_world_size() > 1):
            raise ValueError(
                "Trying to run tensor parallelization but the model does not "
                "support it yet!")

        for child_name, child_module in module.named_children():
            qual_name = maybe_prefix(prefix, child_name)
            for pattern, style in self.config.base_model_tp_plan.items():
                if re.match(pattern, qual_name) and isinstance(
                        child_module, nn.Linear):
                    new_module = replace_linear_class(child_module, style,
                                                      self.quant_config)
                    setattr(module, child_name, new_module)
                    log_replacement(qual_name, child_module, new_module)
            else:
                self.apply_base_model_tp_plan(child_module, prefix=qual_name)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],  # argument not used
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if input_ids is not None:
                input_ids = input_ids[None, ...]
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[None, ...]
        else:
            assert intermediate_tensors is not None
            input_ids = None
            inputs_embeds = intermediate_tensors["hidden_states"][None, ...]

        hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            position_ids=positions[None, ...],
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            attention_instances=self.attention_instances,
            return_dict=False)[0][0, ...]  # we remove batch dimension for now

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        return hidden_states

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

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params = set[str]()
        for name, loaded_weight in weights:
            if is_pp_missing_parameter(name, self):
                continue
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        return loaded_params
