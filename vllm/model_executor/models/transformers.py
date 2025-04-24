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
from itertools import chain
from typing import Iterable, Literal, Optional, Union

import torch
from torch import nn
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, VllmConfig)
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.utils import get_pp_indices
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP, SupportsQuant
from .utils import (AutoWeightsLoader, PPMissingLayer, WeightsMapper,
                    is_pp_missing_parameter,
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
        scaling: Optional[float] = None,
        # vLLM kwargs
        attention_instances: Optional[dict[Attention]] = None,
        **kwargs):
    self_attn = attention_instances[module.layer_idx]
    if scaling is not None:
        self_attn.impl.scale = float(scaling)
    hidden = query.shape[-2]
    query, key, value = (x.transpose(1, 2) for x in (query, key, value))
    query, key, value = (x.reshape(hidden, -1) for x in (query, key, value))
    return self_attn.forward(query, key, value), None


ALL_ATTENTION_FUNCTIONS["vllm"] = vllm_flash_attention_forward


def log_replacement(name: str, old_module: nn.Module, new_module: nn.Module):
    logger.debug("%s: %s -> %s", name, old_module, new_module)


def replace_linear_class(
    linear: nn.Linear, style: Literal["colwise", "rowwise"],
    quant_config: QuantizationConfig
) -> Union[ColumnParallelLinear, RowParallelLinear]:
    """
    Replace nn.Linear with one of vLLM's tensor parallel linear classes.

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
    }.get(style, ReplicatedLinear)

    return vllm_linear_cls(
        input_size=linear.in_features,
        output_size=linear.out_features,
        bias=linear.bias is not None,
        quant_config=quant_config,
        return_bias=False,
    )


class TransformersModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        logger.info("Using Transformers backend.")

        config: PretrainedConfig = vllm_config.model_config.hf_config
        cache_config: CacheConfig = vllm_config.cache_config
        device_config: DeviceConfig = vllm_config.device_config
        model_config: ModelConfig = vllm_config.model_config
        parallel_config: ParallelConfig = vllm_config.parallel_config
        quant_config: QuantizationConfig = vllm_config.quant_config

        self.config = config
        self.cache_config = cache_config
        self.device_config = device_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.quant_config = quant_config

        self.pp_group = get_pp_group()
        self.pp_size = self.pp_group.world_size
        self.pp_rank = self.pp_group.rank_in_group
        self.tp_size = get_tensor_model_parallel_world_size()

        # Use meta device to delay allocating GPU tensors
        with torch.device("meta"):
            # FIXME(Isotr0py): We need to refactor this part in the future to
            # avoid registering an extra model layer, otherwise we will need a
            # weights mapper to rename weights.
            self.model: PreTrainedModel = AutoModel.from_config(
                config,
                attn_implementation="vllm",
                torch_dtype=model_config.dtype,
                trust_remote_code=model_config.trust_remote_code,
            )

        self.pipeline_parallel()
        self.tensor_parallel()

        # Input embeddings
        if not isinstance(self.model.get_input_embeddings(), PPMissingLayer):
            self.model.set_input_embeddings(
                VocabParallelEmbedding(
                    config.vocab_size,
                    config.hidden_size,
                    org_num_embeddings=config.vocab_size,
                    quant_config=quant_config,
                ))

        # Attention layers
        self.attention_instances = self.create_attention_instances()

        # Initialize buffers (e.g. rotary embedding inverse frequency)
        self.init_buffers(self.model)

        # Move remaining meta tensors to device (should happen last)
        self.meta_to_empty(self.model)

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.hidden_size))

    def pipeline_parallel(self):
        """
        Apply the model's pipeline parallelization plan.
        """
        if self.pp_size <= 1:
            return

        if not self.model.supports_pp_plan:
            raise ValueError(
                f"{type(self.model)} does not support pipeline parallel yet!")

        module_lists = []
        module_list_idx = None
        pp_plan = list(self.model._pp_plan.keys())
        for i, name in enumerate(pp_plan):
            if isinstance(getattr(self.model, name), nn.ModuleList):
                module_lists.append(name)
                module_list_idx = i

        if len(module_lists) > 1:
            raise ValueError(
                "Pipeline parallel of models with multiple `ModuleList`s "
                "in the base model are not supported yet!")
        if module_list_idx is None:
            raise ValueError(
                f"Could not find `ModuleList` in {type(self.model)}")

        # Layers before module list
        for name in pp_plan[:module_list_idx]:
            if self.pp_group.is_first_rank or (self.config.tie_word_embeddings
                                               and self.pp_group.is_last_rank):
                continue
            setattr(self.model, name, PPMissingLayer())

        # Module list
        start_layer, end_layer = get_pp_indices(self.config.num_hidden_layers,
                                                self.pp_rank, self.pp_size)
        layers_name = pp_plan[module_list_idx]
        layers = getattr(self.model, layers_name)
        for i in range(len(layers)):
            if start_layer <= i and i < end_layer:
                continue
            layers[i] = PPMissingLayer(return_tuple=True)

        # Layers after module list
        for name in pp_plan[module_list_idx + 1:]:
            # Modules that should be on last rank
            if not self.pp_group.is_last_rank:
                setattr(self.model, name, PPMissingLayer())

    def tensor_parallel(self):
        """
        Apply the model's tensor parallelization plan.
        Currently only supports linear layers.
        """
        if not self.model.supports_tp_plan:
            if self.tp_size <= 1:
                return

            raise ValueError(
                f"{type(self.model)} does not support tensor parallel yet!")

        tp_plan = self.model._tp_plan

        def _tensor_parallel(module: nn.Module, prefix: str = ""):
            for child_name, child_module in module.named_children():
                qual_name = maybe_prefix(prefix, child_name)
                for pattern, style in tp_plan.items():
                    if re.match(pattern, qual_name) and isinstance(
                            child_module, nn.Linear):
                        new_module = replace_linear_class(
                            child_module, style, self.quant_config)
                        setattr(module, child_name, new_module)
                        log_replacement(qual_name, child_module, new_module)
                else:
                    _tensor_parallel(child_module, prefix=qual_name)

        _tensor_parallel(self.model)

    def create_attention_instances(self) -> dict[int, Attention]:
        """
        Create `Attention` instances to inform KV cache allocation.
        """
        num_heads = self.model_config.get_num_attention_heads(
            self.parallel_config)
        head_size = self.model_config.get_head_size()
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        start, end = get_pp_indices(self.config.num_hidden_layers,
                                    self.pp_rank, self.pp_size)
        return {
            i:
            Attention(
                num_heads=num_heads,
                head_size=head_size,
                # NOTE: We use Llama scale as default, if it's set by
                # Transformers, it's updated in vllm_flash_attention_forward
                scale=head_size**-0.5,
                num_kv_heads=num_kv_heads,
                cache_config=self.cache_config,
                quant_config=self.quant_config,
                prefix=f"{i}.attn")
            for i in range(start, end)
        }

    def init_buffers(self, module: nn.Module):
        """
        If a `buffer` is on the `meta` device, then its parent
        `module` is the original module created by:

        ```python
        with torch.device("meta"):
            self.model: PreTrainedModel = AutoModel.from_config(...)
        ```

        This means that:
        - `type(module)` is a class from `transformers`
        - This class is constructed using a `PretrainedConfig`
        """
        for name, buffer in module.named_buffers(recurse=False):
            if buffer.device == torch.device("meta"):
                new_buffer = getattr(type(module)(self.config), name)
                setattr(module, name, new_buffer)
        for child in module.children():
            self.init_buffers(child)

    def meta_to_empty(self, module: nn.Module):
        tensors = list(chain(module.buffers(), module.parameters()))
        if tensors and all(t.device == torch.device("meta") for t in tensors):
            module.to_empty(device=self.device_config.device)
            return  # We can stop recursing because to_empty is recursive
        for child in module.children():
            self.meta_to_empty(child)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if not get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            input_ids = None
            inputs_embeds = intermediate_tensors["hidden_states"]

        if input_ids is not None:
            input_ids = input_ids[None, ...]
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds[None, ...]

        hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            position_ids=positions[None, ...],
            attention_instances=self.attention_instances,
            return_dict=False)[0][0, ...]  # we remove batch dimension for now

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params = set[str]()
        for name, loaded_weight in weights:
            # Use "model" instead of base_model_prefix because
            # the base model attribute in vLLM is always `model`
            if not name.startswith(prefix := "model."):
                name = prefix + name

            if is_pp_missing_parameter(name, self):
                continue
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        return loaded_params


@support_torch_compile
class TransformersForCausalLM(nn.Module, SupportsQuant, SupportsLoRA,
                              SupportsPP):
    embedding_padding_modules = ["lm_head"]
    embedding_modules = ["embed_tokens"
                         ]  # TODO transformers will have a util to get it

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: PretrainedConfig = vllm_config.model_config.hf_config
        quant_config: QuantizationConfig = vllm_config.quant_config

        self.config = config

        self.model = TransformersModel(vllm_config=vllm_config, prefix=prefix)

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
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

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    # FIXME(Isotr0py): Don't use any weights mapper for Transformers backend,
    # this makes thing complicated. We need to remove this mapper after refactor
    # `TransformersModel` in the future.
    @property
    def hf_to_vllm_mapper(self):
        prefix_mapper = {
            name: "model." + name
            for name, _ in self.model.model.named_children()
        }
        return WeightsMapper(
            orig_to_new_substr={"model.": "model.model."},
            orig_to_new_prefix=prefix_mapper,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(input_ids, positions, intermediate_tensors,
                                  inputs_embeds)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
