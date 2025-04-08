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
from transformers import AutoModel, PretrainedConfig, PreTrainedModel, LlavaConfig
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
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processor import cached_get_processor
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry, MultiModalKwargs
from vllm.multimodal.processing import BaseMultiModalProcessor, BaseProcessingInfo
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalInputs, PlaceholderRange

from .interfaces import SupportsLoRA, SupportsPP, SupportsQuant, SupportsMultiModal
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


class MultiModalProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        # NOTE: this means we don't check if return config type is same as requested
        # VLLM on contrary always checks. In whcih cases we can have different config types tho?
        return self.ctx.model_config.hf_config

    def get_supported_mm_limits(self):
        return {"image": None, "video": None}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        return {"image": self.get_max_image_tokens(), "video": 100}

    def get_max_image_tokens(self) -> int:
        # Is already an attribute in some VLMs and now reason to make it a required attribute
        # TODO: @raushan add it for all VLM configs
        return self.get_hf_config().image_seq_length

    def get_hf_processor(self):
        processor = cached_get_processor(self.ctx.model_config.model)
        return processor


class MultiModalDummyInputsBuilder(BaseDummyInputsBuilder):
    def get_dummy_processor_inputs(
        self,
        seq_len,
        mm_counts,
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        num_frames = 8

        processor = self.info.get_hf_processor()
        image_token = getattr(processor, "image_token", None)
        video_token = getattr(processor, "video_token", None)

        # TODO: raushan, we can have processor attr for `processor.max_output_size` which will infer
        # max features for model in HF side. But imo we can just set a veru high resolution
        # and the processor will return us pixels with correct max shape. Resolution 3kx3k is high enough
        target_width = target_height = 3000

        # NOTE: we can pass videos/images/audio to any processor With the new API used in MLLMs,
        # HF processor will take the modality needed for model and ignore all others
        mm_data = {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images
            ),
            "video": self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=num_frames,
                num_videos=num_videos,
            )
        }

        prompt_text = video_token*num_videos if video_token is not None else image_token*num_images    
        return ProcessorInputs(
            prompt_text=prompt_text,
            mm_data=mm_data,
        )


class MultiModalProcessor(BaseMultiModalProcessor):
    def _get_prompt_replacements(
        self,
        mm_items,
        hf_processor_mm_kwargs,
        out_mm_kwargs: MultiModalKwargs,
    ):
        return

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs,
    ):
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_sizes=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
            mm_token_type_ids=MultiModalFieldConfig.batched("image"),
            pixel_values_videos=MultiModalFieldConfig.batched("video"),
            video_embeds=MultiModalFieldConfig.batched("video"),
        )
    
    def _apply_hf_processor_text_mm(
        self,
        prompt_text,
        mm_items,
        hf_processor_mm_kwargs,
    ):
        """
        Apply the HF processor on the prompt text and multi-modal data
        together.

        In addition, return whether prompt replacements have been applied.
        """
        processor_data, passthrough_data = self._get_hf_mm_data(mm_items)
        processor_data["return_mm_token_type_ids"] = True

        processed_data = self._call_hf_processor(
            prompt=prompt_text,
            mm_data=processor_data,
            mm_kwargs=hf_processor_mm_kwargs,
        )
        processed_data.update(passthrough_data)

        prompt_ids, = processed_data.pop("input_ids").tolist()
        mm_token_type_ids = processed_data.pop("mm_token_type_ids")

        mm_kwargs = MultiModalKwargs.from_hf_inputs(
            processed_data,
            self._get_mm_fields_config(processed_data, hf_processor_mm_kwargs),
        )

        return prompt_ids, mm_kwargs, mm_token_type_ids

    def apply(
        self,
        prompt,
        mm_data,
        hf_processor_mm_kwargs,
    ) -> MultiModalInputs:
        """
        Process multi-modal inputs to be used in vLLM.

        Apply HF Processor on prompt text and multi-modal data together,
        outputting token IDs and processed tensors.
        """
        mm_items = self._to_mm_items(mm_data)
        prompt_ids, mm_kwargs, mm_token_type_ids = self._apply_hf_processor_text_mm(
            prompt_text=prompt,
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )

        # HF processor will return `mm_token_type_ids` from which
        # we can infer mm_placeholders. Until then hardcode to make code run
        # Below tested on Llava. Prompts and `mm_token_type_ids` are always bs=1
        mm_positions = torch.where(mm_token_type_ids == 1)[1]
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        mm_tokens_per_modality = hf_processor._get_num_mm_tokens(
            image_inputs=mm_kwargs.get_hf_inputs("image"),
            video_inputs=mm_kwargs.get_hf_inputs("video"),
        )

        mm_placeholders = {}
        for modality in mm_tokens_per_modality:
            split_sizes = mm_tokens_per_modality[modality]
            if split_sizes != 0:
                chunked_mm_positions = torch.split(mm_positions, split_sizes)
                ranges = [
                    PlaceholderRange(offset=positions[0].item(), length=positions.shape[0]) 
                    for positions in chunked_mm_positions
                ]
                mm_placeholders = {modality: ranges}

        print(mm_placeholders)
        return MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=None,
            mm_placeholders=mm_placeholders,
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
        self.text_config = config.get_text_config()
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
                attn_implementation={"text_config": "vllm", "vision_config": "sdpa"},
                torch_dtype=model_config.dtype,
                trust_remote_code=model_config.trust_remote_code,
            )

        self.pipeline_parallel()
        self.tensor_parallel()

        # Input embeddings
        text_config = config.get_text_config()
        if not isinstance(self.model.get_input_embeddings(), PPMissingLayer):
            self.model.set_input_embeddings(
                VocabParallelEmbedding(
                    text_config.vocab_size,
                    text_config.hidden_size,
                    org_num_embeddings=text_config.vocab_size,
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
                                                    text_config.hidden_size))

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
            if self.pp_group.is_first_rank or (self.text_config.tie_word_embeddings
                                               and self.pp_group.is_last_rank):
                continue
            setattr(self.model, name, PPMissingLayer())

        # Module list
        start_layer, end_layer = get_pp_indices(self.text_config.num_hidden_layers,
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
        start, end = get_pp_indices(self.text_config.num_hidden_layers,
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
    embedding_modules = ["embed_tokens"]  # TODO transformers will have a util to get it

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

        self.sampler = get_sampler()

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

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:

        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


@MULTIMODAL_REGISTRY.register_processor(MultiModalProcessor,
                                        info=MultiModalProcessingInfo,
                                        dummy_inputs=MultiModalDummyInputsBuilder)
class TransformersForMultimodalLM(nn.Module, SupportsQuant, SupportsLoRA,
                              SupportsPP, SupportsMultiModal):
    embedding_padding_modules = ["lm_head"]
    embedding_modules = ["embed_tokens"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: PretrainedConfig = vllm_config.model_config.hf_config
        quant_config: QuantizationConfig = vllm_config.quant_config

        self.config = config

        self.model = TransformersModel(vllm_config=vllm_config, prefix=prefix)
        text_config = config.get_text_config()

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = text_config.vocab_size
            self.lm_head = ParallelLMHead(
                text_config.vocab_size,
                text_config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if text_config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.get_input_embeddings())

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    text_config.vocab_size,
                                                    logit_scale)
        else:
            self.lm_head = PPMissingLayer()

        self.sampler = get_sampler()

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

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:

        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params = set[str]()
        for name, loaded_weight in weights:
            if name not in params_dict:
                # In MLLM the head is usually part of the LM so we might want to strip it
                # Very bad workaround, needs smth better
                if "lm_head" in name:
                    name = name.replace("language_model.", "")
                else:
                    name = f"{self.model.base_model_prefix}.{name}"
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        return loaded_params

    def get_multimodal_embeddings(self, **kwargs):
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            vision_embeddings = self.model.get_image_features(
                # Thing about pixels being batched again, adding extra dim
                # TODO: find out do we really need that extra dim
                pixel_values.flatten(0, 1), 
                vision_feature_layer=self.config.vision_feature_layer,
                vision_feature_select_strategy=self.config.vision_feature_select_strategy,
            )
            return vision_embeddings

        if image_embeds is not None:
            return image_embeds

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if multimodal_embeddings is not None:
            # most supported VLMs merge like this, otherwise we can add a special
            # `merge_multimodal_embeddings` method on HF side
            mask = (input_ids == self.config.image_token_index)
            mask = mask.unsqueeze(-1).expand_as(inputs_embeds)
            multimodal_embeddings = torch.cat(multimodal_embeddings)

            # FIXME: The returned multimodal_embeddings must be either a 3D torch.Tensor of shape
            # (num_items, feature_size, hidden_size), or a list / tuple of 2D torch.Tensor’s of shape
            # (feature_size, hidden_size), so that multimodal_embeddings[i] retrieves the embeddings generated
            # from the i-th multimodal data item (e.g, image) of the request.
            inputs_embeds = inputs_embeds.masked_scatter(mask, multimodal_embeddings)
        return inputs_embeds