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
from transformers import AutoModel, PreTrainedModel, LlavaConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from vllm.attention import Attention, AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.utils import divide
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
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

from .interfaces import SupportsQuant, SupportsMultiModal
from .utils import maybe_prefix

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
        attn_metadata: Optional[AttentionMetadata] = None,
        attention_instances: Optional[list[Attention]] = None,
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
    }.get(style, ReplicatedLinear)

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
        quant_config=quant_config,
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


@MULTIMODAL_REGISTRY.register_processor(MultiModalProcessor,
                                        info=MultiModalProcessingInfo,
                                        dummy_inputs=MultiModalDummyInputsBuilder)
class TransformersModel(nn.Module, SupportsQuant, SupportsMultiModal):
    embedding_padding_modules = ["lm_head"]
    embedding_modules = ["embed_tokens"]  # TODO transformers will have a util to get it

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        logger.info("Using Transformers backend.")

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config

        self.config = config
        self.text_config = config.get_text_config()
        self.vocab_size = self.text_config.vocab_size
        self.unpadded_vocab_size = self.text_config.vocab_size

        self.model: PreTrainedModel = AutoModel.from_config(
            self.config,
            attn_implementation={"text_config": "vllm", "vision_config": "eager"},
            torch_dtype=vllm_config.model_config.dtype,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
        )
        prefix = self.model.base_model_prefix

        # MLP modifications
        self.apply_base_model_tp_plan(self.model)

        # Attention modifications (assumes 1 attention op per hidden layer)
        tp_size = get_tensor_model_parallel_world_size()
        self.attention_instances = [
            Attention(
                num_heads=divide(self.text_config.num_attention_heads, tp_size),
                head_size=self.text_config.head_dim,
                # NOTE: We use Llama scale as default, if it's set by
                # Transformers, it's updated in vllm_flash_attention_forward
                scale=self.text_config.head_dim**-0.5,
                num_kv_heads=divide(self.text_config.num_key_value_heads, tp_size),
                cache_config=cache_config,
                quant_config=self.quant_config,
                prefix=f"{i}.attn") for i in range(self.text_config.num_hidden_layers)
        ]

        # Model modifications
        self.replace_vocab_embed_class(self.model)

        # ForCausalLM modifications
        self.lm_head = ParallelLMHead(self.text_config.vocab_size,
                                      self.text_config.hidden_size,
                                      quant_config=self.quant_config,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        if self.text_config.tie_word_embeddings:
            self.lm_head.weight = self.model.get_input_embeddings().weight

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                self.vocab_size, logit_scale)
        self.sampler = get_sampler()

    def apply_base_model_tp_plan(self, module: nn.Module, prefix: str = ""):
        """
        Apply the base model tensor parallelization plan to a module.
        Currently only supports linear layers.
        """
        if (self.text_config.base_model_tp_plan is None
                and get_tensor_model_parallel_world_size() > 1):
            raise ValueError(
                "Trying to run tensor parallelization but the model does not "
                "support it yet!")

        for child_name, child_module in module.named_children():
            qual_name = maybe_prefix(prefix, child_name)
            for pattern, style in self.text_config.base_model_tp_plan.items():
                if re.match(pattern, qual_name) and isinstance(
                        child_module, nn.Linear):
                    new_module = replace_linear_class(child_module, style,
                                                      self.quant_config)
                    setattr(module, child_name, new_module)
                    log_replacement(qual_name, child_module, new_module)
            else:
                self.apply_base_model_tp_plan(child_module, prefix=qual_name)

    def replace_vocab_embed_class(self, module: nn.Module):
        # Use native set input embeddings
        new_module = VocabParallelEmbedding(
            self.vocab_size,
            self.text_config.hidden_size,
            org_num_embeddings=self.vocab_size,
            quant_config=None,
        )
        log_replacement("input embedding", self.model.get_input_embeddings(),
                        new_module)
        self.model.set_input_embeddings(new_module)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],  # argument not used
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(
            input_ids[None, ...] if input_ids is not None else None,
            inputs_embeds=inputs_embeds[None, ...] if inputs_embeds is not None else None,
            use_cache=False,
            position_ids=positions[None, ...],
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