# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# Adapted from https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/modeling_kimi_vl.py
# Copyright 2025 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
#
# The code is based on llava (llava/modeling_llava.py) and DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py), but modified for KimiVL.
#
# Licensing Information:
# - Code derived from llava (llava/modeling_llava.py) and DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py) is licensed under the Apache License, Version 2.0.
# - Other parts of the code are licensed under the MIT License.
#
# Apache License, Version 2.0:
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
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import torch
from torch import nn
from transformers import BatchFeature, DeepseekV2Config
from transformers.activations import GELUActivation

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.moonvit import MoonVitPretrainedModel
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import KimiVLConfig, MoonViTConfig
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .utils import PPMissingLayer, is_pp_missing_parameter, maybe_prefix
from .vision import run_dp_sharded_mrope_vision_model


# For dummy input only
@dataclass
class MaxImageTokenMeta:
    width: int = 1024
    height: int = 1024


class KimiVLMultiModalProjector(nn.Module):
    def __init__(
        self, config: KimiVLConfig, use_data_parallel: bool = False, prefix: str = ""
    ):
        super().__init__()
        self.use_data_parallel = use_data_parallel

        self.hidden_size = (
            config.vision_config.hidden_size
            * config.vision_config.merge_kernel_size[0]
            * config.vision_config.merge_kernel_size[1]
        )

        self.pre_norm = torch.nn.LayerNorm(config.vision_config.hidden_size, eps=1e-5)
        self.linear_1 = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            prefix=maybe_prefix(prefix, "linear_1"),
        )
        self.linear_2 = ReplicatedLinear(
            self.hidden_size,
            config.text_config.hidden_size,
            bias=True,
            prefix=maybe_prefix(prefix, "linear_2"),
        )
        self.act = GELUActivation()

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)
        hidden_states, _ = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class KimiVLImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - nc: Number of channels
        - np: Number of patches
        - ps: Patch size
        - ni: Number of images
    """

    type: Literal["pixel_values"] = "pixel_values"

    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("np", 3, "ps", "ps"),
    ]

    image_grid_hws: Annotated[torch.Tensor, TensorShape("ni", 2)]


# TODO: support embeds too
# We only support pixel input for kimi-vl now
KimiVLImageInputs = KimiVLImagePixelInputs


class KimiVLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(KimiVLConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_processor = self.get_hf_processor()
        patch_size = hf_processor.image_processor.patch_size
        kernel_size = hf_processor.image_processor.merge_kernel_size
        in_token_limit = hf_processor.image_processor.in_token_limit
        height = image_height
        width = image_width
        assert isinstance(height, int), f"height must be int, current height {height}"
        assert isinstance(width, int), f"width must be int, current width {width}"
        assert kernel_size is not None, "kernel_size must be specified"

        if (width // patch_size) * (height // patch_size) > in_token_limit:
            scale = math.sqrt(
                in_token_limit / ((width // patch_size) * (height // patch_size))
            )
            new_w, new_h = int(width * scale), int(height * scale)
            width, height = new_w, new_h

        kernel_height, kernel_width = kernel_size

        pad_height = (
            kernel_height * patch_size - height % (kernel_height * patch_size)
        ) % (kernel_height * patch_size)
        pad_width = (
            kernel_width * patch_size - width % (kernel_width * patch_size)
        ) % (kernel_width * patch_size)

        # Calculate new dimensions after padding and patching
        token_height = (height + pad_height) // (kernel_size[0] * patch_size)
        token_width = (width + pad_width) // (kernel_size[1] * patch_size)
        return int(token_height * token_width)

    @property
    def image_token_id(self) -> int:
        return self.get_hf_config().media_placeholder_token_id


class KimiVLDummyInputsBuilder(BaseDummyInputsBuilder[KimiVLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=MaxImageTokenMeta.width,
                height=MaxImageTokenMeta.height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class KimiVLMultiModalProcessor(BaseMultiModalProcessor[KimiVLProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_hws = hf_inputs.get("image_grid_hws", torch.empty((0, 2)))
        image_grid_sizes = image_grid_hws.prod(-1)

        # pixel_values is merged as a single large tensor
        # image_grid_hws is shapes for each subtensor in pixel_values
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_sizes
            ),
            image_grid_hws=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_token_id = self.info.image_token_id

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )

            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    KimiVLMultiModalProcessor,
    info=KimiVLProcessingInfo,
    dummy_inputs=KimiVLDummyInputsBuilder,
)
class KimiVLForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    merge_by_field_config = True

    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|media_start|>image<|media_content|><|media_pad|><|media_end|>"

        raise ValueError("Only image modality is supported")

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiVLConfig = model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        assert isinstance(config.vision_config, MoonViTConfig)
        self.use_data_parallel = (
            model_config.multimodal_config.mm_encoder_tp_mode == "data"
        )
        self.hidden_size = config.text_config.hidden_size
        self.vision_tower = MoonVitPretrainedModel(
            config.vision_config,
            self.use_data_parallel,
            prefix=maybe_prefix(prefix, "vision_tower"),
        )

        self.multi_modal_projector = KimiVLMultiModalProjector(
            config=config,
            use_data_parallel=self.use_data_parallel,
            prefix=maybe_prefix(prefix, "multi_modal_projector"),
        )

        self.quant_config = quant_config
        sub_vllm_config = copy.deepcopy(vllm_config)
        sub_vllm_config.model_config.hf_config = (
            sub_vllm_config.model_config.hf_config.text_config
        )
        self.language_model = DeepseekV2Model(
            vllm_config=sub_vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.unpadded_vocab_size = config.text_config.vocab_size
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.text_config.hidden_size,
                org_num_embeddings=self.config.text_config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size, config.vocab_size, logit_scale
        )
        self.media_placeholder: int = self.config.media_placeholder_token_id

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> KimiVLImageInputs | None:
        # image input type must be pixel values now
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_hws = kwargs.pop("image_grid_hws", None)

        if pixel_values is None:
            return None

        return KimiVLImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_hws=image_grid_hws,
        )

    # perform vt on processored pixel_values
    @torch.inference_mode()
    def _process_image_pixels(self, inputs: KimiVLImagePixelInputs) -> torch.Tensor:
        assert self.vision_tower is not None

        pixel_values = inputs["pixel_values"]
        image_grid_hws = inputs["image_grid_hws"]
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.vision_tower,
                pixel_values,
                image_grid_hws.tolist(),
                rope_type="rope_2d",
            )
        else:
            return self.vision_tower(pixel_values, image_grid_hws)

    def _process_image_input(self, image_input: KimiVLImageInputs) -> torch.Tensor:
        assert image_input["type"] == "pixel_values"
        image_features = self._process_image_pixels(image_input)
        assert isinstance(image_features, (list, tuple))
        lengths = [x.shape[0] for x in image_features]
        return self.multi_modal_projector(torch.cat(image_features)).split(lengths)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self, **kwargs: object) -> NestedTensors | None:
        # Validate the multimodal input keyword arguments
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        # Run multimodal inputs through encoder and projector
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states, **kwargs)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        config = self.config.text_config
        _KEYS_TO_MODIFY_MAPPING = {
            "language_model.lm_head": "lm_head",
            "language_model.model": "language_model",
        }
        # only doing this for language model part for now.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        if not config.use_mla:
            stacked_params_mapping += [
                (".qkv_proj", ".q_proj", "q"),
                (".qkv_proj", ".k_proj", "k"),
                (".qkv_proj", ".v_proj", "v"),
            ]
        if getattr(config, "n_routed_experts", None):
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=config.n_routed_experts,
            )
        else:
            expert_params_mapping = []

        params_dict = dict(self.named_parameters())

        for args in weights:
            name, loaded_weight = args[:2]
            kwargs = args[2] if len(args) > 2 else {}
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            use_default_weight_loading = False
            if "vision" in name:
                if self.vision_tower is not None:
                    # We only do sharding for language model and
                    # not vision model for now.
                    use_default_weight_loading = True
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    # We have mlp.experts[0].gate_proj in the checkpoint.
                    # Since we handle the experts below in expert_params_mapping,
                    # we need to skip here BEFORE we update the name, otherwise
                    # name will be updated to mlp.experts[0].gate_up_proj, which
                    # will then be updated below in expert_params_mapping
                    # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                    if ("mlp.experts." in name) and name not in params_dict:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id, **kwargs)
                    break
                else:
                    for idx, (
                        param_name,
                        weight_name,
                        expert_id,
                        shard_id,
                    ) in enumerate(expert_params_mapping):
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)

                        if is_pp_missing_parameter(name, self):
                            continue

                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name,
                            expert_id=expert_id,
                            shard_id=shard_id,
                            **kwargs,
                        )
                        break
                    else:
                        use_default_weight_loading = True
            if use_default_weight_loading:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, **kwargs)


def get_spec_layer_idx_from_weight_name(
    config: DeepseekV2Config, weight_name: str
) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and (
        config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx + i}."):
                return layer_idx + i
    return None
