# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/fuyu/modeling_fuyu.py
# Copyright 2023 The vLLM team.
# Copyright 2023 HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Fuyu model."""
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, Optional

import torch
import torch.nn as nn
from transformers import (BatchFeature, FuyuConfig, FuyuImageProcessor,
                          FuyuProcessor)

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.models.persimmon import PersimmonForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import (ImageProcessorItems, ImageSize,
                                   MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, WeightsMapper, flatten_bn, maybe_prefix,
                    merge_multimodal_embeddings)

# Cannot find the following 2 numbers from hf config.
_IMAGE_TOKEN_ID = 71011
_NEWLINE_TOKEN_ID = 71019


class FuyuImagePatchInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - bnp: Batch size * number of images * number of patches
        - fn: patch_size_x * patch_size_y * num_channels
    """

    type: Literal["image_patches"] = "image_patches"

    flat_data: Annotated[
        torch.Tensor,
        TensorShape("bnp", "fn"),
    ]

    patches_per_image: Annotated[list[int], TensorShape("bn")]
    """
    The number of total patches for each image in the batch.
    
    This is used to split the embeddings which has the first two dimensions
    flattened just like `flat_data`.
    """


class FuyuProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(FuyuConfig)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(FuyuProcessor, **kwargs)

    def get_image_processor(self, **kwargs: object) -> FuyuImageProcessor:
        return self.get_hf_processor(**kwargs).image_processor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 1}

    def get_image_feature_grid_size(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int]:
        image_processor = self.get_image_processor()
        target_width = image_processor.size["width"]
        target_height = image_processor.size["height"]
        patch_width = image_processor.patch_size["width"]
        patch_height = image_processor.patch_size["height"]

        if not (image_width <= target_width and image_height <= target_height):
            height_scale_factor = target_height / image_height
            width_scale_factor = target_width / image_width
            optimal_scale_factor = min(height_scale_factor, width_scale_factor)

            image_height = int(image_height * optimal_scale_factor)
            image_width = int(image_width * optimal_scale_factor)

        ncols = math.ceil(image_width / patch_width)
        nrows = math.ceil(image_height / patch_height)
        return ncols, nrows

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        ncols, nrows = self.get_image_feature_grid_size(
            image_width=image_width,
            image_height=image_height,
        )

        return ncols * nrows

    def get_image_size_with_most_features(self) -> ImageSize:
        image_processor = self.get_image_processor()
        return ImageSize(width=image_processor.size["width"],
                         height=image_processor.size["height"])


class FuyuDummyInputsBuilder(BaseDummyInputsBuilder[FuyuProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        num_images = mm_counts.get("image", 0)

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


class FuyuMultiModalProcessor(BaseMultiModalProcessor[FuyuProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
            # Avoid warning from HF logger for text-only input
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        image_patches = processed_outputs.get("image_patches")
        if image_patches is not None:
            images = mm_data["images"]
            assert isinstance(images, list)

            # Original output: (1, num_images, Pn, Px * Py * C)
            # New output: (num_images, Pn, Px * Py * C)
            # image_patches is a list with shape:
            # (1, num_images, Pn, Px * Py * C)
            # before Transformers 4.53
            if isinstance(image_patches, list):
                assert len(image_patches) == 1
                assert (isinstance(image_patches[0], torch.Tensor)
                        and len(image_patches[0]) == len(images))
                processed_outputs["image_patches"] = image_patches[0]
            # image_patches is a tensor with shape:
            # (num_images, Pn, Px * Py * C)
            # after Transformers 4.53
            elif isinstance(image_patches, torch.Tensor):
                assert len(image_patches) == len(images)
            else:
                raise AssertionError("This line should be unreachable.")

        return processed_outputs

    def _apply_hf_processor_tokens_only(
        self,
        prompt_tokens: list[int],
    ) -> list[int]:
        # HF processor adds boa_token_id
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        boa_token_id = vocab["<0x04>"]
        if prompt_tokens[-1] != boa_token_id:
            prompt_tokens.append(boa_token_id)

        return prompt_tokens

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(image_patches=MultiModalFieldConfig.batched("image"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        bos_token_id = hf_config.bos_token_id
        assert isinstance(bos_token_id, int)

        tokenizer = self.info.get_tokenizer()
        eot_token_id = tokenizer.bos_token_id
        assert isinstance(eot_token_id, int)

        def get_replacement_fuyu(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)

            ncols, nrows = self.info.get_image_feature_grid_size(
                image_width=image_size.width,
                image_height=image_size.height,
            )
            image_tokens = ([_IMAGE_TOKEN_ID] * ncols +
                            [_NEWLINE_TOKEN_ID]) * nrows

            return PromptUpdateDetails.select_token_id(
                image_tokens + [bos_token_id],
                embed_token_id=_IMAGE_TOKEN_ID,
            )

        return [
            PromptReplacement(
                modality="image",
                target=[eot_token_id],
                replacement=get_replacement_fuyu,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(FuyuMultiModalProcessor,
                                        info=FuyuProcessingInfo,
                                        dummy_inputs=FuyuDummyInputsBuilder)
class FuyuForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.vision_embed_tokens.": "vision_embed_tokens.",
            "model.language_model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        })

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return None

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        self.vocab_size = config.text_config.vocab_size
        self.image_token_id = _IMAGE_TOKEN_ID
        self.image_feature_size = config.patch_size**2 * config.num_channels

        self.vision_embed_tokens = ColumnParallelLinear(
            self.image_feature_size,
            config.hidden_size,
            quant_config=quant_config,
            gather_output=True,
        )
        self.language_model = PersimmonForCausalLM(
            vllm_config=vllm_config.with_hf_config(config.text_config),
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[FuyuImagePatchInputs]:
        image_patches = kwargs.pop("image_patches", None)
        if image_patches is not None:
            image_patches_flat = flatten_bn(image_patches)
            flat_data = flatten_bn(image_patches_flat, concat=True)

            return FuyuImagePatchInputs(
                type="image_patches",
                flat_data=flat_data,
                patches_per_image=[x.size(0) for x in image_patches_flat],
                resolve_bindings={"fn": self.image_feature_size},
            )

        return None

    def _process_image_input(
            self, image_input: FuyuImagePatchInputs) -> MultiModalEmbeddings:
        image_patches_flat = image_input["flat_data"]
        patches_per_image = image_input["patches_per_image"]

        assert self.vision_embed_tokens is not None
        vision_embeddings_flat, _ = self.vision_embed_tokens(
            image_patches_flat)

        return vision_embeddings_flat.split(patches_per_image, dim=0)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                _IMAGE_TOKEN_ID,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ):
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.language_model.logits_processor(
            self.language_model.lm_head, hidden_states, sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
