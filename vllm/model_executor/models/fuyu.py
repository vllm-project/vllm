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
from typing import (Iterable, List, Literal, Mapping, Optional, Set, Tuple,
                    TypedDict)

import torch
import torch.nn as nn
from transformers import (BatchFeature, FuyuConfig, FuyuImageProcessor,
                          FuyuProcessor)

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.inputs import InputContext
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.models.persimmon import PersimmonForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalInputsV2, MultiModalKwargs,
                                    NestedTensors, PlaceholderRange)
from vllm.multimodal.parse import ImageProcessorItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        MultiModalDataItems, ProcessorInputs,
                                        PromptReplacement)
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, flatten_bn, maybe_prefix,
                    merge_multimodal_embeddings)

# Cannot find the following 2 numbers from hf config.
_IMAGE_TOKEN_ID = 71011
_NEWLINE_TOKEN_ID = 71019

MAX_IMAGE_FEATURE_SIZE_HEIGHT = 1080
MAX_IMAGE_FEATURE_SIZE_WIDTH = 1920


class FuyuImagePatchInputs(TypedDict):
    type: Literal["image_patches"]
    data: torch.Tensor
    """
    Shape: 
    `(batch_size * num_patches, patch_size_x * patch_size_y * num_channels)`
    """

    patches_per_image: List[int]
    """
    List of number of total patches for each image in the batch.
    This is used to restore the first two dimensions of `data`.
    """


def _get_fuyu_num_image_tokens(
    image_height: int,
    image_width: int,
) -> Tuple[int, int]:
    """
    Calculate the number of image tokens needed for a given image size.

    The expected Fuyu image prompts can be expressed as:

    .. code-block::
        (image_token * ncols + newline_token) * nrows

    Args:
        image_size: Tuple[int, int] - `(width, height)` of the image

    Returns:
        ncols: int - number of image tokens in `x` direction
        nrows: int - number of image tokens in `y` direction
    """
    ncols = math.ceil(image_width / 30)
    nrows = math.ceil(image_height / 30)
    return ncols, nrows


def get_max_fuyu_image_tokens(ctx: InputContext):
    ncols, nrows = _get_fuyu_num_image_tokens(
        image_height=MAX_IMAGE_FEATURE_SIZE_HEIGHT,
        image_width=MAX_IMAGE_FEATURE_SIZE_WIDTH,
    )

    return (ncols + 1) * nrows


class FuyuMultiModalProcessor(BaseMultiModalProcessor):

    def _get_hf_processor(self) -> FuyuProcessor:
        return self.ctx.get_hf_processor(FuyuProcessor)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:

        if not mm_data:
            # Avoid warning from HF logger for text-only input
            # Input_ids format: bos_token_id + prompt_token_ids + boa_token_id
            # Tokenizer won't add boa_token_id by default, we add it manually.
            tokenizer = self._get_tokenizer()
            boa_token_id: int = tokenizer.vocab["<0x04>"]  # type: ignore
            prompt_ids = tokenizer.encode(prompt) + [boa_token_id]
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )

        image_patches = processed_outputs.get("image_patches")
        if image_patches is not None:
            images = mm_data["images"]
            assert isinstance(images, list)

            # Original output: (1, num_images, Pn, Px * Py * C)
            # New output: (num_images, Pn, Px * Py * C)
            assert (isinstance(image_patches, list)
                    and len(image_patches) == 1)
            assert (isinstance(image_patches[0], torch.Tensor)
                    and len(image_patches[0]) == len(images))

            processed_outputs["image_patches"] = image_patches[0]

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(image_patches=MultiModalFieldConfig.batched("image"))

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        hf_config = self.ctx.get_hf_config(FuyuConfig)
        bos_token_id = hf_config.bos_token_id

        tokenizer = self._get_tokenizer()
        eot_token_id = tokenizer.bos_token_id
        assert isinstance(eot_token_id, int)

        hf_processor = self._get_hf_processor()
        image_processor: FuyuImageProcessor = hf_processor.image_processor
        target_size = image_processor.size
        target_height, target_width = (target_size["height"],
                                       target_size["width"])

        def get_replacement_fuyu(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)
            width, height = image_size.width, image_size.height
            if not (width <= target_width and height <= target_height):
                height_scale_factor = target_height / height
                width_scale_factor = target_width / width
                optimal_scale_factor = min(height_scale_factor,
                                           width_scale_factor)

                height = int(height * optimal_scale_factor)
                width = int(width * optimal_scale_factor)

            ncols, nrows = _get_fuyu_num_image_tokens(
                image_width=width,
                image_height=height,
            )

            return (([_IMAGE_TOKEN_ID] * ncols + [_NEWLINE_TOKEN_ID]) * nrows +
                    [bos_token_id])

        return [
            PromptReplacement(
                modality="image",
                target=[eot_token_id],
                replacement=get_replacement_fuyu,
            )
        ]

    def apply(
        self,
        prompt_text: str,
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> MultiModalInputsV2:
        result = super().apply(prompt_text, mm_data, hf_processor_mm_kwargs)

        # Only |SPEAKER| (image) tokens should be considered as placeholders,
        # so we ignore the trailing bos_token_id
        result["mm_placeholders"] = {
            modality: [
                PlaceholderRange(offset=p["offset"], length=p["length"] - 1)
                for p in ps
            ]
            for modality, ps in result["mm_placeholders"].items()
        }

        return result

    def _get_dummy_mm_inputs(
        self,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)

        mm_data = {
            "image":
            self._get_dummy_images(width=MAX_IMAGE_FEATURE_SIZE_WIDTH,
                                   height=MAX_IMAGE_FEATURE_SIZE_HEIGHT,
                                   num_images=num_images)
        }

        return ProcessorInputs(
            prompt_text="",
            mm_data=mm_data,
        )


@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_fuyu_image_tokens)
@MULTIMODAL_REGISTRY.register_processor(FuyuMultiModalProcessor)
class FuyuForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        self.padding_idx = config.pad_token_id
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

    @property
    def sampler(self):
        return self.language_model.sampler

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:

        h = w = self.config.patch_size
        num_channels = self.config.num_channels
        expected_dims = num_channels * h * w

        def _validate_shape(d: torch.Tensor):
            actual_dims = d.size(-1)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f" per patch is {expected_expr}. "
                    f"You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data.to(self.vision_embed_tokens.weight.dtype)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[FuyuImagePatchInputs]:
        image_patches = kwargs.pop("image_patches", None)
        if image_patches is not None:
            if not isinstance(image_patches, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image patches. "
                                 f"Got type: {type(image_patches)}")

            image_patches_flat = flatten_bn(image_patches)

            return FuyuImagePatchInputs(
                type="image_patches",
                data=self._validate_pixel_values(
                    flatten_bn(image_patches_flat, concat=True)),
                patches_per_image=[x.size(0) for x in image_patches_flat],
            )

        return None

    def _process_image_input(
            self, image_input: FuyuImagePatchInputs) -> NestedTensors:
        image_patches = image_input["data"]
        patches_per_image = image_input["patches_per_image"]

        assert self.vision_embed_tokens is not None
        vision_embeddings, _ = self.vision_embed_tokens(image_patches)
        return vision_embeddings.split(patches_per_image, dim=0)

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                _IMAGE_TOKEN_ID)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
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
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
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

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.language_model.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
