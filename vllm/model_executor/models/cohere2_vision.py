# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/model_executor/models/aya_vision.py
"""Command-A-Vision (Cohere2Vision) multimodal model implementation for vLLM."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, Optional, Union

import torch
from torch import nn
from transformers import BatchFeature, PretrainedConfig
from transformers.models.cohere2_vision import Cohere2VisionConfig
from transformers.models.cohere2_vision.image_processing_cohere2_vision_fast import (  # noqa: E501
    get_optimal_tiled_canvas)
from transformers.models.cohere2_vision.processing_cohere2_vision import (
    Cohere2VisionProcessor)

from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import MulAndSilu
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict, MultiModalKwargsItems
from vllm.multimodal.parse import (ImageProcessorItems, ImageSize,
                                   MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo,
                                        MultiModalFieldConfig,
                                        PromptReplacement, PromptUpdate,
                                        PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, WeightsMapper, flatten_bn,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings)


class Cohere2VisionImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - np: The total number of patches over each image over each prompt in
              the batch
        - c: Number of channels
        - h: Height of each image patch
        - w: Width of each image patch
        - bn: Batch size * number of images
    """

    type: Literal["pixel_values"]

    pixel_values: Annotated[
        torch.Tensor,
        TensorShape("np", 3, "h", "w"),
    ]

    num_patches: Annotated[
        torch.Tensor,
        TensorShape("bn"),
    ]


class Cohere2VisionMultiModalProjector(nn.Module):
    """Multimodal projector that maps vision features to text embedding space.
    
    Uses pixel shuffle downsampling followed by SwiGLU activation.
    """

    def __init__(self, config: Cohere2VisionConfig, prefix: str = ""):
        super().__init__()
        self.downsample_factor = config.downsample_factor

        # Input dimension after pixel shuffle downsampling
        input_dim = config.vision_config.hidden_size * (
            config.downsample_factor**2)
        # MergedColumnParallelLinear expects the intermediate size to be a list
        # of sizes, so that it will load the weights as two separate linear
        # layers before applying any parallelism.
        # We need to divide the alignment intermediate size by 2 because
        # the weights are merged weights of two linear layers for SwiGLU.
        self.intermediate_size = config.alignment_intermediate_size // 2

        self.linear_1 = MergedColumnParallelLinear(
            input_dim,
            [self.intermediate_size] * 2,
            bias=True,
            return_bias=False,
            prefix=f"{prefix}.linear_1",
        )
        self.act = MulAndSilu()
        self.linear_2 = RowParallelLinear(
            self.intermediate_size,
            config.text_config.hidden_size,
            bias=True,
            return_bias=False,
            prefix=f"{prefix}.linear_2",
        )

    def forward(self, image_features):
        image_features = self.pixel_shuffle(image_features)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def pixel_shuffle(self, image_features: torch.Tensor) -> torch.Tensor:
        """Apply pixel shuffle downsampling to reduce spatial dimensions.
        
        Args:
            image_features: Input tensor of shape [B, S, D] where S = H*W
            
        Returns:
            Downsampled tensor with increased channel dimension
        """
        height = width = int(image_features.shape[1]**0.5)
        x = image_features.reshape(image_features.shape[0], width, height, -1)
        n, h, w, c = x.size()
        scale_factor = 1. / self.downsample_factor
        nh = int(h * scale_factor)
        nw = int(w * scale_factor)
        x = x.reshape(n, nh, self.downsample_factor, nw,
                      self.downsample_factor, c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(n, nh, nw, -1)
        return x


class Cohere2VisionProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> Cohere2VisionConfig:
        return self.ctx.get_hf_config(Cohere2VisionConfig)

    def get_hf_processor(self, **kwargs: object) -> Cohere2VisionProcessor:
        return self.ctx.get_hf_processor(Cohere2VisionProcessor, **kwargs)

    def get_image_processor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).image_processor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_image_size_with_most_features(self) -> ImageSize:
        image_processor = self.get_image_processor()
        height = image_processor.size['height']
        width = image_processor.size['width']
        max_patches = image_processor.max_patches
        return ImageSize(height=height * max_patches, width=width)

    def get_num_patches(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[Cohere2VisionProcessor],
    ) -> int:
        """
        Calculate the number of image patches for a given image.
        Uses the HF processor to determine the actual number of patches.
        """
        if processor is None:
            processor = self.get_hf_processor()

        image_processor = processor.image_processor

        # The current implementation of get_number_of_image_patches
        # is incorrect, so we patch it here.
        # TODO: Revert once
        # https://github.com/huggingface/transformers/pull/40312 is released.
        # return image_processor.get_number_of_image_patches(image_height,
        #                                                    image_width, {})

        min_patches = image_processor.min_patches
        max_patches = image_processor.max_patches
        patch_size = image_processor.size
        crop_to_patches = image_processor.crop_to_patches

        if not crop_to_patches:
            return 1

        num_columns, num_rows = get_optimal_tiled_canvas(
            (image_height, image_width),
            (patch_size["height"], patch_size["width"]),
            min_patches,
            max_patches,
        )
        num_patches = num_columns * num_rows
        if num_patches > 1:
            num_patches += 1  # Thumbnail image

        return num_patches


class Cohere2VisionDummyInputsBuilder(
        BaseDummyInputsBuilder[Cohere2VisionProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        image_size = \
            self.info.get_image_size_with_most_features()

        return {
            "image":
            self._get_dummy_images(width=image_size.width,
                                   height=image_size.height,
                                   num_images=num_images)
        }


class Cohere2VisionMultiModalProcessor(
        BaseMultiModalProcessor[Cohere2VisionProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super()._call_hf_processor(
            prompt,
            mm_data,
            mm_kwargs,
            tok_kwargs,
        )

        # Ensure num_patches is available for proper tensor splitting
        if "num_patches" not in processed_outputs and (
                images := mm_data.get("images")) is not None:
            hf_processor = self.info.get_hf_processor(**mm_kwargs)

            # Fallback calculation if HF processor didn't provide num_patches
            parsed_images = self._get_data_parser().parse_mm_data({
                "image":
                images
            }).get_items("image", ImageProcessorItems)

            num_patches = [
                self.info.get_num_patches(
                    image_width=parsed_images.get_image_size(i).width,
                    image_height=parsed_images.get_image_size(i).height,
                    processor=hf_processor,
                ) for i in range(len(parsed_images))
            ]
            processed_outputs["num_patches"] = torch.tensor(num_patches)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_patches = hf_inputs.get("num_patches", torch.empty(0))
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches),
            num_patches=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_token = hf_processor.image_token
        img_tokens_per_tile = int(hf_processor.patch_size**2)
        img_line_break_token = hf_processor.img_line_break_token
        boi_token = hf_processor.boi_token
        eoi_token = hf_processor.eoi_token

        def get_replacement(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size: ImageSize = images.get_image_size(item_idx)

            num_patches = self.info.get_num_patches(
                image_width=image_size.width,
                image_height=image_size.height,
                processor=hf_processor,
            )
            patch_tokens = (image_token * img_tokens_per_tile +
                            img_line_break_token)
            repl = f"{boi_token}{patch_tokens * num_patches}{eoi_token}"

            return PromptUpdateDetails.select_text(repl, image_token)

        return [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_replacement,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Cohere2VisionMultiModalProcessor,
    info=Cohere2VisionProcessingInfo,
    dummy_inputs=Cohere2VisionDummyInputsBuilder)
class Cohere2VisionForConditionalGeneration(nn.Module, SupportsMultiModal,
                                            SupportsPP):

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "model.language_model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Cohere2VisionConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.quant_config = quant_config
        self.multimodal_config = multimodal_config
        self._patch_quant_config(config, quant_config)

        self.vision_tower = SiglipVisionModel(config.vision_config,
                                              quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "vision_tower"))
        self.vocab_size = config.text_config.vocab_size
        self.multi_modal_projector = \
            Cohere2VisionMultiModalProjector(
                config, prefix=maybe_prefix(prefix, "multi_modal_projector"))
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=config.text_config.architectures)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def _process_image_input(self, image_input: Cohere2VisionImagePixelInputs,
                             **kwargs) -> list[torch.Tensor]:
        """Process image pixels through vision tower and projector.
        
        Args:
            image_input: Validated image input containing pixel values and 
                         patch counts
            
        Returns:
            List of flattened image embeddings, one per image
        """
        assert self.vision_tower is not None, "Vision tower is required"

        pixel_values = image_input["pixel_values"]
        num_patches = image_input["num_patches"]

        # Extract visual features
        image_features = self.vision_tower(pixel_values)

        # Project to text embedding space
        image_embeds = self.multi_modal_projector(image_features)

        # Split and flatten embeddings per image
        return [
            e.flatten(0, 2) for e in image_embeds.split(num_patches.tolist())
        ]

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Cohere2VisionImagePixelInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        num_patches = kwargs.pop("num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)
        assert image_embeds is None, \
            "Cohere2Vision does not support image_embeds."

        if pixel_values is None:
            return None

        return Cohere2VisionImagePixelInputs(
            type="pixel_values",
            pixel_values=flatten_bn(pixel_values, concat=True),
            num_patches=flatten_bn(num_patches, concat=True),
            resolve_bindings={
                "h": self.config.vision_config.image_size,
                "w": self.config.vision_config.image_size,
            })

    def _patch_quant_config(self, config: PretrainedConfig,
                            quant_config: QuantizationConfig):
        # the awq models from OpenGVLab missing `modules_to_not_convert`
        # patch the quant_config to add `modules_to_not_convert` back
        if isinstance(quant_config, AWQConfig):
            text_config = config.text_config
            llm_quant_config = getattr(text_config, "quantization_config",
                                       None)
            if (not quant_config.modules_to_not_convert) and (llm_quant_config
                                                              is not None):
                quant_config.modules_to_not_convert.append("vision_tower")

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input, **kwargs)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=self.config.image_token_id,
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)
