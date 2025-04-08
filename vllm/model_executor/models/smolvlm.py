# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from typing import Dict, Literal, Optional,TypedDict, Union

import torch
from transformers import (BatchFeature, SmolVLMImageProcessor,
                          SmolVLMProcessor)

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.parse import ImageProcessorItems
# yapf conflicts with isort for this block
# yapf: disable
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        MultiModalDataItems,
                                        MultiModalFieldConfig,
                                        PromptReplacement, PromptUpdate,
                                        PromptUpdateDetails)
# yapf: enable
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
# yapf: disable
from .idefics3 import Idefics3ForConditionalGeneration, Idefics3ProcessingInfo
# yapf: enable


class SmolVLMImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    """
    Shape: `(batch_size * num_images * num_patches, 
             num_channels, height, width)`
    """
    pixel_attention_mask: torch.Tensor

    num_patches: torch.Tensor
    """Shape: `(batch_size * num_images)`"""


class SmolVLMImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """
    Shape: `(batch_size * num_images, image_feature_size, hidden_size)`
    `hidden_size` must match the hidden size of language model backbone.
    """


ImageInputs = Union[SmolVLMImagePixelInputs, SmolVLMImageEmbeddingInputs]


class SmolVLMProcessingInfo(Idefics3ProcessingInfo):

    def get_hf_processor(
        self,
        *,
        size: Optional[Dict[str, int]] = None,
        **kwargs: object,
    ) -> SmolVLMProcessor:
        if size is not None:
            kwargs["size"] = size

        return self.ctx.get_hf_processor(SmolVLMProcessor, **kwargs)

    def get_image_repl(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[SmolVLMProcessor],
    ) -> str:
        if processor is None:
            processor = self.get_hf_processor()
        image_token = processor.image_token
        fake_image_token = processor.fake_image_token
        global_img_token = processor.global_image_token
        image_seq_len = processor.image_seq_len
        grid_placeholder = "<row_{n_h}_col_{n_w}>"

        p_img = image_token * image_seq_len
        global_img_placeholder = fake_image_token + global_img_token + p_img
        tile_img_placeholder = fake_image_token + grid_placeholder + p_img

        grid_w, grid_h = self._get_image_feature_grid_size(
            image_width=image_width,
            image_height=image_height,
            processor=processor,
        )
        if grid_w == 0 and grid_h == 0:
            return global_img_placeholder + fake_image_token

        tiles_placeholder = list[str]()
        for i in range(grid_h):
            for j in range(grid_w):
                placeholder_per_tile = tile_img_placeholder.format(n_h=i + 1,
                                                                   n_w=j + 1)
                tiles_placeholder.append(placeholder_per_tile)
                # Add line break if it is the last tile in the row
                if j == grid_w - 1:
                    tiles_placeholder.append("\n")
        return "".join([
            *tiles_placeholder,
            "\n",
            global_img_placeholder,
            fake_image_token,
        ])


class SmolVLMDummyInputsBuilder(BaseDummyInputsBuilder[SmolVLMProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)
        hf_processor = self.info.get_hf_processor()
        image_processor: SmolVLMImageProcessor = hf_processor.image_processor
        longest_edge = image_processor.max_image_size['longest_edge']
        image_token = hf_processor.image_token

        mm_data = {
            "image":
            self._get_dummy_images(width=longest_edge,
                                   height=longest_edge,
                                   num_images=num_images)
        }

        return ProcessorInputs(
            prompt_text=image_token * num_images,
            mm_data=mm_data,
        )


class SmolVLMMultiModalProcessor(BaseMultiModalProcessor[SmolVLMProcessingInfo]
                                 ):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Text-only input not supported in composite processor
        if not (images := mm_data.get("images", [])):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        processed_outputs = super()._call_hf_processor(
            prompt,
            mm_data,
            mm_kwargs,
        )

        parsed_images = (self._get_data_parser().parse_mm_data({
            "image": images
        }).get_items("image", ImageProcessorItems))
        image_sizes = [
            parsed_images.get_image_size(i) for i in range(len(parsed_images))
        ]
        hf_processor = self.info.get_hf_processor(**mm_kwargs)

        num_patches = [
            self.info.get_num_patches(
                image_width=size.width,
                image_height=size.height,
                processor=hf_processor,
            ) for size in image_sizes
        ]
        processed_outputs["num_patches"] = torch.tensor(num_patches)

        # Remove the extra batch dimension
        processed_outputs["pixel_values"].squeeze_(0)
        processed_outputs["pixel_attention_mask"].squeeze_(0)

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
            pixel_attention_mask=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches),
            image_embeds=MultiModalFieldConfig.batched("image"),
            num_patches=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_token = hf_processor.image_token

        def get_replacement_smolvlm(item_idx: int) -> PromptUpdateDetails:
            images = mm_items.get_items("image", ImageProcessorItems)

            image_size = images.get_image_size(item_idx)

            image_repl = self.info.get_image_repl(
                image_width=image_size.width,
                image_height=image_size.height,
                processor=hf_processor,
            )

            return PromptUpdateDetails.select_text(
                image_repl,
                embed_text=image_token,
            )

        return [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_replacement_smolvlm,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(SmolVLMMultiModalProcessor,
                                        info=SmolVLMProcessingInfo,
                                        dummy_inputs=SmolVLMDummyInputsBuilder)
class SmolVLMForConditionalGeneration(Idefics3ForConditionalGeneration):
   

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
        )