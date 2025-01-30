# adapted from https://huggingface.co/h2oai/h2ovl-mississippi-2b/blob/main/modeling_h2ovl_chat.py
# https://huggingface.co/h2oai/h2ovl-mississippi-2b/blob/main/image_process.py
# --------------------------------------------------------
# H2OVL-Mississippi
# Copyright (c) 2024 H2O.AI
# Licensed under Apache 2.0 License [see LICENSE for details]
# --------------------------------------------------------
from typing import Optional

import torch
from PIL import Image
from transformers import PretrainedConfig

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .intern_vit import InternVisionModel
from .internvl import (IMG_CONTEXT, IMG_END, IMG_START,
                       BaseInternVLProcessingInfo, BaseInternVLProcessor,
                       InternVLChatModel, InternVLDummyInputsBuilder,
                       InternVLMultiModalProcessor, build_transform,
                       find_closest_aspect_ratio, get_internvl_target_ratios)


def resolve_h2ovl_min_max_num(
    *,
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: bool,
    use_thumbnail: bool,
    use_msac: bool,
) -> tuple[int, int]:
    max_dynamic_patch = max_dynamic_patch if dynamic_image_size else 1

    if use_msac:
        max_dynamic_patch *= 2

    if use_thumbnail and max_dynamic_patch > 1:
        max_dynamic_patch += 1

    return min_dynamic_patch, max_dynamic_patch


def get_h2ovl_target_ratios(
    min_num: int,
    max_num: int,
    *,
    prior_aspect_ratio: Optional[tuple[int, int]],
) -> list[tuple[int, int]]:
    target_ratios = get_internvl_target_ratios(min_num, max_num)

    # if prior_aspect_ratio is provided, filter the target ratios
    if prior_aspect_ratio is not None:
        target_ratios = [
            ratio for ratio in target_ratios if prior_aspect_ratio[0] %
            ratio[0] != 0 and prior_aspect_ratio[1] % ratio[1] != 0
        ]

    return target_ratios


# modified to include blocks generated in second pass
def calculate_h2ovl_targets(
    *,
    orig_width: int,
    orig_height: int,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> tuple[int, int, int, tuple[int, int]]:
    aspect_ratio = orig_width / orig_height

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        width=orig_width,
        height=orig_height,
        image_size=image_size,
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    # add thumbnail image if num_blocks > 1
    if use_thumbnail and blocks > 1:
        blocks += 1
    return blocks, target_width, target_height, target_aspect_ratio


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
# refactored to handle prior_aspect_ratio
def dynamic_preprocess_h2ovl(
    image: Image.Image,
    *,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> tuple[list[Image.Image], tuple[int, int]]:
    orig_width, orig_height = image.size

    # calculate the number of blocks without thumbnail
    (
        blocks,
        target_width,
        target_height,
        target_aspect_ratio,
    ) = calculate_h2ovl_targets(
        orig_width=orig_width,
        orig_height=orig_height,
        target_ratios=target_ratios,
        image_size=image_size,
        use_thumbnail=False,
    )

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images, target_aspect_ratio


def _preprocess_image(
    image: Image.Image,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
    prior_aspect_ratio: Optional[tuple[int, int]],
) -> tuple[torch.Tensor, tuple[int, int]]:
    target_ratios = get_h2ovl_target_ratios(
        min_num,
        max_num,
        prior_aspect_ratio=prior_aspect_ratio,
    )

    transform = build_transform(input_size=input_size)
    images, target_aspect_ratio = dynamic_preprocess_h2ovl(
        image,
        image_size=input_size,
        use_thumbnail=use_thumbnail,
        target_ratios=target_ratios,
    )

    pixel_values = torch.stack([transform(image) for image in images])
    return pixel_values, target_aspect_ratio


# refactored to use the _preprocess_image function
def image_to_pixel_values_h2ovl(
    image: Image.Image,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
    use_msac: bool,
) -> torch.Tensor:
    # when MSAC is turned on, we need to process the image twice
    if use_msac:
        # first pass
        pixel_values, target_aspect_ratio = _preprocess_image(
            image,
            input_size=input_size,
            min_num=min_num,
            max_num=max_num,
            use_thumbnail=True,
            prior_aspect_ratio=None,
        )
        # second pass
        pixel_values2, _ = _preprocess_image(
            image,
            input_size=input_size,
            min_num=min_num,
            max_num=max_num,
            use_thumbnail=True,
            prior_aspect_ratio=target_aspect_ratio,
        )
        # combine pixel values
        pixel_values = torch.cat(
            [pixel_values2[:-1], pixel_values[:-1], pixel_values2[-1:]], 0)

    else:
        pixel_values, _ = _preprocess_image(
            image,
            input_size=input_size,
            min_num=min_num,
            max_num=max_num,
            use_thumbnail=use_thumbnail,
            prior_aspect_ratio=None,
        )

    return pixel_values


class H2OVLProcessor(BaseInternVLProcessor):

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: AnyTokenizer,
        *,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        use_msac: Optional[bool] = None,
    ) -> None:
        super().__init__(
            config,
            tokenizer,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
        )

        if use_msac is None:
            use_msac = config.use_msac
        assert isinstance(use_msac, bool)

        self.use_msac = use_msac

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[IMG_CONTEXT]

    def get_image_repl_features(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> str:
        return IMG_CONTEXT * feature_size

    def get_image_repl_full(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> str:
        features = self.get_image_repl_features(feature_size, num_patches)
        return IMG_START + features + IMG_END

    def resolve_min_max_num(
        self,
        *,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        use_thumbnail: Optional[bool] = None,
        use_msac: Optional[bool] = None,
    ) -> tuple[int, int]:
        min_dynamic_patch = self.min_dynamic_patch
        max_dynamic_patch = (self.max_dynamic_patch if max_dynamic_patch
                             is None else max_dynamic_patch)
        dynamic_image_size = (self.dynamic_image_size if dynamic_image_size
                              is None else dynamic_image_size)
        use_thumbnail = (self.use_thumbnail
                         if use_thumbnail is None else use_thumbnail)
        use_msac = (self.use_msac if use_msac is None else use_msac)

        return resolve_h2ovl_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            use_msac=use_msac,
        )

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        target_ratios = self.resolve_target_ratios(
            use_thumbnail=False,  # Applied in calculate_targets
        )

        num_patches, _, _, _ = calculate_h2ovl_targets(
            orig_width=image_width,
            orig_height=image_height,
            image_size=self.image_size,
            target_ratios=target_ratios,
            use_thumbnail=self.use_thumbnail,
        )

        return num_patches * self.num_image_token

    def _images_to_pixel_values_lst(
        self,
        images: list[Image.Image],
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> list[torch.Tensor]:
        use_msac = self.use_msac if len(images) > 1 else False

        min_num, max_num = self.resolve_min_max_num(
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=False,  # Applied in image_to_pixel_values
            use_msac=use_msac,
        )

        return [
            image_to_pixel_values_h2ovl(
                image,
                input_size=self.image_size,
                min_num=min_num,
                max_num=max_num,
                use_thumbnail=self.use_thumbnail,
                use_msac=use_msac,
            ) for image in images
        ]


class H2OVLProcessingInfo(BaseInternVLProcessingInfo):

    def get_hf_processor(
        self,
        *,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> H2OVLProcessor:
        return H2OVLProcessor(
            self.get_hf_config(),
            self.get_tokenizer(),
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
        )


@MULTIMODAL_REGISTRY.register_processor(
    InternVLMultiModalProcessor,
    info=H2OVLProcessingInfo,
    dummy_inputs=InternVLDummyInputsBuilder)
class H2OVLChatModel(InternVLChatModel):

    def _init_vision_model(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        *,
        is_mono: bool,
        prefix: str,
    ):
        if not is_mono:
            vision_feature_layer = config.select_layer
            if vision_feature_layer < 0:
                num_hidden_layers = (config.vision_config.num_hidden_layers +
                                     vision_feature_layer + 1)
            else:
                num_hidden_layers = vision_feature_layer + 1

            return InternVisionModel(
                config.vision_config,
                quant_config=quant_config,
                num_hidden_layers_override=num_hidden_layers,
                prefix=prefix,
            )
        else:
            msg = "Monolith mode is not applicable to H2OVL"
            raise NotImplementedError(msg)
