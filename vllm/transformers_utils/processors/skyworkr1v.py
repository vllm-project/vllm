# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://huggingface.co/Skywork/Skywork-R1V-38B/blob/main/modeling_skywork_chat.py
# --------------------------------------------------------
# SkyworkR1V
# Copyright (c) 2025 Skywork
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import BatchFeature, PretrainedConfig, TensorType

from vllm.multimodal.image import convert_image_mode
from vllm.multimodal.processing import PromptUpdateDetails
from vllm.tokenizers import TokenizerLike

IMG_START = "<img>"
IMG_END = "</img>"
IMG_CONTEXT = "<IMG_CONTEXT>"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# adapted from https://huggingface.co/Skywork/Skywork-R1V-38B/
def build_transform(input_size: int):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose(
        [
            T.Lambda(lambda img: convert_image_mode(img, "RGB")),
            T.Resize(
                (input_size, input_size), interpolation=T.InterpolationMode.BICUBIC
            ),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )


# adapted from https://huggingface.co/Skywork/Skywork-R1V-38B/
def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    *,
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def resolve_skyworkr1v_min_max_num(
    *,
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: bool,
    use_thumbnail: bool,
) -> tuple[int, int]:
    min_dynamic_patch = min_dynamic_patch if dynamic_image_size else 1
    max_dynamic_patch = max_dynamic_patch if dynamic_image_size else 1

    if use_thumbnail and max_dynamic_patch != 1:
        max_dynamic_patch += 1

    return min_dynamic_patch, max_dynamic_patch


def get_skyworkr1v_target_ratios(
    min_num: int,
    max_num: int,
) -> list[tuple[int, int]]:
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    return sorted(target_ratios, key=lambda x: x[0] * x[1])


def calculate_skyworkr1v_targets(
    *,
    orig_width: int,
    orig_height: int,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> tuple[int, int, int]:
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

    # add thumbnail image if num_blocks != 1
    if use_thumbnail and blocks != 1:
        blocks += 1

    return blocks, target_width, target_height


def dynamic_preprocess_skyworkr1v(
    image: Image.Image,
    *,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> list[Image.Image]:
    orig_width, orig_height = image.size

    # calculate the number of blocks without thumbnail
    blocks, target_width, target_height = calculate_skyworkr1v_targets(
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

    return processed_images


# adapted from https://huggingface.co/Skywork/Skywork-R1V-38B
def image_to_pixel_values_skyworkr1v(
    image: Image.Image,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor:
    target_ratios = get_skyworkr1v_target_ratios(min_num, max_num)

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess_skyworkr1v(
        image,
        target_ratios=target_ratios,
        image_size=input_size,
        use_thumbnail=use_thumbnail,
    )

    pixel_values = torch.stack([transform(image) for image in images])
    return pixel_values


class SkyworkR1VProcessor:
    """
    This model doesn't define its own HF processor,
    so we implement our own one here.

    The code to insert image tokens is based on:
    https://huggingface.co/Skywork/Skywork-R1V-38B/blob/main/modeling_skywork_chat.py#L252
    """

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: TokenizerLike,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        image_size: int = config.vision_config.image_size
        patch_size: int = config.vision_config.patch_size

        if min_dynamic_patch is None:
            min_dynamic_patch = config.min_dynamic_patch
        assert isinstance(min_dynamic_patch, int)

        if max_dynamic_patch is None:
            max_dynamic_patch = config.max_dynamic_patch
        assert isinstance(max_dynamic_patch, int)

        if dynamic_image_size is None:
            dynamic_image_size = config.dynamic_image_size
        assert isinstance(dynamic_image_size, bool)

        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.image_size = image_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail: bool = config.use_thumbnail

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[IMG_CONTEXT]

    def get_image_repl(
        self,
        feature_size: int,
        num_patches: int | None,
    ) -> PromptUpdateDetails[str]:
        repl_features = IMG_CONTEXT * feature_size
        repl_full = IMG_START + repl_features + IMG_END

        return PromptUpdateDetails.select_text(repl_full, IMG_CONTEXT)

    def resolve_min_max_num(
        self,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
        use_thumbnail: bool | None = None,
    ) -> tuple[int, int]:
        min_dynamic_patch = (
            self.min_dynamic_patch if min_dynamic_patch is None else min_dynamic_patch
        )
        max_dynamic_patch = (
            self.max_dynamic_patch if max_dynamic_patch is None else max_dynamic_patch
        )
        dynamic_image_size = (
            self.dynamic_image_size
            if dynamic_image_size is None
            else dynamic_image_size
        )
        use_thumbnail = self.use_thumbnail if use_thumbnail is None else use_thumbnail

        return resolve_skyworkr1v_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
        )

    def resolve_target_ratios(
        self,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
        use_thumbnail: bool | None = None,
    ) -> list[tuple[int, int]]:
        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
        )

        return get_skyworkr1v_target_ratios(min_num, max_num)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        target_ratios = self.resolve_target_ratios(
            use_thumbnail=False,  # Applied in calculate_targets
        )

        num_patches, _, _ = calculate_skyworkr1v_targets(
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
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> list[torch.Tensor]:
        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=False,  # Applied in image_to_pixel_values
        )

        return [
            image_to_pixel_values_skyworkr1v(
                image,
                input_size=self.image_size,
                min_num=min_num,
                max_num=max_num,
                use_thumbnail=self.use_thumbnail,
            )
            for image in images
        ]

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | None = None,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
        return_tensors: str | TensorType | None = None,
    ) -> BatchFeature:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        if len(images) == 0:
            image_inputs = {}
        else:
            pixel_values_lst = self._images_to_pixel_values_lst(
                images,
                min_dynamic_patch=min_dynamic_patch,
                max_dynamic_patch=max_dynamic_patch,
                dynamic_image_size=dynamic_image_size,
            )
            image_inputs = {
                "pixel_values_flat": torch.cat(pixel_values_lst),
                "image_num_patches": torch.tensor(
                    [len(item) for item in pixel_values_lst]
                ),
            }

            for pixel_values in pixel_values_lst:
                num_patches = pixel_values.shape[0]
                feature_size = num_patches * self.num_image_token

                image_repl = self.get_image_repl(feature_size, num_patches)

                text = [t.replace("<image>", image_repl.full, 1) for t in text]

        text_inputs = self.tokenizer(text)

        combined_outputs = {**text_inputs, **image_inputs}

        return BatchFeature(combined_outputs, tensor_type=return_tensors)
