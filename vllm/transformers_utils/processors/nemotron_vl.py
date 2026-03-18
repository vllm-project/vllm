# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torchvision.transforms as T
from PIL import Image

from vllm.multimodal.image import convert_image_mode
from vllm.tokenizers.hf import HfTokenizer

from .internvl import InternVLImageProcessor, InternVLProcessor

# Configure PIL to handle large images without warnings
# This prevents DecompressionBombWarning for legitimate large images
Image.MAX_IMAGE_PIXELS = None  # Disable the limit entirely
# Alternative: Set a specific higher limit
# Image.MAX_IMAGE_PIXELS = 300000000  # ~300M pixels


def build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: convert_image_mode(img, "RGB")),
            T.Resize(
                (input_size, input_size), interpolation=T.InterpolationMode.BICUBIC
            ),
            T.ToTensor(),
        ]
    )


# adapted from https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1
def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    *,
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_factor = float("-inf")
    best_ratio = (1, 1)
    area = width * height

    for rw, rh in target_ratios:
        target_aspect_ratio = rw / rh
        size_factor = min((rw * rh * image_size * image_size) / area, 0.6)
        ratio_closeness = min(
            target_aspect_ratio / aspect_ratio, aspect_ratio / target_aspect_ratio
        )
        factor = size_factor * ratio_closeness

        if factor > best_factor:
            best_factor = factor
            best_ratio = (rw, rh)

    return best_ratio


def calculate_nemotron_vl_targets(
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


def dynamic_preprocess_nemotron_vl(
    image: Image.Image,
    *,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> list[Image.Image]:
    orig_width, orig_height = image.size

    # calculate the number of blocks without thumbnail
    blocks, target_width, target_height = calculate_nemotron_vl_targets(
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


def get_nemotron_vl_target_ratios(
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


def image_to_pixel_values_nemotron_vl(
    image: Image.Image,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
    transform: T.Compose | None = None,
) -> torch.Tensor:
    target_ratios = get_nemotron_vl_target_ratios(min_num, max_num)

    if transform is None:
        transform = build_transform(input_size=input_size)

    images = dynamic_preprocess_nemotron_vl(
        image,
        target_ratios=target_ratios,
        image_size=input_size,
        use_thumbnail=use_thumbnail,
    )

    pixel_values = torch.stack([transform(image) for image in images])
    return pixel_values


class LlamaNemotronNanoVLImageProcessor(InternVLImageProcessor):
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
            image_to_pixel_values_nemotron_vl(
                image,
                input_size=self.image_size,
                min_num=min_num,
                max_num=max_num,
                use_thumbnail=self.use_thumbnail,
                transform=build_transform(self.image_size),
            )
            for image in images
        ]


class LlamaNemotronNanoVLProcessor(InternVLProcessor):
    """
    This model doesn't define its own HF processor,
    so we implement our own one here.

    The image processor is given by:
    https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1/blob/main/image_processing.py
    """

    def __init__(
        self,
        image_processor: LlamaNemotronNanoVLImageProcessor,
        tokenizer: HfTokenizer,
        *,
        image_seq_length: int,
        image_token: str = "<image>",
        start_image_token: str = "<img>",
        end_image_token: str = "</img>",
    ) -> None:
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            image_seq_length=image_seq_length,
            image_token=image_token,
            start_image_token=start_image_token,
            end_image_token=end_image_token,
        )

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        image_processor = self.image_processor
        target_ratios = self.resolve_target_ratios(
            use_thumbnail=False,  # Applied in calculate_targets
        )

        num_patches, _, _ = calculate_nemotron_vl_targets(
            orig_width=image_width,
            orig_height=image_height,
            image_size=image_processor.image_size,
            target_ratios=target_ratios,
            use_thumbnail=image_processor.use_thumbnail,
        )

        return num_patches * self.image_seq_length


# SigLIP normalization constants
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


def build_siglip_transform(input_size: int):
    """Build transform for SigLIP vision encoder with normalization.

    Extends the base transform from nemotron_vl with SigLIP-specific normalization.
    """
    return T.Compose(
        [
            build_transform(input_size=input_size),
            T.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
        ]
    )


class LlamaNemotronVLEmbedImageProcessor(InternVLImageProcessor):
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
            image_to_pixel_values_nemotron_vl(
                image,
                input_size=self.image_size,
                min_num=min_num,
                max_num=max_num,
                use_thumbnail=self.use_thumbnail,
                transform=build_siglip_transform(self.image_size),
            )
            for image in images
        ]


class LlamaNemotronVLEmbedProcessor(InternVLProcessor):
    """
    Processor for LlamaNemotronVL embedding model.

    Inherits from NemotronVLProcessor and specializes it for embedding tasks:
    - Uses SigLIP transform with normalization instead of base transform
    - Uses different image context token (<IMG_CONTEXT> vs <image>)
    """

    def __init__(
        self,
        image_processor: LlamaNemotronVLEmbedImageProcessor,
        tokenizer: HfTokenizer,
        *,
        image_seq_length: int,
        image_token: str = "<IMG_CONTEXT>",
        start_image_token: str = "<img>",
        end_image_token: str = "</img>",
    ) -> None:
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            image_seq_length=image_seq_length,
            image_token=image_token,
            start_image_token=start_image_token,
            end_image_token=end_image_token,
        )

        self.image_processor: LlamaNemotronVLEmbedImageProcessor

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        image_processor = self.image_processor
        target_ratios = self.resolve_target_ratios(
            use_thumbnail=False,  # Applied in calculate_targets
        )

        num_patches, _, _ = calculate_nemotron_vl_targets(
            orig_width=image_width,
            orig_height=image_height,
            image_size=image_processor.image_size,
            target_ratios=target_ratios,
            use_thumbnail=image_processor.use_thumbnail,
        )

        return num_patches * self.image_seq_length
