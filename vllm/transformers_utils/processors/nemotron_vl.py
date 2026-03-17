# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import BaseImageProcessorFast, BatchFeature, TensorType
from transformers.processing_utils import ProcessorMixin

from vllm.multimodal.image import convert_image_mode
from vllm.multimodal.processing import PromptUpdateDetails
from vllm.tokenizers import TokenizerLike

from .internvl import (
    InternVLImageProcessor,
    InternVLProcessor,
    InternVLProcessorLike,
    get_internvl_target_ratios,
    resolve_internvl_min_max_num,
)

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


class LlamaNemotronNanoVLProcessor(InternVLProcessorLike, ProcessorMixin):
    """
    This model doesn't define its own HF processor,
    so we implement our own one here.

    The image processor is given by:
    https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1/blob/main/image_processing.py
    """

    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor: BaseImageProcessorFast,
        tokenizer: TokenizerLike,
        *,
        image_seq_length: int,
        image_token: str = "<image>",
        start_image_token: str = "<img>",
        end_image_token: str = "</img>",
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        self.image_seq_length = image_seq_length
        self.image_token = image_token
        self.start_image_token = start_image_token
        self.end_image_token = end_image_token

        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        self.start_image_token_id = tokenizer.convert_tokens_to_ids(start_image_token)
        self.end_image_token_id = tokenizer.convert_tokens_to_ids(end_image_token)

    def resolve_target_ratios(
        self,
        *,
        max_num_tiles: int | None = None,
        use_thumbnail: bool | None = None,
    ) -> list[tuple[int, int]]:
        image_processor = self.image_processor
        if max_num_tiles is None:
            max_num_tiles = image_processor.max_num_tiles
        if use_thumbnail is None:
            use_thumbnail = image_processor.use_thumbnail

        min_num, max_num = resolve_internvl_min_max_num(
            min_dynamic_patch=1,
            max_dynamic_patch=max_num_tiles,
            dynamic_image_size=True,
            use_thumbnail=use_thumbnail,
        )

        return get_internvl_target_ratios(min_num, max_num)

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

    def get_image_repl(
        self,
        num_patches: int | None,
        num_features: int | None = None,
    ) -> PromptUpdateDetails[str]:
        if num_patches is None:
            assert num_features is not None
        else:
            num_features = num_patches * self.image_seq_length

        context_token = self.image_token
        repl_features = context_token * num_features
        repl_full = self.start_image_token + repl_features + self.end_image_token

        return PromptUpdateDetails.select_text(repl_full, context_token)

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | None = None,
        *,
        max_num_tiles: int | None = None,
        use_thumbnail: bool | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        if images is not None:
            image_inputs = self.image_processor(
                images=images,
                max_num_tiles=max_num_tiles,
                use_thumbnail=use_thumbnail,
                return_tensors=return_tensors,
            )
            image_inputs["pixel_values_flat"] = image_inputs.pop("pixel_values")
            image_inputs["image_num_patches"] = image_inputs.pop("num_patches")
            image_num_patches = image_inputs["image_num_patches"]
        else:
            image_inputs = {}
            image_num_patches = []

        if text is not None:
            if not isinstance(text, list):
                text = [text]

            if image_inputs:
                image_token = self.image_token
                image_index = 0
                processed_text = list[str]()
                replace_strings = list[str]()

                for prompt in text:
                    new_prompt = prompt

                    while image_token in new_prompt:
                        new_prompt = new_prompt.replace(image_token, "<placeholder>", 1)
                        image_repl = self.get_image_repl(image_num_patches[image_index])
                        replace_strings.append(image_repl.full)
                        image_index += 1

                    while "<placeholder>" in new_prompt:
                        replace_str = replace_strings.pop(0)
                        new_prompt = new_prompt.replace("<placeholder>", replace_str, 1)

                    processed_text.append(new_prompt)

                text = processed_text

            text_inputs = self.tokenizer(text, return_tensors=return_tensors)
        else:
            text_inputs = {}

        combined_outputs = {**text_inputs, **image_inputs}

        return BatchFeature(combined_outputs, tensor_type=return_tensors)


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
        tokenizer: TokenizerLike,
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
