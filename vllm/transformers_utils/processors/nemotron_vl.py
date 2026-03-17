# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import PretrainedConfig
from transformers.image_processing_utils_fast import BaseImageProcessorFast

from vllm.multimodal.image import convert_image_mode
from vllm.multimodal.processing import PromptUpdateDetails
from vllm.tokenizers import TokenizerLike

from .internvl import InternVLProcessor

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


class NemotronVLProcessor(InternVLProcessor):
    IMG_START = "<img>"
    IMG_END = "</img>"
    IMG_CONTEXT = "<image>"

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: TokenizerLike,
        image_processor: BaseImageProcessorFast,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> None:
        ABC.__init__(self)
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        image_size: int = config.force_image_size
        patch_size: int = config.patch_size

        if min_dynamic_patch is None:
            min_dynamic_patch = 1
        assert isinstance(min_dynamic_patch, int)

        if max_dynamic_patch is None:
            max_dynamic_patch = self.image_processor.max_num_tiles
        assert isinstance(max_dynamic_patch, int)

        if dynamic_image_size is None:
            dynamic_image_size = True
        assert isinstance(dynamic_image_size, bool)

        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.image_size = image_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.dynamic_image_size = dynamic_image_size

        if image_processor is not None:
            self.use_thumbnail = image_processor.use_thumbnail
        else:
            self.use_thumbnail = getattr(config, "use_thumbnail", True)

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[self.IMG_CONTEXT]

    def _get_transform(self) -> T.Compose:
        return build_transform(input_size=self.image_size)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        target_ratios = self.resolve_target_ratios(
            use_thumbnail=False,  # Applied in calculate_targets
        )

        num_patches, _, _ = calculate_nemotron_vl_targets(
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
            image_to_pixel_values_nemotron_vl(
                image,
                input_size=self.image_size,
                min_num=min_num,
                max_num=max_num,
                use_thumbnail=self.use_thumbnail,
                transform=self._get_transform(),
            )
            for image in images
        ]

    def _replace_image_tokens(
        self,
        text: list[str],
        pixel_values_lst: list[torch.Tensor],
    ) -> list[str]:
        """Replace <image> placeholders with image tokens."""
        for pixel_values in pixel_values_lst:
            num_patches = pixel_values.shape[0]
            feature_size = num_patches * self.num_image_token
            image_repl = self.get_image_repl(feature_size, num_patches)
            # Use temporary placeholder to avoid replacing tokens we just inserted
            NVL_IMAGE_CONTEXT = image_repl.full.replace("<image>", "<NVL_IMG_CONTEXT>")
            text = [t.replace("<image>", NVL_IMAGE_CONTEXT, 1) for t in text]
        return [t.replace("<NVL_IMG_CONTEXT>", self.IMG_CONTEXT) for t in text]

    def _preprocess_image(
        self,
        text: list[str],
        images: list[Image.Image],
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> tuple[list[str], dict[str, torch.Tensor]]:
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

            text = self._replace_image_tokens(text, pixel_values_lst)
        return text, image_inputs

    def get_image_repl(
        self,
        feature_size: int,
        num_patches: int | None,
    ) -> PromptUpdateDetails[str]:
        repl_features = self.IMG_CONTEXT * feature_size
        repl_full = self.IMG_START + repl_features + self.IMG_END

        return PromptUpdateDetails.select_text(repl_full, self.IMG_CONTEXT)


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


class LlamaNemotronVLEmbedProcessor(NemotronVLProcessor):
    """
    Processor for LlamaNemotronVL embedding model.

    Inherits from NemotronVLProcessor and specializes it for embedding tasks:
    - Uses SigLIP transform with normalization instead of base transform
    - Uses different image context token (<IMG_CONTEXT> vs <image>)
    """

    IMG_CONTEXT = "<IMG_CONTEXT>"

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: TokenizerLike,
        processor_config: dict,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> None:
        if min_dynamic_patch is None:
            min_dynamic_patch = processor_config.get(
                "min_input_tiles",
                getattr(config, "min_dynamic_patch", 1),
            )
        if max_dynamic_patch is None:
            max_dynamic_patch = processor_config.get(
                "max_input_tiles",
                getattr(config, "max_dynamic_patch", 1),
            )
        if dynamic_image_size is None:
            dynamic_image_size = processor_config.get(
                "dynamic_image_size",
                getattr(config, "dynamic_image_size", True),
            )
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            image_processor=None,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
        )

    def _get_transform(self) -> T.Compose:
        """Override to add SigLIP normalization."""
        return build_siglip_transform(input_size=self.image_size)

    def _replace_image_tokens(
        self,
        text: list[str],
        pixel_values_lst: list[torch.Tensor],
    ) -> list[str]:
        """Override with simpler token replacement for embedding model.

        No temporary placeholder needed because IMG_CONTEXT is <IMG_CONTEXT>,
        not <image>, so there's no collision risk.
        """
        for pixel_values in pixel_values_lst:
            num_patches = pixel_values.shape[0]
            feature_size = num_patches * self.num_image_token
            image_repl = self.get_image_repl(feature_size, num_patches)
            text = [t.replace("<image>", image_repl.full, 1) for t in text]
        return text
