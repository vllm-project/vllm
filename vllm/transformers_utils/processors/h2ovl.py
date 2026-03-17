# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from PIL import Image
from transformers import PretrainedConfig

from vllm.multimodal.processing import PromptUpdateDetails
from vllm.tokenizers import TokenizerLike

from .internvl import (
    IMG_CONTEXT,
    IMG_END,
    IMG_START,
    BaseInternVLProcessor,
    build_transform,
    find_closest_aspect_ratio,
    get_internvl_target_ratios,
)


def resolve_h2ovl_min_max_num(
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


def get_h2ovl_target_ratios(
    min_num: int,
    max_num: int,
    *,
    prior_aspect_ratio: tuple[int, int] | None,
) -> list[tuple[int, int]]:
    target_ratios = get_internvl_target_ratios(min_num, max_num)

    # if prior_aspect_ratio is provided, filter the target ratios
    if prior_aspect_ratio is not None:
        target_ratios = [
            ratio
            for ratio in target_ratios
            if prior_aspect_ratio[0] % ratio[0] != 0
            and prior_aspect_ratio[1] % ratio[1] != 0
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

    # add thumbnail image if num_blocks != 1
    if use_thumbnail and blocks != 1:
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
    prior_aspect_ratio: tuple[int, int] | None,
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
        pixel_values1, aspect_ratio1 = _preprocess_image(
            image,
            input_size=input_size,
            min_num=1,
            max_num=max_num,
            use_thumbnail=True,
            prior_aspect_ratio=None,
        )
        # second pass
        pixel_values2, _ = _preprocess_image(
            image,
            input_size=input_size,
            min_num=3,
            max_num=max_num,
            use_thumbnail=True,
            prior_aspect_ratio=aspect_ratio1,
        )
        # combine pixel values
        pixel_values = torch.cat(
            [pixel_values2[:-1], pixel_values1[:-1], pixel_values2[-1:]], 0
        )

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
        tokenizer: TokenizerLike,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
        use_msac: bool | None = None,
    ) -> None:
        super().__init__(
            config,
            tokenizer,
            min_dynamic_patch=min_dynamic_patch,
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

        return resolve_h2ovl_min_max_num(
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
        prior_aspect_ratio: tuple[int, int] | None = None,
        override_min_num: int | None = None,
    ) -> list[tuple[int, int]]:
        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
        )
        if override_min_num is not None:
            min_num = override_min_num

        return get_h2ovl_target_ratios(
            min_num,
            max_num,
            prior_aspect_ratio=prior_aspect_ratio,
        )

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        use_msac: bool | None = None,
    ) -> int:
        use_msac = self.use_msac if use_msac is None else use_msac

        use_thumbnail = self.use_thumbnail

        if use_msac:
            target_ratios_1 = self.resolve_target_ratios(
                use_thumbnail=False,  # Applied in calculate_targets
                override_min_num=1,
            )
            num_patches_1, _, _, aspect_ratio_1 = calculate_h2ovl_targets(
                orig_width=image_width,
                orig_height=image_height,
                image_size=self.image_size,
                target_ratios=target_ratios_1,
                use_thumbnail=True,
            )

            target_ratios_2 = self.resolve_target_ratios(
                use_thumbnail=False,  # Applied in calculate_targets
                prior_aspect_ratio=aspect_ratio_1,
                override_min_num=3,
            )
            num_patches_2, _, _, _ = calculate_h2ovl_targets(
                orig_width=image_width,
                orig_height=image_height,
                image_size=self.image_size,
                target_ratios=target_ratios_2,
                use_thumbnail=True,
            )

            num_patches = num_patches_1 + num_patches_2 - 1
        else:
            target_ratios = self.resolve_target_ratios(
                use_thumbnail=False,  # Applied in calculate_targets
            )
            num_patches, _, _, _ = calculate_h2ovl_targets(
                orig_width=image_width,
                orig_height=image_height,
                image_size=self.image_size,
                target_ratios=target_ratios,
                use_thumbnail=use_thumbnail,
            )

        return num_patches * self.num_image_token

    def _images_to_pixel_values_lst(
        self,
        images: list[Image.Image],
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> list[torch.Tensor]:
        use_msac = self.use_msac if len(images) == 1 else False

        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=False,  # Applied in image_to_pixel_values
        )

        return [
            image_to_pixel_values_h2ovl(
                image,
                input_size=self.image_size,
                min_num=min_num,
                max_num=max_num,
                use_thumbnail=self.use_thumbnail,
                use_msac=use_msac,
            )
            for image in images
        ]
