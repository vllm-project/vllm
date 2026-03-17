# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from typing import Protocol

import numpy.typing as npt
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import BatchFeature, TensorType
from transformers.processing_utils import ProcessorMixin

from vllm.multimodal.image import convert_image_mode
from vllm.multimodal.processing import PromptUpdateDetails
from vllm.tokenizers import TokenizerLike

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
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


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
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


def resolve_internvl_min_max_num(
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


def get_internvl_target_ratios(
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


def calculate_internvl_targets(
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


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
def dynamic_preprocess_internvl(
    image: Image.Image,
    *,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> list[Image.Image]:
    orig_width, orig_height = image.size

    # calculate the number of blocks without thumbnail
    blocks, target_width, target_height = calculate_internvl_targets(
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


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
def image_to_pixel_values_internvl(
    image: Image.Image,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor:
    target_ratios = get_internvl_target_ratios(min_num, max_num)

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess_internvl(
        image,
        target_ratios=target_ratios,
        image_size=input_size,
        use_thumbnail=use_thumbnail,
    )

    pixel_values = torch.stack([transform(image) for image in images])
    return pixel_values


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
def video_to_pixel_values_internvl(
    video: npt.NDArray,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor:
    target_ratios = get_internvl_target_ratios(min_num, max_num)

    transform = build_transform(input_size=input_size)
    frames_list = list[Image.Image]()
    for frame in video:
        pil_frame = dynamic_preprocess_internvl(
            Image.fromarray(frame, mode="RGB"),
            target_ratios=target_ratios,
            image_size=input_size,
            use_thumbnail=use_thumbnail,
        )
        assert len(pil_frame) == 1
        frames_list.extend(pil_frame)

    pixel_values = torch.stack([transform(image) for image in frames_list])
    return pixel_values


class InternVLImageProcessor:
    def __init__(
        self,
        image_size: int,
        min_dynamic_patch: int,
        max_dynamic_patch: int,
        dynamic_image_size: bool,
        use_thumbnail: bool,
    ) -> None:
        self.image_size = image_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail

    def resolve_min_max_num(
        self,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
        use_thumbnail: bool | None = None,
    ) -> tuple[int, int]:
        if min_dynamic_patch is None:
            min_dynamic_patch = self.min_dynamic_patch
        if max_dynamic_patch is None:
            max_dynamic_patch = self.max_dynamic_patch
        if dynamic_image_size is None:
            dynamic_image_size = self.dynamic_image_size
        if use_thumbnail is None:
            use_thumbnail = self.use_thumbnail

        return resolve_internvl_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
        )

    def _images_to_pixel_values_lst(
        self,
        images: list[Image.Image],
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> list[torch.Tensor]:
        if min_dynamic_patch is None:
            min_dynamic_patch = self.min_dynamic_patch
        if max_dynamic_patch is None:
            max_dynamic_patch = self.max_dynamic_patch
        if dynamic_image_size is None:
            dynamic_image_size = self.dynamic_image_size

        min_num, max_num = resolve_internvl_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=False,  # Applied in image_to_pixel_values
        )

        return [
            image_to_pixel_values_internvl(
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
        images: Image.Image | list[Image.Image],
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        images_lst = [images] if not isinstance(images, list) else images

        pixel_values_lst = self._images_to_pixel_values_lst(
            images_lst,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
        )

        image_inputs = {
            "pixel_values_flat": torch.cat(pixel_values_lst),
            "image_num_patches": torch.tensor([len(item) for item in pixel_values_lst]),
        }
        return BatchFeature(image_inputs, tensor_type=return_tensors)


class InternVLVideoProcessor:
    def __init__(
        self,
        image_size: int,
    ) -> None:
        self.image_size = image_size

    def _videos_to_pixel_values_lst(
        self,
        videos: list[npt.NDArray],
    ) -> list[torch.Tensor]:
        return [
            video_to_pixel_values_internvl(
                video,
                input_size=self.image_size,
                min_num=1,
                max_num=1,
                use_thumbnail=False,
            )
            for video in videos
        ]

    def __call__(
        self,
        videos: npt.NDArray | list[npt.NDArray],
        *,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        videos_lst = [videos] if not isinstance(videos, list) else videos

        pixel_values_lst = self._videos_to_pixel_values_lst(videos_lst)

        image_inputs = {
            "pixel_values_flat_video": torch.cat(pixel_values_lst),
            "video_num_patches": torch.tensor([len(item) for item in pixel_values_lst]),
        }
        return BatchFeature(image_inputs, tensor_type=return_tensors)


class InternVLProcessorLike(Protocol):
    image_seq_length: int
    image_token: str
    image_token_id: int
    start_image_token: str
    start_image_token_id: int
    end_image_token: str
    end_image_token_id: int

    def resolve_target_ratios(self) -> list[tuple[int, int]]: ...

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int: ...

    def get_image_repl(
        self,
        num_patches: int | None,
        num_features: int | None = None,
    ) -> PromptUpdateDetails[str]: ...


class InternVLProcessor(InternVLProcessorLike, ProcessorMixin):
    """
    This model doesn't define its own HF processor,
    so we implement our own one here.

    The code to insert image tokens is based on:
    https://huggingface.co/OpenGVLab/InternVL2-1B/blob/main/modeling_internvl_chat.py#L252

    Code for video processing is adapted from video example:
    https://huggingface.co/OpenGVLab/InternVL3-1B#inference-with-transformers
    """

    attributes = ["image_processor", "tokenizer", "video_processor"]

    def __init__(
        self,
        image_processor: InternVLImageProcessor,
        tokenizer: TokenizerLike,
        video_processor: InternVLVideoProcessor | None = None,
        *,
        image_seq_length: int,
        image_token: str = "<IMG_CONTEXT>",
        start_image_token: str = "<img>",
        end_image_token: str = "</img>",
        video_token: str | None = None,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.video_processor = video_processor

        self.image_seq_length = image_seq_length
        self.image_token = image_token
        self.start_image_token = start_image_token
        self.end_image_token = end_image_token
        self.video_token = video_token

        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        self.start_image_token_id = tokenizer.convert_tokens_to_ids(start_image_token)
        self.end_image_token_id = tokenizer.convert_tokens_to_ids(end_image_token)
        self.video_token_id = (
            None
            if video_token is None
            else tokenizer.convert_tokens_to_ids(video_token)
        )

    @property
    def supports_video(self) -> bool:
        return self.video_token_id is not None

    def resolve_target_ratios(
        self,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
        use_thumbnail: bool | None = None,
    ) -> list[tuple[int, int]]:
        min_num, max_num = self.image_processor.resolve_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
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

        num_patches, _, _ = calculate_internvl_targets(
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

    def get_video_repl(self, num_patches: int) -> PromptUpdateDetails[str]:
        assert self.video_token is not None and self.video_processor is not None

        context_token = self.video_token
        repl_features = context_token * self.image_seq_length
        repl_features_with_sep = (
            self.start_image_token + repl_features + self.end_image_token
        )
        # num_patches is equal to num_frames
        repl_full = "".join(
            [f"Frame{i + 1}: {repl_features_with_sep}" for i in range(num_patches)]
        )

        return PromptUpdateDetails.select_text(repl_full, context_token)

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | None = None,
        videos: npt.NDArray | list[npt.NDArray] | None = None,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        if images is not None:
            image_inputs = self.image_processor(
                images=images,
                min_dynamic_patch=min_dynamic_patch,
                max_dynamic_patch=max_dynamic_patch,
                dynamic_image_size=dynamic_image_size,
                return_tensors=return_tensors,
            )
            image_num_patches = image_inputs["image_num_patches"]
        else:
            image_inputs = {}
            image_num_patches = []

        if videos is not None:
            if self.video_processor is None:
                raise ValueError("This model does not support video inputs")

            video_inputs = self.video_processor(
                videos=videos,
                return_tensors=return_tensors,
            )
            video_num_patches = video_inputs["video_num_patches"]
        else:
            video_inputs = {}
            video_num_patches = []

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

            if video_inputs:
                video_token = self.video_token
                video_index = 0
                processed_text = list[str]()
                replace_strings = list[str]()

                assert video_token is not None

                for prompt in text:
                    new_prompt = prompt

                    while video_token in new_prompt:
                        new_prompt = new_prompt.replace(video_token, "<placeholder>", 1)
                        video_repl = self.get_video_repl(video_num_patches[video_index])
                        replace_strings.append(video_repl.full)
                        video_index += 1

                    while "<placeholder>" in new_prompt:
                        replace_str = replace_strings.pop(0)
                        new_prompt = new_prompt.replace("<placeholder>", replace_str, 1)

                    processed_text.append(new_prompt)

                text = processed_text

            text_inputs = self.tokenizer(processed_text, return_tensors=return_tensors)
        else:
            text_inputs = {}

        combined_outputs = {**text_inputs, **image_inputs, **video_inputs}

        return BatchFeature(combined_outputs, tensor_type=return_tensors)
