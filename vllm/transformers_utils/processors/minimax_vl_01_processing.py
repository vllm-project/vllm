# SPDX-License-Identifier: Apache-2.0
"""
Processor class for MiniMaxVL01.
"""
import os
from typing import List, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, get_image_size, to_numpy_array
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging

logger = logging.get_logger(__name__)

LEGACY_PROCESSING = int(os.getenv('LEGACY_PROCESSING', 1))


class MiniMaxVL01ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {},
    }


def get_hw_multiple_of(image_size, multiple, max_size=None):
    w, h = image_size
    new_w = w if w % multiple == 0 else w + (multiple - w % multiple)
    new_h = h if h % multiple == 0 else h + (multiple - h % multiple)
    if max_size is not None:
        assert isinstance(max_size, (list, tuple)) and len(max_size) == 2
        max_w, max_h = max_size
        assert max_w % multiple == 0 and max_h % multiple == 0
        if new_w > max_w or new_h > max_h:
            # ratio = min(max_w / new_w, max_h / new_h)
            # new_w = int(new_w * ratio)
            # new_h = int(new_h * ratio)
            new_w = min((new_w * max_w) // new_w, (new_w * max_h) // new_h)
            new_h = min((new_h * max_w) // new_w, (new_h * max_h) // new_h)

            new_w = new_w if new_w % multiple == 0 else new_w + (
                multiple - new_w % multiple)
            new_h = new_h if new_h % multiple == 0 else new_h + (
                multiple - new_h % multiple)
        assert new_w % multiple == 0 and new_h % multiple == 0
        assert new_w <= max_w and new_h <= max_h
    return new_w, new_h


def split_special_tokens(text, special_tokens):
    # 使用正则表达式匹配所有特殊标记及其前后内容
    import re
    pattern = '|'.join(map(re.escape, special_tokens))
    parts = re.split(f'({pattern})', text)

    # 过滤掉空字符串
    return [p for p in parts if p]


def select_best_resolution(original_size, possible_resolutions):
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height,
                                   original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
                effective_resolution == max_effective_resolution
                and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def get_w_h_num(resolution, best_resolution):
    original_width, original_height = resolution
    current_width, current_height = best_resolution

    current_height = int(current_height)
    current_width = int(current_width)
    original_height = int(original_height)
    original_width = int(original_width)

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        new_height = int(original_height * current_width) // original_width
        padding = (current_height - new_height) // 2
        w_num = current_width
        h_num = current_height - 2 * padding
    else:
        new_width = int(original_width * current_height) // original_height

        padding = (current_width - new_width) // 2
        w_num = current_width - 2 * padding
        h_num = current_height

    return (w_num, h_num)


def get_num_token(img_h, img_w, grid_pinpoints, patch_size):
    best_resolution = select_best_resolution((img_w, img_h), grid_pinpoints)
    resized_w, resized_h = best_resolution
    w_num, h_num = get_w_h_num(
        (img_w, img_h), (resized_w // patch_size, resized_h // patch_size))
    total_token = int((w_num + 1) * h_num) + (336 // patch_size)**2
    return total_token


class MiniMaxVL01Processor(ProcessorMixin):

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template", "patch_size", "vision_feature_select_strategy",
        "image_token"
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<image>",
        **kwargs,
    ):
        self.patch_size = patch_size
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = image_token
        super().__init__(image_processor,
                         tokenizer,
                         chat_template=chat_template)
        self.patch_size = image_processor.patch_size
        self.grid_pinpoints = image_processor.image_grid_pinpoints
        self.max_size = image_processor.size
        self.process_image_mode = image_processor.process_image_mode

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput],
                    List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError(
                "You have to specify at least one of `images` or `text`.")

        output_kwargs = self._merge_kwargs(
            MiniMaxVL01ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(
                images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide " +
                             "a string, or a list of strings")

        prompt_strings = text
        if image_inputs.get("pixel_values") is not None:
            if self.process_image_mode == 'anyres':
                if LEGACY_PROCESSING:
                    pixel_values = image_inputs["pixel_values"]
                    image_sizes = image_inputs["image_sizes"]

                    if isinstance(pixel_values, list) and isinstance(
                            image_sizes,
                            list) and len(pixel_values) != len(image_sizes):
                        if len(pixel_values) > len(image_sizes):
                            image_sizes = [image_sizes[0]] * len(pixel_values)
                        else:
                            pixel_values = pixel_values[:len(image_sizes)]

                    all_image_tokens = []
                    for pixel_value, image_size in zip(pixel_values,
                                                       image_sizes):
                        height, width = image_size
                        num_image_tokens = get_num_token(
                            height, width, self.grid_pinpoints,
                            self.patch_size)
                        all_image_tokens.append(num_image_tokens)

                    prompt_strings = []
                    image_index = 0
                    for sample in text:
                        split_text = split_special_tokens(
                            sample, [self.image_token])
                        final_text = ''
                        for i, _sample in enumerate(split_text):
                            if _sample == self.image_token:
                                if image_index < len(all_image_tokens):
                                    final_text += _sample * all_image_tokens[
                                        image_index]
                                    image_index += 1
                                else:
                                    # 如果图像索引超出范围，保持原始token
                                    final_text += _sample
                            else:
                                final_text += _sample
                        prompt_strings.append(final_text)
            elif self.process_image_mode == 'resize':
                pixel_values = image_inputs["pixel_values"]

                all_image_tokens = []
                for pixel_value in pixel_values:
                    height, width = get_image_size(to_numpy_array(pixel_value))
                    all_image_tokens.append(
                        int(height * width / self.patch_size**2))

                prompt_strings = []
                image_index = 0
                for sample in text:
                    split_text = split_special_tokens(sample,
                                                      [self.image_token])
                    final_text = ''
                    for i, _sample in enumerate(split_text):
                        if _sample == self.image_token:
                            final_text += _sample * all_image_tokens[
                                image_index]
                            image_index += 1
                        else:
                            final_text += _sample
                    prompt_strings.append(final_text)
            else:

                if self.patch_size is not None:
                    pixel_values = image_inputs["pixel_values"]
                    all_image_tokens = []
                    for pixel_value in pixel_values:
                        height, width = get_image_size(
                            to_numpy_array(pixel_value))
                        new_width, new_height = get_hw_multiple_of(
                            (width, height), self.patch_size, self.max_size)
                        num_image_tokens = (new_height // self.patch_size) * (
                            new_width // self.patch_size)  # + 1
                        # if self.vision_feature_select_strategy == "default":
                        #     num_image_tokens -= 1
                        all_image_tokens.append(num_image_tokens)

                    prompt_strings = []
                    image_index = 0
                    for sample in text:
                        split_text = split_special_tokens(
                            sample, [self.image_token])
                        final_text = ''
                        for i, _sample in enumerate(split_text):
                            if _sample == self.image_token:
                                final_text += _sample * all_image_tokens[
                                    image_index]
                                image_index += 1
                            else:
                                final_text += _sample
                        prompt_strings.append(final_text)
                else:

                    raise ValueError(
                        "You need to provide `patch_size` " +
                        "and `vision_feature_select_strategy` " +
                        "in the model's processing config to expand inputs " +
                        "for image tokens.")

        text_inputs = self.tokenizer(prompt_strings,
                                     **output_kwargs["text_kwargs"])
        return {**text_inputs, **image_inputs}

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(
            dict.fromkeys(tokenizer_input_names + image_processor_input_names))
