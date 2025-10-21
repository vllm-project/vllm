# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa: E501
# coding=utf-8
# adapted from https://github.com/AIDC-AI/Ovis/blob/35ab51a1a1e3542fa6db260a1084cefbc8f164bb/ovis/vllm/processing_ovis.py
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import cached_property

import PIL
import torch
from transformers import AutoProcessor, BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from vllm.multimodal.image import convert_image_mode

__all__ = ["OvisProcessor"]
IGNORE_ID = -100


class OvisProcessorKwargs(ProcessingKwargs, total=False):  # type: ignore[call-arg]
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "max_partition": 9,
            "covering_threshold": 0.9,
            "convert_to_rgb": True,
            "return_tensors": "pt",
        },
    }


class OvisProcessor(ProcessorMixin):
    r"""
    Constructs an Ovis processor which wraps an Ovis image processor and a Qwen2 tokenizer into a single processor.
    [`OvisProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`] and [`Qwen2TokenizerFast`]. See the
    [`~OvisProcessor.__call__`] and [`~OvisProcessor.decode`] for more information.
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "image_pad_token", "image_segment_len"]

    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_pad_token=None,
        image_segment_len=255,
        **kwargs,
    ):
        self.image_token = "<image>"
        self.image_pad_token = image_pad_token
        self.image_segment_len = image_segment_len
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    @cached_property
    def extra_special_tokens(self):
        image_pad_token_id = self.tokenizer.get_vocab()[self.image_pad_token]
        extra_special_tokens = {
            "image_token": -200,
            "image_atom": -300,
            "image_start": -301,
            "image_prefix": -302,
            "image_col_sep": -303,
            "image_row_sep": -304,
            "image_end": -305,
            "image_pad": image_pad_token_id,
        }
        return extra_special_tokens

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput] = None,
        **kwargs: Unpack[OvisProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.
            Args:
                images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                    The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                    tensor. Both channels-first and channels-last formats are supported.
                text (`str`, `list[str]`, `list[list[str]]`):
                    The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                    (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                    `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
                videos (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                    The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                    tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
                return_tensors (`str` or [`~utils.TensorType`], *optional*):
                    If set, will return tensors of a particular framework. Acceptable values are:
                    - `'tf'`: Return TensorFlow `tf.constant` objects.
                    - `'pt'`: Return PyTorch `torch.Tensor` objects.
                    - `'np'`: Return NumPy `np.ndarray` objects.
                    - `'jax'`: Return JAX `jnp.ndarray` objects.
            Returns:
                [`BatchFeature`]: A [`BatchFeature`] with the following fields:
                - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
                - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
                  `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
                  `None`).
                - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
                - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
                - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
                - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
                - **second_per_grid_ts** -- List of video seconds per time grid. Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            OvisProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # Process all images first
        image_features = {}
        if images is not None:
            processed_images = []
            image_placeholders_list = []
            grids = []

            # Process each image
            for image in images if isinstance(images, list) else [images]:
                pixel_values, image_placeholders, grid = self.preprocess_image(
                    image=image, **output_kwargs["images_kwargs"]
                )
                processed_images.append(pixel_values)
                image_placeholders_list.append(image_placeholders)
                grids.append(grid)

            # assign all processed images
            if processed_images:
                image_features["image_placeholders"] = image_placeholders_list

        # Process text input
        if text is not None:
            if not isinstance(text, list):
                text = [text]

            tokenized_batched_text = self._tokenize_with_image_symbol(text)
            image_token_id = self.get_token_value("image_token")
            replaced_ids_list = []
            idx = 0
            for ids_tensor in tokenized_batched_text:
                if (
                    image_token_id in ids_tensor
                    and "image_placeholders" in image_features
                ):
                    if idx < len(image_features["image_placeholders"]):
                        # Converts in list for ease of use
                        ids_list = ids_tensor.tolist()

                        new_ids = []

                        # replace placeholders
                        for i, token_id in enumerate(ids_list):
                            if token_id == image_token_id:
                                placeholder_ids = image_features["image_placeholders"][
                                    idx
                                ]
                                new_ids.extend(placeholder_ids)
                                idx += 1
                            else:
                                new_ids.append(token_id)

                        # Converts back to tensors
                        ids_tensor = torch.tensor(new_ids, dtype=torch.long)
                    else:
                        raise RuntimeError(
                            "Mismatch between the images you provided and the number of placeholder present in the text"
                        )

                replaced_ids_list.append(ids_tensor)

            if replaced_ids_list:
                replaced_and_tokenized_ids = torch.stack(replaced_ids_list)
            else:
                replaced_and_tokenized_ids = torch.tensor([], dtype=torch.long)

            # Create the output with text features
            output = BatchFeature(
                data={
                    "input_ids": replaced_and_tokenized_ids,
                }
            )

            # Add image features if present
            if image_features:
                output["pixel_values"] = processed_images
                output["grids"] = grids

            return output

        # If only images were provided
        return BatchFeature(data=image_features)

    def _tokenize_with_image_symbol(self, text_list: list[str]) -> torch.LongTensor:
        batch_token_ids = []
        for text in text_list:
            text_chunks = [
                self.tokenizer(chunk, add_special_tokens=False).input_ids
                for chunk in text.split(self.image_token)
            ]
            token_ids = []
            num_chuck = len(text_chunks)
            for i, chunk in enumerate(text_chunks):
                token_ids.extend(chunk)
                if i < num_chuck - 1:
                    token_ids.append(self.get_token_value("image_token"))
            batch_token_ids.append(token_ids)
        return torch.tensor(batch_token_ids, dtype=torch.long)

    def get_image_size(self):
        size = self.image_processor.size
        if "shortest_edge" in size:
            width = height = size["shortest_edge"]
        elif "height" in size and "width" in size:
            width = size["width"]
            height = size["height"]
        else:
            raise ValueError("Can't parse image size from image_processor config.")
        return height, width

    def get_token_value(self, tok):
        return self.extra_special_tokens[tok]

    def construct_image_indicators(self, grid):
        image_placeholders = [
            self.get_token_value("image_start"),
            self.get_token_value("image_atom"),
            self.get_token_value("image_prefix"),
        ]
        if grid[0] * grid[1] > 1:
            for r in range(grid[0]):
                for c in range(grid[1]):
                    image_placeholders.append(self.get_token_value("image_atom"))
                    if c < grid[1] - 1:
                        image_placeholders.append(self.get_token_value("image_col_sep"))
                if r < grid[0] - 1:
                    image_placeholders.append(self.get_token_value("image_row_sep"))
        image_placeholders.append(self.get_token_value("image_end"))
        return image_placeholders

    def construct_image_placeholders(self, grid):
        image_placeholders = self.construct_image_indicators(grid)

        image_atom_token_id = self.get_token_value("image_atom")
        # Extract the padding token ID from tokenizer
        image_padding_token_id = self.get_token_value("image_pad")

        # Create a new list with padding tokens inserted
        padded_placeholder_tokens = []
        for token in image_placeholders:
            padded_placeholder_tokens.append(image_padding_token_id)
            if token == image_atom_token_id:
                padded_placeholder_tokens.extend(
                    [image_padding_token_id] * self.image_segment_len
                )
        return padded_placeholder_tokens

    def preprocess_image(
        self,
        image: PIL.Image.Image,
        max_partition,
        covering_threshold,
        convert_to_rgb,
        return_tensors,
    ):
        def _preprocess(img: PIL.Image.Image, side):
            # first resize and preprocess
            w, h = img.size
            if w == h:
                new_width = new_height = side
            elif w > h:
                new_width = side
                new_height = int(h / w * new_width)
            else:
                new_height = side
                new_width = int(w / h * new_height)
            new_size = dict(height=new_height, width=new_width)
            pixel_values = self.image_processor.preprocess(
                img, size=new_size, return_tensors=return_tensors
            )["pixel_values"]

            # then pad to square
            square_values = torch.zeros(
                [1, 3, side, side], dtype=pixel_values.dtype, device=pixel_values.device
            )
            new_height, new_width = pixel_values.shape[2:]
            if new_height == new_width:
                square_values[:, :, :, :] = pixel_values
            elif new_height > new_width:
                from_index = (side - new_width) // 2
                square_values[:, :, :, from_index : from_index + new_width] = (
                    pixel_values
                )
            else:
                from_index = (side - new_height) // 2
                square_values[:, :, from_index : from_index + new_height, :] = (
                    pixel_values
                )

            return square_values

        def _partition(img, grid) -> list[tuple[int, int, int, int]]:
            w, h = img.size
            row_height = h // grid[0]
            col_width = w // grid[1]

            partition = []
            for row in range(grid[0]):
                for col in range(grid[1]):
                    left = col * col_width
                    upper = row * row_height
                    right = w if col == grid[1] - 1 else (col + 1) * col_width
                    lower = h if row == grid[0] - 1 else (row + 1) * row_height
                    partition.append((left, upper, right, lower))

            return partition

        def _covering_area(left, upper, right, lower, side):
            w = right - left
            h = lower - upper
            w, h = max(w, h), min(w, h)
            if w > side:
                h = h / w * side
                w = side
            return w * h

        def _get_best_grid(img, side):
            img_area = img.size[0] * img.size[1]

            candidate_grids = []
            for i in range(1, max_partition + 1):
                for j in range(1, max_partition + 1):
                    if i * j <= max_partition:
                        candidate_grids.append((i, j))

            all_grids = []
            good_grids = []
            for grid in candidate_grids:
                partition = _partition(img, grid)
                covering_ratio = (
                    sum([_covering_area(*p, side) for p in partition]) / img_area
                )
                assert covering_ratio <= 1.0
                all_grids.append((grid, covering_ratio))
                if covering_ratio > covering_threshold:
                    good_grids.append((grid, covering_ratio))

            if len(good_grids) > 0:
                # pick the good partition with minimum #sub_images and break the tie using covering_ratio
                return sorted(good_grids, key=lambda x: (x[0][0] * x[0][1], -x[1]))[0][
                    0
                ]
            else:
                # pick the partition with maximum covering_ratio and break the tie using #sub_images
                return sorted(all_grids, key=lambda x: (-x[1], x[0][0] * x[0][1]))[0][0]

        if convert_to_rgb:
            image = convert_image_mode(image, "RGB")

        sides = self.get_image_size()
        if sides[0] != sides[1]:
            raise ValueError("get_image_size() returns non-square size")
        side = sides[0]
        grid = _get_best_grid(image, side)
        partition = _partition(image, grid)
        crops = [image.crop(p) for p in partition]
        if len(crops) > 1:
            crops.insert(0, image)
        pixel_values = torch.cat([_preprocess(crop, side) for crop in crops], dim=0)
        image_placeholders = self.construct_image_placeholders(grid)
        return torch.tensor(pixel_values), image_placeholders, torch.tensor(grid)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(self, generated_outputs):
        """
        Post-process the output of the model to decode the text.
        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        names_from_processor = list(
            dict.fromkeys(tokenizer_input_names + image_processor_input_names)
        )
        return names_from_processor + ["second_per_grid_ts"]


AutoProcessor.register("OvisProcessor", OvisProcessor)
