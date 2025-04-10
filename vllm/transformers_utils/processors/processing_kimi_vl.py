# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
# Adapted from https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/main/processing_kimi_vl.py
# Copyright 2025 The Moonshot Team and HuggingFace Inc. team. All rights reserved.
#
# The code is based on the Qwen2VL processor (qwen2_vl/processing_qwen2_vl.py), but modified for KimiVL.
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
"""
Processor class for KimiVL.
"""

import math
from typing import List, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import (ProcessingKwargs, ProcessorMixin,
                                           Unpack,
                                           _validate_images_text_input_order)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging

logger = logging.get_logger(__name__)


class KimiVLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {},
    }


class KimiVLProcessor(ProcessorMixin):
    r"""
    Constructs a KimiVL processor which wraps a KimiVL image processor and a tokenizer into a single processor.

    [`KimiVLProcessor`] offers all the functionalities of [`KimiVLImageProcessor`] and [`TikTokenTokenizer`]. See the
    [`~KimiVLProcessor.__call__`] and [`~KimiVLProcessor.decode`] for more information.

    Args:
        image_processor ([`KimiVLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`TikTokenTokenizer`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = "<|media_pad|>"
        super().__init__(image_processor,
                         tokenizer,
                         chat_template=chat_template)

    # for vllm
    def get_image_tokens(self, width: int, height: int) -> int:
        patch_size = self.image_processor.patch_size
        kernel_size = self.image_processor.merge_kernel_size
        in_token_limit = self.image_processor.in_token_limit
        assert isinstance(height,
                          int), f'height must be int, current height {height}'
        assert isinstance(width,
                          int), f'width must be int, current width {width}'
        assert kernel_size is not None, 'kernel_size must be specified'

        if (width // patch_size) * (height // patch_size) > in_token_limit:
            scale = math.sqrt(in_token_limit / ((width // patch_size) *
                                                (height // patch_size)))
            new_w, new_h = int(width * scale), int(height * scale)
            width, height = new_w, new_h

        kernel_height, kernel_width = kernel_size

        pad_height = (kernel_height * patch_size - height %
                      (kernel_height * patch_size)) % (kernel_height *
                                                       patch_size)
        pad_width = (kernel_width * patch_size - width %
                     (kernel_width * patch_size)) % (kernel_width * patch_size)

        # Calculate new dimensions after padding and patching
        token_height = (height + pad_height) // (kernel_size[0] * patch_size)
        token_width = (width + pad_width) // (kernel_size[1] * patch_size)
        return int(token_height * token_width)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput],
                    List[PreTokenizedInput]] = None,
        **kwargs: Unpack[KimiVLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to TikTokenTokenizer's [`~TikTokenTokenizer.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
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
        """
        if images is None and text is None:
            raise ValueError(
                "You have to specify at least one of `images` or `text`.")

        # check if images and text inputs are reversed for BC
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            KimiVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(
                images, **output_kwargs["images_kwargs"])
            image_grid_hws = image_inputs["image_grid_hws"]
        else:
            image_inputs = {}
            image_grid_hws = None

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        if image_grid_hws is not None:
            merge_length = self.image_processor.merge_kernel_size[
                0] * self.image_processor.merge_kernel_size[1]
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" *
                        (image_grid_hws[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **image_inputs})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(
            dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["KimiVLProcessorKwargs"]
