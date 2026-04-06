# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2024 The HuggingFace Inc. team.
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
Processor class for MiniCPMV.
"""

from typing import TypeAlias

import regex
import torch
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import TensorType

MiniCPMVBatchFeature: TypeAlias = BatchFeature


class MiniCPMVProcessor(ProcessorMixin):
    r"""
    Constructs a MiniCPMV processor which wraps a MiniCPMV image
    processor and a MiniCPMV tokenizer into a single processor.

    [`MiniCPMVProcessor`] offers all the functionalities of
    [`MiniCPMVImageProcessor`] and [`LlamaTokenizerWrapper`]. See the
    [`~MiniCPMVProcessor.__call__`] and [`~MiniCPMVProcessor.decode`]
    for more information.

    Args:
        image_processor ([`MiniCPMVImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerWrapper`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None):
        super().__init__(image_processor, tokenizer)
        self.version = image_processor.version

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        images: ImageInput = None,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy = None,
        max_length: int | None = None,
        do_pad: bool | None = True,
        return_tensors: str | TensorType | None = TensorType.PYTORCH,
    ) -> MiniCPMVBatchFeature:
        """
        Only support for single input for now. Batched input is coming soon.

        Args:
            text (`str`):
                The sequence or batch of sequences to be encoded. Each
                sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as
                list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a
                batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`,
                `List[PIL.Image.Image]`, `List[np.ndarray]`,
                `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image
                can be a PIL image, NumPy array or PyTorch tensor. Both
                channels-first and channels-last formats are supported.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`],
                *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences
                (according to the model's padding side and padding index)
                among:
                - `True` or `'longest'`: Pad to the longest sequence in
                  the batch (or no padding if only a single sequence if
                  provided).
                - `'max_length'`: Pad to a maximum length specified with
                  the argument `max_length` or to the maximum acceptable
                  input length for the model if that argument is not
                  provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e.,
                  can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally
                padding length (see above).
            do_pad (`bool`, *optional*, defaults to self.do_pad):
                Whether to pad the image. If `True` will pad the images
                in the batch to the largest image in the batch and create
                a pixel mask. Padding will be applied to the bottom and
                right of the image with zeros.
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than
                `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework.
                Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following
            fields:

            - **input_ids** -- List of token ids to be fed to a model.
              Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which
              tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is
              in `self.model_input_names` and if `text` is not `None`).
            - **pixel_values** -- Pixel values to be fed to a model.
              Returned when `images` is not `None`.
        """
        if images is not None:
            image_inputs = self.image_processor(
                images, do_pad=do_pad, return_tensors=return_tensors
            )
        return self._convert_images_texts_to_inputs(
            image_inputs, text, max_length=max_length
        )

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor
    # .batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's
        [`~PreTrainedTokenizer.batch_decode`]. Please refer to the
        docstring of this method for more information.
        """
        output_ids = args[0]
        result_text = []

        bos_id = getattr(
            self.tokenizer,
            "bos_token_id",
            getattr(self.tokenizer, "bos_id", 1),
        )
        eos_id = getattr(
            self.tokenizer,
            "eos_token_id",
            getattr(self.tokenizer, "eos_id", 2),
        )

        for result in output_ids:
            result = result[result != 0]
            if len(result) > 0 and result[0] == bos_id:
                result = result[1:]
            if len(result) > 0 and result[-1] == eos_id:
                result = result[:-1]
            result_text.append(
                self.tokenizer.decode(result, *args[1:], **kwargs).strip()
            )
        return result_text

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor
    # .decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's
        [`~PreTrainedTokenizer.decode`]. Please refer to the docstring
        of this method for more information.
        """
        result = args[0]
        result = result[result != 0]

        bos_id = getattr(
            self.tokenizer,
            "bos_token_id",
            getattr(self.tokenizer, "bos_id", 1),
        )
        eos_id = getattr(
            self.tokenizer,
            "eos_token_id",
            getattr(self.tokenizer, "eos_id", 2),
        )
        eot_id = getattr(self.tokenizer, "eot_id", None)

        if len(result) > 0 and result[0] == bos_id:
            result = result[1:]
        if len(result) > 0 and (
            result[-1] == eos_id or (eot_id is not None and result[-1] == eot_id)
        ):
            result = result[:-1]
        return self.tokenizer.decode(result, *args[1:], **kwargs).strip()

    def _convert(self, input_str, max_inp_length: int | None = None):
        add_bos = getattr(self.tokenizer, "add_bos_token", False)
        if self.version == 2.5 or add_bos:
            input_ids = self.tokenizer.encode(input_str)
        else:
            bos_id = getattr(
                self.tokenizer,
                "bos_token_id",
                getattr(self.tokenizer, "bos_id", 1),
            )
            input_ids = [bos_id] + self.tokenizer.encode(input_str)

        if max_inp_length is not None:
            input_ids = input_ids[:max_inp_length]
        input_ids = torch.tensor(input_ids, dtype=torch.int32)

        im_start_id = getattr(
            self.tokenizer,
            "im_start_id",
            self.tokenizer.convert_tokens_to_ids("<im_start>"),
        )
        im_end_id = getattr(
            self.tokenizer,
            "im_end_id",
            self.tokenizer.convert_tokens_to_ids("<im_end>"),
        )

        image_start_tokens = torch.where(input_ids == im_start_id)[0]
        image_start_tokens += 1
        image_end_tokens = torch.where(input_ids == im_end_id)[0]
        valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))
        image_bounds = torch.hstack(
            [
                image_start_tokens[:valid_image_nums].unsqueeze(-1),
                image_end_tokens[:valid_image_nums].unsqueeze(-1),
            ]
        )
        return input_ids.unsqueeze(0), image_bounds

    def _convert_images_texts_to_inputs(
        self,
        images,
        texts,
        do_pad=False,
        truncation=None,
        max_length=None,
        return_tensors=None,
    ):
        if not len(images):
            model_inputs = self.tokenizer(
                texts,
                return_tensors=return_tensors,
                padding=do_pad,
                truncation=truncation,
                max_length=max_length,
            )
            return MiniCPMVBatchFeature(data={**model_inputs})

        pattern = "(<image>./</image>)"
        images_val = images["pixel_values"]
        image_sizes = images["image_sizes"]
        tgt_sizes = images["tgt_sizes"]

        image_tags = regex.findall(pattern, texts)
        assert len(image_tags) == len(image_sizes[0])
        text_chunks = texts.split(pattern)
        final_texts = ""
        for i in range(len(image_tags)):
            placeholder = self.image_processor.get_slice_image_placeholder(
                image_sizes[0][i]
            )
            final_texts = final_texts + text_chunks[i] + placeholder
        final_texts += text_chunks[-1]
        input_ids, image_bounds = self._convert(final_texts, max_length)
        return MiniCPMVBatchFeature(
            data={
                "input_ids": input_ids,
                "pixel_values": images_val,
                "image_sizes": image_sizes,
                "image_bound": [image_bounds],
                "tgt_sizes": tgt_sizes,
            }
        )

    @property
    # Copied from
    # transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def pad(
        self,
        orig_items,
        key,
        max_length=None,
        padding_value=0,
        padding_side="left",
    ):
        items = []
        if isinstance(orig_items[0][key], list):
            assert isinstance(orig_items[0][key][0], torch.Tensor)
            for it in orig_items:
                for tr in it[key]:
                    items.append({key: tr})
        else:
            assert isinstance(orig_items[0][key], torch.Tensor)
            items = orig_items

        batch_size = len(items)
        shape = items[0][key].shape
        dim = len(shape)
        assert dim <= 3
        if max_length is None:
            max_length = 0
        max_length = max(max_length, max(item[key].shape[-1] for item in items))
        min_length = min(item[key].shape[-1] for item in items)
        dtype = items[0][key].dtype

        if dim == 1:
            return torch.cat([item[key] for item in items], dim=0)
        elif dim == 2:
            if max_length == min_length:
                return torch.cat([item[key] for item in items], dim=0)
            tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
        else:
            tensor = (
                torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype)
                + padding_value
            )

        for i, item in enumerate(items):
            if dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0])] = item[key][0].clone()
            elif dim == 3:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :, :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0]), :] = item[key][0].clone()

        return tensor
