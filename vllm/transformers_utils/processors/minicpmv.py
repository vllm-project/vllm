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
        # Newer (transformers v5.7+) MiniCPM-V image processors, e.g.
        # MiniCPMV4_6ImageProcessor, no longer carry a `version` attribute.
        # Fall back to None instead of hard-crashing: `version` is only used
        # to special-case the 2.5 tokenization path in `_convert`, and any
        # value other than 2.5 takes the default branch anyway.
        self.version = getattr(image_processor, "version", None)

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
        """Run the vendored MiniCPMV processor on a (text, images) pair.

        Batched inputs are supported following the upstream MiniCPM-V
        processor flow. ``images`` is forwarded to the underlying image
        processor and ``text`` is tokenized with image placeholders
        replaced by the appropriate slice tokens. Returns a
        ``MiniCPMVBatchFeature`` with at minimum ``input_ids`` and (when
        images are provided) ``pixel_values``, ``image_sizes``,
        ``image_bound`` and ``tgt_sizes``.
        """
        if images is not None:
            image_inputs = self.image_processor(
                images, do_pad=do_pad, return_tensors=return_tensors
            )
        else:
            image_inputs = {}
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
        assert len(image_start_tokens) == len(image_end_tokens), (
            f"The number of image start tokens ({len(image_start_tokens)}) "
            f"and end tokens ({len(image_end_tokens)}) must match."
        )
        image_bounds = torch.hstack(
            [
                image_start_tokens.unsqueeze(-1),
                image_end_tokens.unsqueeze(-1),
            ]
        )
        return input_ids, image_bounds

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

        if isinstance(texts, str):
            texts = [texts]

        input_ids_list = []
        image_bounds_list = []

        for index, text in enumerate(texts):
            image_tags = regex.findall(pattern, text)
            assert len(image_tags) == len(image_sizes[index])
            text_chunks = text.split(pattern)
            final_text = ""
            for i in range(len(image_tags)):
                placeholder = self.image_processor.get_slice_image_placeholder(
                    image_sizes[index][i]
                )
                final_text = final_text + text_chunks[i] + placeholder
            final_text += text_chunks[-1]
            input_ids, image_bounds = self._convert(final_text, max_length)
            input_ids_list.append(input_ids)
            image_bounds_list.append(image_bounds)

        padded_input_ids, padding_lengths = self.pad(
            input_ids_list,
            padding_side="left",
        )
        for i, length in enumerate(padding_lengths):
            image_bounds_list[i] = image_bounds_list[i] + length

        return MiniCPMVBatchFeature(
            data={
                "input_ids": padded_input_ids,
                "attention_mask": padded_input_ids.ne(0),
                "pixel_values": images_val,
                "image_sizes": image_sizes,
                "image_bound": image_bounds_list,
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

    # Copied from openbmb/MiniCPM-V-4_5 processing_minicpmv.py.
    def pad(self, inputs, max_length=None, padding_value=0, padding_side="left"):
        if not inputs:
            return torch.empty(0), []

        items = []
        if isinstance(inputs[0], list):
            assert isinstance(inputs[0][0], torch.Tensor)
            for it in inputs:
                for tr in it:
                    items.append(tr)
        else:
            assert isinstance(inputs[0], torch.Tensor)
            items = inputs

        batch_size = len(items)
        shape = items[0].shape
        dim = len(shape)
        assert dim <= 2
        if max_length is None:
            max_length = 0
        max_length = max(max_length, max(item.shape[-1] for item in items))
        min_length = min(item.shape[-1] for item in items)
        dtype = items[0].dtype

        if dim == 0:
            return torch.stack([item for item in items], dim=0), [0]
        elif dim == 1:
            if max_length == min_length:
                return torch.stack([item for item in items], dim=0), [0] * batch_size
            tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
        else:
            tensor = (
                torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype)
                + padding_value
            )

        padding_lengths = []
        for i, item in enumerate(items):
            if dim == 1:
                if padding_side == "left":
                    tensor[i, -len(item) :] = item.clone()
                else:
                    tensor[i, : len(item)] = item.clone()
            elif dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item) :, :] = item.clone()
                else:
                    tensor[i, : len(item), :] = item.clone()
            padding_lengths.append(tensor.shape[-1] - len(item))

        return tensor, padding_lengths
