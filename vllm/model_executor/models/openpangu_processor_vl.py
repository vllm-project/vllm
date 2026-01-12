#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The HuggingFace Inc. team
# Adapted from transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py
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

from typing import List, Optional, Union, Tuple
from transformers.utils import logging
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor, Qwen2_5_VLProcessorKwargs
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput

logger = logging.get_logger(__name__)


class OpenPanguVLProcessor(Qwen2_5_VLProcessor):
    tokenizer_class = ("AutoTokenizer")

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        self.tokenizer = tokenizer

        self.image_token = "[unused19]"
        self.video_token = "[unused32]"
        self.vision_start_token = "[unused18]"
        self.vision_end_token = "[unused20]"

        self.image_token_id = (
            self.tokenizer.image_token_id
            if getattr(self.tokenizer, "image_token_id", None)
            else self.tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            self.tokenizer.video_token_id
            if getattr(self.tokenizer, "video_token_id", None)
            else self.tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.vision_start_token_id = (
            self.tokenizer.vision_start_token_id
            if getattr(self.tokenizer, "vision_start_token_id", None)
            else self.tokenizer.convert_tokens_to_ids(self.vision_start_token)
        )
        self.vision_end_token_id = (
            self.tokenizer.vision_end_token_id
            if getattr(self.tokenizer, "vision_end_token_id", None)
            else self.tokenizer.convert_tokens_to_ids(self.vision_end_token)
        )

        self.image_processor = image_processor
        self.video_processor = video_processor
        self.chat_template = chat_template

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if not isinstance(text, list):
            text = [text]
        image_inputs = {}
        videos_inputs = {}
        if images is not None:
            print(f"self.image_processor:{self.image_processor}")
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            self._process_vision_placeholders(
                text=text,
                vision_token=self.image_token,
                grid_thw=image_inputs["image_grid_thw"],
                merge_size=self.image_processor.merge_size,
                vision_start_token=self.vision_start_token,
                vision_end_token=self.vision_end_token,
            )
        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            self._process_vision_placeholders(
                text=text,
                vision_token=self.video_token,
                grid_thw=videos_inputs["video_grid_thw"],
                merge_size=self.video_processor.merge_size,
                vision_start_token=self.vision_start_token,
                vision_end_token=self.vision_end_token,
            )
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

    @staticmethod
    def _process_vision_placeholders(
        text: List[str],
        vision_token: str,
        grid_thw: List[Tuple[int, int, int]],
        merge_size: int,
        vision_start_token: str,
        vision_end_token: str,
    ) -> None:
        """
        Replace placeholder tokens (e.g., <image> or <video>) in text with structured vision token sequences.
        For images:
            - `grid_thw[i] = (1, H, W)` → treated as a single time slice.
        For videos:
            - `grid_thw[i] = (T, H, W)` → repeated `T` times.

        Args:
            text (`List[str]`):
                List of input text strings that may contain placeholder tokens.
            vision_token (`str`):
                The vision token to replace (e.g., "<|image|>" or "<|video|>").
            grid_thw (`List[Tuple[int, int, int]]`):
                List of (T, H, W) grid dimensions for each media item.
            merge_size (`int`):
                Spatial merging factor used to compute token count per frame.
            vision_start_token (`str`):
                Special token marking the start of a vision sequence.
            vision_end_token (`str`):
                Special token marking the end of a vision sequence.

        Returns:
            `None`. The `text` is modified in-place.
        """
        index = 0
        for i in range(len(text)):
            while vision_token in text[i]:
                if index >= len(grid_thw):
                    raise ValueError(
                        f"Found more vision tokens than entries in 'grid_thw'. "
                        f"Expected one (T, H, W) tuple per '{vision_token}' token."
                    )
                grid_t, grid_h, grid_w = grid_thw[index]
                # Calculate the sequence length per time slice based on the grid size and merge length.
                seq_length_per_time = (grid_h * grid_w) // (merge_size ** 2)
                # Prepare a placeholder string that includes start and end tokens,
                # and then calculate the number of media tokens to replace.
                placeholder_string = (
                    vision_start_token
                    + ("<|vision_placeholder|>" * seq_length_per_time)
                    + vision_end_token
                )
                if grid_t > 1:
                    # For videos only, repeat the placeholder string for each time slice.
                    placeholder_string *= grid_t
                placeholder_string = placeholder_string.removeprefix(vision_start_token)
                placeholder_string = placeholder_string.removesuffix(vision_end_token)
                text[i] = text[i].replace(vision_token, placeholder_string, 1)
                index += 1
            text[i] = text[i].replace("<|vision_placeholder|>", vision_token)


__all__ = ["OpenPanguVLProcessor"]
