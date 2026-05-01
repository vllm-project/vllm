# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://huggingface.co/Qwen/Qwen-VL/blob/main/modeling_qwen.py
# Copyright (c) Alibaba Cloud.
from transformers.image_processing_utils_fast import BaseImageProcessorFast
from transformers.image_utils import PILImageResampling
from transformers.processing_utils import ProcessorMixin

from vllm.tokenizers.qwen_vl import QwenVLTokenizer


class QwenVLImageProcessorFast(BaseImageProcessorFast):
    """
    Port of https://huggingface.co/Qwen/Qwen-VL/blob/main/visual.py#L354
    to HF Transformers.
    """

    resample = PILImageResampling.BICUBIC
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    size = {"height": 448, "width": 448}
    do_resize = True
    do_rescale = True
    do_normalize = True


class QwenVLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor: QwenVLImageProcessorFast,
        tokenizer: QwenVLTokenizer,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        self.image_start_tag = tokenizer.image_start_tag
        self.image_end_tag = tokenizer.image_end_tag
        self.image_pad_tag = tokenizer.image_pad_tag
