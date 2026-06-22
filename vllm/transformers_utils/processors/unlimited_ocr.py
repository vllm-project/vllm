# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from transformers import LlamaTokenizerFast

from vllm.transformers_utils.processors.deepseek_ocr import (
    BASE_SIZE,
    CROP_MODE,
    IMAGE_SIZE,
    DeepseekOCRProcessor,
)


class UnlimitedOCRHFProcessor(DeepseekOCRProcessor):
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")
    attributes = ["tokenizer"]

    def __init__(
        self,
        tokenizer: LlamaTokenizerFast,
        image_size: int = IMAGE_SIZE,
        base_size: int = BASE_SIZE,
        crop_mode: bool = CROP_MODE,
        sft_format: str = "unlimitedocr",
        mask_prompt: bool = False,
        candidate_resolutions: list[list[int]] | None = None,
        processor_class: str | None = None,
        **kwargs: Any,
    ):
        self.crop_mode = crop_mode
        self.candidate_resolutions = candidate_resolutions or [[1024, 1024]]
        self.processor_class = processor_class

        super().__init__(
            tokenizer=tokenizer,
            image_size=image_size,
            base_size=base_size,
            sft_format=sft_format,
            mask_prompt=mask_prompt,
            **kwargs,
        )

    def __call__(
        self,
        *,
        prompt: str,
        images: list,
        crop_mode: bool | None = None,
        **kwargs: Any,
    ):
        return super().__call__(
            prompt=prompt,
            images=images,
            crop_mode=self.crop_mode if crop_mode is None else crop_mode,
            **kwargs,
        )
