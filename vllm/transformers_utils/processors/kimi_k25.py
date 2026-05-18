# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deprecated thin wrapper around :class:`KimiK25Preprocessor`.

Edit preprocessing logic in
``vllm.model_executor.models.preprocessing.kimi_k25_preprocessing``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import BatchFeature, TensorType
from transformers.processing_utils import ProcessorMixin

from vllm.multimodal.inputs import VisionChunk

if TYPE_CHECKING:
    from vllm.model_executor.models.preprocessing.kimi_k25_preprocessing import (
        KimiK25Preprocessor,
    )


class KimiK25Processor(ProcessorMixin):
    """HF-compatible processor facade delegating to ``KimiK25Preprocessor``."""

    attributes = ["image_processor", "tokenizer"]

    def __init__(self, preprocessor: KimiK25Preprocessor) -> None:
        self._preprocessor = preprocessor
        self.image_processor = preprocessor.image_processor
        self.tokenizer = preprocessor.tokenizer
        self.media_token_id = preprocessor.media_token_id

    def __call__(
        self,
        text: str | list[str] | None = None,
        vision_chunks: list[VisionChunk] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            raise ValueError("text is required")
        if isinstance(text, list):
            if len(text) != 1:
                raise ValueError("KimiK25Processor only supports a single text prompt")
            text = text[0]

        result = self._preprocessor.preprocess(
            text,
            vision_chunks=vision_chunks,
            return_tensors=return_tensors,
        )
        return self._preprocessor.to_batch_feature(result, return_tensors=return_tensors)
