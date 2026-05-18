# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone CPU preprocessing for Kimi-K2.5.

Algorithm engineers should edit this file for preprocessing changes. The vLLM
integration layer in ``kimi_k25.py`` only adapts outputs to
``BaseMultiModalProcessor``.

Corresponds to HuggingFace ``moonshotai/Kimi-K2.5``:
- ``kimi_k25_processor.py`` (``preprocess_medias``, ``update_raw_text``)
- Image patchify via the checkpoint's ``image_processor.preprocess``

vLLM serving uses ``preprocess()`` with pre-resolved ``vision_chunk`` items;
``preprocess_from_medias()`` matches the HF ``medias`` + ``text`` API.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import BaseImageProcessor, BatchFeature, TensorType

from vllm.tokenizers.hf import HfTokenizer

# Default video placeholder from HF KimiK25Processor.
DEFAULT_VIDEO_PLACEHOLDER = "<|kimi_k25_video_placeholder|>"


@dataclass
class KimiK25PreprocessConfig:
    """Runtime configuration for Kimi-K2.5 preprocessing."""

    media_token_id: int
    video_placeholder: str = DEFAULT_VIDEO_PLACEHOLDER


@dataclass
class KimiK25PreprocessResult:
    """Output of a full preprocess pass."""

    input_ids: list[int]
    pixel_values: torch.Tensor | None
    grid_thws: torch.Tensor | None
    num_tokens_per_chunk: list[int]
    attention_mask: list[int] | None = None


class KimiK25Preprocessor:
    """Self-contained Kimi-K2.5 multimodal preprocessor (CPU).

    Uses the HF hub ``image_processor`` for NaViT resize/patchify while keeping
    token expansion and media orchestration in this module.
    """

    def __init__(
        self,
        tokenizer: HfTokenizer,
        image_processor: BaseImageProcessor,
        config: KimiK25PreprocessConfig,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.config = config
        self.media_token_id = config.media_token_id
        self.video_placeholder = config.video_placeholder
        self.media_tokens_calculator: Callable[[Any], int] = (
            image_processor.media_tokens_calculator
        )

    def preprocess_medias(
        self, medias: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Port of HF ``KimiK25Processor.preprocess_medias``.

        Images pass through; videos are split into chunks via the HF image
        processor and per-video prompt strings are collected.
        """
        updated_medias: list[dict[str, Any]] = []
        video_prompts: list[str] = []
        for media in medias:
            media_type = media["type"]
            if media_type == "image":
                updated_medias.append(media)
            elif media_type == "video":
                video_chunks = self.image_processor.split_video_chunks(media["video"])
                updated_medias.extend(video_chunks)
                video_prompts.append(
                    "".join(vc["prompt"] for vc in video_chunks),
                )
            else:
                raise ValueError(f"unsupported media type: {media_type}")
        return updated_medias, video_prompts

    def update_raw_text(self, text: str, video_prompts: list[str]) -> str:
        """Port of HF ``KimiK25Processor.update_raw_text``."""
        video_count = text.count(self.video_placeholder)
        if video_count == 0:
            return text
        assert video_count == len(video_prompts)
        text_parts = text.split(self.video_placeholder)
        assert len(text_parts) == len(video_prompts) + 1
        text = "".join(
            text_parts[i] + video_prompts[i] for i in range(len(video_prompts))
        )
        return text + text_parts[-1]

    def expand_media_tokens(self, input_ids: list[int], num_tokens: list[int]) -> list[int]:
        """Expand each ``media_token_id`` placeholder to ``num_tokens`` copies."""
        num_tokens_queue = list(num_tokens)
        expanded: list[int] = []
        for token in input_ids:
            if token == self.media_token_id:
                count = num_tokens_queue.pop(0)
                expanded.extend([self.media_token_id] * count)
            else:
                expanded.append(token)
        if num_tokens_queue:
            raise ValueError(
                "num_tokens has more entries than media placeholders in input_ids"
            )
        return expanded

    def num_tokens_for_chunks(self, vision_chunks: Sequence[Any]) -> list[int]:
        return [self.media_tokens_calculator(chunk) for chunk in vision_chunks]

    def preprocess(
        self,
        text: str,
        vision_chunks: Sequence[Any] | None = None,
        return_tensors: str | TensorType | None = None,
    ) -> KimiK25PreprocessResult:
        """vLLM path: text prompt + pre-resolved vision chunks.

        Matches the previous ``KimiK25Processor.__call__`` behavior used by
        ``KimiK25MultiModalProcessor``.
        """
        mm_inputs: dict[str, Any] = {}
        num_tokens_per_chunk: list[int] = []
        if vision_chunks is not None:
            num_tokens_per_chunk = self.num_tokens_for_chunks(vision_chunks)
            mm_inputs = dict(
                self.image_processor.preprocess(
                    list(vision_chunks),
                    return_tensors=return_tensors,
                )
            )

        text_inputs = self.tokenizer([text])
        input_ids: list[int] = text_inputs["input_ids"][0]  # type: ignore[index]
        attention_mask: list[int] | None = text_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask[0]  # type: ignore[index]

        if vision_chunks is not None:
            input_ids = self.expand_media_tokens(input_ids, num_tokens_per_chunk)

        pixel_values = mm_inputs.get("pixel_values")
        grid_thws = mm_inputs.get("grid_thws")
        return KimiK25PreprocessResult(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            num_tokens_per_chunk=num_tokens_per_chunk,
            attention_mask=attention_mask,
        )

    def preprocess_from_medias(
        self,
        text: str,
        medias: list[dict[str, Any]],
        return_tensors: str | TensorType | None = "pt",
    ) -> KimiK25PreprocessResult:
        """HF ``medias`` + ``text`` path (no per-pad token expansion)."""
        updated_medias, video_prompts = self.preprocess_medias(medias)
        preprocessed = self.image_processor.preprocess(
            updated_medias,
            return_tensors=return_tensors,
        )
        text = self.update_raw_text(text, video_prompts)
        text_inputs = self.tokenizer([text], return_tensors=return_tensors)

        input_ids_tensor = text_inputs["input_ids"]
        input_ids = (
            input_ids_tensor[0].tolist()
            if isinstance(input_ids_tensor, torch.Tensor)
            else input_ids_tensor[0]
        )
        attention_mask_tensor = text_inputs.get("attention_mask")
        attention_mask = None
        if attention_mask_tensor is not None:
            attention_mask = (
                attention_mask_tensor[0].tolist()
                if isinstance(attention_mask_tensor, torch.Tensor)
                else attention_mask_tensor[0]
            )

        pixel_values = preprocessed.get("pixel_values")
        grid_thws = preprocessed.get("grid_thws")
        num_tokens_per_chunk = [
            self.media_tokens_calculator(media) for media in updated_medias
        ]
        return KimiK25PreprocessResult(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            num_tokens_per_chunk=num_tokens_per_chunk,
            attention_mask=attention_mask,
        )

    def to_batch_feature(
        self,
        result: KimiK25PreprocessResult,
        return_tensors: str | TensorType | None = None,
    ) -> BatchFeature:
        """Convert a result to ``BatchFeature`` for the vLLM MM processor."""
        data: dict[str, Any] = {"input_ids": [result.input_ids]}
        if result.attention_mask is not None:
            data["attention_mask"] = [result.attention_mask]
        if result.pixel_values is not None:
            data["pixel_values"] = result.pixel_values
        if result.grid_thws is not None:
            data["grid_thws"] = result.grid_thws
        return BatchFeature(data=data, tensor_type=return_tensors)
