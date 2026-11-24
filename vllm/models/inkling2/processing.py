# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 image preprocessing and vLLM multimodal processing.

The renderer emits one `<|content_image|>` token per image. The processor
replaces it with `<|content_image|>` followed by `rows * cols` copies of
`IMAGE_PAD_ID`; only the pad positions receive vision embeddings.
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any, TypeAlias

import numpy as np
import torch
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature

from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.media import MediaWithBytes
from vllm.multimodal.parse import (
    ImageProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

# <|content_image|>: block marker emitted by the renderer, kept verbatim.
IMAGE_MARKER_ID = 200005
IMAGE_PLACEHOLDER = "<|content_image|>"
# <|unused_200054|>: one per decoder image token, overwritten by embeddings.
IMAGE_PAD_ID = 200054

PATCH_SIZE = 40
# Dummy (profiling) images are at most this many pixels per side.
MAX_DUMMY_IMAGE_SIDE = 2048

ImageInput: TypeAlias = bytes | Image.Image | MediaWithBytes | np.ndarray


def _composite_on_checkerboard(rgba: np.ndarray) -> np.ndarray:
    rgba = rgba.astype(np.uint16)
    height, width, _ = rgba.shape
    y, x = np.ogrid[:height, :width]
    background = np.where((x // 8 + y // 8) % 2 == 0, 204, 153).astype(np.uint16)
    alpha = rgba[:, :, 3:4]
    blended = rgba[:, :, :3] * alpha + background[:, :, None] * (255 - alpha) + 127
    return (blended // 255).astype(np.uint8)


def _pil_to_rgb(image: Image.Image) -> np.ndarray:
    if not image.has_transparency_data:
        return np.array(image.convert("RGB"), dtype=np.uint8)
    return _composite_on_checkerboard(np.array(image.convert("RGBA")))


def decode_image(image: ImageInput) -> np.ndarray:
    """Decode an image to HWC uint8 RGB, compositing alpha on a checkerboard.

    Encoded bytes (including `MediaWithBytes.original_bytes`) are preferred
    over decoded media, so no vLLM-side mode conversion or EXIF handling
    leaks into the pixels.
    """
    if isinstance(image, MediaWithBytes):
        image = image.original_bytes
    if isinstance(image, (bytes, bytearray, memoryview)):
        data = bytes(image)
        if data.startswith(b"\xff\xd8") and not data.endswith(b"\xff\xd9"):
            data += b"\xff\xd9"
        with Image.open(io.BytesIO(data)) as pil:
            pil.load()
            return _pil_to_rgb(pil)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return _pil_to_rgb(image)


class Inkling2ImageProcessor:
    """Pads each image to a 40-pixel grid and emits one patch per cell.

    Returns `vision_pixels` (bf16 `[sum(rows*cols), 40, 40, 3]` raw RGB
    values, padding pixels at -1; the encoder normalizes them) and
    `vision_grid_thw`.
    """

    @staticmethod
    def get_grid(width: int, height: int) -> tuple[int, int]:
        return math.ceil(height / PATCH_SIZE), math.ceil(width / PATCH_SIZE)

    def preprocess(self, images: Sequence[ImageInput]) -> dict[str, torch.Tensor]:
        decoded = [torch.from_numpy(decode_image(image)) for image in images]
        grids = [self.get_grid(pixels.shape[1], pixels.shape[0]) for pixels in decoded]
        out = torch.empty(
            (sum(rows * cols for rows, cols in grids), PATCH_SIZE, PATCH_SIZE, 3),
            dtype=torch.bfloat16,
        )
        start = 0
        for pixels, (rows, cols) in zip(decoded, grids):
            height, width, _ = pixels.shape
            if (height, width) != (rows * PATCH_SIZE, cols * PATCH_SIZE):
                padded = pixels.new_full(
                    (rows * PATCH_SIZE, cols * PATCH_SIZE, 3), -1, dtype=torch.bfloat16
                )
                padded[:height, :width] = pixels
                pixels = padded
            cells = pixels.view(rows, PATCH_SIZE, cols, PATCH_SIZE, 3)
            out[start : start + rows * cols].view(
                rows, cols, PATCH_SIZE, PATCH_SIZE, 3
            ).copy_(cells.permute(0, 2, 1, 3, 4))
            start += rows * cols
        return {
            "vision_pixels": out,
            "vision_grid_thw": torch.tensor(
                [[1, rows, cols] for rows, cols in grids], dtype=torch.int64
            ),
        }


class Inkling2MultiModalDataParser(MultiModalDataParser):
    """Keeps PIL image modes so transparency reaches the checkerboard step."""

    def _parse_image_data(self, data: Any) -> Any:
        if isinstance(data, Image.Image):
            return ImageProcessorItems([data])
        if (
            isinstance(data, Sequence)
            and data
            and all(isinstance(item, Image.Image) for item in data)
        ):
            return ImageProcessorItems(list(data))
        return super()._parse_image_data(data)


class Inkling2ProcessingInfo(BaseProcessingInfo):
    @cached_property
    def image_processor(self) -> Inkling2ImageProcessor:
        return Inkling2ImageProcessor()

    def get_data_parser(self) -> MultiModalDataParser:
        return Inkling2MultiModalDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_supported_mm_processor_kwargs(self) -> dict[str, set[str]]:
        return {}


class Inkling2DummyInputsBuilder(BaseDummyInputsBuilder[Inkling2ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        # Largest square image whose tokens still fit in seq_len.
        side = min(MAX_DUMMY_IMAGE_SIDE, PATCH_SIZE * max(1, math.isqrt(seq_len)))
        return {
            "image": self._get_dummy_images(
                width=side,
                height=side,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            )
        }


class Inkling2MultiModalProcessor(BaseMultiModalProcessor[Inkling2ProcessingInfo]):
    # Prompts arrive as token ids and the update target is a single token id.
    requires_tokenizer = False

    def get_dummy_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> ProcessorInputs:
        # Build marker ids directly: the tokenizer may not parse special text.
        mm_data = self.dummy_inputs.get_dummy_mm_data(seq_len, mm_counts, mm_options)
        return ProcessorInputs(
            prompt=[IMAGE_MARKER_ID] * mm_counts.get("image", 0),
            mm_data_items=self.info.parse_mm_data(mm_data, validate=False),
        )

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        hf_data, hf_kwargs, passthrough_data = self._get_hf_mm_inputs(
            mm_items, hf_kwargs
        )
        images = mm_items.get("image")
        processed = None
        if isinstance(images, ImageProcessorItems) and images.get_count():
            # Raw items keep MediaWithBytes.original_bytes for exact decoding.
            processed = BatchFeature(self.info.image_processor.preprocess(images.data))
        return self._finalize_hf_mm_data(
            hf_data, hf_kwargs, passthrough_data, processed
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        grid_thw = hf_inputs.get("vision_grid_thw", torch.empty((0, 3)))
        return dict(
            vision_pixels=MultiModalFieldConfig.flat_from_sizes(
                "image", grid_thw.prod(-1)
            ),
            vision_grid_thw=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        out_items = out_mm_kwargs.get("image", [])

        def replacement(item_idx: int) -> PromptUpdateDetails:
            item = out_items[item_idx]
            assert item is not None
            grid_thw = item["vision_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            num_tokens = int(grid_thw.prod())
            return PromptUpdateDetails.select_token_id(
                [IMAGE_MARKER_ID] + [IMAGE_PAD_ID] * num_tokens, IMAGE_PAD_ID
            )

        return [
            PromptReplacement(
                modality="image", target=[IMAGE_MARKER_ID], replacement=replacement
            )
        ]
