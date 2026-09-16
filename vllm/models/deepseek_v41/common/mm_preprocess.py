# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multimodal preprocessing for the DeepSeek-V4.1 vision variant.

The image transform and image-span construction are ported from the
official repository's ``image_processor.py`` so that token counts bit-match
the reference. Each ``<｜deepseek_image｜>`` placeholder in the prompt expands
to ``[IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h +
[IMAGE_END]``; every span position carries ``image_token_id`` (129264) in
``input_ids`` and the roles ride along in a per-image ``types`` tensor
(the reference's out-of-band ``token_types``). IMAGE slots receive aligner
rows in reading order; the delimiters take the learned ``image_start`` /
``image_newline`` / ``image_end`` vectors.

The token stream matches the reference exactly: no compressor-alignment
pad is inserted (the reference pools image tokens across compressor-group
boundaries freely).
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import BatchFeature

from vllm.config.multimodal import BaseDummyOptions, ImageDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

# The checkpoint tokenizer's id for IMAGE_PLACEHOLDER (the config's
# ``image_token_id``). Every image-span position carries this id, exactly
# like the reference; the MoE router detects image tokens as the range
# [IMAGE_SENTINEL_BASE_ID, IMAGE_SENTINEL_BASE_ID + 5).
IMAGE_SENTINEL_BASE_ID = 129264


def image_sentinel_mask(token_ids: torch.Tensor) -> torch.Tensor:
    """Boolean mask for image-span positions."""
    return token_ids == IMAGE_SENTINEL_BASE_ID


def validate_image_sentinel_ids(tokenizer) -> None:
    """Check the image token id against the tokenizer."""
    image_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    if image_id != IMAGE_SENTINEL_BASE_ID:
        raise ValueError(
            f"Image placeholder {IMAGE_PLACEHOLDER!r} has id {image_id}, "
            f"expected {IMAGE_SENTINEL_BASE_ID} (the config's "
            "image_token_id); the DeepSeek-V4.1 vision path keys image "
            "routing and engram masking off this id."
        )


def llm_grid(best_height, best_width, patch_size, downsample_ratio):
    """Token grid the aligner produces from a patch grid of this pixel size."""
    return (
        math.ceil((best_height // patch_size) / downsample_ratio),
        math.ceil((best_width // patch_size) / downsample_ratio),
    )


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    """Largest aspect-preserving pixel size whose token grid still fits in
    max_n_token. Returns (best_height, best_width)."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:  # very tall: collapse to a single column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:  # very wide: collapse to a single row
        return cell, (max_n_token - 3) * cell
    beta = min(
        math.floor(max_w_float) * cell / width,
        math.floor(max_h_float) * cell / height,
    )
    return (
        math.floor(height * beta / patch_size) * patch_size,
        math.floor(width * beta / patch_size) * patch_size,
    )


def safe_resize(
    height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token
):
    """Shrink the pixel size until the image costs at most max_n_token LLM
    tokens."""
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, max_n_token
        )
        n_llm_h, n_llm_w = llm_grid(
            best_height, best_width, patch_size, downsample_ratio
        )
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def load_image(
    image: Image.Image,
    *,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
    min_pixels: int,
    max_wh_ratio: float | None,
):
    """Transform one PIL image into ViT patches.

    Same math as the reference ``load_image``, except the image is already
    decoded (vLLM supplies PIL images instead of a record dict).
    """
    p = patch_size
    image = image.convert("RGB")
    width, height = image.size
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height, width, best_height, best_width, p, downsample_ratio, max_n_token
    )
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if max_wh_ratio is not None and image.width >= max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        x.reshape(3, n_vit_h, p, n_vit_w, p)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, p, p)
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    """Reading-order span layout: one IMAGE_NEW_LINE per row."""
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return torch.tensor(types, dtype=torch.int64)


class DeepseekV4VLImageProcessor:
    """Per-image transform (the PIL-input equivalent of the reference
    ``load_image``)."""

    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__()
        self.patch_size = config.vision_patch_size
        self.downsample_ratio = config.vision_downsample_ratio
        self.max_n_token = config.vision_max_n_token
        self.min_pixels = config.vision_min_pixels
        self.max_wh_ratio = config.vision_max_wh_ratio

    def __call__(self, image: Image.Image):
        return load_image(
            image,
            patch_size=self.patch_size,
            downsample_ratio=self.downsample_ratio,
            max_n_token=self.max_n_token,
            min_pixels=self.min_pixels,
            max_wh_ratio=self.max_wh_ratio,
        )


class DeepseekV4VLProcessor:
    """Minimal stand-in for the HF processor of DeepSeek-V4.1 vision models.

    The official repository ships image preprocessing as plain functions in
    ``image_processor.py`` (no ``auto_map`` processor), so this class wraps
    their ports directly and the model loads without ``--trust-remote-code``.

    ``__call__`` returns a ``BatchFeature`` with one entry per image
    (flattened across images):

    - ``patches``: ``(sum(n_vit_h * n_vit_w), 3, p, p)`` bf16 ViT patches.
    - ``vit_grid``: ``(num_images, 2)`` int64 ``[n_vit_h, n_vit_w]``.
    - ``llm_grid``: ``(num_images, 2)`` int64 ``[n_llm_h, n_llm_w]``.
    - ``types``: concatenated per-image pad-free span roles
      (IMAGE_START/IMAGE/IMAGE_NEW_LINE/IMAGE_END); every span position
      carries ``image_token_id`` in the prompt's token ids.
    """

    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__()
        self.config = config
        self.image_processor = DeepseekV4VLImageProcessor(config)

    def __call__(
        self,
        text: str | None = None,
        images: Sequence[Image.Image] | None = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> BatchFeature:
        patches_list = []
        vit_grid = []
        llm_grid_list = []
        types_list = []
        for image in images or []:
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = self.image_processor(image)
            patches_list.append(patches)
            vit_grid.append((n_vit_h, n_vit_w))
            llm_grid_list.append((n_llm_h, n_llm_w))
            types_list.append(image_token_types(n_llm_h, n_llm_w))

        if not patches_list:
            return BatchFeature({})

        return BatchFeature(
            {
                "patches": torch.cat(patches_list),
                "vit_grid": torch.tensor(vit_grid, dtype=torch.int64),
                "llm_grid": torch.tensor(llm_grid_list, dtype=torch.int64),
                "types": torch.cat(types_list),
            }
        )


class DeepseekV4VLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> DeepseekV41Config:
        return self.ctx.get_hf_config(DeepseekV41Config)

    def get_hf_processor(self, **kwargs: object) -> DeepseekV4VLProcessor:
        if kwargs:
            raise ValueError(f"Unexpected processor kwargs: {sorted(kwargs)}")
        return DeepseekV4VLProcessor(self.get_hf_config())

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_hf_config().vision_max_n_token}

    def get_image_placeholder_token_id(self) -> int:
        token_id = self.get_tokenizer().convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        if token_id is None:
            raise ValueError(f"Token not found in tokenizer: {IMAGE_PLACEHOLDER}")
        return token_id

    def get_image_size_with_most_features(self) -> ImageSize:
        hf_config = self.get_hf_config()
        patch_size = hf_config.vision_patch_size
        downsample_ratio = hf_config.vision_downsample_ratio
        # A square maximizes the ViT patch count (area) within the token
        # budget; solve the budget-derived size directly to keep the dummy
        # image small.
        budget = hf_config.vision_max_n_token
        side = budget * patch_size * downsample_ratio
        best_h, best_w = solve_resize_ratio(
            side, side, patch_size, downsample_ratio, budget
        )
        return ImageSize(width=best_w, height=best_h)


class DeepseekV4VLDummyInputsBuilder(
    BaseDummyInputsBuilder[DeepseekV4VLProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        size = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
                overrides=cast(ImageDummyOptions | None, mm_options.get("image")),
            ),
        }


class DeepseekV4VLMultiModalProcessor(
    BaseMultiModalProcessor[DeepseekV4VLProcessingInfo]
):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        vit_grid = hf_inputs.get("vit_grid")
        llm_grid = hf_inputs.get("llm_grid")

        if vit_grid is None or llm_grid is None:
            empty = torch.empty(0, dtype=torch.long)
            patch_sizes = types_sizes = empty
        else:
            patch_sizes = vit_grid.prod(-1)
            n_llm_h, n_llm_w = llm_grid[:, 0], llm_grid[:, 1]
            types_sizes = n_llm_h * (n_llm_w + 1) + 2

        return {
            "patches": MultiModalFieldConfig.flat_from_sizes("image", patch_sizes),
            "vit_grid": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            "llm_grid": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            "types": MultiModalFieldConfig.flat_from_sizes(
                "image", types_sizes, keep_on_cpu=True
            ),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_token_id = self.info.get_image_placeholder_token_id()
        validate_image_sentinel_ids(self.info.get_tokenizer())

        def get_image_replacement(item_idx: int) -> PromptUpdateDetails:
            types: torch.Tensor = out_mm_kwargs["image"][item_idx]["types"].data
            # Every span position carries image_token_id; the roles live in
            # ``types``. All of them are embed positions (delimiters get the
            # learned vectors from embed_multimodal, not the embed table).
            full = [image_token_id] * types.numel()
            return PromptUpdateDetails.select_token_id(full, image_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_image_replacement,
            ),
        ]
