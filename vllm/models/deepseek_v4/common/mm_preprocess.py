# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multimodal preprocessing for the DeepSeek-V4 vision variants
(DeepSeek-V4-Flash-Vision-Exp).

The image transform and sentinel-block construction are ported from the
official repository's ``image_processor.py`` so that token counts bit-match
the reference. Each ``<｜deepseek_image｜>`` placeholder in the prompt expands
to a variable-length block of sentinel tokens; only positions with
``type == IMAGE`` receive vision embeddings, the other sentinels are looked
up from learned embedding vectors in the model.

Unlike the reference (which uses out-of-vocab ids ``vocab_size + type``),
the sentinel block borrows five consecutive reserved tokenizer tokens
(``<|place_holder_mm_span_0431|>`` .. ``_0435|>``): they are special tokens
the tokenizer never emits from plain text, so the ids stay in-vocabulary and
work with stock token validation, logprobs and detokenization. The ids are
pure markers — every sentinel position's embedding is overwritten with
vision/sentinel vectors before the decoder layers see it, exactly like the
reference's out-of-vocab scheme.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import BatchFeature
from typing_extensions import assert_never

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
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates,
    MultiModalPromptUpdatesApplyResult,
    PlaceholderFeaturesInfo,
    UpdateMode,
    _plan_prompt_updates,
)
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

# Sentinel roles borrow five consecutive ``<|place_holder_mm_span_XXXX|>``
# tokens (reserved special tokens, never emitted from plain text). The order
# must match IMAGE_START..IMAGE_END above so that
# ``id == IMAGE_SENTINEL_BASE_ID + type``.
IMAGE_SENTINEL_BASE_ID = 129257
IMAGE_SENTINEL_TOKEN_NAMES = (
    "<|place_holder_mm_span_0431|>",  # IMAGE_START
    "<|place_holder_mm_span_0432|>",  # IMAGE_PAD
    "<|place_holder_mm_span_0433|>",  # IMAGE
    "<|place_holder_mm_span_0434|>",  # IMAGE_NEW_LINE
    "<|place_holder_mm_span_0435|>",  # IMAGE_END
)


def image_sentinel_mask(token_ids: torch.Tensor) -> torch.Tensor:
    """Boolean mask for image-block sentinel positions (in-vocab ids)."""
    return (token_ids >= IMAGE_SENTINEL_BASE_ID) & (
        token_ids < IMAGE_SENTINEL_BASE_ID + len(IMAGE_SENTINEL_TOKEN_NAMES)
    )


def validate_image_sentinel_ids(tokenizer) -> None:
    """Check the borrowed sentinel ids against the tokenizer."""
    for i, name in enumerate(IMAGE_SENTINEL_TOKEN_NAMES):
        token_id = tokenizer.convert_tokens_to_ids(name)
        if token_id != IMAGE_SENTINEL_BASE_ID + i:
            raise ValueError(
                f"Image sentinel token {name!r} has id {token_id}, expected "
                f"{IMAGE_SENTINEL_BASE_ID + i}; the DeepSeek-V4 vision "
                "sentinel block requires these consecutive reserved ids."
            )


def grid_tokens(best_height, best_width, patch_size, downsample_ratio):
    """Number of LLM tokens the aligner grid occupies (N-layout, including
    row/align padding)."""
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(
    height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token
):
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget
        )
        budget -= 1
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


def build_image_block(n_llm_h: int, n_llm_w: int, start_pos: int):
    """Builds the N-layout token types (final order) and the aligner-row order
    for IMAGE slots."""
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
        + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = torch.arange(rows * row_len).view(rows // 2, 2, row_len)
    order = order.transpose(1, 2).reshape(-1)
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(
        n_llm_h * n_llm_w
    ).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START]),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END]),
        ]
    )
    return types, perm


def build_image_block_pad_free(n_llm_h: int, n_llm_w: int):
    """``build_image_block`` without the position-dependent compressor pad.

    ``start_pos`` is chosen so that ``compress_pad == 0``; the pad is instead
    prepended when the block is spliced into the final prompt, where its
    position is known.
    """
    return build_image_block(n_llm_h, n_llm_w, COMPRESS_PAD_TO - 1)


class DeepseekV4VLImageProcessor:
    """Per-image transform (the PIL-input equivalent of the reference
    ``load_image``)."""

    def __init__(self, config: DeepseekV4Config) -> None:
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
    """Minimal stand-in for the HF processor of DeepSeek-V4 vision models.

    The official repository ships image preprocessing as plain functions in
    ``image_processor.py`` (no ``auto_map`` processor), so this class wraps
    their ports directly and the model loads without ``--trust-remote-code``.

    ``__call__`` returns a ``BatchFeature`` with one entry per image
    (flattened across images):

    - ``patches``: ``(sum(n_vit_h * n_vit_w), 3, p, p)`` bf16 ViT patches.
    - ``vit_grid``: ``(num_images, 2)`` int64 ``[n_vit_h, n_vit_w]``.
    - ``llm_grid``: ``(num_images, 2)`` int64 ``[n_llm_h, n_llm_w]``.
    - ``perm``: concatenated per-image ``(n_llm_h * n_llm_w,)`` int64 index
      selecting aligner outputs into the final N-layout order.
    - ``types``: concatenated per-image pad-free sentinel block types;
      ``block ids = IMAGE_SENTINEL_BASE_ID + types``.
    """

    def __init__(self, config: DeepseekV4Config) -> None:
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
        llm_grid = []
        perm_list = []
        types_list = []
        for image in images or []:
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = self.image_processor(image)
            types, perm = build_image_block_pad_free(n_llm_h, n_llm_w)
            patches_list.append(patches)
            vit_grid.append((n_vit_h, n_vit_w))
            llm_grid.append((n_llm_h, n_llm_w))
            perm_list.append(perm)
            types_list.append(types)

        if not patches_list:
            return BatchFeature({})

        return BatchFeature(
            {
                "patches": torch.cat(patches_list),
                "vit_grid": torch.tensor(vit_grid, dtype=torch.int64),
                "llm_grid": torch.tensor(llm_grid, dtype=torch.int64),
                "perm": torch.cat(perm_list),
                "types": torch.cat(types_list),
            }
        )


class DeepseekV4VLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> DeepseekV4Config:
        return self.ctx.get_hf_config(DeepseekV4Config)

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
        # ``safe_resize`` reserves COMPRESS_PAD_TO - 1 tokens of
        # vision_max_n_token for the compressor-alignment pad, so the full
        # sentinel block is bounded by vision_max_n_token; the margin is kept
        # in case that reservation changes.
        return {
            "image": self.get_hf_config().vision_max_n_token + COMPRESS_PAD_TO - 1,
        }

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
        budget = hf_config.vision_max_n_token - (COMPRESS_PAD_TO - 1)
        side = budget * patch_size * downsample_ratio
        _, _, best_h, best_w, _ = solve_resize_ratio(
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
            patch_sizes = perm_sizes = types_sizes = empty
        else:
            patch_sizes = vit_grid.prod(-1)
            perm_sizes = llm_grid.prod(-1)
            n_llm_h, n_llm_w = llm_grid[:, 0], llm_grid[:, 1]
            # Pad-free block length; same formula as ``grid_tokens`` given
            # the LLM grid.
            types_sizes = (
                n_llm_h * (n_llm_w + 1)
                + 2
                + (n_llm_h % 2) * (n_llm_w + 1)
                + (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
            )

        return {
            "patches": MultiModalFieldConfig.flat_from_sizes("image", patch_sizes),
            "vit_grid": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            "llm_grid": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            "perm": MultiModalFieldConfig.flat_from_sizes("image", perm_sizes),
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
        image_embed_id = IMAGE_SENTINEL_BASE_ID + IMAGE

        def get_image_replacement(item_idx: int) -> PromptUpdateDetails:
            types: torch.Tensor = out_mm_kwargs["image"][item_idx]["types"].data
            full = (IMAGE_SENTINEL_BASE_ID + types).tolist()
            return PromptUpdateDetails.select_token_id(full, image_embed_id)

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_image_replacement,
            ),
        ]

    def _apply_token_matches_with_placeholders(
        self,
        token_ids: list[int],
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> tuple[
        list[int],
        MultiModalPromptUpdatesApplyResult,
        Mapping[str, list[PlaceholderFeaturesInfo]],
    ]:
        # Same as the base implementation, except that each image block gets
        # its compressor-alignment pad (``3 - start_pos % 4`` IMAGE_PAD
        # sentinels) prepended while splicing: the pad depends on the block's
        # final position in the prompt, which is unknown when the (cacheable)
        # prompt updates are built.
        matched_updates, result = _plan_prompt_updates(token_ids, mm_prompt_updates)
        placeholders: dict[str, list[PlaceholderFeaturesInfo]] = {
            modality: [] for modality in mm_prompt_updates
        }

        pad_id = IMAGE_SENTINEL_BASE_ID + IMAGE_PAD

        new_token_ids = list[int]()
        prev_end_idx = 0
        for matched_update in matched_updates:
            update = matched_update.update
            match = matched_update.match

            if update.mode == UpdateMode.INSERT:
                end_idx_to_insert = match.end_idx
            elif update.mode == UpdateMode.REPLACE:
                end_idx_to_insert = match.start_idx
            else:
                assert_never(update.mode)

            new_token_ids.extend(token_ids[prev_end_idx:end_idx_to_insert])
            start_idx = len(new_token_ids)

            tokens = list(update.content.full)
            if tokens and update.modality == "image":
                compress_pad = COMPRESS_PAD_TO - 1 - start_idx % COMPRESS_PAD_TO
                tokens = [pad_id] * compress_pad + tokens

            if tokens:
                content_is_embed = update.content.is_embed
                is_embed = (
                    content_is_embed(tokens) if content_is_embed is not None else None
                )
                placeholders[update.modality].append(
                    PlaceholderFeaturesInfo(
                        modality=update.modality,
                        item_idx=update.item_idx,
                        start_idx=start_idx,
                        tokens=tokens,
                        is_embed=is_embed,
                    )
                )
                new_token_ids.extend(tokens)

            prev_end_idx = match.end_idx

        new_token_ids.extend(token_ids[prev_end_idx:])
        return new_token_ids, result, placeholders
