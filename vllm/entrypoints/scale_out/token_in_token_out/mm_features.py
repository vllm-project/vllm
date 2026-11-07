# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for render-time multimodal feature extraction and generate input."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from typing import cast

import torch
from transformers import PreTrainedConfig

from vllm.entrypoints.generate.base.protocol import MMPayload
from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import (
    decode_mm_kwargs_item,
    encode_mm_kwargs_item,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    MultiModalFeatures,
    PlaceholderRangeInfo,
)
from vllm.inputs import (
    EngineInput,
    MultiModalHashes,
    MultiModalInput,
    MultiModalPlaceholders,
)
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalKwargsOptionalItems,
)


def merge_mm_kwargs_items(
    kwargs_item: MultiModalKwargsItem | None,
    metadata_item: MultiModalKwargsItem | None,
) -> MultiModalKwargsItem | None:
    """Merge full multimodal data with its metadata-only counterpart."""
    if kwargs_item is None:
        return metadata_item
    if metadata_item is None:
        return kwargs_item
    return MultiModalKwargsItem({**kwargs_item, **metadata_item})


def _encode_metadata_items(
    items: Sequence[MultiModalKwargsItem | None],
    *,
    declared: Collection[str],
) -> list[str | None]:
    """Serialize placeholder-metadata and ``keep_on_cpu`` fields per item."""
    metadata_items: list[str | None] = []
    for item in items:
        if item is None:
            metadata_items.append(None)
            continue

        metadata_item = MultiModalKwargsItem(
            {
                key: elem
                for key, elem in item.items()
                if elem.field.keep_on_cpu or key in declared
            }
        )
        metadata_items.append(
            encode_mm_kwargs_item(metadata_item) if metadata_item else None
        )
    return metadata_items


def _encode_mm_kwargs_with_metadata(
    raw_mm_kwargs: MultiModalKwargsOptionalItems,
    *,
    metadata_fields_for: Callable[[str], Collection[str]] | None = None,
) -> tuple[dict[str, list[str | None]], dict[str, list[str | None]] | None]:
    """Serialize full kwargs and their metadata-only subsets per modality."""
    kwargs_data: dict[str, list[str | None]] = {}
    metadata_by_modality: dict[str, list[str | None]] = {}

    for modality, items in raw_mm_kwargs.items():
        kwargs_data[modality] = [
            encode_mm_kwargs_item(item) if item is not None else None for item in items
        ]

        declared = (
            set(metadata_fields_for(modality))
            if metadata_fields_for is not None
            else set()
        )
        metadata_items = _encode_metadata_items(items, declared=declared)
        if any(item is not None for item in metadata_items):
            metadata_by_modality[modality] = metadata_items

    mm_metadata = metadata_by_modality or None
    return kwargs_data, mm_metadata


def mm_kwargs_from_features(
    features: MultiModalFeatures,
) -> dict[str, list[MultiModalKwargsItem | None]]:
    """Deserialize ``features`` into per-modality kwargs for ``mm_input``."""
    mm_kwargs: dict[str, list[MultiModalKwargsItem | None]] = {}
    kwargs_data = features.kwargs_data or {}
    mm_metadata = features.mm_metadata or {}
    for modality, hashes in features.mm_hashes.items():
        n = len(hashes)
        kwargs_items = [
            decode_mm_kwargs_item(item) if item is not None else None
            for item in kwargs_data.get(modality, [None] * n)
        ]
        metadata_items = [
            decode_mm_kwargs_item(item) if item is not None else None
            for item in mm_metadata.get(modality, [None] * n)
        ]
        mm_kwargs[modality] = [
            merge_mm_kwargs_items(kwargs_item, metadata_item)
            for kwargs_item, metadata_item in zip(
                kwargs_items, metadata_items, strict=True
            )
        ]
    return mm_kwargs


def validate_qwen_image_features(
    features: MultiModalFeatures,
    items: Sequence[MultiModalKwargsItem | None],
    config: PreTrainedConfig,
    *,
    allow_ec_metadata: bool,
    allow_shm_handles: bool,
) -> None:
    """Validate decoded Qwen2/2.5-VL image items before engine admission."""
    if config.model_type not in ("qwen2_vl", "qwen2_5_vl"):
        return

    vision = config.vision_config
    merge_size = vision.spatial_merge_size
    pixel_width = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    embed_width = config.get_text_config().hidden_size
    serialized_items = (features.kwargs_data or {}).get("image", [None] * len(items))
    for item, serialized, placeholder in zip(
        items, serialized_items, features.mm_placeholders["image"], strict=True
    ):
        # Cache-only entries and authenticated SHM references have no inline tensor.
        if item is None or (allow_shm_handles and "address" in item):
            continue

        grid_elem = item.get("image_grid_thw")
        if (
            grid_elem is None
            or not isinstance(grid_elem.field, MultiModalBatchedField)
            or not isinstance(grid := grid_elem.data, torch.Tensor)
            or grid.shape != (3,)
            or grid.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "image_grid_thw must be a batched integer tensor of size 3"
            )
        t, h, w = grid.tolist()
        if min(t, h, w) <= 0 or h % merge_size or w % merge_size:
            raise ValueError("image_grid_thw has invalid patch dimensions")

        patches = t * h * w
        embeds = patches // merge_size**2
        if placeholder.length != embeds:
            raise ValueError(
                "image_grid_thw does not match the image placeholder length"
            )

        keys = item.keys()
        if keys == {"image_grid_thw"} and serialized is None and allow_ec_metadata:
            continue
        if keys == {"image_grid_thw", "pixel_values"}:
            key, shape = "pixel_values", (patches, pixel_width)
        elif keys == {"image_grid_thw", "image_embeds"}:
            key, shape = "image_embeds", (embeds, embed_width)
        else:
            raise ValueError("Image features require pixel_values or image_embeds")

        elem = item[key]
        if (
            not isinstance(elem.field, MultiModalFlatField)
            or elem.field.dim != 0
            or not isinstance(elem.data, torch.Tensor)
            or elem.data.shape != shape
        ):
            raise ValueError(
                f"{key} must be a tensor of shape {shape} matching the grid"
            )


def placeholder_ranges_from_engine_input(
    engine_input: EngineInput,
) -> dict[str, list[PlaceholderRangeInfo]] | None:
    """Return per-modality placeholder ranges, or ``None`` for text-only prompts."""
    if engine_input.get("type") != "multimodal":
        return None

    raw_placeholders: MultiModalPlaceholders = cast(MultiModalInput, engine_input)[
        "mm_placeholders"
    ]
    return {
        modality: [
            PlaceholderRangeInfo(offset=p.offset, length=p.length) for p in ranges
        ]
        for modality, ranges in raw_placeholders.items()
    }


def extract_mm_features(
    engine_input: EngineInput,
    *,
    metadata_fields_for: Callable[[str], Collection[str]] | None = None,
    mm_payload: MMPayload = "tensors",
) -> MultiModalFeatures | None:
    """Extract multimodal features from a rendered engine prompt.

    Returns ``None`` for text-only prompts. ``mm_metadata`` keeps the
    intersection of processed kwargs that prefill needs after EC transfer:
    fields declared as embedding metadata, plus fields marked
    ``keep_on_cpu`` (for example M-RoPE grid dims). With
    ``mm_payload="none"`` neither is serialized.
    """
    if engine_input.get("type") != "multimodal":
        return None

    mm_engine_input = cast(MultiModalInput, engine_input)
    mm_hashes: MultiModalHashes = mm_engine_input["mm_hashes"]
    mm_placeholders = placeholder_ranges_from_engine_input(engine_input)
    assert mm_placeholders is not None

    kwargs_data: dict[str, list[str | None]] | None = None
    mm_metadata: dict[str, list[str | None]] | None = None
    if mm_payload == "tensors":
        raw_mm_kwargs = mm_engine_input.get("mm_kwargs")
        if raw_mm_kwargs:
            kwargs_data, mm_metadata = _encode_mm_kwargs_with_metadata(
                raw_mm_kwargs,
                metadata_fields_for=metadata_fields_for,
            )

    return MultiModalFeatures(
        mm_hashes=mm_hashes,
        mm_placeholders=mm_placeholders,
        kwargs_data=kwargs_data,
        mm_metadata=mm_metadata,
    )
