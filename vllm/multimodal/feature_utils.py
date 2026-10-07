# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for render-time multimodal feature extraction and generate input."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from typing import cast

import torch

from vllm.inputs import (
    EngineInput,
    MultiModalHashes,
    MultiModalInput,
    MultiModalPlaceholders,
    mm_input,
)
from vllm.multimodal.feature_serde import (
    decode_mm_kwargs_item,
    encode_mm_kwargs_item,
)
from vllm.multimodal.feature_specs import (
    MultiModalFeatures,
    PlaceholderRangeInfo,
)
from vllm.multimodal.inputs import (
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalKwargsOptionalItems,
    PlaceholderRange,
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
            PlaceholderRangeInfo(
                offset=p.offset,
                length=p.length,
                is_embed=p.is_embed.tolist() if p.is_embed is not None else None,
            )
            for p in ranges
        ]
        for modality, ranges in raw_placeholders.items()
    }


def extract_mm_features(
    engine_input: EngineInput,
    *,
    metadata_fields_for: Callable[[str], Collection[str]] | None = None,
    include_mm_kwargs: bool = True,
) -> MultiModalFeatures | None:
    """Extract multimodal features from a rendered engine prompt.

    Returns ``None`` for text-only prompts. ``mm_metadata`` keeps the
    intersection of processed kwargs that prefill needs after EC transfer:
    fields declared as embedding metadata, plus fields marked
    ``keep_on_cpu`` (for example M-RoPE grid dims). With
    ``include_mm_kwargs=False`` neither is serialized.
    """
    if engine_input.get("type") != "multimodal":
        return None

    mm_engine_input = cast(MultiModalInput, engine_input)
    mm_hashes: MultiModalHashes = mm_engine_input["mm_hashes"]
    mm_placeholders = placeholder_ranges_from_engine_input(engine_input)
    assert mm_placeholders is not None

    kwargs_data: dict[str, list[str | None]] | None = None
    mm_metadata: dict[str, list[str | None]] | None = None
    if include_mm_kwargs:
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
        requires_kv=mm_engine_input.get("mm_requires_kv", False),
        kv_handoff=mm_engine_input.get("mm_kv_handoff"),
    )


def engine_input_from_features(
    token_ids: list[int],
    features: MultiModalFeatures,
    *,
    cache_salt: str | None = None,
) -> MultiModalInput:
    """Restore processed input for both Python and Generate API callers."""
    if not token_ids or any(type(t) is not int or t < 0 for t in token_ids):
        raise ValueError("features require valid expanded token_ids")
    mm_placeholders = {}
    for modality, ranges in features.mm_placeholders.items():
        if any(p.offset + p.length > len(token_ids) for p in ranges):
            raise ValueError(
                "mm_placeholders must remain within the token_ids sequence"
            )
        mm_placeholders[modality] = [
            PlaceholderRange(
                offset=p.offset,
                length=p.length,
                is_embed=(
                    torch.tensor(p.is_embed, dtype=torch.bool)
                    if p.is_embed is not None
                    else None
                ),
            )
            for p in ranges
        ]
    result = mm_input(
        prompt_token_ids=list(token_ids),
        mm_kwargs=MultiModalKwargsItems(mm_kwargs_from_features(features)),
        mm_hashes=features.mm_hashes,
        mm_placeholders=mm_placeholders,
        cache_salt=cache_salt,
    )
    if features.requires_kv:
        result["mm_requires_kv"] = True
    if features.kv_handoff is not None:
        result["mm_kv_handoff"] = features.kv_handoff
    return result
