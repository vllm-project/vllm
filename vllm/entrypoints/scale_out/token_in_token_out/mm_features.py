# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for render-time multimodal feature extraction and generate input."""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import cast

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
from vllm.multimodal.inputs import MultiModalKwargsItem


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


def extract_mm_features(
    engine_input: EngineInput,
    *,
    metadata_fields_for: Callable[[str], Collection[str]] | None = None,
) -> MultiModalFeatures | None:
    """Extract multimodal features from a rendered engine prompt.

    Returns ``None`` for text-only prompts. ``mm_metadata`` keeps the
    intersection of processed kwargs that prefill needs after EC transfer:
    fields declared as embedding metadata, plus fields marked
    ``keep_on_cpu`` (for example M-RoPE grid dims).
    """
    if engine_input.get("type") != "multimodal":
        return None

    mm_engine_input = cast(MultiModalInput, engine_input)
    mm_hashes: MultiModalHashes = mm_engine_input["mm_hashes"]
    raw_placeholders: MultiModalPlaceholders = mm_engine_input["mm_placeholders"]

    mm_placeholders = {
        modality: [
            PlaceholderRangeInfo(offset=p.offset, length=p.length) for p in ranges
        ]
        for modality, ranges in raw_placeholders.items()
    }

    kwargs_data: dict[str, list[str | None]] | None = None
    mm_metadata: dict[str, list[str | None]] | None = None
    if raw_mm_kwargs := mm_engine_input.get("mm_kwargs"):
        kwargs_data = {}
        metadata_by_modality: dict[str, list[str | None]] = {}
        for modality, items in raw_mm_kwargs.items():
            kwargs_data[modality] = [
                encode_mm_kwargs_item(item) if item is not None else None
                for item in items
            ]

            declared = (
                set(metadata_fields_for(modality))
                if metadata_fields_for is not None
                else set()
            )
            metadata_items: list[str | None] = []
            has_metadata = False
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
                has_metadata = has_metadata or bool(metadata_item)
                metadata_items.append(
                    encode_mm_kwargs_item(metadata_item) if metadata_item else None
                )

            if has_metadata:
                metadata_by_modality[modality] = metadata_items

        if metadata_by_modality:
            mm_metadata = metadata_by_modality

    return MultiModalFeatures(
        mm_hashes=mm_hashes,
        mm_placeholders=mm_placeholders,
        kwargs_data=kwargs_data,
        mm_metadata=mm_metadata,
    )
