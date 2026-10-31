# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for render-time multimodal feature extraction and generate input."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from typing import TYPE_CHECKING, Any, cast

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
    mm_input,
)
from vllm.multimodal.inputs import (
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalKwargsOptionalItems,
    PlaceholderRange,
)

if TYPE_CHECKING:
    from vllm.multimodal.cache.base import BaseMultiModalProcessorCache


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


def validate_feature_bounds(
    token_ids: Sequence[int], features: MultiModalFeatures
) -> None:
    """Check placeholder ranges stay within the token-id sequence.

    Shared by the ``GenerateRequest`` parse-time validator and
    ``engine_input_from_features`` so the two paths cannot drift.
    """
    prompt_len = len(token_ids)
    for ranges in features.mm_placeholders.values():
        for placeholder in ranges:
            if placeholder.offset + placeholder.length > prompt_len:
                raise ValueError(
                    "mm_placeholders must remain within the token_ids sequence"
                )


def validate_metadata_only(
    features: MultiModalFeatures, ec_transfer_params: dict[str, Any] | None
) -> None:
    """Check metadata-only items are only used with EC transfer.

    A ``None`` entry in ``kwargs_data`` means the item's embeddings are
    expected to arrive through the encoder-cache connector; without
    ``ec_transfer_params`` the engine has no pixel values and would fail
    deep in engine core instead of returning a 400. Shared by the
    ``GenerateRequest`` parse-time validator and
    ``engine_input_from_features``.
    """
    if ec_transfer_params:
        return
    if not any(
        item is not None
        for items in (features.mm_metadata or {}).values()
        for item in items
    ):
        return
    kwargs_data = features.kwargs_data or {}
    for modality, metadata_items in (features.mm_metadata or {}).items():
        kwargs_items = kwargs_data.get(modality, [None] * len(metadata_items))
        if any(
            metadata is not None and kwargs is None
            for metadata, kwargs in zip(metadata_items, kwargs_items, strict=True)
        ):
            raise ValueError(
                "metadata-only multimodal items require ec_transfer_params"
            )


def engine_input_from_features(
    token_ids: Sequence[int],
    features: MultiModalFeatures,
    *,
    cache_salt: str | None = None,
    mm_cache: BaseMultiModalProcessorCache | None = None,
    ec_transfer_params: dict[str, Any] | None = None,
) -> MultiModalInput:
    """Build a multimodal engine input from forwarded render features.

    Single implementation shared by the generate endpoint and the chat
    reuse path, performing the validations the generate protocol applies
    to the same payload:

    1. Placeholder ranges remain within ``token_ids``
       (``validate_feature_bounds``).
    2. Metadata-only items require ``ec_transfer_params``
       (``validate_metadata_only``).
    3. Deserialized items pass the multimodal processor cache's handle
       authentication when a cache is configured. SHM cache handles are
       untrusted client input (cf. #59357): ``address``, ``monotonic_id``
       and ``signature`` must match what this process actually published.

    Raises ``ValueError`` on the first violation so callers can map it
    to an HTTP 400.

    Note: the ``GenerateRequest`` parse-time validators run these same
    two protocol checks when the request body is parsed, so the generate
    path validates twice (parse + here). That is intentional and cheap;
    do not remove one side — the chat path only has this gate.
    """
    validate_feature_bounds(token_ids, features)
    validate_metadata_only(features, ec_transfer_params)

    # Deserialize full tensor data and optional metadata-only data.
    mm_kwargs = mm_kwargs_from_features(features)
    if mm_cache is not None:
        for modality, items in mm_kwargs.items():
            for mm_hash, item in zip(
                features.mm_hashes[modality], items, strict=True
            ):
                if item is not None:
                    mm_cache.validate_input_item(item, mm_hash)

    # Convert PlaceholderRangeInfo → PlaceholderRange per modality.
    mm_placeholders: MultiModalPlaceholders = {
        modality: [
            PlaceholderRange(offset=p.offset, length=p.length) for p in ranges
        ]
        for modality, ranges in features.mm_placeholders.items()
    }

    return mm_input(
        prompt_token_ids=list(token_ids),
        mm_kwargs=MultiModalKwargsItems(mm_kwargs),
        mm_hashes=features.mm_hashes,
        mm_placeholders=mm_placeholders,
        cache_salt=cache_salt,
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
