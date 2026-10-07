# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared wire types for processed multimodal inputs."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StrictBool, model_validator
from typing_extensions import TypedDict


class MultiModalKvInfo(TypedDict):
    """Compatibility information for continuing from decoder KV, not transport."""

    version: Literal[1]
    model_fingerprint: Annotated[str, Field(min_length=1)]
    position_state: Literal["sequence"]


class PlaceholderRangeInfo(BaseModel):
    """Serializable placeholder location for a single multi-modal item."""

    offset: int = Field(ge=0, strict=True)
    """Start index of the placeholder tokens in the prompt."""

    length: int = Field(gt=0, strict=True)
    """Number of placeholder tokens."""

    is_embed: list[StrictBool] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """Embedding-token mask within the placeholder span, when sparse."""

    @model_validator(mode="after")
    def _validate_mask(self) -> "PlaceholderRangeInfo":
        if self.is_embed is not None and len(self.is_embed) != self.length:
            raise ValueError("is_embed must have the same length as the placeholder")
        return self


class MultiModalFeatures(BaseModel):
    """Lightweight multimodal metadata produced by the render step.

    Carries hashes (for cache lookup / identification) and placeholder
    positions so the downstream `/generate` service knows *where* in
    the token sequence each multimodal item lives.
    """

    mm_hashes: dict[str, list[str]]
    """Per-modality item hashes, e.g. `{"image": ["abc", "def"]}`."""

    mm_placeholders: dict[str, list[PlaceholderRangeInfo]]
    """Per-modality placeholder ranges in the token sequence."""

    kwargs_data: dict[str, list[str | None]] | None = None
    """Per-modality serialized tensor data.

    Each value is a list parallel to `mm_hashes[modality]`.  A `str`
    entry is a base64-encoded `MultiModalKwargsItem`; `None` means
    the item should be resolved from cache.  The entire field is
    `None` for metadata-only (cache-hit) responses.
    """

    mm_metadata: dict[str, list[str | None]] | None = None
    """Per-modality serialized metadata for disaggregated prefill.

    Each value is a list parallel to `mm_hashes[modality]`. A `str`
    entry is a base64-encoded `MultiModalKwargsItem` containing only
    placeholder-metadata and `keep_on_cpu` fields. `None` means that
    the metadata is unavailable for that item. Prefill can use this
    instead of `kwargs_data` only when `ec_transfer_params` is also
    set, or `requires_kv` explicitly requires decoder KV coverage. The
    engine must also support the model's continuation state.
    """

    requires_kv: StrictBool = Field(default=False, exclude_if=lambda value: not value)
    """All items require decoder KV coverage instead of encoder computation.

    This is a requirement, not proof of KV availability. The scheduler checks
    actual coverage, including after preemption and failed loads. It is valid
    with local KV and does not require KV transfer parameters.
    """

    kv_handoff: MultiModalKvInfo | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    """Source compatibility information, required when `requires_kv` is true."""

    @model_validator(mode="after")
    def _validate_kv_requirement(self) -> "MultiModalFeatures":
        if not self.requires_kv:
            return self
        if self.kv_handoff is None:
            raise ValueError(
                "requires_kv requires kv_handoff compatibility information"
            )
        if not self.mm_hashes or any(
            not hashes or any(not h for h in hashes)
            for hashes in self.mm_hashes.values()
        ):
            raise ValueError("requires_kv requires nonempty multimodal hashes")
        if self.kwargs_data and any(
            item is not None for items in self.kwargs_data.values() for item in items
        ):
            raise ValueError("requires_kv does not accept full kwargs_data")
        return self

    @model_validator(mode="after")
    def _validate_parallel_fields(self) -> "MultiModalFeatures":
        modalities = set(self.mm_hashes)
        if set(self.mm_placeholders) != modalities:
            raise ValueError(
                "mm_hashes and mm_placeholders must use the same modalities"
            )
        if self.kwargs_data is not None and set(self.kwargs_data) != modalities:
            raise ValueError("kwargs_data must use the same modalities as mm_hashes")
        if self.mm_metadata is not None and set(self.mm_metadata) != modalities:
            raise ValueError("mm_metadata must use the same modalities as mm_hashes")

        flattened_ranges: list[tuple[int, int]] = []
        for modality in modalities:
            num_hashes = len(self.mm_hashes[modality])
            num_placeholders = len(self.mm_placeholders[modality])
            if num_hashes != num_placeholders:
                raise ValueError(
                    f"{modality} mm_hashes and mm_placeholders must have "
                    "the same length"
                )
            if (
                self.kwargs_data is not None
                and len(self.kwargs_data[modality]) != num_hashes
            ):
                raise ValueError(
                    f"{modality} kwargs_data and mm_hashes must have the same length"
                )
            if (
                self.mm_metadata is not None
                and len(self.mm_metadata[modality]) != num_hashes
            ):
                raise ValueError(
                    f"{modality} mm_metadata and mm_hashes must have the same length"
                )
            flattened_ranges.extend(
                (placeholder.offset, placeholder.offset + placeholder.length)
                for placeholder in self.mm_placeholders[modality]
            )

        flattened_ranges.sort()
        for (offset, end), (next_offset, _) in zip(
            flattened_ranges, flattened_ranges[1:]
        ):
            if next_offset < end:
                raise ValueError(
                    "mm_placeholders must be globally non-overlapping and sorted"
                )
        return self
