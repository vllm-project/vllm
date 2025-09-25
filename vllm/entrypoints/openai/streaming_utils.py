# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utility helpers shared across OpenAI-compatible streaming code."""

from __future__ import annotations

from typing import Any

__all__ = ["EmptyDeltaTracker", "_is_empty_content_only_delta"]

_NO_INDEX_SENTINEL = object()


def _is_empty_content_only_delta(choice: Any) -> bool:
    """Return True if ``choice`` has a delta exactly {"content": ""}."""

    delta = getattr(choice, "delta", None)
    if delta is None:
        return False

    delta_payload = delta.model_dump(exclude_none=True, exclude_unset=True)
    return (delta_payload.get("content") == ""
            and len(delta_payload) == 1)


class EmptyDeltaTracker:
    """Track empty deltas so repeated blanks can be filtered per-choice."""

    def __init__(self) -> None:
        self._seen_indexes: set[object] = set()

    def _index_key(self, choice: Any) -> object:
        index = getattr(choice, "index", None)
        return index if index is not None else _NO_INDEX_SENTINEL

    def should_suppress(self, choice: Any) -> bool:
        """Return True if the ``choice`` should be filtered out."""

        if not _is_empty_content_only_delta(choice):
            return False

        key = self._index_key(choice)
        if getattr(choice, "finish_reason", None) is not None:
            self._seen_indexes.add(key)
            return False

        if key in self._seen_indexes:
            return True

        self._seen_indexes.add(key)
        return False

