# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utility helpers shared across OpenAI-compatible streaming code."""

from __future__ import annotations

from typing import Any


def _is_empty_content_only_delta(choice: Any) -> bool:
    """Return True if ``choice`` has a delta exactly {"content": ""}."""

    delta = getattr(choice, "delta", None)
    if delta is None:
        return False

    delta_payload = delta.model_dump(exclude_none=True, exclude_unset=True)
    return (delta_payload.get("content") == ""
            and len(delta_payload) == 1)

