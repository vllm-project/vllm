# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Structured recipe conversion outcomes for automation consumers.

The converter owns all feasibility decisions; CI only interprets these results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RecipeSkipError(ValueError):
    """Recipe deployment cannot be executed with the selected resources."""

    def __init__(self, reason_code: str, reason: str, **details: Any):
        super().__init__(reason)
        self.reason_code = reason_code
        self.details = details


def write_validation_result(
    output: str | None,
    status: str,
    *,
    reason_code: str | None = None,
    reason: str | None = None,
    **details: Any,
) -> None:
    if not output:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"schema_version": 1, "status": status}
    if reason_code:
        payload["reason_code"] = reason_code
    if reason:
        payload["reason"] = reason
    payload.update(details)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
