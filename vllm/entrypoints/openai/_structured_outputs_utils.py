# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from vllm.sampling_params import StructuredOutputsParams


def merge_structured_outputs_params(
    base: StructuredOutputsParams | None,
    updates: dict[str, Any],
) -> StructuredOutputsParams:
    """Merge structured-output fields without relying on dataclasses.replace.

    `StructuredOutputsParams` is a pydantic dataclass, and `dataclasses.replace`
    does not type-check cleanly under mypy.

    This helper keeps the merge logic explicit and mypy-friendly.
    """

    if base is None:
        return StructuredOutputsParams(**updates)

    merged = dict(base.__dict__)
    merged.update(updates)
    return StructuredOutputsParams(**merged)
