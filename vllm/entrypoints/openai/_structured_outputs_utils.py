# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Helpers for working with StructuredOutputsParams.

We use `pydantic.dataclasses.dataclass` for StructuredOutputsParams, which is
runtime-compatible with `dataclasses.replace`, but mypy does not treat it as a
standard dataclass. Keep the merge/update logic in one place to avoid repeated
`type: ignore` at call sites.
"""

from __future__ import annotations

from typing import Any

from vllm.sampling_params import StructuredOutputsParams


def merge_structured_outputs_params(
    base: StructuredOutputsParams,
    updates: dict[str, Any],
) -> StructuredOutputsParams:
    """Return a new StructuredOutputsParams with `updates` applied.

    Preserves non-init internal fields (e.g. backend selection flags).
    """

    fields = base.__dataclass_fields__  # type: ignore[attr-defined]

    init_keys = [name for name, f in fields.items() if f.init]
    non_init_keys = [name for name, f in fields.items() if not f.init]

    merged: dict[str, Any] = {k: getattr(base, k) for k in init_keys}
    merged.update(updates)

    new_params = StructuredOutputsParams(**merged)

    for k in non_init_keys:
        setattr(new_params, k, getattr(base, k))

    return new_params
