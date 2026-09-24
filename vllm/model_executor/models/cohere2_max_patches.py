# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounds for Cohere2 Vision request ``max_patches`` overrides."""

from collections.abc import Mapping

from vllm.exceptions import VLLMValidationError


def iter_max_patches_values(
    mm_kwargs: Mapping[str, object],
) -> list[tuple[str, object]]:
    """Yield request ``max_patches`` from flat and image-scoped kwargs."""
    values: list[tuple[str, object]] = []
    if "max_patches" in mm_kwargs:
        values.append(("max_patches", mm_kwargs["max_patches"]))
    nested = mm_kwargs.get("images_kwargs")
    if isinstance(nested, Mapping) and "max_patches" in nested:
        values.append(("images_kwargs.max_patches", nested["max_patches"]))
    return values


def validate_cohere2_max_patches(
    mm_kwargs: Mapping[str, object],
    limit: int,
) -> None:
    """Reject request ``max_patches`` that are not ints in ``[1, limit]``."""
    if not isinstance(mm_kwargs, Mapping):
        raise VLLMValidationError(
            "mm_processor_kwargs must be a mapping",
            parameter="mm_processor_kwargs",
            value=mm_kwargs,
        )
    for name, value in iter_max_patches_values(mm_kwargs):
        if not isinstance(value, int) or isinstance(value, bool):
            raise VLLMValidationError(
                f"{name} must be a positive integer not greater than {limit}",
                parameter=name,
                value=value,
            )
        if value < 1 or value > limit:
            raise VLLMValidationError(
                f"{name} must be a positive integer not greater than {limit}",
                parameter=name,
                value=value,
            )
