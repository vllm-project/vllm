# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Ownership and lookup of checkpoint-backed rotations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from .base import RotationTransform
from .composite_hadamard import CompositeHadamardRotation

_ROTATION_TYPES: dict[str, type[RotationTransform]] = {
    "composite_hadamard": CompositeHadamardRotation,
}


def build_rotation(config: Mapping[str, Any]) -> RotationTransform:
    """Build one concrete rotation from its tagged configuration."""
    rotation_type = config.get("type")
    if not isinstance(rotation_type, str) or not rotation_type:
        raise TypeError("rotation type must be a non-empty string")
    rotation_class = _ROTATION_TYPES.get(rotation_type)
    if rotation_class is None:
        raise ValueError(f"unsupported rotation type: {rotation_type!r}")
    return rotation_class.from_config(config)


class RotationRegistry(nn.ModuleDict):
    """The sole owner of checkpoint-backed rotation parameters."""

    def __init__(self, rotations: Mapping[str, Mapping[str, Any]]) -> None:
        super().__init__({tag: build_rotation(spec) for tag, spec in rotations.items()})

    def __getitem__(self, key: str) -> RotationTransform:
        return super().__getitem__(key)

    def validate_loaded(self) -> None:
        """Verify that every checkpoint-backed rotation was loaded."""
        for tag, rotation in self.items():
            assert isinstance(rotation, RotationTransform)
            rotation.validate_loaded(tag)


__all__ = ["RotationRegistry", "build_rotation"]
