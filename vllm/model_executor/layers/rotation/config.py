# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Configuration model for checkpoint-backed rotation transforms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RotationApplication:
    """A rotation attached to one or more model modules."""

    tag: str
    targets: tuple[str, ...]
    mode: str
    inverse: bool


@dataclass(frozen=True)
class RotationConfig:
    """Validated rotation definitions and their application points."""

    rotations: Mapping[str, Mapping[str, Any]]
    applications: tuple[RotationApplication, ...]


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def parse_rotation_config(config: Any) -> RotationConfig | None:
    """Parse and validate ``hf_config.rotation_config``."""
    raw = getattr(config, "rotation_config", None)
    if raw is None and isinstance(config, Mapping):
        raw = config.get("rotation_config")
    if raw is None:
        return None

    raw = require_mapping(raw, "rotation_config")
    raw_rotations = require_mapping(raw.get("rotations"), "rotation_config.rotations")
    rotations: dict[str, Mapping[str, Any]] = {}
    for raw_tag, raw_spec in raw_rotations.items():
        tag = str(raw_tag)
        if tag in rotations:
            raise ValueError(f"duplicate rotation tag: {tag}")
        spec = require_mapping(raw_spec, f"rotation_config.rotations[{tag!r}]")
        rotations[tag] = spec

    raw_apply = raw.get("apply")
    if not isinstance(raw_apply, list):
        raise TypeError("rotation_config.apply must be a list")
    applications: list[RotationApplication] = []
    claimed_targets: set[str] = set()
    for index, raw_application in enumerate(raw_apply):
        application = require_mapping(
            raw_application, f"rotation_config.apply[{index}]"
        )
        tag = str(application["tag"])
        if tag not in rotations:
            raise ValueError(
                f"rotation_config.apply[{index}] refers to unknown tag {tag}"
            )
        mode = application["mode"]
        inverse = application.get("inverse")
        if not isinstance(inverse, bool):
            raise TypeError(f"rotation_config.apply[{index}].inverse must be a boolean")
        raw_targets = application.get("targets")
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ValueError(
                f"rotation_config.apply[{index}].targets must be a non-empty list"
            )
        targets: list[str] = []
        for target in raw_targets:
            if not isinstance(target, str) or not target:
                raise TypeError("rotation targets must be non-empty strings")
            if target in claimed_targets:
                raise ValueError(f"rotation target {target!r} is duplicated")
            claimed_targets.add(target)
            targets.append(target)
        applications.append(RotationApplication(tag, tuple(targets), mode, inverse))

    if not rotations:
        raise ValueError("rotation_config.rotations must not be empty")
    if not applications:
        raise ValueError("rotation_config.apply must not be empty")
    return RotationConfig(rotations, tuple(applications))


__all__ = [
    "RotationApplication",
    "RotationConfig",
    "parse_rotation_config",
]
