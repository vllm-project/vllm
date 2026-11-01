# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Config-driven, checkpoint-backed rotation transforms."""

from .adapter import RotationAdapter
from .base import RotationTransform
from .composite_hadamard import CompositeHadamardRotation
from .config import RotationApplication, RotationConfig, parse_rotation_config
from .norm import PreRotationRMSNorm
from .registry import RotationRegistry, build_rotation
from .replacement import bind_rotation_transforms, prepare_rotation_transforms

__all__ = [
    "CompositeHadamardRotation",
    "PreRotationRMSNorm",
    "RotationAdapter",
    "RotationApplication",
    "RotationConfig",
    "RotationRegistry",
    "RotationTransform",
    "bind_rotation_transforms",
    "build_rotation",
    "parse_rotation_config",
    "prepare_rotation_transforms",
]
