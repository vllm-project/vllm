# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from typing import TypeVar

import vllm.platforms as platforms
from vllm.platforms import CpuArchEnum, PlatformEnum

PlatformKey = tuple[PlatformEnum, CpuArchEnum | None]
_ConfigT = TypeVar("_ConfigT")


def select_platform_config(
    *,
    default: _ConfigT,
    overrides: Mapping[PlatformKey, _ConfigT],
) -> _ConfigT:
    """Select platform-specific Triton tuning values, such as block size."""
    current_platform = platforms.current_platform
    architecture = (
        current_platform.get_cpu_architecture() if current_platform.is_cpu() else None
    )
    platform = current_platform._enum
    return overrides.get(
        (platform, architecture),
        overrides.get((platform, None), default),
    )
