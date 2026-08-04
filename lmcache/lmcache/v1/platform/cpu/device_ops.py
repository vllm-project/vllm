# SPDX-License-Identifier: Apache-2.0
"""CPU ops backend: the torch baseline, registered under ``device_type="cpu"``.

:class:`CpuDeviceOps` adds no overrides -- all ops are inherited from
:class:`DeviceOps` (the torch baseline) via MRO.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.v1.platform.base.device_ops import DeviceOps


class CpuDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "cpu"
