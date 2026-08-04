# SPDX-License-Identifier: Apache-2.0
"""XPU ops backend: bind the SYCL ops over the torch baseline.

:class:`XpuDeviceOps` calls :meth:`bind_native` in :meth:`ensure_native`
to layer the SYCL extension on top of the torch baseline.  If the extension
is not built, a warning is logged and the instance stays on the torch
fallback.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps

logger = init_logger(__name__)


class XpuDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "xpu"

    def ensure_native(self) -> None:
        if self._native_bound:
            return
        self._native_bound = True  # set early to prevent repeated attempts
        try:
            # First Party
            import lmcache.xpu_ops as sycl
        except ImportError:
            logger.warning(
                "lmcache.xpu_ops not built; XpuDeviceOps stays on the "
                "torch baseline for all ops."
            )
            return
        self.bind_native(sycl)
