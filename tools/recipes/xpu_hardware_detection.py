# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Discover XPU devices visible to the current vLLM process/container.

Hardware detection records facts only; tuning belongs in sweep/runtime_tuning.py.
Import torch lazily so plain recipe conversion does not require an XPU runtime.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class XPUDeviceInfo:
    index: int
    name: str
    total_memory_bytes: int
    free_memory_bytes: int


@dataclass(frozen=True, slots=True)
class XPUHardwareInfo:
    devices: tuple[XPUDeviceInfo, ...]
    ze_affinity_mask: str | None

    @property
    def gpu_count(self) -> int:
        return len(self.devices)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_xpu_hardware() -> XPUHardwareInfo:
    """Report visible XPU count, names and device memory through PyTorch.

    Indices are process-relative; respect Level Zero/PyTorch visibility rather
    than guessing physical GPU count from PCI inventory or xpu-smi.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "XPU detection needs PyTorch with Intel XPU support"
        ) from exc

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError(
            "Intel XPU unavailable in this process/container; verify drivers, "
            "device passthrough and the XPU-enabled PyTorch installation"
        )

    devices = []
    for index in range(torch.xpu.device_count()):
        properties = torch.xpu.get_device_properties(index)
        # mem_get_info returns (free_bytes, total_bytes) for the selected device.
        try:
            free, total = torch.xpu.mem_get_info(index)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot query XPU {index} memory; refusing to guess VRAM"
            ) from exc
        free, total = int(free), int(total)
        if total <= 0 or free < 0 or free > total:
            raise RuntimeError(
                f"XPU {index} reported invalid memory: free={free}, total={total}"
            )
        devices.append(
            XPUDeviceInfo(
                index=index,
                name=properties.name,
                total_memory_bytes=total,
                free_memory_bytes=free,
            )
        )

    if not devices:
        raise RuntimeError("No Intel XPU devices visible to this process")

    return XPUHardwareInfo(
        devices=tuple(devices),
        ze_affinity_mask=os.environ.get("ZE_AFFINITY_MASK"),
    )


if __name__ == "__main__":
    print(json.dumps(detect_xpu_hardware().to_dict(), indent=2))
