# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared data structures for GPU discovery and bounded layout sweeps."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class GPUDeviceInfo:
    index: int
    name: str
    total_memory_bytes: int
    free_memory_bytes: int


@dataclass(frozen=True, slots=True)
class GPUHardwareInfo:
    devices: tuple[GPUDeviceInfo, ...]

    @property
    def gpu_count(self) -> int:
        return len(self.devices)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GPULayoutPolicy:
    """Backend capability limits for the shared single-node GPU planner."""

    supported_tp: tuple[int, ...] = (1, 2, 4, 8)
    allow_dense_dp: bool = True
    device_label: str = "GPU"
    title: str = "GPU"
    memory_label: str = "GPU"
