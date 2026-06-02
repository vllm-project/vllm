# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from pydantic import BaseModel


class ComputeCapability(BaseModel):
    major: int
    minor: int


class DeviceInfo(BaseModel):
    rank: int
    name: str
    total_memory_bytes: int
    compute_capability: ComputeCapability | None = None
    num_compute_units: int


class DevicesResponse(BaseModel):
    devices: list[DeviceInfo]
