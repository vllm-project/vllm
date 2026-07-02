# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Protocol

from vllm.v1.kv_offload.base import OffloadKey

PinHandle = str


@dataclass(frozen=True)
class TransportEndpoint:
    """In-process transport endpoint owned by the primary tier."""

    name: str
    end_point: Any
    info: str = ""


@dataclass(frozen=True)
class MemDescriptor:
    """Transport-addressable memory span for a pinned KV block."""

    end_point_name: str
    mem_type: str
    addr: int
    size: int
    device_Id: int
    info: str


class PrimaryPinningAPI(Protocol):
    """Primary-tier capability shared with components that pin KV blocks."""

    def get_transport_endpoint(self) -> TransportEndpoint:
        ...

    def search_and_pin(
        self, keys: Collection[OffloadKey]
    ) -> tuple[PinHandle, dict[OffloadKey, MemDescriptor]] | None:
        ...

    def unpin(self, pin_handle: PinHandle) -> bool:
        ...
