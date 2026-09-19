# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-memory layout primitives for EPLB expert weight backups."""

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ExpertSlotLocation:
    """Location and tensor metadata for one weight in a backup region."""

    addr: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if self.addr < 0:
            raise ValueError("Expert backup address must be non-negative.")
        if self.nbytes < 0:
            raise ValueError("Expert backup size must be non-negative.")
        if any(dim < 0 for dim in self.shape):
            raise ValueError("Expert backup tensor dimensions must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "addr": self.addr,
            "nbytes": self.nbytes,
            "shape": list(self.shape),
            "dtype": str(self.dtype).removeprefix("torch."),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExpertSlotLocation":
        dtype_name = value["dtype"]
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported expert backup dtype: {dtype_name}")
        return cls(
            addr=int(value["addr"]),
            nbytes=int(value["nbytes"]),
            shape=tuple(int(dim) for dim in value["shape"]),
            dtype=dtype,
        )


@dataclass(frozen=True)
class ExpertBackupDescriptor:
    """Wire-safe description of a node's registered expert backup region."""

    owner_node_rank: int
    backup_region_base: int
    weight_pointer_map: dict[str, ExpertSlotLocation]
    nixl_agent_metadata: bytes = b""

    def __post_init__(self) -> None:
        if self.owner_node_rank < 0:
            raise ValueError("Expert backup owner node rank must be non-negative.")
        if self.backup_region_base < 0:
            raise ValueError("Expert backup region base must be non-negative.")

    def to_bytes(self) -> bytes:
        payload = {
            "owner_node_rank": self.owner_node_rank,
            "backup_region_base": self.backup_region_base,
            "weight_pointer_map": {
                name: location.to_dict()
                for name, location in sorted(self.weight_pointer_map.items())
            },
            "nixl_agent_metadata": base64.b64encode(self.nixl_agent_metadata).decode(
                "ascii"
            ),
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> "ExpertBackupDescriptor":
        try:
            value = json.loads(payload)
            pointer_map = {
                name: ExpertSlotLocation.from_dict(location)
                for name, location in value["weight_pointer_map"].items()
            }
            metadata = base64.b64decode(
                value["nixl_agent_metadata"],
                validate=True,
            )
            return cls(
                owner_node_rank=int(value["owner_node_rank"]),
                backup_region_base=int(value["backup_region_base"]),
                weight_pointer_map=pointer_map,
                nixl_agent_metadata=metadata,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid expert backup descriptor.") from exc


@dataclass(frozen=True)
class ExpertBackupRegion:
    """A contiguous CPU byte buffer and its publishable descriptor."""

    buffer: torch.Tensor
    descriptor: ExpertBackupDescriptor

    def tensor(self, name: str) -> torch.Tensor:
        """Return a zero-copy typed view of a weight in this region."""
        location = self.descriptor.weight_pointer_map[name]
        offset = location.addr - self.descriptor.backup_region_base
        end = offset + location.nbytes
        if offset < 0 or end > self.buffer.numel():
            raise ValueError(
                f"Expert backup location for {name} is outside the region."
            )
        view = self.buffer[offset:end].view(location.dtype).view(location.shape)
        if view.numel() * view.element_size() != location.nbytes:
            raise ValueError(f"Expert backup metadata size mismatch for {name}.")
        return view


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def build_expert_backup_region(
    weights: Mapping[str, torch.Tensor],
    *,
    owner_node_rank: int,
    nixl_agent_metadata: bytes = b"",
    pin_memory: bool = False,
    alignment: int = 64,
) -> ExpertBackupRegion:
    """Copy named expert weights into one aligned, contiguous CPU region."""
    if not weights:
        raise ValueError("At least one expert weight is required.")
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("Expert backup alignment must be a positive power of two.")

    cpu_weights: dict[str, torch.Tensor] = {}
    offsets: dict[str, int] = {}
    total_nbytes = 0
    for name, weight in sorted(weights.items()):
        if not name:
            raise ValueError("Expert backup weight names must be non-empty.")
        if weight.layout != torch.strided:
            raise ValueError(f"Expert backup weight {name} must use strided layout.")
        cpu_weight = weight.detach().to(device="cpu").contiguous()
        offset = _align_up(total_nbytes, alignment)
        cpu_weights[name] = cpu_weight
        offsets[name] = offset
        total_nbytes = offset + cpu_weight.numel() * cpu_weight.element_size()

    buffer = torch.empty(
        total_nbytes,
        dtype=torch.uint8,
        device="cpu",
        pin_memory=pin_memory,
    )
    base_addr = buffer.data_ptr()
    pointer_map: dict[str, ExpertSlotLocation] = {}
    for name, weight in cpu_weights.items():
        offset = offsets[name]
        nbytes = weight.numel() * weight.element_size()
        buffer[offset : offset + nbytes].copy_(weight.view(torch.uint8).reshape(-1))
        pointer_map[name] = ExpertSlotLocation(
            addr=base_addr + offset,
            nbytes=nbytes,
            shape=tuple(weight.shape),
            dtype=weight.dtype,
        )

    descriptor = ExpertBackupDescriptor(
        owner_node_rank=owner_node_rank,
        backup_region_base=base_addr,
        weight_pointer_map=pointer_map,
        nixl_agent_metadata=nixl_agent_metadata,
    )
    return ExpertBackupRegion(buffer=buffer, descriptor=descriptor)
