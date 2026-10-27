# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA VMM mappings for immutable Engram tables in HOST_NUMA memory.

FABRIC handles can be sent to processes on another host in the same IMEX and
NVLink domain. The exporter has a CPU mapping for checkpoint loading; an
importer only maps its local GPU. Neither role depends on a vLLM process group.
"""

import ctypes
import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from cuda.bindings import driver as cu

from vllm.utils.cpu_resource_utils import get_memory_node_info
from vllm.utils.math_utils import round_up

FABRIC_HANDLE = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
CHUNK_BYTES = 2 << 30
NUMA_RESERVE_FRACTION = 0.05


def _call(fn, *args):
    error, *result = fn(*args)
    if error != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{fn.__name__} failed: {error.name}")
    return result[0] if result else None


def _ranges(size: int, chunk_bytes: int = CHUNK_BYTES) -> list[tuple[int, int]]:
    return [
        (offset, min(chunk_bytes, size - offset))
        for offset in range(0, size, chunk_bytes)
    ]


def _node_free_bytes(numa_node: int) -> int:
    """Use physically free pages for admission; cuMemCreate may not reclaim cache."""
    with open(f"/sys/devices/system/node/node{numa_node}/meminfo") as source:
        for line in source:
            parts = line.split()
            if len(parts) >= 5 and parts[2] == "MemFree:":
                return int(parts[3]) * 1024
    raise RuntimeError(f"cannot read MemFree for NUMA node {numa_node}")


def gpu_host_numa_nodes() -> tuple[int, ...]:
    """GPU-local HOST_NUMA nodes visible to this owner process."""
    nodes = tuple(
        dict.fromkeys(
            _call(
                cu.cuDeviceGetAttribute,
                cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
                device,
            )
            for device in range(torch.accelerator.device_count())
        )
    )
    if not nodes:
        raise RuntimeError("no GPU-local HOST_NUMA nodes are visible")
    return nodes


def _allocation_prop(numa_node: int) -> cu.CUmemAllocationProp:
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
    prop.location.id = numa_node
    prop.requestedHandleTypes = FABRIC_HANDLE
    return prop


def _access(location_type, location_id: int, *, writable: bool) -> cu.CUmemAccessDesc:
    desc = cu.CUmemAccessDesc()
    desc.location.type = location_type
    desc.location.id = location_id
    desc.flags = (
        cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        if writable
        else cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
    )
    return desc


@dataclass(frozen=True)
class TableHandles:
    """Serializable metadata for one physical table allocation."""

    size_bytes: int
    padded_bytes: int
    numa_node: int
    fabric_handles: tuple[bytes, ...]
    chunk_bytes: int = CHUNK_BYTES
    chunk_numa_nodes: tuple[int, ...] = ()

    @property
    def placement(self) -> tuple[int, ...]:
        return self.chunk_numa_nodes or (self.numa_node,) * len(self.fabric_handles)

    def to_wire(self) -> dict:
        return {
            "size_bytes": self.size_bytes,
            "padded_bytes": self.padded_bytes,
            "numa_node": self.numa_node,
            "chunk_bytes": self.chunk_bytes,
            "chunk_numa_nodes": self.chunk_numa_nodes,
            "fabric_handles": [handle.hex() for handle in self.fabric_handles],
        }

    @classmethod
    def from_wire(cls, value: dict) -> "TableHandles":
        size = int(value["size_bytes"])
        padded = int(value["padded_bytes"])
        handles = tuple(bytes.fromhex(item) for item in value["fabric_handles"])
        chunk_bytes = int(value.get("chunk_bytes", CHUNK_BYTES))
        nodes = tuple(int(node) for node in value.get("chunk_numa_nodes", ()))
        if (
            size <= 0
            or padded < size
            or chunk_bytes <= 0
            or len(handles) != len(_ranges(padded, chunk_bytes))
            or any(len(handle) != 64 for handle in handles)
            or (nodes and len(nodes) != len(handles))
        ):
            raise ValueError("invalid shared Engram table extent")
        return cls(size, padded, int(value["numa_node"]), handles, chunk_bytes, nodes)


class TableMapping:
    """Keep one local VA mapping alive while its tensors or graphs may use it."""

    def __init__(
        self,
        address: int,
        table: TableHandles,
        mapped: list[tuple[int, int]],
        device: int,
        *,
        cpu_access: bool,
    ) -> None:
        self.address = address
        self.table = table
        self._mapped = mapped
        self.device = device
        self._closed = False
        self.cpu = None
        if cpu_access:
            self._cpu_buffer = (ctypes.c_uint8 * table.size_bytes).from_address(address)
            self.cpu = torch.frombuffer(self._cpu_buffer, dtype=torch.uint8)
        # The local VA is stable for the entire mapping lifetime. Tensor objects
        # alone must never be used as an ownership or graph-quiescence signal.
        self.cuda = torch.as_tensor(
            SimpleNamespace(
                __cuda_array_interface__={
                    "shape": (table.size_bytes,),
                    "typestr": "|u1",
                    # PyTorch's CUDA-array importer rejects the read-only bit.
                    # The imported VA itself has driver-level read access only.
                    "data": (address, False),
                    "version": 3,
                },
                owner=self,
            ),
            device=torch.device("cuda", device),
        )

    def close(self) -> None:
        """Release only after all local GPU work, including graphs, has stopped."""
        if self._closed:
            return
        try:
            torch.accelerator.synchronize(self.device)
        except Exception:
            # Retain a potentially live mapping rather than invalidating it.
            raise RuntimeError("cannot quiesce shared Engram CUDA mapping") from None
        while self._mapped:
            offset, length = self._mapped[-1]
            _call(cu.cuMemUnmap, self.address + offset, length)
            self._mapped.pop()
        _call(cu.cuMemAddressFree, self.address, self.table.padded_bytes)
        self._closed = True


def _map(
    table: TableHandles,
    handles: list,
    device: int,
    *,
    cpu_access: bool,
) -> TableMapping:
    if len(handles) != len(_ranges(table.padded_bytes, table.chunk_bytes)):
        raise ValueError("shared Engram VMM handle count mismatch")
    address = None
    mapped: list[tuple[int, int]] = []
    try:
        address = int(_call(cu.cuMemAddressReserve, table.padded_bytes, 0, 0, 0))
        for (offset, length), handle, numa_node in zip(
            _ranges(table.padded_bytes, table.chunk_bytes),
            handles,
            table.placement,
        ):
            _call(cu.cuMemMap, address + offset, length, 0, handle, 0)
            mapped.append((offset, length))
            location = cu.CUmemLocationType
            access = [
                _access(
                    location.CU_MEM_LOCATION_TYPE_DEVICE,
                    device,
                    writable=cpu_access,
                )
            ]
            if cpu_access:
                access.append(
                    _access(
                        location.CU_MEM_LOCATION_TYPE_HOST_NUMA,
                        numa_node,
                        writable=True,
                    )
                )
            _call(cu.cuMemSetAccess, address + offset, length, access, len(access))
        return TableMapping(address, table, mapped, device, cpu_access=cpu_access)
    except BaseException:
        if address is not None:
            for offset, length in reversed(mapped):
                cu.cuMemUnmap(address + offset, length)
            cu.cuMemAddressFree(address, table.padded_bytes)
        raise


def create_table(
    size_bytes: int,
    device: int,
    numa_node: int | None = None,
    *,
    numa_nodes: tuple[int, ...] | None = None,
    chunk_bytes: int = CHUNK_BYTES,
    chunk_placement: tuple[int, ...] | None = None,
) -> TableMapping:
    """Create one contiguous VA backed by FABRIC chunks on available NUMA nodes."""
    if size_bytes <= 0:
        raise ValueError("shared Engram table must be nonempty")
    if numa_nodes is None:
        if numa_node is None:
            numa_node = _call(
                cu.cuDeviceGetAttribute,
                cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
                device,
            )
        numa_nodes = (numa_node,)
    if not numa_nodes or (numa_node is not None and numa_node not in numa_nodes):
        raise ValueError("invalid HOST_NUMA placement")
    props = {node: _allocation_prop(node) for node in numa_nodes}
    granularities = {
        node: _call(cu.cuMemGetAllocationGranularity, prop, 0)
        for node, prop in props.items()
    }
    granularity = math.lcm(*granularities.values())
    if chunk_bytes < granularity or chunk_bytes % granularity:
        raise ValueError("shared Engram chunk must align to HOST_NUMA granularity")
    padded = round_up(size_bytes, granularity)
    ranges = _ranges(padded, chunk_bytes)
    if chunk_placement is not None and len(chunk_placement) != len(ranges):
        raise ValueError("shared Engram chunk placement length mismatch")
    reserves = {
        node: get_memory_node_info(node).total_memory * NUMA_RESERVE_FRACTION
        for node in numa_nodes
    }
    handles = []
    placement = []
    try:
        for chunk_index, (offset, length) in enumerate(ranges):
            candidates = (
                (chunk_placement[chunk_index],)
                if chunk_placement is not None
                else tuple(
                    sorted(
                        numa_nodes,
                        key=lambda node: _node_free_bytes(node) - reserves[node],
                        reverse=True,
                    )
                )
            )
            errors = []
            for node in candidates:
                if node not in props:
                    raise ValueError(f"unknown HOST_NUMA node {node}")
                free = _node_free_bytes(node)
                if free - length < reserves[node]:
                    errors.append(f"node {node} free {free / 2**30:.1f} GiB")
                    continue
                try:
                    handle = _call(cu.cuMemCreate, length, props[node], 0)
                except RuntimeError as exc:
                    errors.append(f"node {node}: {exc}")
                    continue
                handles.append(handle)
                placement.append(node)
                break
            else:
                raise MemoryError(
                    f"HOST_NUMA allocation failed at {offset / 2**30:.1f}/"
                    f"{padded / 2**30:.1f} GiB: {', '.join(errors)}"
                )
        exported = tuple(
            bytes(_call(cu.cuMemExportToShareableHandle, handle, FABRIC_HANDLE, 0).data)
            for handle in handles
        )
        table = TableHandles(
            size_bytes, padded, placement[0], exported, chunk_bytes, tuple(placement)
        )
        return _map(table, handles, device, cpu_access=True)
    finally:
        for handle in handles:
            cu.cuMemRelease(handle)


def import_table(table: TableHandles, device: int) -> TableMapping:
    """Import on a local GPU, including when the backing is on another host."""
    handles = []
    try:
        for exported in table.fabric_handles:
            handles.append(
                _call(cu.cuMemImportFromShareableHandle, exported, FABRIC_HANDLE)
            )
        return _map(table, handles, device, cpu_access=False)
    finally:
        for handle in handles:
            cu.cuMemRelease(handle)
