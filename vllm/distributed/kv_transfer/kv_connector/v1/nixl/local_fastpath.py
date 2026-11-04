# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-host CUDA IPC primitives for the NIXL pull fast path."""

import ctypes
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import numpy as np


class LocalIpcError(RuntimeError):
    pass


class _CudaIpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ubyte * 64)]


@dataclass(frozen=True)
class IpcAllocation:
    handle: bytes
    size: int
    exported_base: int
    offset: int
    length: int

    def validate(self) -> None:
        if not isinstance(self.handle, bytes) or len(self.handle) != 64:
            raise ValueError("Invalid CUDA IPC memory handle")
        values = (self.size, self.exported_base, self.offset, self.length)
        if any(type(value) is not int for value in values):
            raise ValueError("CUDA IPC allocation bounds must be integers")
        if not 0 < self.length <= self.size < 2**63:
            raise ValueError("Invalid CUDA IPC allocation size")
        if not 0 <= self.offset <= self.size - self.length:
            raise ValueError("CUDA IPC export exceeds its allocation")


@dataclass(frozen=True)
class LocalIpcMetadata:
    version: int
    boot_id: str
    device_uuid: str
    allocations: tuple[IpcAllocation, ...]


@dataclass(frozen=True)
class CopySpan:
    src: int
    dst: int
    length: int


def normalize_gpu_uuid(value: object) -> str:
    return "GPU-" + str(UUID(str(value).removeprefix("GPU-")))


def get_boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    if not path.is_file():
        raise RuntimeError("NIXL local fast path requires Linux boot identity")
    return path.read_text().strip()


class CudaIpcDriver:
    """Export and import allocations in the caller's existing CUDA context."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so.1")
        self._range = self._bind(
            "cuMemGetAddressRange_v2",
            [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_uint64,
            ],
        )
        self._export = self._bind(
            "cuIpcGetMemHandle", [ctypes.POINTER(_CudaIpcHandle), ctypes.c_uint64]
        )
        self._open = self._bind(
            "cuIpcOpenMemHandle_v2",
            [ctypes.POINTER(ctypes.c_uint64), _CudaIpcHandle, ctypes.c_uint],
        )
        self._close = self._bind("cuIpcCloseMemHandle", [ctypes.c_uint64])
        self._imports: dict[bytes, tuple[int, int]] = {}

    def _bind(self, name: str, args: list[object]):
        fn = getattr(self.lib, name)
        fn.argtypes = args
        fn.restype = ctypes.c_int
        return fn

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result != 0:
            raise LocalIpcError(f"{operation} failed with CUDA driver error {result}")

    def export(self, pointer: int, length: int) -> IpcAllocation:
        base = ctypes.c_uint64()
        size = ctypes.c_size_t()
        self._check(
            self._range(ctypes.byref(base), ctypes.byref(size), pointer),
            "cuMemGetAddressRange",
        )
        handle = _CudaIpcHandle()
        self._check(self._export(ctypes.byref(handle), base.value), "cuIpcGetMemHandle")
        allocation = IpcAllocation(
            handle=bytes(handle.reserved),
            size=size.value,
            exported_base=pointer,
            offset=pointer - base.value,
            length=length,
        )
        allocation.validate()
        return allocation

    def open(self, allocation: IpcAllocation) -> int:
        allocation.validate()
        cached = self._imports.get(allocation.handle)
        if cached is None:
            pointer = ctypes.c_uint64()
            self._check(
                self._open(
                    ctypes.byref(pointer),
                    _CudaIpcHandle.from_buffer_copy(allocation.handle),
                    1,
                ),
                "cuIpcOpenMemHandle",
            )
            cached = (pointer.value, allocation.size)
            self._imports[allocation.handle] = cached
        if cached[1] != allocation.size:
            raise ValueError("Conflicting sizes for one CUDA IPC allocation")
        return cached[0] + allocation.offset

    def close(self) -> None:
        for handle, (pointer, _) in list(self._imports.items()):
            self._check(self._close(pointer), "cuIpcCloseMemHandle")
            del self._imports[handle]


def expand_strided_descriptors(
    descriptors: np.ndarray, descriptor_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve flattened NIXL descriptor IDs to addresses and lengths."""
    if descriptors.ndim != 2 or descriptors.shape[1] not in (3, 5):
        raise ValueError("NIXL descriptor table must have shape [N, 3] or [N, 5]")
    ids = np.asarray(descriptor_ids, dtype=np.int64)
    if descriptors.shape[1] == 3:
        if ids.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        if ids.min() < 0 or ids.max() >= len(descriptors):
            raise ValueError("NIXL descriptor ID is out of range")
        return (
            descriptors[ids, 0].astype(np.int64),
            descriptors[ids, 1].astype(np.int64),
        )
    counts = descriptors[:, 4].astype(np.int64, copy=False)
    if np.any(counts < 0):
        raise ValueError("NIXL descriptor counts must be non-negative")
    ends = np.cumsum(counts)
    if ids.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if ids.min() < 0 or ids.max() >= ends[-1]:
        raise ValueError("NIXL descriptor ID is out of range")
    rows = np.searchsorted(ends, ids, side="right")
    starts = np.concatenate((np.array([0], dtype=np.int64), ends[:-1]))
    offsets = ids - starts[rows]
    addresses = descriptors[rows, 0] + offsets * descriptors[rows, 3]
    lengths = descriptors[rows, 1]
    return addresses.astype(np.int64), lengths.astype(np.int64)


def translate_remote_addresses(
    addresses: np.ndarray,
    allocations: tuple[IpcAllocation, ...],
    mapped_bases: tuple[int, ...],
    lengths: np.ndarray | None = None,
) -> np.ndarray:
    if len(allocations) != len(mapped_bases):
        raise ValueError("CUDA IPC allocation mapping is incomplete")
    if lengths is None:
        lengths = np.ones(len(addresses), dtype=np.int64)
    if len(lengths) != len(addresses) or np.any(lengths <= 0):
        raise ValueError("Invalid CUDA IPC descriptor lengths")
    translated = np.empty(len(addresses), dtype=np.int64)
    for index, (address, length) in enumerate(
        zip(addresses.tolist(), lengths.tolist(), strict=True)
    ):
        for allocation, mapped_base in zip(allocations, mapped_bases, strict=True):
            start = allocation.exported_base
            end = start + allocation.length
            if start <= address and address + length <= end:
                translated[index] = mapped_base + address - start
                break
        else:
            raise ValueError("Remote descriptor is outside CUDA IPC allocations")
    return translated


def coalesce_copy_spans(
    src_addresses: np.ndarray,
    dst_addresses: np.ndarray,
    src_lengths: np.ndarray,
    dst_lengths: np.ndarray,
) -> list[CopySpan]:
    """Merge adjacent or aliased ranges only when their byte mapping agrees."""
    lengths = np.asarray(src_lengths, dtype=np.int64)
    if not (
        len(src_addresses) == len(dst_addresses) == len(lengths) == len(dst_lengths)
    ):
        raise ValueError("CUDA IPC source and destination descriptors differ")
    if not np.array_equal(lengths, np.asarray(dst_lengths, dtype=np.int64)):
        raise ValueError("CUDA IPC descriptor lengths differ")
    ranges = sorted(
        zip(
            src_addresses.tolist(),
            dst_addresses.tolist(),
            lengths.tolist(),
            strict=True,
        ),
        key=lambda span: span[1],
    )
    spans: list[CopySpan] = []
    for src, dst, length in ranges:
        if length <= 0:
            raise ValueError("CUDA IPC copy span must be non-empty")
        if spans:
            previous = spans[-1]
            previous_end = previous.dst + previous.length
            same_mapping = src - dst == previous.src - previous.dst
            if dst < previous_end and not same_mapping:
                raise ValueError(
                    "CUDA IPC overlapping destinations map to different sources"
                )
            if dst <= previous_end and same_mapping:
                # Hybrid FA/SSM descriptors can alias one physical cache page.
                spans[-1] = CopySpan(
                    previous.src,
                    previous.dst,
                    max(previous_end, dst + length) - previous.dst,
                )
                continue
        spans.append(CopySpan(src, dst, length))
    return spans
