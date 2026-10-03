# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ctypes bindings for the CUDA (``cuMem*``) and HIP (``hipMem*``) virtual memory
management APIs, used by :class:`vllm.utils.extensible_tensor.ExtensibleTensor`.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Iterable
from functools import cache
from typing import Any

_SUCCESS = 0
_MEM_ALLOCATION_TYPE_PINNED = 1
_MEM_LOCATION_TYPE_DEVICE = 1
_MEM_ALLOC_GRANULARITY_MINIMUM = 0
_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
_MEM_ALLOCATION_COMP_NONE = 0
_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110

DevicePtr = ctypes.c_ulonglong
MemHandle = ctypes.c_ulonglong
_Context = ctypes.c_void_p


class _MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _MemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _MemAllocationProp(ctypes.Structure):
    # CUmemAllocationProp. hipMemAllocationProp matches it up to allocFlags,
    # whose HIP variant lacks the trailing reserved bytes; the driver never
    # reads past the end of its own layout, so the CUDA shape serves both.
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _MemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _MemAllocFlags),
    ]


class _MemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _MemLocation), ("flags", ctypes.c_int)]


def _find_loaded_library(
    lib_name: str, maps_lines: Iterable[str] | None = None
) -> str | None:
    """Path of the mapped shared object named ``lib_name`` (e.g. ``libcuda``),
    matched on the file name so that ``libcudart`` does not stand in for it."""
    try:
        if maps_lines is None:
            with open("/proc/self/maps") as f:
                return _find_loaded_library(lib_name, f.readlines())
        for line in maps_lines:
            if "/" not in line:
                continue
            path = line[line.index("/") :].strip()
            if os.path.basename(path).startswith(f"{lib_name}.so"):
                return path
    except (OSError, ValueError):
        return None
    return None


class VmmDriver:
    """Uniform interface over a GPU driver's VMM API.

    Subclasses supply library names and symbols; signatures and structs are
    shared between CUDA and HIP.
    """

    # DLPack device type for tensors viewing driver-mapped memory.
    dlpack_device_type: int
    _lib_candidates: tuple[str, ...]
    _lib_search_name: str
    # Logical name -> library symbol.
    _symbols: dict[str, str]

    def __init__(self) -> None:
        self._lib = self._load_library()
        self._fns: dict[str, Any] = {
            logical: getattr(self._lib, symbol)
            for logical, symbol in self._symbols.items()
        }
        self._configure_signatures()

    def _load_library(self) -> ctypes.CDLL:
        # Prefer the copy already in the process, e.g. the HIP runtime a ROCm
        # torch wheel bundles: loading one by name could add a second runtime.
        if path := _find_loaded_library(self._lib_search_name):
            return ctypes.CDLL(path)
        for name in self._lib_candidates:
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        raise RuntimeError(
            f"Could not load {self._lib_candidates[0]}. The GPU driver "
            "library is required for VMM-backed tensors."
        )

    def _configure_signatures(self) -> None:
        pointer = ctypes.POINTER
        fns = self._fns
        fns["get_granularity"].argtypes = [
            pointer(ctypes.c_size_t),
            pointer(_MemAllocationProp),
            ctypes.c_int,
        ]
        fns["address_reserve"].argtypes = [
            pointer(DevicePtr),
            ctypes.c_size_t,
            ctypes.c_size_t,
            DevicePtr,
            ctypes.c_ulonglong,
        ]
        fns["create"].argtypes = [
            pointer(MemHandle),
            ctypes.c_size_t,
            pointer(_MemAllocationProp),
            ctypes.c_ulonglong,
        ]
        fns["map"].argtypes = [
            DevicePtr,
            ctypes.c_size_t,
            ctypes.c_size_t,
            MemHandle,
            ctypes.c_ulonglong,
        ]
        fns["set_access"].argtypes = [
            DevicePtr,
            ctypes.c_size_t,
            pointer(_MemAccessDesc),
            ctypes.c_size_t,
        ]
        fns["unmap"].argtypes = [DevicePtr, ctypes.c_size_t]
        fns["release"].argtypes = [MemHandle]
        fns["address_free"].argtypes = [DevicePtr, ctypes.c_size_t]
        for fn in fns.values():
            fn.restype = ctypes.c_int

    def error_string(self, code: int) -> str:
        raise NotImplementedError

    def unusable_runtime_reason(self, device_index: int) -> str | None:
        """Why this driver's runtime cannot back a growable allocation, if so."""
        return None

    def ensure_context(self, device_index: int) -> None:
        """Make sure a driver context for `device_index` is current."""
        raise NotImplementedError

    def _check(self, result: int) -> None:
        if result == _SUCCESS:
            return
        raise RuntimeError(f"GPU driver error {result}: {self.error_string(result)}")

    def _make_alloc_prop(self, device_index: int) -> _MemAllocationProp:
        prop = _MemAllocationProp()
        prop.type = _MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = _MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device_index
        prop.allocFlags.compressionType = _MEM_ALLOCATION_COMP_NONE
        return prop

    def granularity(self, device_index: int) -> int:
        prop = self._make_alloc_prop(device_index)
        granularity = ctypes.c_size_t()
        self._check(
            self._fns["get_granularity"](
                ctypes.byref(granularity),
                ctypes.byref(prop),
                _MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        )
        return granularity.value

    def reserve(self, size: int) -> int:
        """Reserve a virtual address range and return its base pointer."""
        dptr = DevicePtr()
        self._check(self._fns["address_reserve"](ctypes.byref(dptr), size, 0, 0, 0))
        return dptr.value

    def free_reserved(self, ptr: int, size: int) -> None:
        self._check(self._fns["address_free"](ptr, size))

    def create(self, size: int, device_index: int) -> int:
        """Create a physical memory handle of `size` bytes."""
        prop = self._make_alloc_prop(device_index)
        handle = MemHandle()
        self._check(
            self._fns["create"](ctypes.byref(handle), size, ctypes.byref(prop), 0)
        )
        return handle.value

    def map(self, ptr: int, size: int, handle: int) -> None:
        self._check(self._fns["map"](ptr, size, 0, handle, 0))

    def set_access(self, ptr: int, size: int, device_index: int) -> None:
        desc = _MemAccessDesc()
        desc.location.type = _MEM_LOCATION_TYPE_DEVICE
        desc.location.id = device_index
        desc.flags = _MEM_ACCESS_FLAGS_PROT_READWRITE
        self._check(self._fns["set_access"](ptr, size, ctypes.byref(desc), 1))

    def unmap(self, ptr: int, size: int) -> None:
        self._check(self._fns["unmap"](ptr, size))

    def release(self, handle: int) -> None:
        self._check(self._fns["release"](handle))


class CudaVmmDriver(VmmDriver):
    dlpack_device_type = 2  # kDLCUDA
    _lib_candidates = ("libcuda.so.1", "libcuda.so")
    _lib_search_name = "libcuda"
    _symbols = {
        "get_granularity": "cuMemGetAllocationGranularity",
        "address_reserve": "cuMemAddressReserve",
        "create": "cuMemCreate",
        "map": "cuMemMap",
        "set_access": "cuMemSetAccess",
        "unmap": "cuMemUnmap",
        "release": "cuMemRelease",
        "address_free": "cuMemAddressFree",
    }

    def __init__(self) -> None:
        super().__init__()
        lib = self._lib
        lib.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        lib.cuGetErrorString.restype = ctypes.c_int
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuInit.restype = ctypes.c_int
        lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(_Context)]
        lib.cuCtxGetCurrent.restype = ctypes.c_int
        lib.cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cuCtxGetDevice.restype = ctypes.c_int
        lib.cuDevicePrimaryCtxRetain.argtypes = [
            ctypes.POINTER(_Context),
            ctypes.c_int,
        ]
        lib.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
        lib.cuCtxSetCurrent.argtypes = [_Context]
        lib.cuCtxSetCurrent.restype = ctypes.c_int
        lib.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.cuDeviceGetAttribute.restype = ctypes.c_int
        self._rdma_capable: dict[int, bool] = {}

    def _make_alloc_prop(self, device_index: int) -> _MemAllocationProp:
        # Request GPUDirect RDMA support where available, as the sleep-mode
        # allocator does: without it, KV connectors cannot register the
        # memory with RDMA NICs (ibv_reg_mr fails with EFAULT).
        prop = super()._make_alloc_prop(device_index)
        if (capable := self._rdma_capable.get(device_index)) is None:
            flag = ctypes.c_int()
            result = self._lib.cuDeviceGetAttribute(
                ctypes.byref(flag),
                _DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
                device_index,
            )
            capable = result == _SUCCESS and bool(flag.value)
            self._rdma_capable[device_index] = capable
        prop.allocFlags.gpuDirectRDMACapable = int(capable)
        return prop

    def error_string(self, code: int) -> str:
        msg = ctypes.c_char_p()
        self._lib.cuGetErrorString(code, ctypes.byref(msg))
        return msg.value.decode() if msg.value else "unknown error"

    def ensure_context(self, device_index: int) -> None:
        """Make the primary context of ``device_index`` current, as torch does
        for the device it runs on; usable before torch has touched the device."""
        lib = self._lib
        self._check(lib.cuInit(0))
        pctx = _Context()
        self._check(lib.cuCtxGetCurrent(ctypes.byref(pctx)))
        if pctx.value:
            device = ctypes.c_int()
            self._check(lib.cuCtxGetDevice(ctypes.byref(device)))
            if device.value == device_index:
                return
        self._check(lib.cuDevicePrimaryCtxRetain(ctypes.byref(pctx), device_index))
        self._check(lib.cuCtxSetCurrent(pctx))


class HipVmmDriver(VmmDriver):
    """HIP mirrors the CUDA VMM API with identical signatures, structs and constants."""

    dlpack_device_type = 10  # kDLROCM
    _lib_candidates = (
        "libamdhip64.so.7",
        "libamdhip64.so",
        "libamdhip64.so.6",
        "libamdhip64.so.5",
    )
    _lib_search_name = "libamdhip64"
    _symbols = {
        "get_granularity": "hipMemGetAllocationGranularity",
        "address_reserve": "hipMemAddressReserve",
        "create": "hipMemCreate",
        "map": "hipMemMap",
        "set_access": "hipMemSetAccess",
        "unmap": "hipMemUnmap",
        "release": "hipMemRelease",
        "address_free": "hipMemAddressFree",
    }

    def __init__(self) -> None:
        super().__init__()
        lib = self._lib
        lib.hipGetErrorString.argtypes = [ctypes.c_int]
        lib.hipGetErrorString.restype = ctypes.c_char_p
        lib.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.hipGetDevice.restype = ctypes.c_int
        lib.hipSetDevice.argtypes = [ctypes.c_int]
        lib.hipSetDevice.restype = ctypes.c_int
        lib.hipRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.hipRuntimeGetVersion.restype = ctypes.c_int
        lib.hipGetLastError.argtypes = []
        lib.hipGetLastError.restype = ctypes.c_int

    def _check(self, result: int) -> None:
        if result != _SUCCESS:
            # A failed hipMem* call leaves the error sticky for the next
            # torch launch on this thread; reading it clears it.
            self._lib.hipGetLastError()
        super()._check(result)

    def runtime_version(self) -> tuple[int, int]:
        """(major, minor) of the loaded HIP runtime."""
        version = ctypes.c_int()
        self._check(self._lib.hipRuntimeGetVersion(ctypes.byref(version)))
        return divmod(version.value // 100000, 100)

    def unusable_runtime_reason(self, device_index: int) -> str | None:
        if not _set_access_fails_within_reservation(self, device_index):
            return None
        major, minor = self.runtime_version()
        return (
            f"the HIP runtime ({major}.{minor}) rejects hipMemSetAccess for an "
            "allocation that is not the first in its virtual-address "
            "reservation (ROCm/rocm-systems#2516, fixed by "
            "ROCm/rocm-systems#2451 in TheRock 7.12 and later)"
        )

    def error_string(self, code: int) -> str:
        msg = self._lib.hipGetErrorString(code)
        return msg.decode() if msg else "unknown error"

    def ensure_context(self, device_index: int) -> None:
        # The HIP runtime manages contexts implicitly; just make sure the
        # buffer's device is current on this thread.
        device = ctypes.c_int()
        self._check(self._lib.hipGetDevice(ctypes.byref(device)))
        if device.value != device_index:
            self._check(self._lib.hipSetDevice(device_index))


def _set_access_fails_within_reservation(driver: VmmDriver, device_index: int) -> bool:
    """Whether set-access rejects one of two allocations mapped side by side in
    a reservation, as HIP runtimes before ROCm/rocm-systems#2451 do.

    Those check the requested size against the reservation's allocations
    counted from the first in their list rather than from the target. With
    sizes of one and two granules, one of the two calls fails whichever comes
    first in that list, so the check does not depend on allocation order.
    """
    granularity = driver.granularity(device_index)
    sizes = (granularity, 2 * granularity)
    base = driver.reserve(sum(sizes))
    mapped: list[tuple[int, int, int]] = []
    try:
        ptr = base
        for size in sizes:
            handle = driver.create(size, device_index)
            try:
                driver.map(ptr, size, handle)
            except RuntimeError:
                driver.release(handle)
                raise
            mapped.append((ptr, size, handle))
            ptr += size
        try:
            for ptr, size, _ in mapped:
                driver.set_access(ptr, size, device_index)
        except RuntimeError:
            return True
        return False
    finally:
        for ptr, size, handle in mapped:
            driver.unmap(ptr, size)
            driver.release(handle)
        driver.free_reserved(base, sum(sizes))


@cache
def get_vmm_driver() -> VmmDriver:
    import torch

    if torch.version.hip is not None:
        return HipVmmDriver()
    return CudaVmmDriver()


@cache
def vmm_unavailable_reason() -> str | None:
    """Why VMM cannot be used on the current device, or None if it can.

    Checks library load, entry points, known runtime defects (by probing), and
    one reserve/create/map/set-access/unmap/release round trip of a single
    granule. Covers WSL2 and non-CUDA/ROCm builds.
    """
    try:
        import torch

        if not torch.accelerator.is_available():
            return "no CUDA/ROCm device is available"
        driver = get_vmm_driver()
        device_index = torch.accelerator.current_device_index()
        driver.ensure_context(device_index)
        if reason := driver.unusable_runtime_reason(device_index):
            return reason
        granularity = driver.granularity(device_index)
        ptr = driver.reserve(granularity)
        try:
            handle = driver.create(granularity, device_index)
            try:
                driver.map(ptr, granularity, handle)
                try:
                    driver.set_access(ptr, granularity, device_index)
                finally:
                    driver.unmap(ptr, granularity)
            finally:
                driver.release(handle)
        finally:
            driver.free_reserved(ptr, granularity)
    except Exception as e:
        return str(e)
    return None
