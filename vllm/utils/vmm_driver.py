# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ctypes bindings for GPU virtual-memory-management (VMM) driver APIs.

Exposes a uniform driver interface over the CUDA driver's ``cuMem*`` entry
points and HIP's mirrored ``hipMem*`` entry points, used by
:class:`vllm.utils.extensible_tensor.ExtensibleTensor`: reserve a virtual
address range, create physical memory handles, map/unmap them into the
reservation, and set access permissions.
"""

from __future__ import annotations

import ctypes
from functools import cache

from vllm.logger import init_logger

logger = init_logger(__name__)

_SUCCESS = 0
_MEM_ALLOCATION_TYPE_PINNED = 1
_MEM_LOCATION_TYPE_DEVICE = 1
_MEM_ALLOC_GRANULARITY_MINIMUM = 0
_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
_MEM_ALLOCATION_COMP_NONE = 0

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
    # Layout shared by CUmemAllocationProp and hipMemAllocationProp.
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _MemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _MemAllocFlags),
    ]


class _MemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _MemLocation), ("flags", ctypes.c_int)]


def _find_loaded_library(lib_name: str) -> str | None:
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if lib_name not in line:
                    continue
                start = line.index("/")
                return line[start:].strip()
    except (OSError, ValueError):
        return None
    return None


class VmmDriver:
    """Uniform interface over a GPU driver's virtual memory management API.

    Subclasses supply the driver library candidates and symbol names; the
    call signatures and struct layouts are shared between CUDA and HIP.
    """

    # DLPack device type for tensors viewing driver-mapped memory.
    dlpack_device_type: int
    _lib_candidates: tuple[str, ...]
    _lib_search_name: str
    # Logical name -> library symbol.
    _symbols: dict[str, str]

    def __init__(self) -> None:
        self._lib = self._load_library()
        self._fns: dict[str, ctypes._CFuncPtr] = {}
        for logical, symbol in self._symbols.items():
            self._fns[logical] = getattr(self._lib, symbol)
        self._configure_signatures()

    def _load_library(self) -> ctypes.CDLL:
        for name in self._lib_candidates:
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        if path := _find_loaded_library(self._lib_search_name):
            return ctypes.CDLL(path)
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

    def ensure_context(self, device_index: int) -> None:
        """Make sure a driver context for `device_index` is current."""
        raise NotImplementedError

    def _check(self, result: int) -> None:
        if result == _SUCCESS:
            return
        raise RuntimeError(f"GPU driver error {result}: {self.error_string(result)}")

    def _make_alloc_prop(
        self, device_index: int, rdma_capable: bool = False
    ) -> _MemAllocationProp:
        prop = _MemAllocationProp()
        prop.type = _MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = _MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device_index
        prop.allocFlags.compressionType = _MEM_ALLOCATION_COMP_NONE
        prop.allocFlags.gpuDirectRDMACapable = 1 if rdma_capable else 0
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

    def create(self, size: int, device_index: int, rdma_capable: bool = False) -> int:
        """Create a physical memory handle of `size` bytes."""
        prop = self._make_alloc_prop(device_index, rdma_capable)
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
        lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(_Context)]
        lib.cuCtxGetCurrent.restype = ctypes.c_int
        lib.cuDevicePrimaryCtxRetain.argtypes = [
            ctypes.POINTER(_Context),
            ctypes.c_int,
        ]
        lib.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
        lib.cuCtxSetCurrent.argtypes = [_Context]
        lib.cuCtxSetCurrent.restype = ctypes.c_int

    def error_string(self, code: int) -> str:
        msg = ctypes.c_char_p()
        self._lib.cuGetErrorString(code, ctypes.byref(msg))
        return msg.value.decode() if msg.value else "unknown error"

    def ensure_context(self, device_index: int) -> None:
        pctx = _Context()
        self._check(self._lib.cuCtxGetCurrent(ctypes.byref(pctx)))
        if pctx.value:
            return
        self._check(
            self._lib.cuDevicePrimaryCtxRetain(ctypes.byref(pctx), device_index)
        )
        self._check(self._lib.cuCtxSetCurrent(pctx))


class HipVmmDriver(VmmDriver):
    """HIP mirrors the CUDA driver's VMM API (``hipMem*``) with identical
    call signatures, struct layouts, and constants; PyTorch's
    expandable-segments allocator uses the same entry points on ROCm.
    """

    dlpack_device_type = 10  # kDLROCM
    _lib_candidates = (
        "libamdhip64.so",
        "libamdhip64.so.7",
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


@cache
def get_vmm_driver() -> VmmDriver:
    import torch

    if torch.version.hip is not None:
        return HipVmmDriver()
    return CudaVmmDriver()


@cache
def vmm_unavailable_reason() -> str | None:
    """Probe VMM support; returns None if usable, else a reason string.

    Checks that the driver library loads, exposes the VMM entry points, and
    can actually reserve (and release) a virtual address range on the current
    device. Notably returns a reason on platforms whose driver lacks VMM
    support (e.g. WSL2) and on non-CUDA/ROCm builds.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "no CUDA/ROCm device is available"
        torch.cuda.init()
        driver = get_vmm_driver()
        device_index = torch.cuda.current_device()
        driver.ensure_context(device_index)
        granularity = driver.granularity(device_index)
        ptr = driver.reserve(granularity)
        driver.free_reserved(ptr, granularity)
    except Exception as e:
        return str(e)
    return None
