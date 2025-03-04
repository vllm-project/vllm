# SPDX-License-Identifier: Apache-2.0
"""This file is a pure Python wrapper for the hipamd64 library.
Hippiefied version of vllm.hip_wrapper
It avoids the need to compile a separate shared library, and is
convenient for use when we just need to call a few functions.
"""

import ctypes
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# this line makes it possible to directly load `libhiprt.so` using `ctypes`
import torch  # noqa

hipError_t = ctypes.c_int
hipMemcpyKind = ctypes.c_int


class hipIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


def find_loaded_library(lib_name) -> Optional[str]:
    """
    According to according to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    a loaded library.
    """ # noqa
    found = False
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                found = True
                break
    if not found:
        # the library is not loaded in the current process
        return None
    start = line.index("/")
    path = line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(lib_name), \
        f"Unexpected filename: {filename} for library {lib_name}"
    return path


class RocmLibrary:
    exported_functions = [
        # ​hipError_t hipSetDevice ( int  device )
        Function("hipSetDevice", hipError_t, [ctypes.c_int]),
        # hipError_t 	hipDeviceSynchronize ( void )
        Function("hipDeviceSynchronize", hipError_t, []),
        # ​hipError_t hipDeviceReset ( void )
        Function("hipDeviceReset", hipError_t, []),

        # const char* 	hipGetErrorString ( hipError_t error )
        Function("hipGetErrorString", ctypes.c_char_p, [hipError_t]),

        # ​hipError_t 	hipMalloc ( void** devPtr, size_t size )
        Function("hipMalloc", hipError_t,
                 [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]),
        # ​hipError_t 	hipFree ( void* devPtr )
        Function("hipFree", hipError_t, [ctypes.c_void_p]),
        # ​hipError_t hipMemset ( void* devPtr, int  value, size_t count )
        Function("hipMemset", hipError_t,
                 [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]),
        # ​hipError_t hipMemcpy ( void* dst, const void* src, size_t count, hipMemcpyKind kind ) # noqa
        Function(
            "hipMemcpy", hipError_t,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, hipMemcpyKind
             ]),

        # hipError_t hipIpcGetMemHandle ( hipIpcMemHandle_t* handle, void* devPtr ) # noqa
        Function("hipIpcGetMemHandle", hipError_t,
                 [ctypes.POINTER(hipIpcMemHandle_t), ctypes.c_void_p]),
        # ​hipError_t hipIpcOpenMemHandle ( void** devPtr, hipIpcMemHandle_t handle, unsigned int  flags ) # noqa
        Function("hipIpcOpenMemHandle", hipError_t, [
            ctypes.POINTER(ctypes.c_void_p), hipIpcMemHandle_t, ctypes.c_uint
        ]),
        # ​hipError_t hipExtMallocWithFlags(void** devPtr, size_t size, int flags) # noqa
        Function(
            "hipExtMallocWithFlags", hipError_t,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):
        if so_file is None:
            so_file = find_loaded_library("libamdhip64")
            assert so_file is not None, \
                "libamdhip64 is not loaded in the current process"
        if so_file not in RocmLibrary.path_to_library_cache:
            lib = ctypes.CDLL(so_file)
            RocmLibrary.path_to_library_cache[so_file] = lib
        self.lib = RocmLibrary.path_to_library_cache[so_file]

        if so_file not in RocmLibrary.path_to_dict_mapping:
            _funcs = {}
            for func in RocmLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            RocmLibrary.path_to_dict_mapping[so_file] = _funcs
        self.funcs = RocmLibrary.path_to_dict_mapping[so_file]

    def hipRT_CHECK(self, result: hipError_t) -> None:
        if result != 0:
            error_str = self.hipGetErrorString(result)
            raise RuntimeError(f"hipRT error: {error_str}")

    def hipGetErrorString(self, error: hipError_t) -> str:
        return self.funcs["hipGetErrorString"](error).decode("utf-8")

    def hipSetDevice(self, device: int) -> None:
        self.hipRT_CHECK(self.funcs["hipSetDevice"](device))

    def hipDeviceSynchronize(self) -> None:
        self.hipRT_CHECK(self.funcs["hipDeviceSynchronize"]())

    def hipDeviceReset(self) -> None:
        self.hipRT_CHECK(self.funcs["hipDeviceReset"]())

    def hipMalloc(self, size: int) -> ctypes.c_void_p:
        devPtr = ctypes.c_void_p()
        self.hipRT_CHECK(self.funcs["hipMalloc"](ctypes.byref(devPtr), size))
        return devPtr

    def hipMallocUncached(self, size: int) -> ctypes.c_void_p:
        devPtr = ctypes.c_void_p()
        kind = 3  #hipDeviceMallocUncached
        self.hipRT_CHECK(self.funcs["hipExtMallocWithFlags"](
            ctypes.byref(devPtr), size, kind))
        return devPtr

    def hipFree(self, devPtr: ctypes.c_void_p) -> None:
        self.hipRT_CHECK(self.funcs["hipFree"](devPtr))

    def hipMemset(self, devPtr: ctypes.c_void_p, value: int,
                  count: int) -> None:
        self.hipRT_CHECK(self.funcs["hipMemset"](devPtr, value, count))

    def hipMemcpy(self, dst: ctypes.c_void_p, src: ctypes.c_void_p,
                  count: int) -> None:
        hipMemcpyDefault = 4
        kind = hipMemcpyDefault
        self.hipRT_CHECK(self.funcs["hipMemcpy"](dst, src, count, kind))

    def hipIpcGetMemHandle(self, devPtr: ctypes.c_void_p) -> hipIpcMemHandle_t:
        handle = hipIpcMemHandle_t()
        self.hipRT_CHECK(self.funcs["hipIpcGetMemHandle"](ctypes.byref(handle),
                                                          devPtr))
        return handle

    def hipIpcOpenMemHandle(self,
                            handle: hipIpcMemHandle_t) -> ctypes.c_void_p:
        hipIpcMemLazyEnablePeerAccess = 1
        devPtr = ctypes.c_void_p()
        self.hipRT_CHECK(self.funcs["hipIpcOpenMemHandle"](
            ctypes.byref(devPtr), handle, hipIpcMemLazyEnablePeerAccess))
        return devPtr
