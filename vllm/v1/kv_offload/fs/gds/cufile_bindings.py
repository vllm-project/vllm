# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ctypes bindings for the NVIDIA cuFile (GDS) synchronous API."""

import ctypes
import ctypes.util
import os
from ctypes import (
    POINTER,
    Structure,
    Union,
    c_int,
    c_size_t,
    c_ssize_t,
    c_uint,
    c_void_p,
)
from enum import IntEnum

CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1
CU_FILE_SUCCESS = 0


class CUfileOpError(IntEnum):
    CU_FILE_SUCCESS = 0
    CU_FILE_DRIVER_NOT_INITIALIZED = 1
    CU_FILE_DRIVER_INVALID_PROPS = 2
    CU_FILE_DRIVER_UNSUPPORTED_LIMIT = 3
    CU_FILE_DRIVER_VERSION_MISMATCH = 4
    CU_FILE_DRIVER_VERSION_READ_ERROR = 5
    CU_FILE_DRIVER_CLOSING = 6
    CU_FILE_PLATFORM_NOT_SUPPORTED = 7
    CU_FILE_IO_NOT_SUPPORTED = 8
    CU_FILE_DEVICE_NOT_SUPPORTED = 9
    CU_FILE_INTERNAL_ERROR = 10
    CU_FILE_GETNEWFD_FAILED = 11
    CU_FILE_INVAL_FLAGS = 12
    CU_FILE_HANDLE_NOT_REGISTERED = 13


class CUfileError_t(Structure):
    _fields_ = [
        ("err", c_int),  # CUfileOpError
        ("cu_err", c_int),  # CUDA driver error
    ]


class _HandleUnion(Union):
    _fields_ = [
        ("fd", c_int),  # Linux POSIX fd
        ("handle", c_void_p),  # Windows handle
    ]


class CUfileDescr_t(Structure):
    _fields_ = [
        ("type", c_uint),  # CUfileFileHandleType
        ("handle", _HandleUnion),
        ("fs_ops", c_void_p),  # const CUfileFSOps_t*
    ]


CUfileHandle_t = c_void_p

_lib: ctypes.CDLL | None = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    lib_path = ctypes.util.find_library("cufile")
    if lib_path is None:
        for candidate in ("libcufile.so.0", "libcufile.so"):
            try:
                _lib = ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                return _lib
            except OSError:
                continue
        raise OSError(
            "Cannot find libcufile.so. Ensure nvidia-cufile is installed "
            "and the library is on LD_LIBRARY_PATH."
        )
    _lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    return _lib


def _check_error(err: CUfileError_t, func_name: str) -> None:
    if err.err != CU_FILE_SUCCESS:
        try:
            err_name = CUfileOpError(err.err).name
        except ValueError:
            err_name = f"UNKNOWN({err.err})"
        raise RuntimeError(
            f"{func_name} failed: err={err.err} ({err_name}), cu_err={err.cu_err}"
        )


def cuFileDriverOpen() -> None:
    lib = _get_lib()
    lib.cuFileDriverOpen.restype = CUfileError_t
    lib.cuFileDriverOpen.argtypes = []
    err = lib.cuFileDriverOpen()
    _check_error(err, "cuFileDriverOpen")


def cuFileDriverClose() -> None:
    lib = _get_lib()
    lib.cuFileDriverClose.restype = CUfileError_t
    lib.cuFileDriverClose.argtypes = []
    err = lib.cuFileDriverClose()
    _check_error(err, "cuFileDriverClose")


def cuFileHandleRegister(fd: int) -> CUfileHandle_t:
    """Register a POSIX file descriptor for cuFile I/O."""
    lib = _get_lib()
    lib.cuFileHandleRegister.restype = CUfileError_t
    lib.cuFileHandleRegister.argtypes = [
        POINTER(CUfileHandle_t),
        POINTER(CUfileDescr_t),
    ]

    descr = CUfileDescr_t()
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD
    descr.handle.fd = fd
    descr.fs_ops = None

    handle = CUfileHandle_t()
    err = lib.cuFileHandleRegister(ctypes.byref(handle), ctypes.byref(descr))
    _check_error(err, "cuFileHandleRegister")
    return handle


def cuFileHandleDeregister(handle: CUfileHandle_t) -> None:
    lib = _get_lib()
    lib.cuFileHandleDeregister.restype = None
    lib.cuFileHandleDeregister.argtypes = [CUfileHandle_t]
    lib.cuFileHandleDeregister(handle)


def cuFileRead(
    handle: CUfileHandle_t,
    dev_ptr: int,
    size: int,
    file_offset: int = 0,
    buf_offset: int = 0,
) -> int:
    """Synchronous read from file to GPU memory. Returns bytes read."""
    lib = _get_lib()
    lib.cuFileRead.restype = c_ssize_t
    lib.cuFileRead.argtypes = [CUfileHandle_t, c_void_p, c_size_t, c_ssize_t, c_ssize_t]
    ret = lib.cuFileRead(
        handle,
        c_void_p(dev_ptr),
        c_size_t(size),
        c_ssize_t(file_offset),
        c_ssize_t(buf_offset),
    )
    if ret < 0:
        raise RuntimeError(f"cuFileRead failed: ret={ret}")
    return ret


def cuFileWrite(
    handle: CUfileHandle_t,
    dev_ptr: int,
    size: int,
    file_offset: int = 0,
    buf_offset: int = 0,
) -> int:
    """Synchronous write from GPU memory to file. Returns bytes written."""
    lib = _get_lib()
    lib.cuFileWrite.restype = c_ssize_t
    lib.cuFileWrite.argtypes = [
        CUfileHandle_t,
        c_void_p,
        c_size_t,
        c_ssize_t,
        c_ssize_t,
    ]
    ret = lib.cuFileWrite(
        handle,
        c_void_p(dev_ptr),
        c_size_t(size),
        c_ssize_t(file_offset),
        c_ssize_t(buf_offset),
    )
    if ret < 0:
        raise RuntimeError(f"cuFileWrite failed: ret={ret}")
    return ret


def open_for_gds(path: str, flags: int) -> int:
    """Open a file with O_DIRECT for GDS I/O. Returns the fd."""
    O_DIRECT = getattr(os, "O_DIRECT", 0)
    return os.open(path, flags | O_DIRECT, 0o644)
