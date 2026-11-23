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
    c_bool,
    c_int,
    c_size_t,
    c_ssize_t,
    c_uint,
    c_void_p,
)
from enum import IntEnum

CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1
CU_FILE_SUCCESS = 0
# cufile.h numbers its errors CUFILEOP_BASE_ERR + n; 0 is success.
_CUFILEOP_BASE_ERR = 5000


class CUfileOpError(IntEnum):
    CU_FILE_SUCCESS = 0
    CU_FILE_DRIVER_NOT_INITIALIZED = _CUFILEOP_BASE_ERR + 1
    CU_FILE_DRIVER_INVALID_PROPS = _CUFILEOP_BASE_ERR + 2
    CU_FILE_DRIVER_UNSUPPORTED_LIMIT = _CUFILEOP_BASE_ERR + 3
    CU_FILE_DRIVER_VERSION_MISMATCH = _CUFILEOP_BASE_ERR + 4
    CU_FILE_DRIVER_VERSION_READ_ERROR = _CUFILEOP_BASE_ERR + 5
    CU_FILE_DRIVER_CLOSING = _CUFILEOP_BASE_ERR + 6
    CU_FILE_PLATFORM_NOT_SUPPORTED = _CUFILEOP_BASE_ERR + 7
    CU_FILE_IO_NOT_SUPPORTED = _CUFILEOP_BASE_ERR + 8
    CU_FILE_DEVICE_NOT_SUPPORTED = _CUFILEOP_BASE_ERR + 9
    CU_FILE_NVFS_DRIVER_ERROR = _CUFILEOP_BASE_ERR + 10
    CU_FILE_CUDA_DRIVER_ERROR = _CUFILEOP_BASE_ERR + 11
    CU_FILE_CUDA_POINTER_INVALID = _CUFILEOP_BASE_ERR + 12
    CU_FILE_CUDA_MEMORY_TYPE_INVALID = _CUFILEOP_BASE_ERR + 13
    CU_FILE_CUDA_POINTER_RANGE_ERROR = _CUFILEOP_BASE_ERR + 14
    CU_FILE_CUDA_CONTEXT_MISMATCH = _CUFILEOP_BASE_ERR + 15
    CU_FILE_INVALID_MAPPING_SIZE = _CUFILEOP_BASE_ERR + 16
    CU_FILE_INVALID_MAPPING_RANGE = _CUFILEOP_BASE_ERR + 17
    CU_FILE_INVALID_FILE_TYPE = _CUFILEOP_BASE_ERR + 18
    CU_FILE_INVALID_FILE_OPEN_FLAG = _CUFILEOP_BASE_ERR + 19
    CU_FILE_DIO_NOT_SET = _CUFILEOP_BASE_ERR + 20
    CU_FILE_INVALID_VALUE = _CUFILEOP_BASE_ERR + 22
    CU_FILE_MEMORY_ALREADY_REGISTERED = _CUFILEOP_BASE_ERR + 23
    CU_FILE_MEMORY_NOT_REGISTERED = _CUFILEOP_BASE_ERR + 24
    CU_FILE_PERMISSION_DENIED = _CUFILEOP_BASE_ERR + 25
    CU_FILE_DRIVER_ALREADY_OPEN = _CUFILEOP_BASE_ERR + 26
    CU_FILE_HANDLE_NOT_REGISTERED = _CUFILEOP_BASE_ERR + 27
    CU_FILE_HANDLE_ALREADY_REGISTERED = _CUFILEOP_BASE_ERR + 28
    CU_FILE_DEVICE_NOT_FOUND = _CUFILEOP_BASE_ERR + 29
    CU_FILE_INTERNAL_ERROR = _CUFILEOP_BASE_ERR + 30
    CU_FILE_GETNEWFD_FAILED = _CUFILEOP_BASE_ERR + 31
    CU_FILE_NVFS_SETUP_ERROR = _CUFILEOP_BASE_ERR + 33
    CU_FILE_IO_DISABLED = _CUFILEOP_BASE_ERR + 34
    CU_FILE_BATCH_SUBMIT_FAILED = _CUFILEOP_BASE_ERR + 35
    CU_FILE_GPU_MEMORY_PINNING_FAILED = _CUFILEOP_BASE_ERR + 36
    CU_FILE_BATCH_FULL = _CUFILEOP_BASE_ERR + 37
    CU_FILE_ASYNC_NOT_SUPPORTED = _CUFILEOP_BASE_ERR + 38
    CU_FILE_IO_MAX_ERROR = _CUFILEOP_BASE_ERR + 39


class CUfileBoolConfigParameter(IntEnum):
    """CUFileBoolConfigParameter_t from cufile.h.

    libcufile reads the numeric value, so these must track the C enum.
    Values only ever get appended, which is why allow_compat_mode is
    stable across cuFile releases.
    """

    CUFILE_PARAM_PROPERTIES_USE_POLL_MODE = 0
    CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE = 1
    CUFILE_PARAM_FORCE_COMPAT_MODE = 2
    CUFILE_PARAM_FS_MISC_API_CHECK_AGGRESSIVE = 3
    CUFILE_PARAM_EXECUTION_PARALLEL_IO = 4
    CUFILE_PARAM_PROFILE_NVTX = 5
    CUFILE_PARAM_PROPERTIES_ALLOW_SYSTEM_MEMORY = 6
    CUFILE_PARAM_USE_PCIP2PDMA = 7
    CUFILE_PARAM_PREFER_IO_URING = 8
    CUFILE_PARAM_FORCE_ODIRECT_MODE = 9
    CUFILE_PARAM_SKIP_TOPOLOGY_DETECTION = 10
    CUFILE_PARAM_STREAM_MEMOPS_BYPASS = 11


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

    # use_errno so sync-API failures can report the real errno.
    lib_path = ctypes.util.find_library("cufile")
    if lib_path is None:
        for candidate in ("libcufile.so.0", "libcufile.so"):
            try:
                _lib = ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL, use_errno=True)
                return _lib
            except OSError:
                continue
        raise OSError(
            "Cannot find libcufile.so. Ensure nvidia-cufile is installed "
            "and the library is on LD_LIBRARY_PATH."
        )
    _lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL, use_errno=True)
    return _lib


def _get_function(lib: ctypes.CDLL, name: str, min_version: str):
    try:
        return getattr(lib, name)
    except AttributeError as e:
        raise RuntimeError(
            f"libcufile does not export {name}; nvidia-cufile >= "
            f"{min_version} is required."
        ) from e


def _check_error(err: CUfileError_t, func_name: str) -> None:
    if err.err != CU_FILE_SUCCESS:
        try:
            err_name = CUfileOpError(err.err).name
        except ValueError:
            err_name = f"UNKNOWN({err.err})"
        raise RuntimeError(
            f"{func_name} failed: err={err.err} ({err_name}), cu_err={err.cu_err}"
        )


def cuFileSetParameterBool(param: CUfileBoolConfigParameter, value: bool) -> None:
    """Stage a boolean cuFile parameter.

    Must be called before cuFileDriverOpen(): staged values are applied at
    driver load, and the call fails with CU_FILE_DRIVER_ALREADY_OPEN once
    the driver is open.
    """
    lib = _get_lib()
    fn = _get_function(lib, "cuFileSetParameterBool", "1.14")
    fn.restype = CUfileError_t
    fn.argtypes = [c_int, c_bool]
    err = fn(int(param), c_bool(value))
    _check_error(err, f"cuFileSetParameterBool({param.name})")


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


def cuFileBufRegister(dev_ptr: int, size: int, flags: int = 0) -> None:
    """Register a device memory range for cuFile I/O."""
    lib = _get_lib()
    lib.cuFileBufRegister.restype = CUfileError_t
    lib.cuFileBufRegister.argtypes = [c_void_p, c_size_t, c_int]
    err = lib.cuFileBufRegister(c_void_p(dev_ptr), c_size_t(size), c_int(flags))
    _check_error(err, "cuFileBufRegister")


def cuFileBufDeregister(dev_ptr: int) -> None:
    lib = _get_lib()
    lib.cuFileBufDeregister.restype = CUfileError_t
    lib.cuFileBufDeregister.argtypes = [c_void_p]
    err = lib.cuFileBufDeregister(c_void_p(dev_ptr))
    _check_error(err, "cuFileBufDeregister")


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


def _io_error(op: str, ret: int) -> RuntimeError:
    # Sync I/O returns -1 with errno set for OS errors, or the negated
    # CUfileOpError for cuFile errors.
    if ret != -1:
        try:
            return RuntimeError(f"{op} failed: ret={ret} ({CUfileOpError(-ret).name})")
        except ValueError:
            pass
    err = ctypes.get_errno()
    return RuntimeError(f"{op} failed: ret={ret}, errno={err} ({os.strerror(err)})")


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
    lib.cuFileRead.argtypes = [
        CUfileHandle_t,
        c_void_p,
        c_size_t,
        c_ssize_t,
        c_ssize_t,
    ]
    ctypes.set_errno(0)
    ret = lib.cuFileRead(
        handle,
        c_void_p(dev_ptr),
        c_size_t(size),
        c_ssize_t(file_offset),
        c_ssize_t(buf_offset),
    )
    if ret < 0:
        raise _io_error("cuFileRead", ret)
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
    ctypes.set_errno(0)
    ret = lib.cuFileWrite(
        handle,
        c_void_p(dev_ptr),
        c_size_t(size),
        c_ssize_t(file_offset),
        c_ssize_t(buf_offset),
    )
    if ret < 0:
        raise _io_error("cuFileWrite", ret)
    return ret


def open_for_gds(path: str, flags: int) -> int:
    """Open a file with O_DIRECT for GDS I/O. Returns the fd."""
    O_DIRECT = getattr(os, "O_DIRECT", 0)
    return os.open(path, flags | O_DIRECT, 0o644)
