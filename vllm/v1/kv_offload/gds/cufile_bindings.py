# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ctypes bindings for NVIDIA cuFile (GDS) API."""

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

from vllm.logger import init_logger

logger = init_logger(__name__)

# --- Constants ---

CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1
CU_FILE_SUCCESS = 0

# Batch API constants
CUFILE_BATCH = 1
CUFILE_READ = 0
CUFILE_WRITE = 1

# Batch status flags
CUFILE_WAITING = 0x01
CUFILE_PENDING = 0x02
CUFILE_INVALID = 0x04
CUFILE_CANCELED = 0x08
CUFILE_COMPLETE = 0x10
CUFILE_TIMEOUT = 0x20
CUFILE_FAILED = 0x40


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


# --- Structures ---


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


# Opaque handle type
CUfileHandle_t = c_void_p

# CUstream is an opaque pointer
CUstream_t = c_void_p

# Batch handle type
CUfileBatchHandle_t = c_void_p


class _BatchParams(Structure):
    _fields_ = [
        ("devPtr_base", c_void_p),
        ("file_offset", c_ssize_t),  # off_t
        ("devPtr_offset", c_ssize_t),  # off_t
        ("size", c_size_t),
    ]


class _BatchUnion(Union):
    _fields_ = [
        ("batch", _BatchParams),
    ]


class CUfileIOParams_t(Structure):
    _fields_ = [
        ("mode", c_int),
        ("u", _BatchUnion),
        ("fh", CUfileHandle_t),
        ("opcode", c_int),
        ("cookie", c_void_p),
    ]


class CUfileIOEvents_t(Structure):
    _fields_ = [
        ("cookie", c_void_p),
        ("status", c_int),
        ("ret", c_ssize_t),
    ]


# --- Library loading ---

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


# --- Public API ---


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


def cuFileStreamRegister(stream_ptr: int, flags: int = 0) -> None:
    """Register a CUDA stream for cuFile async operations."""
    lib = _get_lib()
    lib.cuFileStreamRegister.restype = CUfileError_t
    lib.cuFileStreamRegister.argtypes = [CUstream_t, c_uint]
    err = lib.cuFileStreamRegister(CUstream_t(stream_ptr), c_uint(flags))
    _check_error(err, "cuFileStreamRegister")


def cuFileStreamDeregister(stream_ptr: int) -> None:
    """Deregister a CUDA stream from cuFile."""
    lib = _get_lib()
    lib.cuFileStreamDeregister.restype = CUfileError_t
    lib.cuFileStreamDeregister.argtypes = [CUstream_t]
    err = lib.cuFileStreamDeregister(CUstream_t(stream_ptr))
    _check_error(err, "cuFileStreamDeregister")


def cuFileReadAsync(
    handle: CUfileHandle_t,
    dev_ptr: int,
    size_p: ctypes.Array,
    file_offset_p: ctypes.Array,
    buf_offset_p: ctypes.Array,
    bytes_read_p: ctypes.Array,
    stream_ptr: int,
) -> None:
    """Enqueue an async read from file to GPU memory on a CUDA stream."""
    lib = _get_lib()
    lib.cuFileReadAsync.restype = CUfileError_t
    lib.cuFileReadAsync.argtypes = [
        CUfileHandle_t,
        c_void_p,
        POINTER(c_size_t),
        POINTER(c_ssize_t),
        POINTER(c_ssize_t),
        POINTER(c_ssize_t),
        CUstream_t,
    ]
    err = lib.cuFileReadAsync(
        handle,
        c_void_p(dev_ptr),
        size_p,
        file_offset_p,
        buf_offset_p,
        bytes_read_p,
        CUstream_t(stream_ptr),
    )
    _check_error(err, "cuFileReadAsync")


def cuFileWriteAsync(
    handle: CUfileHandle_t,
    dev_ptr: int,
    size_p: ctypes.Array,
    file_offset_p: ctypes.Array,
    buf_offset_p: ctypes.Array,
    bytes_written_p: ctypes.Array,
    stream_ptr: int,
) -> None:
    """Enqueue an async write from GPU memory to file on a CUDA stream."""
    lib = _get_lib()
    lib.cuFileWriteAsync.restype = CUfileError_t
    lib.cuFileWriteAsync.argtypes = [
        CUfileHandle_t,
        c_void_p,
        POINTER(c_size_t),
        POINTER(c_ssize_t),
        POINTER(c_ssize_t),
        POINTER(c_ssize_t),
        CUstream_t,
    ]
    err = lib.cuFileWriteAsync(
        handle,
        c_void_p(dev_ptr),
        size_p,
        file_offset_p,
        buf_offset_p,
        bytes_written_p,
        CUstream_t(stream_ptr),
    )
    _check_error(err, "cuFileWriteAsync")


def cuFileRead(
    handle: CUfileHandle_t,
    dev_ptr: int,
    size: int,
    file_offset: int = 0,
    buf_offset: int = 0,
) -> int:
    """Synchronous read from file to GPU memory. Returns bytes read."""
    # if size > 65536:
    #    logger.info(
    #        "cuFileRead: dev_ptr=0x%x size=%d file_offset=%d buf_offset=%d",
    #        dev_ptr,
    #        size,
    #        file_offset,
    #        buf_offset,
    #    )
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


# --- Batch API ---


def cuFileBatchIOSetUp(nr: int) -> CUfileBatchHandle_t:
    """Initialize a batch I/O context for nr operations."""
    lib = _get_lib()
    lib.cuFileBatchIOSetUp.restype = CUfileError_t
    lib.cuFileBatchIOSetUp.argtypes = [POINTER(CUfileBatchHandle_t), c_uint]
    batch_id = CUfileBatchHandle_t()
    err = lib.cuFileBatchIOSetUp(ctypes.byref(batch_id), c_uint(nr))
    _check_error(err, "cuFileBatchIOSetUp")
    return batch_id


def cuFileBatchIOSubmit(
    batch_id: CUfileBatchHandle_t,
    nr: int,
    params: ctypes.Array,
    flags: int = 0,
) -> None:
    """Submit nr batch I/O operations."""
    lib = _get_lib()
    lib.cuFileBatchIOSubmit.restype = CUfileError_t
    lib.cuFileBatchIOSubmit.argtypes = [
        CUfileBatchHandle_t,
        c_uint,
        POINTER(CUfileIOParams_t),
        c_uint,
    ]
    err = lib.cuFileBatchIOSubmit(batch_id, c_uint(nr), params, c_uint(flags))
    _check_error(err, "cuFileBatchIOSubmit")


def cuFileBatchIOGetStatus(
    batch_id: CUfileBatchHandle_t,
    min_nr: int,
    max_nr: int,
    events: ctypes.Array,
) -> int:
    """Poll for completed I/Os. Returns number of events retrieved."""
    lib = _get_lib()
    lib.cuFileBatchIOGetStatus.restype = CUfileError_t
    lib.cuFileBatchIOGetStatus.argtypes = [
        CUfileBatchHandle_t,
        c_uint,
        POINTER(c_uint),
        POINTER(CUfileIOEvents_t),
        c_void_p,
    ]
    nr = c_uint(max_nr)
    err = lib.cuFileBatchIOGetStatus(
        batch_id, c_uint(min_nr), ctypes.byref(nr), events, None
    )
    _check_error(err, "cuFileBatchIOGetStatus")
    return nr.value


def cuFileBatchIODestroy(batch_id: CUfileBatchHandle_t) -> None:
    """Destroy a batch I/O context."""
    lib = _get_lib()
    lib.cuFileBatchIODestroy.restype = None
    lib.cuFileBatchIODestroy.argtypes = [CUfileBatchHandle_t]
    lib.cuFileBatchIODestroy(batch_id)


def open_for_gds(path: str, flags: int) -> int:
    """Open a file with O_DIRECT for GDS I/O. Returns the fd."""
    O_DIRECT = getattr(os, "O_DIRECT", 0)
    return os.open(path, flags | O_DIRECT, 0o644)
