# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025 FlyDSL Project Contributors
# mypy: ignore-errors
#
# This file contains code copied from FlyDSL (ROCm/FlyDSL PR #1204 at 21a3d1ee), as vendored by
# ROCm/ATOM PR #2435 (head 45e4b55d, atom/model_ops/monokernel/ipc.py). The original source code was
# licensed under the Apache License 2.0 and included the following copyright notice:
# Copyright (c) 2025 FlyDSL Project Contributors
# Modified by the vLLM project contributors (Apache-2.0 sec. 4(b)): import paths rewritten to this package;
#   the unused allocation helpers removed.

"""Small ctypes wrapper for the HIP IPC memory operations of the peer buffer."""

from __future__ import annotations

import ctypes
import os

HIP_IPC_HANDLE_BYTES = 64
HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1
_RANGE_START_ADDR = 11


class _HipIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * HIP_IPC_HANDLE_BYTES)]


class HipRuntime:
    """Load HIP once and expose the host operations used by peer-memory wrappers."""

    def __init__(self, library=None) -> None:
        self._library = library
        self._configured_library = None

    def library(self):
        """Return the configured HIP ctypes library, loading it on first use."""

        if self._library is None:
            sonames = ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so.5")
            candidates = []
            rocm_path = os.environ.get("ROCM_PATH")
            if rocm_path:
                candidates.extend(
                    os.path.join(rocm_path, lib_dir, name) for lib_dir in ("lib", "lib64") for name in sonames
                )
            candidates.extend(sonames)
            for name in candidates:
                try:
                    self._library = ctypes.CDLL(name)
                    break
                except OSError:
                    continue
            if self._library is None:
                raise RuntimeError("Failed to load HIP runtime library")
        self._configure(self._library)
        return self._library

    def _configure(self, library) -> None:
        if self._configured_library is library:
            return
        library.hipIpcGetMemHandle.restype = ctypes.c_int
        library.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_HipIpcMemHandle), ctypes.c_void_p]
        library.hipIpcOpenMemHandle.restype = ctypes.c_int
        library.hipIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _HipIpcMemHandle, ctypes.c_uint]
        library.hipIpcCloseMemHandle.restype = ctypes.c_int
        library.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        library.hipGetErrorString.restype = ctypes.c_char_p
        library.hipGetErrorString.argtypes = [ctypes.c_int]
        library.hipPointerGetAttribute.restype = ctypes.c_int
        library.hipPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self._configured_library = library

    def check(self, error: int, *, operation: str) -> None:
        """Raise a descriptive exception when a HIP call fails."""

        if int(error) == 0:
            return
        try:
            value = self.library().hipGetErrorString(int(error))
            message = value.decode("utf-8", errors="replace") if value else f"hipError({error})"
        except Exception:
            message = f"hipError({error})"
        raise RuntimeError(f"{operation} failed: {message}")

    def get_allocation_base(self, device_pointer: int) -> int:
        """Return the hipMalloc allocation base containing ``device_pointer``."""

        base = ctypes.c_void_p()
        error = self.library().hipPointerGetAttribute(
            ctypes.byref(base), ctypes.c_int(_RANGE_START_ADDR), ctypes.c_void_p(int(device_pointer))
        )
        self.check(error, operation="hipPointerGetAttribute(RANGE_START_ADDR)")
        if base.value is None:
            raise RuntimeError("hipPointerGetAttribute(RANGE_START_ADDR) returned a null allocation base")
        return int(base.value)

    def get_ipc_handle(self, allocation_base: int) -> bytes:
        """Export a fixed-size HIP IPC handle for an allocation base pointer."""

        handle = _HipIpcMemHandle()
        error = self.library().hipIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(int(allocation_base)))
        self.check(error, operation="hipIpcGetMemHandle")
        return bytes(ctypes.string_at(ctypes.byref(handle), HIP_IPC_HANDLE_BYTES))

    def open_ipc_handle(self, handle_bytes: bytes) -> int:
        """Map a peer allocation and return its local base pointer."""

        if len(handle_bytes) != HIP_IPC_HANDLE_BYTES:
            raise ValueError(f"expected a {HIP_IPC_HANDLE_BYTES}-byte HIP IPC handle, got {len(handle_bytes)} bytes")
        handle = _HipIpcMemHandle()
        ctypes.memmove(ctypes.byref(handle), bytes(handle_bytes), HIP_IPC_HANDLE_BYTES)
        mapped_base = ctypes.c_void_p()
        error = self.library().hipIpcOpenMemHandle(
            ctypes.byref(mapped_base), handle, ctypes.c_uint(HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)
        )
        self.check(error, operation="hipIpcOpenMemHandle")
        if mapped_base.value is None:
            raise RuntimeError("hipIpcOpenMemHandle returned a null mapped pointer")
        return int(mapped_base.value)

    def close_ipc_handle(self, mapped_base: int) -> None:
        """Close a peer mapping returned by :meth:`open_ipc_handle`."""

        error = self.library().hipIpcCloseMemHandle(ctypes.c_void_p(int(mapped_base)))
        self.check(error, operation="hipIpcCloseMemHandle")


_DEFAULT_RUNTIME = HipRuntime()


def get_allocation_base(device_pointer: int) -> int:
    """Return the base of the HIP allocation containing ``device_pointer``."""

    return _DEFAULT_RUNTIME.get_allocation_base(device_pointer)


def get_ipc_handle(allocation_base: int) -> bytes:
    """Export an IPC handle for ``allocation_base``."""

    return _DEFAULT_RUNTIME.get_ipc_handle(allocation_base)


def open_ipc_handle(handle_bytes: bytes) -> int:
    """Open a peer IPC handle and return its mapped base pointer."""

    return _DEFAULT_RUNTIME.open_ipc_handle(handle_bytes)


def close_ipc_handle(mapped_base: int) -> None:
    """Close a peer mapping returned by :func:`open_ipc_handle`."""

    _DEFAULT_RUNTIME.close_ipc_handle(mapped_base)


__all__ = ["close_ipc_handle", "get_allocation_base", "get_ipc_handle", "open_ipc_handle"]
