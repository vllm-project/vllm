# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Symmetric peer buffers: one allocation per TP rank, mapped on every rank."""

from __future__ import annotations

import ctypes
import os

import torch

_IPC_HANDLE_BYTES = 64
_LAZY_ENABLE_PEER_ACCESS = 0x1
_RANGE_START_ADDR = 11


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * _IPC_HANDLE_BYTES)]


_hip = None


def _lib():
    global _hip
    if _hip is not None:
        return _hip
    names = [f"libamdhip64.so{v}" for v in ("", ".7", ".6")]
    root = os.environ.get("ROCM_PATH")
    paths = [os.path.join(root, "lib", n) for n in names] if root else []
    for name in paths + names:
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        raise RuntimeError("MonoKernel peer memory needs the HIP runtime library")
    lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
    lib.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _IpcHandle,
        ctypes.c_uint,
    ]
    lib.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    lib.hipPointerGetAttribute.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.hipGetErrorString.restype = ctypes.c_char_p
    _hip = lib
    return lib


def _check(error: int, what: str) -> None:
    if error:
        raise RuntimeError(f"{what} failed: {_lib().hipGetErrorString(error).decode()}")


def _export(pointer: int) -> tuple[bytes, int]:
    base = ctypes.c_void_p()
    _check(
        _lib().hipPointerGetAttribute(
            ctypes.byref(base), _RANGE_START_ADDR, ctypes.c_void_p(pointer)
        ),
        "hipPointerGetAttribute",
    )
    handle = _IpcHandle()
    _check(_lib().hipIpcGetMemHandle(ctypes.byref(handle), base), "hipIpcGetMemHandle")
    return bytes(handle.reserved), pointer - (base.value or 0)


def _open(handle: bytes) -> int:
    h = _IpcHandle()
    ctypes.memmove(ctypes.byref(h), handle, _IPC_HANDLE_BYTES)
    mapped = ctypes.c_void_p()
    _check(
        _lib().hipIpcOpenMemHandle(ctypes.byref(mapped), h, _LAZY_ENABLE_PEER_ACCESS),
        "hipIpcOpenMemHandle",
    )
    return mapped.value or 0


class PeerBuffers:
    """``nbytes`` of zeroed device memory on every TP rank; ``addresses[r]`` is
    rank r's buffer as mapped in this process. Collective: every rank builds it
    at the same point, and every rank must call ``close`` in the same order."""

    def __init__(self, nbytes: int, rank: int, npes: int, group) -> None:
        import torch.distributed as dist

        self.rank, self.npes, self.group = rank, npes, group
        device = torch.device("cuda", torch.accelerator.current_device_index())
        self.storage = torch.zeros(nbytes, dtype=torch.uint8, device=device)
        local = self.storage.data_ptr()
        self._remote: list[int] = []
        if npes == 1:
            pointers = [local]
        else:
            mine: tuple | str
            try:
                mine = _export(local)
            except RuntimeError as err:
                mine = str(err)
            exported: list = [None] * npes
            dist.all_gather_object(exported, mine, group=group)
            failed = [
                f"rank {r}: {e}" for r, e in enumerate(exported) if isinstance(e, str)
            ]
            if failed:
                raise RuntimeError(
                    "MonoKernel peer export failed: " + "; ".join(failed)
                )
            pointers, error = [], None
            for r, (handle, offset) in enumerate(exported):
                if r == rank:
                    pointers.append(local)
                    continue
                try:
                    base = _open(handle)
                except RuntimeError as e:
                    error = str(e)
                    break
                self._remote.append(base)
                pointers.append(base + offset)
            status: list = [None] * npes
            dist.all_gather_object(status, error, group=group)
            failed = [f"rank {r}: {e}" for r, e in enumerate(status) if e]
            if failed:
                self._close_remote()
                raise RuntimeError("MonoKernel peer open failed: " + "; ".join(failed))
            dist.barrier(group=group)
        self.addresses = torch.tensor(pointers, dtype=torch.int64, device=device)

    def _close_remote(self) -> None:
        for base in self._remote:
            _lib().hipIpcCloseMemHandle(ctypes.c_void_p(base))
        self._remote.clear()

    def close(self) -> None:
        torch.accelerator.synchronize(self.storage.device)
        if self.npes > 1:
            import torch.distributed as dist

            dist.barrier(group=self.group)
        self._close_remote()
