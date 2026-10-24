# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The HIP driver's occupancy of a compiled FlyDSL kernel: a persistent grid
needs every CTA resident at once."""

import ctypes
import functools
import os
from pathlib import Path

import regex as re


def code_object(ir_text: str) -> bytes:
    """The linked ELF of a FlyDSL artifact's ``gpu.binary`` (MLIR escapes
    non-printable bytes as ``\\XX``)."""
    m = re.search(r'bin = "((?:[^"\\]|\\.)*)"', ir_text, re.S)
    if m is None:
        raise RuntimeError("FlyDSL artifact holds no gpu.binary")
    src, out, i = m.group(1), bytearray(), 0
    while i < len(src):
        if src[i] != "\\":
            out.append(ord(src[i]))
            i += 1
        elif src[i + 1] == "\\":
            out.append(0x5C)
            i += 2
        else:
            out.append(int(src[i + 1 : i + 3], 16))
            i += 3
    if out[:4] != b"\x7fELF":
        raise RuntimeError("FlyDSL gpu.binary is not an ELF code object")
    return bytes(out)


@functools.cache
def _hip():
    hip = ctypes.CDLL("libamdhip64.so")
    ptr, out = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
    hip.hipModuleLoadData.argtypes = [out, ctypes.c_char_p]
    hip.hipModuleGetFunction.argtypes = [out, ptr, ctypes.c_char_p]
    hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ptr,
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    hip.hipModuleUnload.argtypes = [ptr]
    hip.hipGetErrorName.argtypes = [ctypes.c_int]
    hip.hipGetErrorName.restype = ctypes.c_char_p
    for f in (
        hip.hipModuleLoadData,
        hip.hipModuleGetFunction,
        hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor,
        hip.hipModuleUnload,
    ):
        f.restype = ctypes.c_int
    return hip


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what}: {_hip().hipGetErrorName(rc).decode()}")


def active_workgroups_per_cu(compiled, name: str, threads: int) -> int:
    """``compiled``: a ``flyc.compile`` result; ``name``: its kernel symbol."""
    hip = _hip()
    module, func, count = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_int()
    image = code_object(compiled._keepalive._ir_text)
    _check(hip.hipModuleLoadData(ctypes.byref(module), image), f"load {name}")
    try:
        _check(
            hip.hipModuleGetFunction(ctypes.byref(func), module, name.encode()),
            f"find {name}",
        )
        _check(
            hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                ctypes.byref(count), func, threads, 0
            ),
            f"occupancy of {name}",
        )
    finally:
        hip.hipModuleUnload(module)
    return count.value


def _writable_flydsl_cache() -> None:
    """AITER points FlyDSL at its bundled read-only cache."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)
