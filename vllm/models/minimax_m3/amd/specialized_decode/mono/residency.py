# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The HIP driver's occupancy of a compiled FlyDSL kernel: a persistent grid
needs every CTA resident at once."""

import ctypes
import functools

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
    return ctypes.CDLL("libamdhip64.so")


def active_workgroups_per_cu(compiled, name: str, threads: int) -> int:
    """``compiled``: a ``flyc.compile`` result; ``name``: its kernel symbol."""
    hip = _hip()
    module, func, count = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_int()
    image = code_object(compiled._keepalive._ir_text)
    if hip.hipModuleLoadData(ctypes.byref(module), image) != 0:
        raise RuntimeError(f"hipModuleLoadData failed for {name}")
    try:
        if hip.hipModuleGetFunction(ctypes.byref(func), module, name.encode()) != 0:
            raise RuntimeError(f"hipModuleGetFunction failed for {name}")
        rc = hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(count), func, threads, 0
        )
        if rc != 0:
            raise RuntimeError(
                f"hipModuleOccupancyMaxActiveBlocksPerMultiprocessor: {rc}"
            )
    finally:
        hip.hipModuleUnload(module)
    return count.value
