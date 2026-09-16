#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that DeepSeek V4 and the JIT monitor leave the HIP symbols alone.

TileLang ships a libhip_stub.so that takes over `hipFree` in the global symbol
table once it is loaded, so neither the model import nor the JIT monitor may
pull TileLang in on ROCm. Both claims are about process-global state, which is
why this runs in an interpreter of its own.
"""

import ctypes
import sys

from vllm.model_executor.layers import mhc  # noqa: F401
from vllm.models import deepseek_v4  # noqa: F401
from vllm.utils import jit_monitor


class DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def _must_not_run() -> None:
    raise AssertionError("TileLang JIT monitor must not run on ROCm")


jit_monitor._active = False
jit_monitor._setup_triton_autotuning_print = lambda: None
jit_monitor._setup_triton_jit_hook = lambda: None
jit_monitor._setup_cutedsl_jit_hook = lambda: None
jit_monitor._setup_tilelang_jit_hook = _must_not_run
jit_monitor.activate()

imported = [
    name for name in sys.modules if name == "tilelang" or name.startswith("tilelang.")
]
assert not imported, f"TileLang was imported: {imported}"

libdl = ctypes.CDLL("libdl.so.2")
dlsym = libdl.dlsym
dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
dlsym.restype = ctypes.c_void_p
dladdr = libdl.dladdr
dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
dladdr.restype = ctypes.c_int

address = dlsym(None, b"hipFree")
assert address is not None, "hipFree is not available in the global symbol table"
info = DlInfo()
assert dladdr(address, ctypes.byref(info))
source = info.dli_fname.decode() if info.dli_fname else "<unknown>"
assert "libhip_stub.so" not in source, source
assert "libamdhip64.so" in source, source
print("OK")
