# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""POSIX ``madvise`` that releases the GIL.

``MADV_POPULATE_WRITE`` over a multi-hundred-GB region spends minutes in the
kernel faulting in pages. CPython calls the syscall without
``Py_BEGIN_ALLOW_THREADS`` (``Modules/mmapmodule.c``) — unlike ``mmap()`` in the
same module, which does release it — so ``mmap.madvise()`` on a background
thread would still stall the engine. ``ctypes.CDLL`` releases the GIL around
foreign calls, so we invoke ``madvise(2)`` directly instead.

Note that ``cudaHostRegister`` needs no such treatment: PyTorch already wraps it
in ``py::gil_scoped_release``.
"""

import ctypes
from functools import cache
from typing import Any


@cache
def _get_madvise() -> Any:
    library = ctypes.CDLL(None, use_errno=True)
    function = library.madvise
    function.restype = ctypes.c_int
    function.argtypes = (
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    )
    return function


def madvise(data_ptr: int, nbytes: int, advice: int) -> None:
    """Apply memory advice through a ctypes call that releases the GIL."""
    if _get_madvise()(data_ptr, nbytes, advice) != 0:
        error = ctypes.get_errno()
        raise OSError(error, "madvise failed")
