# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/runtime/abi.py
"""A kernel's argument list, named once.

A FlyDSL kernel and its launcher each spell out their positional parameters, and
the host passes them by position: an argument out of order is a pointer read as
another, silently. ``KernelAbi`` is the one list of names; the build checks the
kernel's and the launcher's parameters against it, and the host packs by name.
"""

import inspect
from dataclasses import dataclass


def _python_params(fn) -> tuple[str, ...]:
    """Parameter names of a ``@flyc.kernel`` / ``@flyc.jit`` object's Python
    function (FlyDSL keeps it as ``_func`` / ``func``)."""
    py = getattr(fn, "func", None) or getattr(fn, "_func", None)
    if py is None:
        raise TypeError(f"no Python function behind {fn!r}")
    return tuple(inspect.signature(py).parameters)


@dataclass(frozen=True)
class KernelAbi:
    names: tuple[str, ...]

    def check(self, kernel, launcher) -> None:
        """The kernel takes exactly ``names``; the launcher too, then ``stream``."""
        got = _python_params(kernel)
        if got != self.names:
            raise TypeError(f"kernel parameters {got} != ABI {self.names}")
        got = _python_params(launcher)
        if got != (*self.names, "stream"):
            raise TypeError(f"launcher parameters {got} != ABI {self.names} + stream")

    def pack(self, values: dict) -> list:
        """``values`` (every name, nothing else) in ABI order."""
        try:
            if len(values) == len(self.names):
                return [values[n] for n in self.names]
        except KeyError:
            pass
        extra = sorted(set(values) - set(self.names))
        missing = sorted(set(self.names) - set(values))
        raise TypeError(f"ABI mismatch: missing {missing}, unexpected {extra}")

    def zeros(self) -> list:
        """Every argument 0: what a compile-only call passes."""
        return [0] * len(self.names)
