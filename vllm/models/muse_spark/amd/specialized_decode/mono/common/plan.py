# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/plan/build_key.py
# atom/mono/plan/execution.py
# atom/mono/plan/layout.py
# atom/mono/runtime/abi.py
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/plan/arith.py
"""A kernel build's parameters, all of them in FlyDSL's JIT cache key.

FlyDSL keys a compiled kernel by the scalar values (int, float, bool, str, enum,
tuple) its launcher's closures capture; an object in a closure is not keyed, and
a parameter used only by build-time Python (a placement table, a layout) is in no
closure at all. Two builds differing only there would share one binary -- every
TP rank ran rank 3's kernel once that way. So a build's parameters are one frozen
dataclass of scalars, ``key_tuple`` turns it into one (name, value) tuple, and the
launcher references that tuple: every field is keyed, whatever the body uses.

The tuple also carries the model's kernel sources (``source_digest``): FlyDSL
keys a helper's source only when its walk reaches it, and it does not reach a
helper called inside a traced function's dynamic loop or branch (the rewriter
moves those bodies into nested functions it does not scan) nor one outside the
kernel's directory -- an edit there kept the old binary.
"""

import enum
import functools
import hashlib
import inspect
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path

from vllm.models.muse_spark.amd.specialized_decode.mono.abi import health_words

_SCALARS = (int, float, bool, str, enum.Enum)
_ROOT = Path(__file__).resolve().parents[1]


@functools.cache
def source_digest(*paths: str) -> str:
    """sha256 of every ``.py`` file under ``paths`` (files or directories,
    relative to the ``mono`` package), each file's path and contents: one model's
    kernel sources, so an edit to another model's rebuilds none of these."""
    h = hashlib.sha256()
    for rel in paths:
        root = _ROOT / rel
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        if not files:
            raise FileNotFoundError(f"no kernel sources at {root}")
        for f in files:
            h.update(str(f.relative_to(_ROOT)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _key_items(key, prefix: str = "") -> list:
    out = []
    for f in fields(key):
        value = getattr(key, f.name)
        if is_dataclass(value):
            out += _key_items(value, f"{prefix}{f.name}.")
            continue
        if not isinstance(value, _SCALARS):
            raise TypeError(
                f"{type(key).__name__}.{f.name} = {value!r}: a build parameter must be"
                " a scalar (or a dataclass of them) to reach the JIT cache key"
            )
        out.append((prefix + f.name, value))
    return out


def key_tuple(key, sources: str) -> tuple:
    """``key``'s (field name, value) pairs, every value a scalar (a nested
    dataclass flattened, its fields named ``outer.inner``), then ("sources",
    ``sources``): the model's ``source_digest``."""
    return tuple(_key_items(key) + [("sources", sources)])


# The execution model every mono kernel is written for.
#
# One CTA per CU of an MI355X, all of them co-resident -- a spin wait on another
# CTA can only make progress if that CTA is running, so the runner refuses a GPU
# with another CU count -- and THREADS threads each.
#

BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64
LDS_BYTES = 160 * 1024  # a CU's LDS: one CTA's whole


# Arithmetic that runs on Python ints and traced values alike.


def sel(pred, a, b):
    """A if pred else b: a conditional on Python values, ``select`` on traced
    ones (rewrapped in the traced operand's type)."""
    if isinstance(pred, bool):
        return a if pred else b
    out = pred.select(a, b)
    traced = [x for x in (a, b) if not isinstance(x, int)]
    return type(traced[0])(out) if traced else out


def i32(x):
    """``x``, asserted inside the device's Int32 range when it is a Python int."""
    if isinstance(x, int):
        assert -(2**31) <= x < 2**31, x
    return x


def cdiv(a, b):
    """ceil(a / b) for a >= 0, b > 0."""
    return i32((a + b - 1) // b)


def padded(n: int, align: int) -> int:
    return cdiv(n, align) * align


# A kernel's mailbox regions laid out in its scratch or peer buffer.

PAIR_BYTES = 8  # a (value, tag) pair
# A rank's health words (``mono.abi``), for this grid.
HEALTH_WORDS = health_words(BLOCKS)


def pair_layout(items, align: int = PAIR_BYTES, start: int = 0) -> dict:
    """``items`` (name, pairs) laid out in order: name -> (byte offset, bytes),
    each region from an ``align``-byte boundary at or past ``start``."""
    out, off = {}, start
    for name, pairs in items:
        off = (off + align - 1) // align * align
        out[name] = (off, pairs * PAIR_BYTES)
        off += pairs * PAIR_BYTES
    return out


# A kernel's argument list, named once.
#
# A FlyDSL kernel and its launcher each spell out their positional parameters, and
# the host passes them by position: an argument out of order is a pointer read as
# another, silently. ``KernelAbi`` is the one list of names; the build checks the
# kernel's and the launcher's parameters against it, and the host packs by name.
#


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


# The two spaces a mailbox region lives in (``sync.Addr``).


class Space(Enum):
    SCRATCH = "scratch"  # this GPU's memory: device-scope hand-off between CTAs
    PEER = "peer"  # a peer-symmetric region: system-scope hand-off between GPUs
