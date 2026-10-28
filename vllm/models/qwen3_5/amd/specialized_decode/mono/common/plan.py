# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/plan/build_key.py
# atom/mono/plan/execution.py
# atom/mono/plan/layout.py
# atom/mono/runtime/abi.py
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
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path

from vllm.models.qwen3_5.amd.specialized_decode.mono.abi import health_words

_SCALARS = (int, float, bool, str, enum.Enum)
_ROOT = Path(__file__).resolve().parents[1]


@functools.cache
def source_digest() -> str:
    """sha256 of every ``.py`` file of the ``mono`` package, each file's path and
    contents."""
    h = hashlib.sha256()
    for f in sorted(_ROOT.rglob("*.py")):
        h.update(str(f.relative_to(_ROOT)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def key_tuple(key, sources: str) -> tuple:
    """``key``'s (field name, value) pairs, every value a scalar, then
    ("sources", ``sources``): the model's ``source_digest``."""
    out = []
    for f in fields(key):
        value = getattr(key, f.name)
        if not isinstance(value, _SCALARS):
            raise TypeError(
                f"{type(key).__name__}.{f.name} = {value!r}: a build parameter must be"
                " a scalar to reach the JIT cache key"
            )
        out.append((f.name, value))
    out.append(("sources", sources))
    return tuple(out)


# The execution model every mono kernel is written for.
#
# One CTA per CU of an MI355X, all of them co-resident -- a spin wait on another
# CTA can only make progress if that CTA is running, so the runner refuses a GPU
# with another CU count -- and THREADS threads each.
#

BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64


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


class Space(Enum):
    SCRATCH = "scratch"  # this GPU's memory: device-scope hand-off between CTAs
    PEER = "peer"  # a peer-symmetric region: system-scope hand-off between GPUs
