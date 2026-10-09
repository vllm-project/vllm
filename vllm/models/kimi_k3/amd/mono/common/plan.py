# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/plan/build_key.py
# atom/mono/plan/execution.py
# atom/mono/plan/layout.py
# atom/mono/runtime/abi.py
# atom/mono/plan/trace.py
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
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path

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


def symbol_params(key) -> dict:
    """The fields that name the kernel symbol (``metadata={"sym": short name}``),
    short name -> value, in field order."""
    return {
        f.metadata["sym"]: getattr(key, f.name)
        for f in fields(key)
        if "sym" in f.metadata
    }


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


def first_task(bid, base):
    """CTA ``bid``'s first task of a stage placed from CTA ``base`` (any count of
    CTAs past 0: placements that add task counts run past the grid), then every
    BLOCKS-th."""
    return (bid - base % BLOCKS + BLOCKS) % BLOCKS


# A kernel's mailbox regions laid out in its scratch or peer buffer.

PAIR_BYTES = 8  # a (value, tag) pair
# a debug build's wait record, one a CTA: flag, region id, pair index, tag,
# seen tag, CTA, clock, pad (``vllm.models.kimi_k3.amd.mono.common.sync._spin_bounded``)
DIAG_WORDS = 8


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


# A kernel's mailbox accesses, recorded while FlyDSL traces it.
#
# Tracing runs the kernel body's Python once, so every mailbox put / poll executes
# once as Python: noting it here costs the device nothing (no IR is emitted). The
# stage currently being traced is set by ``enter_stage``, called where the program
# starts each stage, so the record is ``(stage, region, put | poll)`` in program
# order.
#
# Everything here is plain functions on a module-level recorder: FlyDSL's AST
# rewriter carries every local whose method is called inside a dynamic if / for as
# loop state, so a kernel body must not call methods on these objects.
#


class Space(Enum):
    SCRATCH = "scratch"  # this GPU's memory: device-scope hand-off between CTAs
    PEER = "peer"  # a peer-symmetric region: system-scope hand-off between GPUs


@dataclass(frozen=True)
class Access:
    stage: str
    region: str
    space: Space
    kind: str  # "put" | "poll"


@dataclass
class Record:
    stages: list[str] = field(default_factory=list)  # in the order entered
    accesses: list[Access] = field(default_factory=list)
    unnamed: list[tuple[str, str]] = field(default_factory=list)  # (stage, kind)


_record: Record | None = None


@contextmanager
def recording():
    """Record the accesses of kernels traced inside this block."""
    global _record
    outer, _record = _record, Record()
    try:
        yield _record
    finally:
        _record = outer


def enter_stage(name: str) -> None:
    if _record is not None:
        _record.stages.append(name)


def _stage() -> str:
    assert _record is not None
    return _record.stages[-1] if _record.stages else "<before any stage>"


def note(region: str, space: Space, kind: str) -> None:
    if _record is not None:
        _record.accesses.append(Access(_stage(), region, space, kind))


def note_unnamed(kind: str) -> None:
    """A put / poll on a raw address: the check reports it (every mailbox access of
    a checked kernel names its region)."""
    if _record is not None:
        _record.unnamed.append((_stage(), kind))
