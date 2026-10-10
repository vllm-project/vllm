# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/plan/execution.py
# atom/mono/plan/layout.py
# atom/mono/plan/arith.py
# atom/mono/plan/shard.py
# atom/mono/plan/build_key.py
"""Plain-Python planning of the mono kernels: the execution model (one CTA a CU,
all resident), mailbox layouts, arithmetic that runs on ints and traced values
alike, TP shards, and build keys (every build parameter, and this package's
sources, in FlyDSL's JIT cache key)."""

import enum
import functools
import hashlib
from dataclasses import dataclass, fields
from pathlib import Path

# ---------------------------------------------------------------- execution model
BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64


def first_task(bid, base):
    """CTA ``bid``'s first task of a stage placed from CTA ``base`` (any count of
    CTAs past 0: placements that add task counts run past the grid), then every
    BLOCKS-th."""
    return (bid - base % BLOCKS + BLOCKS) % BLOCKS


# ---------------------------------------------------------------- mailbox layouts
PAIR_BYTES = 8  # a (value, tag) pair


def pair_layout(items, align: int = PAIR_BYTES, start: int = 0) -> dict:
    """``items`` (name, pairs) laid out in order: name -> (byte offset, bytes),
    each region from an ``align``-byte boundary at or past ``start``."""
    out, off = {}, start
    for name, pairs in items:
        off = (off + align - 1) // align * align
        out[name] = (off, pairs * PAIR_BYTES)
        off += pairs * PAIR_BYTES
    return out


# ---------------------------------------------------------------- arithmetic
def sel(pred, a, b):
    """A if pred else b: a conditional on Python values, ``select`` on traced
    ones (rewrapped in the traced operand's type)."""
    if isinstance(pred, bool):
        return a if pred else b
    out = pred.select(a, b)
    traced = [x for x in (a, b) if not isinstance(x, int)]
    return type(traced[0])(out) if traced else out


def imin(a, b):
    return sel(a < b, a, b)


def i32(x):
    """``x``, asserted inside the device's Int32 range when it is a Python int."""
    if isinstance(x, int):
        assert -(2**31) <= x < 2**31, x
    return x


def cdiv(a, b):
    """ceil(a / b) for a >= 0, b > 0."""
    return i32((a + b - 1) // b)


# ---------------------------------------------------------------- TP shards
class ShardError(ValueError):
    """A width the TP size does not divide, or a TP size the model refuses."""


@dataclass(frozen=True)
class Shard:
    tp: int

    def __post_init__(self):
        if self.tp < 1:
            raise ShardError(f"TP {self.tp}")

    def split(self, full: int, what: str) -> int:
        """``full`` / tp, refusing a remainder: a rank's share of ``what``."""
        if full % self.tp:
            raise ShardError(f"{what} {full} is not divisible by TP {self.tp}")
        return full // self.tp


def padded(n: int, align: int) -> int:
    return cdiv(n, align) * align


# ---------------------------------------------------------------- build keys
_SCALARS = (int, float, bool, str, enum.Enum)
_ROOT = Path(__file__).resolve().parents[1]  # the mono package


@functools.cache
def source_digest(*paths: str) -> str:
    """sha256 of every ``.py`` file under ``paths`` (files or directories,
    relative to the mono package), each file's path and contents: one model's
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
