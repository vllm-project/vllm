# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/plan/build_key.py
# atom/mono/runtime/abi.py
"""What the kernels and the host share: the health words, a build's JIT
cache key and a kernel's argument list.

A rank's health words: word FAILED is set once any in-kernel wait of the rank
gave up (sticky: its later waits exit at once, so a failed step runs to its
end), then a record a CTA of the first wait it gave up on: flag, the wide
kernels' site (`SITE_*`) or `mono`'s mailbox region id and pair index, the
counter or tag seen, the target, CTA, clock.

The output rank's flags: one int32 word a TP rank in the first PEER_HEAD bytes
of rank 0's peer buffer. A rank whose wait gives up raises its word there and
drains the store before it sets FAILED. `runtime.guards.FailStop` reads
them after every step.

FlyDSL keys a compiled kernel by the scalars its launcher's closures capture;
a parameter used only by build-time Python would not be keyed, and two builds
differing only there would share one binary (every TP rank once ran rank 3's
kernel that way). So a build's parameters are one frozen dataclass of
scalars, `key_tuple` turns it into one (name, value) tuple, and the launcher
references that tuple. The tuple also carries the digest of the kernel
sources (`source_digest`): FlyDSL does not reach every helper's source when it
hashes a kernel.

A FlyDSL kernel and its launcher each spell out their positional parameters
and the host passes them by position, so `KernelAbi` is the one list of
names: the build checks both signatures against it and the host packs by name.
"""

import enum
import functools
import hashlib
import inspect
from dataclasses import dataclass, fields
from pathlib import Path

RECORD_WORDS = 8
FAILED = 0
PEER_HEAD = 256  # bytes: the output rank's flags, a word a rank
WAIT_SECONDS = 10  # an in-kernel wait gives up after this long
WAIT_TICKS = WAIT_SECONDS * 100_000_000  # s_memrealtime ticks at 100 MHz

# the wide kernels' wait sites a record names
SITE_GRID = 1  # a grid barrier
SITE_PEER = 2  # a TP exchange line
SITE_FOLD = 3  # a split-K fold's arrival count


def health_words(ctas: int) -> int:
    """A rank's health words for a grid of `ctas`: the header, a record a CTA."""
    return (ctas + 1) * RECORD_WORDS


_SCALARS = (int, float, bool, str, enum.Enum)
_ROOT = Path(__file__).resolve().parent


@functools.cache
def source_digest() -> str:
    """sha256 of every `.py` file of the package, path and contents."""
    h = hashlib.sha256()
    for f in sorted(_ROOT.rglob("*.py")):
        h.update(str(f.relative_to(_ROOT)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def key_tuple(key) -> tuple:
    """`key`'s (field name, value) pairs, every value a scalar, then the
    kernel sources' digest."""
    out = []
    for f in fields(key):
        value = getattr(key, f.name)
        if not isinstance(value, _SCALARS):
            raise TypeError(
                f"{type(key).__name__}.{f.name} = {value!r}: a build parameter must "
                "be a scalar to reach the JIT cache key"
            )
        out.append((f.name, value))
    out.append(("sources", source_digest()))
    return tuple(out)


def _python_params(fn) -> tuple[str, ...]:
    """Parameter names of a `@flyc.kernel` / `@flyc.jit` object's Python
    function (FlyDSL keeps it as `_func` / `func`)."""
    py = getattr(fn, "func", None) or getattr(fn, "_func", None)
    if py is None:
        raise TypeError(f"no Python function behind {fn!r}")
    return tuple(inspect.signature(py).parameters)


@dataclass(frozen=True)
class KernelAbi:
    names: tuple[str, ...]

    def check(self, kernel, launcher) -> None:
        """The kernel takes exactly `names`; the launcher too, then `stream`."""
        got = _python_params(kernel)
        if got != self.names:
            raise TypeError(f"kernel parameters {got} != ABI {self.names}")
        got = _python_params(launcher)
        if got != (*self.names, "stream"):
            raise TypeError(f"launcher parameters {got} != ABI {self.names} + stream")

    def pack(self, values: dict) -> list:
        """`values` (every name, nothing else) in ABI order."""
        if set(values) != set(self.names):
            extra = sorted(set(values) - set(self.names))
            missing = sorted(set(self.names) - set(values))
            raise TypeError(f"ABI mismatch: missing {missing}, unexpected {extra}")
        return [values[n] for n in self.names]

    def zeros(self) -> list:
        """Every argument 0: what a compile-only call passes."""
        return [0] * len(self.names)
