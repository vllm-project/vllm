# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from ROCm/ATOM: atom/mono/plan/build_key.py
# Original code is MIT-licensed; see mono/LICENSE and the provenance in
# specialized_decode/README.md.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A kernel build's parameters, all of them in FlyDSL's JIT cache key.

FlyDSL keys a compiled kernel by the scalar values (int, float, bool, str, enum,
tuple) its launcher's closures capture; an object in a closure is not keyed, and
a parameter used only by build-time Python (a placement table, a layout) is in no
closure at all. Two builds differing only there would share one binary -- every
TP rank ran rank 3's kernel once that way. So a build's parameters are one frozen
dataclass of scalars, ``key_tuple`` turns it into one (name, value) tuple, and the
launcher references that tuple: every field is keyed, whatever the body uses.

The tuple also carries the model's kernel sources (``mono.sources.SOURCES``): FlyDSL
keys a helper's source only when its walk reaches it, and it does not reach a
helper called inside a traced function's dynamic loop or branch (the rewriter
moves those bodies into nested functions it does not scan) nor one outside the
kernel's directory -- an edit there kept the old binary.
"""

import enum
from dataclasses import fields

_SCALARS = (int, float, bool, str, enum.Enum)


def key_tuple(key, sources: str) -> tuple:
    """``key``'s (field name, value) pairs, every value a scalar, then
    ("sources", ``sources``): the model's source digest."""
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
