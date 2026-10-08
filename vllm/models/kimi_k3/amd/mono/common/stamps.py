# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/stamps.py
"""A timeline build's per-CTA phase stamps (debug only; ``ATOM_MONO_TIMELINE``,
saved by ``atom.mono.runtime.timeline``): the s_memrealtime (100 MHz) at which
the CTA passes point k, kept in LDS until the kernel ends -- a global store per
stamp would sit on every later ``s_waitcnt vmcnt(0)``.

``on`` is the build's compile-time switch: off, every call traces to nothing.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr

from vllm.models.kimi_k3.amd.mono.common.ops import memrealtime, traced


@traced
def stamp_begin(on, tls, tid, points):
    """Zero the LDS record, then point 0."""
    if const_expr(on):  # noqa: SIM102
        if tid < points:  # the stamping thread's wave: ordered
            fx.ptr_store(fx.Int64(0), tls + tid)
    stamp(on, tls, tid, 0)


@traced
def stamp(on, tls, tid, k):
    # compile-time gate outside, traced condition inside: they cannot be one `and`
    if const_expr(on):  # noqa: SIM102
        if tid == 0:
            fx.ptr_store(memrealtime(), tls + k)


@traced
def stamp_flush(on, tls, tl, tid, bid, points):
    """The CTA's record -> ``tl`` [CTA][points] int64."""
    if const_expr(on):  # noqa: SIM102
        if tid < points:
            fx.generic_store(
                fx.inttoptr(
                    fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                    tl + fx.Int64((bid * points + tid) * 8),
                ),
                fx.ptr_load(tls + tid),
            )
