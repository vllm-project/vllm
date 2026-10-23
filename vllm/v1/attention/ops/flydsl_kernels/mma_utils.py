# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MMA-atom calls on register vectors for the FlyDSL decode kernels."""

import flydsl.expr as fx


def _fragment(v):
    v = fx.Vector(v)
    frag = fx.make_rmem_tensor(v.numel, v.dtype)
    frag.store(v)
    return frag


def mma(atom, acc, a, b, **atom_state):
    """Issue ``atom`` once on per-lane operand vectors; return the new acc.

    ``atom_state`` is forwarded to ``fx.gemm`` (e.g. ``scale_a``/``scale_b``
    for scaled MFMA).
    """
    c = _fragment(acc)
    fx.gemm(atom, c, _fragment(a), _fragment(b), c, **atom_state)
    return fx.Vector(c.load())
