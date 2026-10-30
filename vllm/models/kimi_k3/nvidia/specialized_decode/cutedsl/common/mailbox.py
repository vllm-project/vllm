# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lamport mailboxes: BF16 rows published in 16-byte fragments; a reader polls until no
fragment holds the -0.0 sentinel, then re-arms."""

import cutlass
import cutlass.cute as cute
from cutlass import Uint32

from .primitives import load_words, published, store_relaxed

SENTINEL = 0x80000000


def rearm(pointer):
    store_relaxed(pointer, (Uint32(SENTINEL),) * 4)


@cute.jit
def fetch(values, i: cutlass.Constexpr[int], pointer):
    """Load fragment ``i`` into ``values[i]``; returns its dirty flag at bit ``i``."""
    got = load_words(pointer, count=4, volatile=True)
    for k in cutlass.range_constexpr(4):
        values[i, k] = got[k]
    return Uint32(not published(got)) << i


@cute.jit
def poll(base, stride, count: cutlass.Constexpr[int]):
    """Lamport-poll ``count`` 16-byte fragments at base + i*stride (elements); all
    loads issue before any check and only dirty fragments re-poll."""
    values = cute.make_rmem_tensor((count, 4), Uint32)
    pending = Uint32(0)
    for i in cutlass.range_constexpr(count):
        pending = pending | fetch(values, i, base + i * stride)
    while pending != 0:
        again = Uint32(0)
        for i in cutlass.range_constexpr(count):
            if (pending >> i) & 1:
                again = again | fetch(values, i, base + i * stride)
        pending = again
    return values
