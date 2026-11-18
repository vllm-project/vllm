# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: B008
"""Standalone launches of the verify-attention fragment (`attn_frag.py`) for the
boundary layers (`FIRST_LAYER` and `LAST_LAYER + 1`), whose attention no
persistent launch fuses. A separate module, so the fused builds' cache keys
(the hash of `attn_frag.py`) do not change."""

import flydsl.expr as fx
from flydsl.expr.typing import Int32, Int64

from .dev import BLOCKS, CM_DEV, THREADS, ld_i32, uniform

HEALTH_WORDS = 8 + 8 * BLOCKS


def standalone(fa):
    """A one-layer launch of fragment `fa` on BLOCKS CTAs.

    The tag is `epoch * 64 + layer + 1` from the device epoch word, as in the
    persistent launches: a host argument would be fixed at CUDA graph capture.
    """
    import flydsl.compiler as flyc

    # FlyDSL does not key on the closure: the build key goes into the traced
    # source, so each fragment build gets its own binary.
    keystr = fa.key

    @flyc.kernel(name=f"muse_bnd_{fa.key}", known_block_size=[THREADS, 1, 1])
    def _k(dptr: fx.Int64, health: fx.Int64, epoch: fx.Int64, layer: fx.Int32):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        alloc = fx.SharedAllocator()
        L = alloc.allocate(fa.Lds).peek()
        tag = (uniform(ld_i32(epoch, 0, CM_DEV)) & 0x1FFFFFF) * 64 + layer + 1
        fa.run(dptr, tag, health, tid, bid, lane, wave, L)

    @flyc.jit
    def _launch(
        dptr: Int64,
        health: Int64,
        epoch: Int64,
        layer: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = keystr
        _k(dptr, health, epoch, layer).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    return _launch
