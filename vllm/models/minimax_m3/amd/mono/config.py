# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shapes the mono kernels are compiled for: one MiniMax-M3 TP4 rank.

Every value here is checked against the loaded model in ``weights.py``; a
model that does not match is never routed to the mono path.
"""

from __future__ import annotations

from dataclasses import dataclass

# One CTA per MI355X CU; the kernels rely on every CTA being co-resident.
BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64

HIDDEN = 6144
HEAD_DIM = 128
ROTARY_DIM = 64
LOCAL_Q_HEADS = 16


def qkv_rows(idx_heads: int) -> int:
    """Rows of a rank's fused projection, q | k | v | index_q | index_k: 16 q
    heads, one k and v head, ``idx_heads`` index q heads (the rank's one, or every
    one under indexer context parallelism) and the index k head."""
    return (LOCAL_Q_HEADS + 3 + idx_heads) * HEAD_DIM


O_K = LOCAL_Q_HEADS * HEAD_DIM

N_ROUTED = 128
TOP_K = 4
SHARED_EXPERT = N_ROUTED  # the fused shared expert's id
MOE_SLOTS = TOP_K + 1
INTER = 768  # expert intermediate per rank

SPARSE_BLOCK = 128
TOPK_BLOCKS = 16
MAX_SPARSE_KEYS = SPARSE_BLOCK * TOPK_BLOCKS
# the longest context served: the score region spans it
MAX_CONTEXT = 1 << 20
MAX_INDEX_BLOCKS = MAX_CONTEXT // SPARSE_BLOCK
PAGE16 = 16

MAX_TOKENS = 16  # tokens one mono step serves (the MFMA B operand holds 16)
TP = 4
# indexer context parallelism: a rank computes every index q head
MAX_QKV_ROWS = qkv_rows(TP)
# indexer context parallelism serves a step's requests past this many index
# blocks: its selection costs a fixed ~7 us (S = 1: ~43.5 us/layer at any length)
# that the one-head path's O(n^2) ranking passes between 258 and 297 blocks
# (S = 1 and S = 16 alike; logs/cp_threshold*.log)
INDEX_CP_FROM_BLOCKS = 288


@dataclass(frozen=True)
class IndexHeads:
    """The index q heads in a rank's fused projection, a build parameter:
    ``count`` of them (1, or all TP in head order under indexer context
    parallelism) and ``own``, the one this rank's selection scores."""

    count: int = 1
    own: int = 0

    def __post_init__(self):
        assert self.count in (1, TP) and 0 <= self.own < self.count

    @property
    def rows(self) -> int:
        return qkv_rows(self.count)

    @property
    def iq_off(self) -> int:
        return (LOCAL_Q_HEADS + 2 + self.own) * HEAD_DIM

    @property
    def ik_off(self) -> int:
        return (LOCAL_Q_HEADS + 2 + self.count) * HEAD_DIM


# without indexer context parallelism: the rank's own index q head only
ONE_INDEX_HEAD = IndexHeads()

LAYER_SLOTS = 128  # mailbox epochs: step * LAYER_SLOTS + layer + 1

# host K/V cache geometry, set by the runner before the kernels are built: the
# page-16 ids one sparse block spans (8: K and V as separate planes; 16: a block
# interleaves K|V and V is reached from the same id at +8 pages, as vLLM lays
# it out) and
# whether fp8 K/V carry the host's fixed unit scale instead of per-token scales,
# and the cache format: "fp8" (AITER sparse PA page-16 K/V, E4M3 index-K) or
# "nvfp4" (the ROCm NVFP4 pages of sparse_attention_nvfp4 / indexer_nvfp4: a page
# id is then page128 * 8 + its page-16 within, see NVFP4_* below)
KV_PAGE_STRIDE = SPARSE_BLOCK // PAGE16
STATIC_KV_SCALE = False
KV_FORMAT = "fp8"

# NVFP4 page geometry (128 tokens a page, 72 B a token: 64 B of e2m1 codes for
# 128 dims + 8 e4m3 scales of 16 dims): K side, then V side, of the one kv head
NVFP4_ROW = 72
NVFP4_CODES = HEAD_DIM // 2
NVFP4_SIDE = SPARSE_BLOCK * NVFP4_ROW
NVFP4_PAGE_BYTES = 2 * NVFP4_SIDE
NVFP4_INDEX_PAGE_BYTES = SPARSE_BLOCK * NVFP4_ROW


def configure(
    *, kv_page_stride: int, static_kv_scale: bool, kv_format: str = "fp8"
) -> None:
    global KV_PAGE_STRIDE, STATIC_KV_SCALE, KV_FORMAT
    assert kv_page_stride in (SPARSE_BLOCK // PAGE16, 2 * SPARSE_BLOCK // PAGE16)
    assert kv_format in ("fp8", "nvfp4")
    assert kv_format == "fp8" or kv_page_stride == SPARSE_BLOCK // PAGE16
    KV_PAGE_STRIDE = kv_page_stride
    STATIC_KV_SCALE = static_kv_scale
    KV_FORMAT = kv_format


def cache_geometry() -> tuple:
    """The cache geometry the kernels read while tracing. FlyDSL's JIT cache key
    does not see module values, so each builder captures this in its kernel body
    (``check_cache_geometry``) to keep one binary per geometry."""
    return (KV_PAGE_STRIDE, STATIC_KV_SCALE, KV_FORMAT)


def check_cache_geometry(geometry: tuple) -> None:
    """At trace time: the kernel is traced for the geometry it was built for."""
    assert geometry == cache_geometry(), (geometry, cache_geometry())


class MonoUnsupported(Exception):
    """The loaded model or runtime configuration is outside what mono serves."""
