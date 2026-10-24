# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from ROCm/ATOM: atom/models/minimax_m3/mono/config.py
# Original code is MIT-licensed; see mono/LICENSE and the provenance in
# specialized_decode/README.md.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shapes the mono kernels are compiled for: one MiniMax-M3 TP4 rank.

Every value here is checked against the loaded model in ``weights.py``; a
model that does not match is never routed to the mono path.
"""

from __future__ import annotations

from dataclasses import dataclass

from .support.plan.shard import (
    Shard,
    ShardError,
)

# the execution model (one CTA per CU, all co-resident) is the framework's

# the model's full widths
Q_HEADS = 64
KV_HEADS = 4
INDEX_HEADS = 4
EXPERT_INTER = 3072  # a routed expert's intermediate width


@dataclass(frozen=True)
class Dims:
    """A TP rank's share of M3's widths. The kernels hold one k / v head and one
    index q head a rank: a TP size that gives a rank another count is refused."""

    tp: int

    def __post_init__(self):
        shard = Shard(self.tp)
        shard.split(Q_HEADS, "query heads")
        shard.split(EXPERT_INTER, "expert intermediate")
        for full, what in ((KV_HEADS, "kv heads"), (INDEX_HEADS, "index heads")):
            if shard.split(full, what) != 1:
                raise ShardError(
                    f"{full // self.tp} {what} a rank at TP {self.tp}: the kernels"
                    " hold one"
                )

    @property
    def q_heads(self) -> int:
        return Q_HEADS // self.tp

    @property
    def inter(self) -> int:
        return EXPERT_INTER // self.tp


# the TP sizes the kernels are built for
SUPPORTED_TP = (4,)
TP = 4
HIDDEN = 6144
HEAD_DIM = 128
ROTARY_DIM = 64
LOCAL_Q_HEADS = Dims(TP).q_heads


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
INTER = Dims(TP).inter  # expert intermediate per rank

SPARSE_BLOCK = 128
TOPK_BLOCKS = 16
MAX_SPARSE_KEYS = SPARSE_BLOCK * TOPK_BLOCKS
# the longest context served: the score region spans it
MAX_CONTEXT = 1 << 20
MAX_INDEX_BLOCKS = MAX_CONTEXT // SPARSE_BLOCK
PAGE16 = 16

MAX_TOKENS = 16  # tokens one mono step serves (the MFMA B operand holds 16)
# indexer context parallelism: a rank computes every index q head
MAX_QKV_ROWS = qkv_rows(TP)
# a request past this many index blocks is selected by the split stage's long
# path (with every index head: context-parallel); up to it one CTA ranks every
# block, one a thread (THREADS: a second a thread slows every context)
LONG_FROM_BLOCKS = 512


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
