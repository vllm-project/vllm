# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# How many K/V sides one block's page-16 ids span. vLLM keeps both sides of a
# block in one allocation, so a block covers both sides' pages and V is reached
# from the same page id through a cache view offset by one side's pages; a layout
# holding the sides as separate planes spans one. Only the main KV cache pages
# this way: the index cache is one plane of whole blocks.
PAGE16_SIDES = 2
BLOCK_PAGES = PAGE16_SIDES * SPARSE_BLOCK // PAGE16  # page ids a block spans

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


class MonoUnsupported(Exception):
    """The loaded model or runtime configuration is outside what mono serves."""
