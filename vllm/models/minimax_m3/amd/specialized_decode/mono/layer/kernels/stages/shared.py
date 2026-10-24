# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from ROCm/ATOM: atom/models/minimax_m3/mono/kernels/stages/shared.py
# Original code is MIT-licensed; see mono/layer/LICENSE and the provenance in
# specialized_decode/README.md.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K4's module-level constants and helpers, shared by post_attn.py and the stage
modules."""

import struct

from ...config import (
    HEAD_DIM,
    HIDDEN,
    INTER,
    LONG_FROM_BLOCKS,
    MAX_INDEX_BLOCKS,
    MAX_TOKENS,
    N_ROUTED,
    PAGE16,
    SPARSE_BLOCK,
    TOPK_BLOCKS,
    TP,
)
from ...layout import (
    N_ROUTER,
    N_SPLIT,
)
from ...support.plan.execution import (
    THREADS,
)

NEG = -3.4e38  # the gluon decode's masked-score value
PAGE_BYTES = PAGE16 * HEAD_DIM
XN_SLICE = HIDDEN // N_ROUTER  # normalized-input elements each router task publishes
W13_ROWS = 2 * INTER  # gate rows then up rows (GGUU)
W13_BYTES = W13_ROWS * HIDDEN // 2  # one expert, fp4
W2_BYTES = HIDDEN * INTER // 2
W13_SCALE_COLS = HIDDEN // 32
W2_SCALE_COLS = INTER // 32
S2_BYTES = (N_ROUTED + 1) * HIDDEN * W2_SCALE_COLS  # e8m0, every expert
S13_BYTES = (N_ROUTED + 1) * W13_ROWS * W13_SCALE_COLS


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


TL_POINTS = 32  # timeline stamps per CTA
K1_STAMPS = {1: 27, 2: 28, 3: 29, 4: 30, 6: 31}  # fuse_k1: K1's points -> K4's
CAND_CAP = 256  # a long share's candidates counted against each other
# blocks a thread keys in its split task's eighth of a context (<= MAX_INDEX_BLOCKS)
CAND_BATCH = 2
assert N_SPLIT * THREADS * CAND_BATCH >= MAX_INDEX_BLOCKS
# Indexer context parallelism: a long context's split task p scans index head
# p // 2's scores of half p % 2 of this rank's blocks -- as many as a TP split task
# scans -- for the head's rank. A request is long past LONG_FROM_BLOCKS blocks in
# either build (``step_rows``; in the context-parallel one every long request is
# context-parallel), its rows ending less than a block apart are >= LONG_FROM_BLOCKS
# blocks each, and a split task then scans >= TOPK_BLOCKS blocks
assert N_SPLIT == 2 * TP
# and a short request's blocks fit one CTA, one a thread
assert N_SPLIT * TOPK_BLOCKS <= LONG_FROM_BLOCKS <= THREADS
assert MAX_TOKENS <= SPARSE_BLOCK
