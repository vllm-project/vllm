# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-batch tables and per-TP-size choices of the specialized decode."""

from typing import NamedTuple


def batch_table(spec):
    """Per-batch table of integers ("1-2:96,3-8:120"), tuples ("16:64.16.4.8") or names
    ("16-64:trtllm"); later entries override earlier ones."""
    table = {}
    for item in filter(None, spec.split(",")):
        span, value = item.split(":")
        lo, _, hi = span.partition("-")
        if "." in value:
            value = tuple(map(int, value.split(".")))
        elif value.lstrip("-").isdigit():
            value = int(value)
        table.update({b: value for b in range(int(lo), int(hi or lo) + 1)})
    return table


# Rows every mailbox is sized for (the executor's PREPARED_MAX).
MAILBOX_CAPACITY = 16

# The rank-order sums of the stock all-reduces the fused boundaries reproduce:
# the node-local one-shot's BF16 adds, the MNNVL all-reduce's FP32 sum.
BF16_ADDS = "bf16"
FP32_ADDS = "fp32"
SUM_ORDERS = (BF16_ADDS, FP32_ADDS)

# The TP sizes whose attention boundary (and DSpark draft boundaries) fuse the
# all-reduce, summing in the order of the backend the stock all-reduce resolved
# to, read once at build (TP4, one node: MNNVL by default, the node-local
# one-shot when set or as vLLM's fallback; TP8, across trays: MNNVL). TP16
# calls the op the executor is given.
FUSED_TPS = (4, 8)
ALL_REDUCE_TPS = (16,)
# The TP sizes whose o_proj (and the DSpark draft's branch GEMMs) run the
# package GEMM, which publishes the partial into the fused boundary's mailbox;
# their boundaries run only the consumer. A subset of FUSED_TPS.
PACKAGE_GEMM_TPS = (8,)
assert set(PACKAGE_GEMM_TPS) <= set(FUSED_TPS)
# The TP sizes whose NVFP4 layers run expert TP (every expert at 1/tp of the
# intermediate width on every rank, resharded at build from the stock EP
# experts). Other sizes, and BF16 layers, keep the stock EP experts.
EXPERT_TP_TPS = (8,)


class UnpoolerShape(NamedTuple):
    """Launch shape of the unpooler GEMM: K split over ``splits`` CTAs of one
    cluster, ``kw`` consecutive K per warp, ``accs`` accumulators per warp, and
    the token tile per batch."""

    splits: int
    kw: int
    accs: int
    token_tile: dict[int, int]


# The MoE unpooler's column block ([hidden / tp, 4096] per rank).
UNPOOLER = UnpoolerShape(
    splits=4, kw=256, accs=4, token_tile=batch_table("1-8:8,9-16:16")
)
# The o_proj partial at TP8 ([8192, 1024] per rank): one CTA of 4 warps per 16
# features.
OPROJ = UNPOOLER._replace(splits=1)
# The DSpark draft's branch partials at TP8 by role (o_proj [8192, 1024] and
# down_proj [8192, 2048] per rank), publishing into the draft boundary's mailbox.
DRAFT_DOWN = OPROJ._replace(kw=128)
DRAFT_GEMMS = {"o_proj": OPROJ, "down_proj": DRAFT_DOWN}
