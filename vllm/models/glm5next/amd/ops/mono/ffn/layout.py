# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/aiter#6173 at c39b56c36 (Apache-2.0 License),
# Copyright (c) 2025 FlyDSL Project Contributors:
# aiter/ops/flydsl/kernels/glm5_mono/glm/layout.py

"""Compile-time storage layout and CTA schedule of the GLM-5.3-Flash FFN launch."""

from vllm.models.glm5next.amd.ops.mono.config import HIDDEN, MOE_SLOTS, N_EXPERTS
from vllm.models.glm5next.amd.ops.mono.layout import (
    BLOCKS,
    ROUTER_TILE,
    ROW_TILE,
    UG_TILE,
    WAVES,
)

# Stages a bounded mailbox poll can expire in, one error word each.
POLL_STAGES = ("seam", "router", "ug", "down")

N_ROW_TILES = HIDDEN // ROW_TILE
N_ROUTER = N_EXPERTS // ROUTER_TILE
XQ_BLOCKS = HIDDEN // 128
XQ_GROUPS = HIDDEN // 32
XQ_WAVES = (XQ_BLOCKS + N_ROUTER - 1) // N_ROUTER
assert N_EXPERTS % ROUTER_TILE == 0 and XQ_WAVES * 4 <= WAVES


def dn_tile(samples: int) -> int:
    """Rows per expert-down / FFN-reduce task."""
    return 32 if samples == 1 or samples > 4 else HIDDEN // BLOCKS


def sample_wave_batches(samples: int) -> int:
    return (samples + WAVES - 1) // WAVES


def ug_task_rounds(inter: int) -> int:
    return (inter + BLOCKS - 1) // BLOCKS


def down_prefetch_batch(samples: int) -> int:
    if samples <= 4:
        return 9
    return 4 if samples <= 6 else 3


def down_x_words(samples: int, inter: int) -> int:
    return samples * MOE_SLOTS * inter // 4


def _align(size: int, alignment: int = 256) -> int:
    return (size + alignment - 1) // alignment * alignment


def layout(samples: int, npes: int, inter: int):
    """Byte offsets of the per-rank scratch and the symmetric peer buffer."""
    pair_bytes = 8
    items = [
        ("a", samples * HIDDEN // 2 * pair_bytes),
        ("scores", samples * N_EXPERTS * pair_bytes),
        ("xq", samples * HIDDEN // 4 * pair_bytes),
        ("xqs", samples * XQ_GROUPS * pair_bytes),
        ("sel", samples * MOE_SLOTS * pair_bytes),
        ("prob", samples * MOE_SLOTS * pair_bytes),
        ("mid", samples * MOE_SLOTS * inter * pair_bytes),
        ("poll_abort", 4),
        ("poll_err_addr", 8),
        ("poll_xrank_addr", 8),
    ]
    offset, scratch = 0, {}
    for name, size in items:
        scratch[name] = offset
        offset += _align(size)
    scratch["_bytes"] = offset

    part = npes * samples * HIDDEN * pair_bytes
    region = 2 * part
    symmetric = {
        "ffn": 0,
        "_part_stride": part,
        # one word per source rank, set on rank 0 by a rank whose poll expired
        "poll_xrank": region,
        "_bytes": region + _align(4 * npes),
    }
    return scratch, symmetric


def stage_tasks(samples: int, inter: int):
    """``(stage name, task count)`` pairs in execution order."""
    return [
        ("seam", N_ROW_TILES),
        ("router", samples * N_ROUTER),
        (
            "ug",
            BLOCKS
            if samples == 1
            else samples
            * max(
                BLOCKS,
                ((MOE_SLOTS * inter // UG_TILE + BLOCKS - 1) // BLOCKS) * BLOCKS,
            ),
        ),
        ("down", HIDDEN // dn_tile(samples)),
    ]
