# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025 FlyDSL Project Contributors
# mypy: ignore-errors
#
# This file contains code copied from FlyDSL (ROCm/FlyDSL PR #1204 at 21a3d1ee), as vendored by
# ROCm/ATOM PR #2435 (head 45e4b55d, atom/model_ops/monokernel/glm/layout.py). The original source code was
# licensed under the Apache License 2.0 and included the following copyright notice:
# Copyright (c) 2025 FlyDSL Project Contributors
# Modified by the vLLM project contributors (Apache-2.0 sec. 4(b)): import paths rewritten to this package;
#   POLL_STAGES and the poll_err / poll_abort scratch words, scratch regions of the in-kernel indexer,
#   a split_keys override of the sparse-MLA split task size; the DCP region and unused helpers removed.

"""Compile-time storage layout and CTA schedule for the GLM-5 MonoKernel."""

from vllm.models.deepseek_v32.amd.mono.kernel.config import (
    HIDDEN,
    INTER,
    KV_LORA,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    QKV_A_ROWS,
    TOP_K,
    V_DIM,
)
from vllm.models.deepseek_v32.amd.mono.kernel.layout import (
    BLOCKS,
    Q_B_TILE,
    QKV_A_TILE,
    ROUTER_TILE,
    ROW_TILE,
    UG_TILE,
    UK_TILE,
    UV_TILE,
    WAVES,
)

INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_Q_ROWS = INDEX_HEADS * INDEX_DIM
INDEX_TILE = 16
INDEX_KEYS_PER_TASK = 64


def split_acc_head(head_group, lane_group: int, element: int):
    """Map one split-PV MFMA row to its local attention head."""

    return head_group * WAVES + lane_group * 4 + element


def split_score_column(wave: int, lane: int):
    """Locate this wave's head column in a 16x16 MFMA score tile."""

    return wave + 16 * ((lane % 16) // 4)


N_QKV_A = QKV_A_ROWS // QKV_A_TILE
N_ROW_TILES = HIDDEN // ROW_TILE
N_ROUTER = N_EXPERTS // ROUTER_TILE
XQ_BLOCKS = HIDDEN // 128
XQ_WAVES = (XQ_BLOCKS + N_ROUTER - 1) // N_ROUTER
assert XQ_WAVES * 4 <= WAVES


def dn_tile(samples: int) -> int:
    """Rows per expert-down / FFN-reduce task."""

    return 32 if samples == 1 or samples > 4 else HIDDEN // BLOCKS


def sparse_keys_per_task(samples: int, heads: int = WAVES) -> int:
    """Use narrower sparse tiles when samples or attention heads fill LDS."""

    return 32 if samples > 4 or heads > 2 * WAVES else 64


def sample_wave_batches(samples: int) -> int:
    return (samples + WAVES - 1) // WAVES


def ug_task_rounds(inter: int) -> int:
    return (inter + BLOCKS - 1) // BLOCKS


def down_x_words(samples: int, inter: int) -> int:
    return samples * MOE_SLOTS * inter // 2


def ug_split(samples: int, inter: int = INTER):
    """Return the balanced up/gate leftover split for batches two and four."""

    full_tiles, remainder = divmod((samples * TOP_K + 1) * (inter // UG_TILE), BLOCKS)
    if samples not in (2, 4) or remainder == 0 or BLOCKS % remainder:
        return None
    segments = BLOCKS // remainder
    if (HIDDEN // 128) % segments or (HIDDEN // 128) // segments > WAVES // 2:
        return None
    return full_tiles, segments


# Stages that poll mailboxes, in execution order.  A bounded poll (``poll_limit``,
# back-ported from FlyDSL #1214's gfx1250 kernel) that expires sets its stage's
# ``poll_err`` scratch word instead of spinning forever.
POLL_STAGES = (
    "qkv_a",
    "q_norm",
    "cache",
    "q_b",
    "index_q",
    "uk",
    "index_score",
    "index_select",
    "split",
    "uv",
    "o",
    "router",
    "ug",
    "down",
)


def _align(size: int, alignment: int = 256) -> int:
    return (size + alignment - 1) // alignment * alignment


def layout(
    samples: int,
    heads: int,
    npes: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    inter: int = INTER,
    split_keys: int | None = None,
):
    """Return byte offsets for per-rank scratch and symmetric peer buffers. ``split_keys``: keys per sparse-MLA
    split task (default ``sparse_keys_per_task``; 64 with ``split_keys64``)."""

    split_count = sparse_attention_topk // (split_keys or sparse_keys_per_task(samples, heads))
    pair_bytes = 8
    items = [
        ("q_a", samples * Q_LORA * pair_bytes),
        ("q_an", samples * Q_LORA // 2 * pair_bytes),
        ("kv_a", samples * (KV_LORA + PE_DIM) * pair_bytes),
        ("kvnew", samples * KV_LORA * pair_bytes),
        ("penew", samples * PE_DIM * pair_bytes),
        ("q_nope", samples * heads * NOPE_DIM * pair_bytes),
        ("q_pe", samples * heads * PE_DIM * pair_bytes),
        ("q_lat", samples * heads * KV_LORA * pair_bytes),
        ("sp_acc", samples * split_count * heads * KV_LORA * pair_bytes),
        ("sp_m", samples * split_count * heads * pair_bytes),
        ("sp_l", samples * split_count * heads * pair_bytes),
        ("o", samples * heads * V_DIM * pair_bytes),
        ("a", samples * HIDDEN * pair_bytes),
        ("scores", samples * N_EXPERTS * pair_bytes),
        ("xq", samples * HIDDEN // 4 * pair_bytes),
        ("xqs", samples * XQ_BLOCKS * pair_bytes),
        ("sel", samples * MOE_SLOTS * pair_bytes),
        ("prob", samples * MOE_SLOTS * pair_bytes),
        ("mid", samples * MOE_SLOTS * inter * pair_bytes),
        ("ugp", BLOCKS * samples * 2 * UG_TILE * pair_bytes),
        ("xqd", samples * HIDDEN * 4),
    ]
    if with_indexer:
        items += [
            ("index_k", samples * INDEX_DIM * pair_bytes),
            ("index_k_new", samples * INDEX_DIM // 2 * pair_bytes),
            ("index_ready", samples * pair_bytes),
            ("index_w", samples * INDEX_HEADS * pair_bytes),
            ("index_q", samples * INDEX_Q_ROWS // 2 * pair_bytes),
            ("index_scores", samples * index_max_seq * pair_bytes),
            ("indices", samples * sparse_attention_topk * 4),
            ("indices_ready", samples * pair_bytes),
        ]
    # One word per stage, set by bounded mailbox polls that time out; appended
    # last so every other offset is unchanged.
    items.append(("poll_err", 4 * len(POLL_STAGES)))
    # ``step + 1`` of the step whose first bounded wait expired on this rank (early-out mark);
    # also appended after everything else so no existing offset moves.
    items.append(("poll_abort", 4))
    if with_indexer:
        # paged fused indexer: this launch's ue8m0 index-K scale per sample
        # (the unscaled FP8 values go through index_k_new); appended last, no other offset moves
        items.append(("index_k_scale", samples * pair_bytes))

    # address of this rank's poll_xrank word on rank 0, written once by the host
    items.append(("poll_xrank_addr", 8))

    offset, scratch = 0, {}
    for name, size in items:
        scratch[name] = offset
        offset += _align(size)
    scratch["_bytes"] = offset

    part = npes * samples * HIDDEN * pair_bytes
    region = 2 * part
    symmetric = {
        "attn": 0,
        "ffn": region,
        "_part_stride": part,
        # one word per source rank: set on rank 0 by a rank whose bounded wait expired
        "poll_xrank": 2 * region,
        "_bytes": 2 * region + _align(4 * npes),
    }
    return scratch, symmetric


def stage_tasks(
    samples: int,
    heads: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    inter: int = INTER,
    split_keys: int | None = None,
):
    """Return ``(stage name, task count)`` pairs in execution order."""

    tasks = [("qkv_a", N_QKV_A), ("q_norm", samples), ("cache", 1), ("q_b", heads * (NOPE_DIM + PE_DIM) // Q_B_TILE)]
    if with_indexer:
        tasks += [("index_q", INDEX_Q_ROWS // INDEX_TILE)]
    tasks += [("uk", heads * KV_LORA // UK_TILE)]
    if with_indexer:
        tasks += [
            ("index_score", samples * ((index_max_seq + INDEX_KEYS_PER_TASK - 1) // INDEX_KEYS_PER_TASK)),
            ("index_select", samples),
        ]
    tasks += [
        (
            "split",
            samples
            * (heads // WAVES)
            * (sparse_attention_topk // (split_keys or sparse_keys_per_task(samples, heads))),
        ),
        ("uv", samples * (heads * V_DIM // UV_TILE)),
        ("o", N_ROW_TILES),
        ("router", samples * N_ROUTER),
        (
            "ug",
            (
                BLOCKS
                if samples == 1
                else samples * max(BLOCKS, ((MOE_SLOTS * inter // UG_TILE + BLOCKS - 1) // BLOCKS) * BLOCKS)
            ),
        ),
        ("down", HIDDEN // dn_tile(samples)),
    ]
    return tasks
