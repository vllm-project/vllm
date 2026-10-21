# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/aiter#6173 at c39b56c36 (Apache-2.0 License),
# Copyright (c) 2025 FlyDSL Project Contributors:
# aiter/ops/flydsl/kernels/glm5_mono/glm/layout.py

"""Compile-time storage layout and CTA schedule of the GLM-5.3-Flash FFN launch."""

from vllm.models.glm5next.amd.ops.mono.config import (
    HC_MULT,
    HIDDEN,
    INDEX_DIM,
    INDEX_HEADS,
    KV_LORA,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    Q_LORA,
    TOPK_TOKENS,
    V_DIM,
    FlashShard,
)
from vllm.models.glm5next.amd.ops.mono.layout import (
    BLOCKS,
    ROUTER_TILE,
    ROW_TILE,
    UG_TILE,
    WAVES,
)

# Stages a bounded mailbox poll can expire in, one error word each.
POLL_STAGES = (
    "aseam",
    "agate",
    "inproj",
    "fgb",
    "kda",
    "oproj",
    "seam",
    "gate",
    "router",
    "ug",
    "down",
    "qkv_a",
    "qnorm",
    "cache",
    "q_b",
    "index_q",
    "uk",
    "iscore",
    "iselect",
    "split",
    "uv",
)
# rows of a projection task: one MFMA tile, or one tile a wave when K is 128
IN_TILE = 16
FGB_TILE = 16 * WAVES
O_TILE = 16

# sparse attention: q_a | kv_a (FP8) then wk | weights_proj | kpool gate (bf16)
QKV_TILE = 16
N_QKV_FP8 = (Q_LORA + KV_LORA) // QKV_TILE
INDEX_ROWS = INDEX_DIM + INDEX_HEADS + INDEX_DIM
N_QKV = N_QKV_FP8 + INDEX_ROWS // QKV_TILE
QB_TILE = 32
UK_TILE = 128
UV_TILE = 64
POOLS_PER_TASK = 256
# split attention: one task 64 selected keys x 8 heads (one head a wave)
SPLIT_KEYS = 64
N_SPLIT = TOPK_TOKENS // SPLIT_KEYS
# per-step row table of a sparse-attention launch (int32 [S, ROW_WORDS])
ROW_POS, ROW_SLOT, ROW_POOL_SLOT, ROW_TAIL_SLOT, ROW_BT = range(5)
ROW_WORDS = 8
# entries of a sparse-attention layer's int64 argument table
DSA_TAB = (
    "w_qkv",
    "s_qkv",
    "w_idx",
    "w_gate",
    "g_qa",
    "g_kv",
    "ln_w",
    "ln_b",
    "ape",
    "w_qb",
    "s_qb",
    "w_iq",
    "w_uk",
    "w_kvb",
    "w_o",
    "s_o",
    "kv_cache",
    "idx_cache",
    "tail_cache",
    "block_table",
    "idx_block_table",
    "bt_stride",
    "idx_bt_stride",
    "kv_block_elems",
    "kv_block_tokens",
    "idx_block_bytes",
    "idx_block_pools",
    "tail_block_elems",
    "tail_gate_elems",
    # f32 scalar: the MLA cache's per-tensor scale when it holds e4m3 (vLLM's
    # ``_k_scale``); unused for a bf16 cache
    "k_scale",
)

N_ROW_TILES = HIDDEN // ROW_TILE
# mix projection outputs (pre, post, comb) plus one sum of squares a tile
N_MIX = HC_MULT * (2 + HC_MULT)
PMIX = N_MIX + 1
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


def layout(
    samples: int,
    npes: int,
    inter: int,
    kda: bool = False,
    dsa: bool = False,
    max_pools: int = 0,
):
    """Byte offsets of the per-rank scratch and the symmetric peer buffer."""
    pair_bytes = 8
    shard = FlashShard(npes)
    items = []
    if kda or dsa:
        items = [
            ("rnew_a", samples * HC_MULT * HIDDEN // 2 * pair_bytes),
            ("pmix_a", samples * N_ROW_TILES * PMIX * pair_bytes),
            ("a_in", samples * HIDDEN // 2 * pair_bytes),
            ("mix_a", samples * HC_MULT * (1 + HC_MULT) * pair_bytes),
        ]
    if kda:
        items += [
            ("proj", samples * shard.kda_in_rows * pair_bytes),
            ("gates", samples * 2 * shard.kda_proj * pair_bytes),
            ("o_in", samples * shard.kda_proj // 2 * pair_bytes),
        ]
    if dsa:
        H = shard.mla_heads
        items += [
            ("qa", samples * Q_LORA * pair_bytes),
            ("kva", samples * KV_LORA * pair_bytes),
            ("ik", samples * INDEX_DIM * pair_bytes),
            ("iw", samples * INDEX_HEADS * pair_bytes),
            ("ig", samples * INDEX_DIM * pair_bytes),
            ("q_an", samples * Q_LORA // 2 * pair_bytes),
            ("q_nope", samples * H * NOPE_DIM // 2 * pair_bytes),
            ("iq8", samples * INDEX_HEADS * INDEX_DIM // 4 * pair_bytes),
            ("iqw", samples * INDEX_HEADS * pair_bytes),
            ("q_lat", samples * H * KV_LORA // 2 * pair_bytes),
            ("kv_ready", samples * pair_bytes),
            ("iscores", samples * max(max_pools, 1) * pair_bytes),
            ("tok_sel", samples * TOPK_TOKENS * 4),
            ("tok_ready", samples * pair_bytes),
            ("sp_acc", samples * N_SPLIT * H * KV_LORA // 2 * pair_bytes),
            ("sp_m", samples * N_SPLIT * H * pair_bytes),
            ("sp_l", samples * N_SPLIT * H * pair_bytes),
            ("o_in", samples * H * V_DIM // 2 * pair_bytes),
        ]
    items += [
        ("rnew", samples * HC_MULT * HIDDEN // 2 * pair_bytes),
        ("pmix", samples * N_ROW_TILES * PMIX * pair_bytes),
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
    assert len({name for name, _ in items}) == len(items), "mailbox names"
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
        # one word per source rank, set on rank 0 by a rank whose poll expired
        "poll_xrank": 2 * region,
        "_bytes": 2 * region + _align(4 * npes),
    }
    return scratch, symmetric


def stage_tasks(
    samples: int, inter: int, npes: int = 1, kda: bool = False, dsa: bool = False
):
    """``(stage name, task count)`` pairs in execution order. ``iscore`` is sized
    per row at run time; its count here only places the next stage."""
    shard = FlashShard(npes)
    attention = []
    if dsa:
        H = shard.mla_heads
        attention = [
            ("aseam", N_ROW_TILES),
            ("agate", samples),
            ("qkv_a", N_QKV),
            ("qnorm", samples),
            ("cache", samples),
            ("q_b", H * NOPE_DIM // QB_TILE),
            ("index_q", INDEX_HEADS),
            ("uk", H * KV_LORA // UK_TILE),
            ("iscore", samples * 8),
            ("iselect", samples),
            ("split", samples * (H // WAVES) * N_SPLIT),
            ("uv", samples * H * (V_DIM // UV_TILE)),
            ("oproj", HIDDEN // O_TILE),
        ]
    if kda:
        attention = [
            ("aseam", N_ROW_TILES),
            ("agate", samples),
            ("inproj", (shard.kda_in_rows + IN_TILE - 1) // IN_TILE),
            ("fgb", 2 * shard.kda_proj // FGB_TILE),
            ("kda", samples * shard.kda_heads),
            ("oproj", HIDDEN // O_TILE),
        ]
    return attention + [
        ("seam", N_ROW_TILES),
        ("gate", samples),
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
