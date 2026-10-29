# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Compile-time storage layout and CTA schedule for the GLM-5 MonoKernel."""

import math
import os

from .geometry import (
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
from .geometry_config import (
    HIDDEN,
    INTER,
    KV_LORA,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    Q_PE_DIM,
    QKV_A_ROWS,
    TOP_K,
    V_DIM,
)

INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_Q_ROWS = INDEX_HEADS * INDEX_DIM
INDEX_TILE = 16
INDEX_KEYS_PER_TASK = 64
# ATOM's FP8 index cache row stride: 128 key bytes plus one fp32 scale,
# padded to a 16-byte row.  It is an allocation width, not a record layout --
# see index_fp8_data_dword.
INDEX_ALIGNED_DIM = (INDEX_DIM + 4 + INDEX_TILE - 1) // INDEX_TILE * INDEX_TILE
# vLLM indexer cache row: 128 E4M3 key bytes + one fp32 scale (no padding).
INDEX_PAGE_ROW = INDEX_DIM + 4
# Parameter-table slots (int64 pointers) of the separate shared expert: w_ug, s_ug, w_dn, s_dn.
# Slots 0..6 are the indexer weights, 7 a zero sentinel.
SHARED_EXPERT_ARGS = 8
PARAM_TABLE_SLOTS = SHARED_EXPERT_ARGS + 4
DCP_SUMMARY_PAIRS = KV_LORA // 2 + 2


def split_acc_head(head_group, lane_group: int, element: int):
    """Map one split-PV MFMA row to its local attention head."""
    return head_group * WAVES + lane_group * 4 + element


def split_score_column(wave: int, lane: int):
    """Locate this wave's head column in a 16x16 MFMA score tile."""
    return wave + 16 * ((lane % 16) // 4)


def fp8_kv_upper_pair_lane(lane):
    return (lane & -4) + 2


def fp8_pe_upper_pair_lane(lane):
    return (lane & -2) + 1


def index_fp8_data_dword(page, row, k, page_size):
    """Dword offset of FP8 index elements ``k`` .. ``k + 3`` of one cache row.

    vLLM's ROCm DSA indexer page (``indexer_k_quant_and_cache_triton`` for
    ``page_size > 1``): the ``page_size x INDEX_DIM`` E4M3 keys in 16 x 16-byte
    SHUFFLE tiles ``[row // 16, k // 16, row % 16, k % 16]``, then ``page_size``
    fp32 row scales (``INDEX_PAGE_ROW`` bytes per row). ``k % 4 == 0``, so the
    four bytes never straddle a tile.
    """
    tiled = (
        row // INDEX_TILE * (INDEX_TILE * INDEX_DIM)
        + k // INDEX_TILE * (INDEX_TILE * INDEX_TILE)
        + row % INDEX_TILE * INDEX_TILE
        + k % INDEX_TILE
    )
    return page * (page_size * INDEX_PAGE_ROW // 4) + tiled // 4


def index_fp8_scale_dword(page, row, page_size):
    """Dword offset of the fp32 scale of one FP8 index cache row."""
    return page * (page_size * INDEX_PAGE_ROW // 4) + page_size * (INDEX_DIM // 4) + row


def sparse_cache_rows(
    sparse_kv_indices,
    *,
    sample: int,
    topk: int,
    cur_pos: int = 0,
    sparse_kv_indptr=None,
):
    """Resolve the cache rows consumed by one flat or paged attention row."""
    if sparse_kv_indptr is not None:
        start, end = sparse_kv_indptr[sample : sample + 2]
        return sparse_kv_indices[start:end]
    context = cur_pos + sample + 1
    count = min(context, topk)
    if context <= topk:
        return range(count)
    start = sample * topk
    return sparse_kv_indices[start : start + count]


def paged_row_contract(
    sparse_kv_indices, sparse_kv_indptr, sample: int, slot_mapping=None
):
    """Reference the device contract for one ATOM-layout request row."""
    start, end = sparse_kv_indptr[sample : sample + 2]
    active = end > start
    rows = sparse_kv_indices[start:end] if active else ()
    slot_owned = slot_mapping is None or slot_mapping[sample] >= 0
    return {
        "active": active,
        "context": end - start,
        "index_base": start if active else 0,
        "safe_row": rows[0] if active else 0,
        "write_cache": active if slot_mapping is None else slot_owned,
    }


INDEX_CP_TILE = 4096
INDEX_RADIX_DIGITS = 4
INDEX_RADIX_BINS = 256
INDEX_LOCAL_LDS_CAPACITY = 16384


def indexer_uses_tiles(index_max_seq: int, indexer_cp: bool) -> bool:
    """Use the distributed tiled radix only to shard a context ACROSS RANKS.

    It used to answer "is this context past what one selector CTA can stage in
    LDS", because the local selector staged every key of the context.  It no
    longer does -- the selector is a merge tree over fixed ``INDEX_SEG`` leaves
    (``index_levels``), so one CTA serves any context.  Tiles are therefore
    purely the context-parallel feature now: shard the scorer and the selection
    over ``index_cp_world`` ranks, which only a CP build can do.
    """
    return indexer_cp and index_max_seq > INDEX_LOCAL_LDS_CAPACITY


# ---------------------------------------------------------------- selection tree
# The selector ranks a FIXED ``INDEX_SEG`` keys in LDS; capacity past that comes
# from STACKING leaves, not from growing the leaf.  Level 0: segment g of
# ``INDEX_SEG`` scores -> its own top-``topk``, emitted dense and ascending by
# key index.  Level l > 0: one list merges ``index_fan(topk)`` lists of level
# l - 1 laid end to end -- every list but a level's last holds exactly ``topk``
# valid entries, so the merged input is dense and in index order, and a global
# winner is always inside its own list's top-``topk``, so no level drops one.
# The tree ends at the level holding a single list.  This is DeepSeek-V4.1's
# design (``atom/models/deepseek_v41/mono/index_plan.py:95-154``) at GLM's
# constants: INDEX_SEG 16384, topk 2048 -> fan 8 -> two merge levels reach
# 16384 * 8 * 8 = 1048576, with no LDS beyond today's.
INDEX_SEG = INDEX_LOCAL_LDS_CAPACITY


def index_seg(index_max_seq: int) -> int:
    """Keys one select task stages in LDS: the leaf size, not a capacity."""
    return min(index_max_seq, INDEX_SEG)


def index_select_capacity(index_max_seq: int, indexer_cp: bool) -> int:
    """Context the LOCAL (merge-tree) selector must serve in this build.

    A tiled build keeps the local selector as its short-query fallback, bounded
    by ``use_short_index``; it reads the peer-exchanged score plane, which is
    one leaf wide, so its tree is always a bare leaf.  Every other build sends
    the whole context through the tree.
    """
    return (
        index_seg(index_max_seq)
        if indexer_uses_tiles(index_max_seq, indexer_cp)
        else index_max_seq
    )


def index_fan(index_max_seq: int, topk: int) -> int:
    """Lists one merge level consumes per list it emits."""
    return index_seg(index_max_seq) // topk


def index_lists(index_max_seq: int, level: int, topk: int) -> int:
    """Lists the tree holds at ``level`` for a build of ``index_max_seq``."""
    seg = index_seg(index_max_seq)
    fan = index_fan(index_max_seq, topk)
    n = (index_max_seq + seg - 1) // seg
    for _ in range(level):
        n = (n + fan - 1) // fan
    return n


def index_levels(index_max_seq: int, topk: int) -> int:
    """Merge levels a build needs: the depth at which one list remains.

    ``index_max_seq <= INDEX_SEG`` gives 0, i.e. today's single flat select.
    """
    level = 0
    while index_lists(index_max_seq, level, topk) > 1:
        level += 1
    return level


def index_select_stage(level: int, depth: int) -> str:
    """Stage name of tree ``level``; the final level keeps ``index_select``."""
    return "index_select" if level == depth else f"index_sel{level}"


def index_cp_world(npes: int, samples: int, index_max_seq: int) -> int:
    """Use two independent CP4 groups for TP8's single-query medium contexts."""
    if (
        npes == 8
        and samples == 1
        and INDEX_LOCAL_LDS_CAPACITY < index_max_seq <= 131072
    ):
        return 4
    return npes


def index_cp_capacity(index_max_seq: int, npes: int) -> int:
    """Maximum context shard, rounded to a complete scoring tile."""
    unit = npes * INDEX_KEYS_PER_TASK
    return (index_max_seq + unit - 1) // unit * INDEX_KEYS_PER_TASK


def index_cp_partitions(index_max_seq: int, npes: int) -> int:
    return (index_cp_capacity(index_max_seq, npes) + INDEX_CP_TILE - 1) // INDEX_CP_TILE


N_QKV_A = QKV_A_ROWS // QKV_A_TILE
N_ROW_TILES = HIDDEN // ROW_TILE
N_ROUTER = N_EXPERTS // ROUTER_TILE
N_UG_PER_SLOT = INTER // UG_TILE
XQ_BLOCKS = HIDDEN // 128
XQ_GROUPS = HIDDEN // 32
XQ_WAVES = (XQ_BLOCKS + N_ROUTER - 1) // N_ROUTER
assert (
    XQ_WAVES <= WAVES
)  # ML4: 128 experts -> 16 router tasks, 3 quant waves each (GLM: 2 of 32 tasks)


def dn_tile(samples: int, expert_mxfp4: bool = False) -> int:
    """Return rows per expert-down/FFN-reduce task for the tuned schedule."""
    return 32 if samples == 1 or (samples > 4 and expert_mxfp4) else HIDDEN // BLOCKS


def sparse_keys_per_task(samples: int, heads: int = WAVES) -> int:
    """Use narrower sparse tiles when samples or attention heads fill LDS."""
    return 32 if heads > 2 * WAVES or (heads == WAVES and samples > 4) else 64


def sample_wave_batches(samples: int) -> int:
    return (samples + WAVES - 1) // WAVES


def ug_task_rounds(inter: int) -> int:
    return (inter + BLOCKS - 1) // BLOCKS


def down_prefetch_batch(
    samples: int, native_fp4_mfma: bool = False, atom_experts: bool = False
) -> int:
    """How many expert-down weight tiles a wave keeps in flight.

    ``run_units`` issues round r+1's buffer loads before consuming round r's,
    so this sets the stage's memory-level parallelism. Changing it is
    bit-identical -- ``run_units`` walks the units in index order whatever the
    batch size, so the MFMA accumulation order does not move.

    Measured on the TP4 GLM-5.2 shard, this stage is **not** latency-starved:
    sweeping 2..8 at S=6 and S=12 is monotone the *wrong* way (bigger is
    slower, presumably register pressure), and 2 is the best of the values
    tried at both widths.

    ``MONO_DN_BATCH`` overrides it for an A/B (see
    ``tests/models/glm52/kernels/bench_glm52_atom_shape.py --ab-env``).
    """
    override = os.environ.get("MONO_DN_BATCH", "")
    if override:
        return int(override)
    if samples <= 4:
        return 9
    if native_fp4_mfma:
        return 4 if samples <= 6 else 3
    return 8 if atom_experts else 4


def ptpc_stage_chunk(samples: int) -> int:
    """How many (sample, group) cells ``stage_x_pairs_ptpc`` stages together.

    The helper's two per-cell costs are per-*call* costs: ``poll`` resolves a
    whole batch in one round trip and ``block_maxs`` reduces a whole list in
    two barriers.  Batching cells is bit-identical and removes a clean O(S)
    term from ``q_b``, ``uk`` and ``o`` -- stages whose task count and weight
    bytes do not depend on ``S`` at all.

    Measured on the TP4 GLM-5.2 shard (``bench_glm52_atom_shape --ab-env
    MONO_PTPC_CHUNK=...``, 7 trials, interleaved, spread <= 0.27%):

    ==========  ===============  ================
    chunk       S=6 us/launch    S=12 us/launch
    ==========  ===============  ================
    1 (old)     113.785          201.790
    **2**       **112.507**      196.527
    **4**       113.238          **194.113**
    6           116.048          194.134
    all cells   116.012          214.911
    ==========  ===============  ================

    So the optimum moves with the width, and batching *every* cell is much
    worse than batching four -- it holds every cell's values live across one
    reduction and spills.  ``MONO_PTPC_CHUNK`` overrides this for an A/B;
    ``1`` restores the per-cell walk exactly.
    """
    override = os.environ.get("MONO_PTPC_CHUNK", "")
    if override:
        return int(override)
    return 2 if samples <= 6 else 4


def down_x_words(
    samples: int, inter: int, expert_mxfp4: bool, native_fp4_mfma: bool = False
) -> int:
    return (
        samples * MOE_SLOTS * inter // (4 if native_fp4_mfma or not expert_mxfp4 else 2)
    )


def dcp_softmax_weights(maxima, sums):
    global_max = max(maxima)
    scaled = [
        total * math.exp(maximum - global_max) for maximum, total in zip(maxima, sums)
    ]
    denominator = sum(scaled)
    return [value / denominator if denominator else 0.0 for value in scaled]


def dcp_summary_index(source, sample, tile, item, samples, n_uv):
    return ((source * samples + sample) * n_uv + tile) * DCP_SUMMARY_PAIRS + item


def dcp_uv_owner(tile, output_heads, rank):
    return tile // (V_DIM // UV_TILE) // output_heads == rank


def dcp_local_uv_tile(tile, output_heads):
    return tile % (output_heads * V_DIM // UV_TILE)


def ug_split(samples: int, inter: int = INTER):
    """Return the balanced up/gate leftover split for batches two and four."""
    full_tiles, remainder = divmod((samples * TOP_K + 1) * (inter // UG_TILE), BLOCKS)
    if samples not in (2, 4) or remainder == 0 or BLOCKS % remainder:
        return None
    segments = BLOCKS // remainder
    if (HIDDEN // 128) % segments or (HIDDEN // 128) // segments > WAVES // 2:
        return None
    return full_tiles, segments


def _align(size: int, alignment: int = 256) -> int:
    return (size + alignment - 1) // alignment * alignment


def layout(
    samples: int,
    heads: int,
    npes: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    indexer_cp: bool = False,
    inter: int = INTER,
    output_heads: int | None = None,
    dcp_size: int = 1,
    native_fp4_mfma: bool = False,
):
    """Return byte offsets for per-rank scratch and symmetric peer buffers."""
    split_count = sparse_attention_topk // sparse_keys_per_task(samples, heads)
    output_heads = heads if output_heads is None else output_heads
    tiled_indexer = with_indexer and indexer_uses_tiles(index_max_seq, indexer_cp)
    index_ranks = index_cp_world(npes, samples, index_max_seq) if indexer_cp else 1
    pair_bytes = 8
    items = [
        ("q_a", samples * Q_LORA * pair_bytes),
        ("q_an", samples * HIDDEN // 2 * pair_bytes),
        ("iq_an", samples * HIDDEN // 2 * pair_bytes),
        ("kv_a", samples * (KV_LORA + PE_DIM) * pair_bytes),
        ("kvnew", samples * KV_LORA * pair_bytes),
        ("penew", samples * PE_DIM * pair_bytes),
        ("q_nope", samples * heads * NOPE_DIM * pair_bytes),
        ("q_pe", samples * heads * PE_DIM * pair_bytes),
        ("q_lat", samples * heads * KV_LORA * pair_bytes),
        ("sp_acc", samples * split_count * heads * KV_LORA * pair_bytes),
        ("sp_m", samples * split_count * heads * pair_bytes),
        ("sp_l", samples * split_count * heads * pair_bytes),
        ("o", samples * output_heads * V_DIM * pair_bytes),
        ("a", samples * HIDDEN * pair_bytes),
        ("scores", samples * N_EXPERTS * pair_bytes),
        ("xq", samples * HIDDEN // 4 * pair_bytes),
        ("xqs", samples * (XQ_GROUPS if native_fp4_mfma else XQ_BLOCKS) * pair_bytes),
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
            (
                "index_scores",
                samples
                * (
                    index_cp_capacity(index_max_seq, index_ranks)
                    if indexer_cp
                    else index_max_seq
                )
                * pair_bytes,
            ),
            ("indices", samples * sparse_attention_topk * 4),
            ("indices_ready", samples * pair_bytes),
        ]
        # One region a NON-final selection-tree level: its lists' (score, key
        # index) entries, two tagged pairs each, so one 16-byte store publishes
        # both and one 16-byte poll validates both.  The final level writes
        # ``indices`` instead, so a build of ``index_levels`` = 0 (every context
        # one leaf can serve, and every tiled build) adds nothing at all.
        select_capacity = index_select_capacity(index_max_seq, indexer_cp)
        for level in range(index_levels(select_capacity, sparse_attention_topk)):
            items += [
                (
                    f"isel{level}",
                    samples
                    * index_lists(select_capacity, level, sparse_attention_topk)
                    * sparse_attention_topk
                    * 2
                    * pair_bytes,
                )
            ]
        if tiled_indexer:
            parts = index_cp_partitions(index_max_seq, index_ranks)
            items += [
                (
                    "index_cp_hist",
                    INDEX_RADIX_DIGITS
                    * samples
                    * parts
                    * INDEX_RADIX_BINS
                    * pair_bytes,
                ),
                ("index_cp_counts", samples * parts * 2 * pair_bytes),
                ("index_cp_offsets", samples * parts * 2 * pair_bytes),
            ]

    offset, scratch = 0, {}
    for name, size in items:
        scratch[name] = offset
        offset += _align(size)
    scratch["_bytes"] = offset

    part = npes * samples * HIDDEN * pair_bytes
    region = 2 * part
    dcp_part = (
        dcp_size * samples * (heads * V_DIM // UV_TILE) * DCP_SUMMARY_PAIRS * pair_bytes
        if dcp_size > 1
        else 0
    )
    symmetric = {
        "attn": 0,
        "ffn": region,
        "dcp": 2 * region,
        "_part_stride": part,
        "_dcp_part_stride": dcp_part,
        "_bytes": 2 * region + 2 * dcp_part,
    }
    if tiled_indexer or (with_indexer and indexer_cp):
        # Two launch slots, like the attention/FFN peer buffers. Distinct radix
        # rounds must not overwrite a histogram or threshold still being read.
        offset = symmetric["_bytes"]
        cp_items = (
            [
                (
                    "index_cp_hist",
                    index_ranks
                    * INDEX_RADIX_DIGITS
                    * samples
                    * INDEX_RADIX_BINS
                    * pair_bytes,
                ),
                ("index_cp_threshold", INDEX_RADIX_DIGITS * samples * 2 * pair_bytes),
                ("index_cp_counts", index_ranks * samples * 2 * pair_bytes),
                ("index_cp_offsets", samples * 2 * pair_bytes),
                ("index_cp_indices", samples * sparse_attention_topk * pair_bytes),
            ]
            if tiled_indexer
            else []
        )
        if indexer_cp:
            cp_items += [
                (
                    "index_cp_scores",
                    samples * min(index_max_seq, INDEX_LOCAL_LDS_CAPACITY) * pair_bytes,
                )
            ]
        if indexer_cp and not tiled_indexer and npes > 1:
            cp_items += [("index_cp_q", samples * INDEX_Q_ROWS // 2 * pair_bytes)]
        for name, size in cp_items:
            symmetric[name] = offset
            symmetric[name + "_stride"] = _align(size)
            offset += 2 * _align(size)
        symmetric["_bytes"] = offset
    return scratch, symmetric


def stage_tasks(
    samples: int,
    heads: int,
    sparse_attention_topk: int,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    expert_mxfp4: bool = False,
    inter: int = INTER,
    indexer_cp: bool = False,
    npes: int = 1,
    split_head_merge: bool = False,
):
    """Return ``(stage name, task count)`` pairs in execution order.

    ``split_head_merge`` folds the two head groups of an (sample, key-chunk)
    cell into one task, halving ``split``. It must be given the same value the
    kernel was built with: these counts set the CTA placement rotation and the
    timeline's per-stage offsets, so a disagreement silently mis-seats both.
    """
    index_ranks = index_cp_world(npes, samples, index_max_seq) if indexer_cp else 1
    tasks = [
        ("qkv_a", N_QKV_A),
        ("q_norm", samples),
        ("cache", 1),
        ("q_b", heads * (NOPE_DIM + Q_PE_DIM) // Q_B_TILE),
    ]
    if with_indexer:
        shard_q = indexer_cp and not indexer_uses_tiles(index_max_seq, indexer_cp)
        tasks += [
            ("index_q", INDEX_Q_ROWS // INDEX_TILE // (index_ranks if shard_q else 1))
        ]
    tasks += [("uk", heads * KV_LORA // UK_TILE)]
    if with_indexer:
        capacity = (
            index_cp_capacity(index_max_seq, index_ranks)
            if indexer_cp
            else index_max_seq
        )
        tasks += [
            (
                "index_score",
                samples * ((capacity + INDEX_KEYS_PER_TASK - 1) // INDEX_KEYS_PER_TASK),
            )
        ]
        if indexer_uses_tiles(index_max_seq, indexer_cp):
            parts = index_cp_partitions(index_max_seq, index_ranks)
            for digit in range(INDEX_RADIX_DIGITS):
                tasks += [
                    (f"index_hist_{digit}", samples * parts),
                    (f"index_prefix_{digit}", samples),
                ]
            tasks += [
                ("index_count", samples * parts),
                ("index_offsets", samples),
                ("index_emit", samples * parts),
            ]
        # One stage a selection-tree level; the last is named ``index_select``
        # so the timeline, the CTA rotation base and every reader of this table
        # keep the name they had when the tree was a single flat select.  A
        # tiled build is depth 0, so its table is unchanged.
        select_capacity = index_select_capacity(index_max_seq, indexer_cp)
        depth = index_levels(select_capacity, sparse_attention_topk)
        tasks += [
            (
                index_select_stage(level, depth),
                samples * index_lists(select_capacity, level, sparse_attention_topk),
            )
            for level in range(depth + 1)
        ]
    tasks += [
        (
            "split",
            samples
            * (1 if split_head_merge else heads // WAVES)
            * (sparse_attention_topk // sparse_keys_per_task(samples, heads)),
        ),
        ("uv", samples * (heads * V_DIM // UV_TILE)),
        ("o", N_ROW_TILES),
        ("router", samples * N_ROUTER),
        # one 8-row column tile of one routed slot per task (the per-sample loop
        # is inside the task); == inter for GLM's top-8
        ("ug", TOP_K * (inter // 8)),
        ("down", HIDDEN // dn_tile(samples, expert_mxfp4)),
    ]
    return tasks
