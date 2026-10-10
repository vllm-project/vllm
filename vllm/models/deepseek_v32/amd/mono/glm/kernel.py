# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/aiter#6173 at 56d6390de (Apache-2.0 License),
# Copyright (c) 2025 FlyDSL Project Contributors:
# aiter/ops/flydsl/kernels/glm5_mono/glm/kernel.py
# Closures built in the device-code loops run while that iteration is traced.
# ruff: noqa: B008, B023, E501, SIM102

"""GLM-5 indexed decode MonoKernel: one persistent launch per TP rank.

One launch of ``grid = 256 CTAs x 512 threads`` (one CTA per MI355X CU) runs the
whole decoder-layer body for this rank's TP shard.  The indexed path adds the
selection refresh to the same resident grid instead of consuming externally
prepared sparse indices::

    input RMSNorm -> q_a / kv_a projection -> q_a RMSNorm -> q_b (+RoPE)
      -> KV RMSNorm / k_pe RoPE -> KV/PE cache publish
      -> index K/Q/W projection -> index K norm/RoPE/cache
      -> index score -> exact radix top-2048
      -> absorbed q (W_UK) -> sparse MLA split softmax -> merge -> W_UV -> W_o
      -> attention TP peer reduce + residual                      (sym_attn)
      -> post-attention RMSNorm -> router sigmoid + activation FP8 quant
      -> top-8 -> 1 shared + 8 routed expert up/gate/SiLU
      -> mid FP8 quant -> expert down + route weighting
      -> MoE TP peer reduce + residual -> x_out                   (sym_ffn)

Scheduling: every stage is a list of tasks; task ``t`` of a stage runs on CTA
``(stage_base + t) % 256`` and every CTA walks the stages in order.  There is
no grid-wide barrier: dependencies only point to earlier stages and all CTAs
are co-resident, so every spin wait makes progress.

Mailboxes are *tagged pairs*: every 32-bit value a task hands to another CTA
(or GPU) is stored next to this launch's epoch tag, ``(value, tag)``, with
device- (``sc1``) or system-coherent (``sc0 sc1``) 8 / 16-byte stores.  A
consumer polls the payload itself until the tags match, so a hand-off costs
one memory round trip: no store drain, no separate flag, no second load.

GEMVs run on the matrix cores: weights are host-packed (``pack_fp8`` /
``pack_bf16``) so one wave loads 16 rows x 64 k as one contiguous 1 KB, FP8 is
widened exactly to bf16 and fed to ``mfma_f32_16x16x32_bf16`` with the samples
as the N dimension.  Each 64-k chunk's partial is scaled by its f32 block
scale (times any activation scale / route weight) into the accumulator, so the
math is exact block-scaled FP8 on bf16 activations.  Weight loads that do not
depend on upstream results are issued before the task waits for its inputs.

Cross-GPU: each rank pushes its partial rows as tagged pairs into every peer's
symmetric buffer and polls its own; every rank sums the TP partials in rank
order, so all ranks produce bit-identical hidden states (and routing).

TileRT shared/reuse MonoKernel reference (fusion-boundary comparison):
https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.deepseek_v32.amd.mono.config import (
    EPS,
    FP8_MAX,
    HIDDEN,
    INTER,
    KV_LORA,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    ROUTE_SCALE,
    SCALE_BM,
    SHARED_EXPERT,
    SOFTMAX_SCALE,
    TOP_K,
    V_DIM,
    AttentionWeight,
    KvCacheLayout,
    as_kv_cache_layout,
)
from vllm.models.deepseek_v32.amd.mono.glm.layout import (
    BLOCKS,
    DCP_SUMMARY_PAIRS,
    INDEX_DIM,
    INDEX_HEADS,
    INDEX_KEYS_PER_TASK,
    INDEX_Q_ROWS,
    INDEX_TILE,
    N_QKV_A,
    N_ROUTER,
    N_ROW_TILES,
    Q_B_TILE,
    QKV_A_TILE,
    ROUTER_TILE,
    ROW_TILE,
    UG_TILE,
    UK_TILE,
    UV_TILE,
    WAVES,
    XQ_BLOCKS,
    XQ_GROUPS,
    XQ_WAVES,
    dcp_local_uv_tile,
    dcp_summary_index,
    dcp_uv_owner,
    dn_tile,
    down_prefetch_batch,
    down_x_words,
    fp8_kv_upper_pair_lane,
    fp8_pe_upper_pair_lane,
    layout,
    sample_wave_batches,
    sparse_keys_per_task,
    split_acc_head,
    split_score_column,
    stage_tasks,
    ug_split,
    ug_task_rounds,
)
from vllm.models.deepseek_v32.amd.mono.layout import (
    CM_DEV,
    CM_SYS,
    LAYER_SLOTS,
    NEG,
    POLL_MAX,
    THREADS,
    TL_COLS,
    atom_mxfp4_scale_index,
)
from vllm.models.deepseek_v32.amd.mono.ops import (
    bpermute_i32,
    div_rn,
    f8_word,
    fp8_pack4,
    mem_realtime,
    read_lane_i32,
    spin_pause,
    wave_umax_dpp,
    write_lane_i32,
)
from vllm.models.deepseek_v32.amd.mono.ops import (
    exp as _exp,
)
from vllm.models.deepseek_v32.amd.mono.ops import (
    fp8_roundtrip as _fp8_roundtrip,
)
from vllm.models.deepseek_v32.amd.mono.ops import (
    fp8_to_bf16x8 as _fp8_to_bf16x8,
)
from vllm.models.deepseek_v32.amd.mono.ops import (
    mxfp4_to_bf16x8 as _mxfp4_to_bf16x8,
)
from vllm.models.deepseek_v32.amd.mono.ops import (
    rcp as _rcp,
)
from vllm.models.deepseek_v32.amd.mono.ops import (
    rsq as _rsq,
)
from vllm.models.deepseek_v32.amd.mono.ops import (
    rsrc as _rsrc,
)
from vllm.models.deepseek_v32.amd.mono.ops import (
    uniform as _uniform,
)
from vllm.models.deepseek_v32.amd.mono.ops import (
    uniform_f32 as _uniform_f32,
)
from vllm.models.deepseek_v32.amd.mono.ops import (
    xred as _xred,
)
from vllm.models.deepseek_v32.amd.mono.ops import (
    xshfl as _xshfl,
)


def build_glm5_monokernel(
    S: int = 1,
    heads: int = 8,
    npes: int = 8,
    topk: int = 2048,
    launches_per_step: int = 1,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    expert_mxfp4: bool = False,
    atom_experts: bool = False,
    attention_weight: AttentionWeight | str = AttentionWeight.FP8_BLOCK128,
    kv_cache_layout: KvCacheLayout | str = KvCacheLayout.SPLIT,
    kv_cache_dtype: str = "bf16",
    inter: int = INTER,
    output_heads: int | None = None,
    dcp_size: int = 1,
    scale: float = SOFTMAX_SCALE,
    uv_scale_rows: int = 128,
    native_fp4_mfma: bool = False,
    timeline: bool = False,
    index_paged: bool = False,
    index_block_size: int = 64,
    index_block_bytes: int = 0,
    index_shuffled: bool = False,
    block_table_stride: int = 0,
    index_k_bf16: bool = False,
    dense_experts: int = 0,
):
    """Return the ``@flyc.jit`` launcher for one rank's whole layer.

    ``timeline=True`` records ``s_memrealtime`` (100 MHz) at the start and end of
    every task, and once its inputs have arrived, into the ``timeline`` buffer:
    int64 ``[sum(task counts), TL_COLS]`` (start, hint seen, inputs staged, compute
    done, end, then free debug marks) in ``stage_tasks`` order.

    ``index_paged=True`` keeps the index keys in vLLM's paged FP8 cache: per
    block, ``index_block_size x 128`` E4M3 values (``index_shuffled``: 16-token x
    16-byte tiles) followed by one power-of-two f32 scale per token, with blocks
    ``index_block_bytes`` apart.  Every row is its own request: ``req_ids`` picks
    its ``block_table`` row (``block_table_stride`` entries), keys cover
    positions ``0 .. positions[s]``, and the top-k is published as global cache
    slots to ``out_indices`` at the row's ``sparse_kv_indptr`` offset.
    ``index_k_bf16`` reads the index K projection as packed bf16.

    ``dense_experts > 0`` runs a dense MLP instead of the MoE: its intermediate
    dimension is stored as that many expert-shaped slices, every row routes to all
    of them with weight 1, and the router GEMV is skipped.
    """
    assert uv_scale_rows in (64, 128)
    assert heads % WAVES == 0, "split attention maps one local head to each wave"
    assert kv_cache_dtype in ("bf16", "fp8")
    attention_weight = AttentionWeight(attention_weight)
    cache_layout = as_kv_cache_layout(kv_cache_layout)
    attention_bf16 = attention_weight is AttentionWeight.BF16
    attention_ptpc = attention_weight is AttentionWeight.FP8_PTPC
    use_atom_kv_cache = cache_layout is KvCacheLayout.ATOM
    cache_fp8 = kv_cache_dtype == "fp8"
    assert not cache_fp8 or use_atom_kv_cache
    assert not atom_experts or expert_mxfp4
    assert not native_fp4_mfma or atom_experts
    attention_k_chunks_per_unit = 1 if attention_bf16 else 2
    SPLIT_KEYS = sparse_keys_per_task(S, heads)
    assert topk % SPLIT_KEYS == 0 and 1 <= S <= 12
    assert 1 <= launches_per_step <= LAYER_SLOTS
    assert not with_indexer or (
        topk == 2048 and index_max_seq % INDEX_KEYS_PER_TASK == 0
    )
    assert not (attention_bf16 and with_indexer), (
        "BF16 attention uses the external indexer"
    )
    assert not index_paged or (
        with_indexer
        and use_atom_kv_cache
        and block_table_stride > 0
        and index_block_bytes >= index_block_size * (INDEX_DIM + 4)
        and index_block_bytes % 4 == 0
        and (not index_shuffled or index_block_size % 16 == 0)
    )
    assert 0 <= dense_experts <= MOE_SLOTS
    assert not dense_experts or (S > 1 and atom_experts)
    H = heads
    L = H if output_heads is None else output_heads
    W = npes
    D = dcp_size
    expert_inter = inter
    SHARED = 0 if dense_experts else SHARED_EXPERT
    assert D == 1 or (D == W and H == L * D)
    I_PER_SLOT = expert_inter // UG_TILE
    G = BLOCKS
    SC, SY = layout(
        S,
        H,
        W,
        topk,
        with_indexer,
        index_max_seq,
        inter=expert_inter,
        output_heads=L,
        dcp_size=D,
        native_fp4_mfma=native_fp4_mfma,
    )
    N_SPLIT = topk // SPLIT_KEYS
    QB_ROWS = H * (NOPE_DIM + PE_DIM)
    N_QB = QB_ROWS // Q_B_TILE
    assert not with_indexer or (INDEX_Q_ROWS // INDEX_TILE == G and N_QB in (G // 2, G))
    QB_PER_HEAD = (NOPE_DIM + PE_DIM) // Q_B_TILE
    N_UK = H * KV_LORA // UK_TILE
    UK_PER_HEAD = KV_LORA // UK_TILE
    N_UV = H * V_DIM // UV_TILE
    O_K = L * V_DIM
    N_UG = S * MOE_SLOTS * I_PER_SLOT
    QK_DIM = KV_LORA + PE_DIM
    # split LDS: bf16 q of all heads, then the KV latent / k_pe tiles (bf16 pairs); row
    # strides are padded by 4 words so the MFMA operand rows spread over the banks
    QS = QK_DIM // 2 + 4
    KS = KV_LORA // 2 + 4
    PS = PE_DIM // 2 + 4
    KT_OFF = H * QS
    PT_OFF = KT_OFF + SPLIT_KEYS * KS
    # The input projection is processed four samples at a time.  Besides keeping
    # the MFMA N dimension dense, this caps its normalized activation tile at
    # 48 KiB for S=8.  Later stages either consume a smaller tensor or use the
    # FP8 representation and therefore fit all samples at once.
    SAMPLE_TILE = min(S, 4)
    DN_TILE = dn_tile(S, expert_mxfp4)
    N_DN_TILES = HIDDEN // DN_TILE
    RED_WORDS = WAVES * 64 * 4
    LDS_KEYS = max(
        (288 if index_paged else 264) if with_indexer else 0,
        SPLIT_KEYS,
        S * MOE_SLOTS,
    )

    # TileRT lineage: use one phase-overlaid arena instead of summing every
    # stage's LDS requirement.  The Kimi kernel reuses this same fusion pattern.
    # The largest X users are the four-sample input projection, all-sample MoE FP8
    # activations, and sparse attention.  Metadata, reductions, and outputs live
    # after that common X region because they are simultaneously live in GEMVs.
    SPLIT_X_WORDS = PT_OFF + SPLIT_KEYS * PS
    X_WORDS = max(
        SAMPLE_TILE * HIDDEN // 2,
        S * HIDDEN // 4,
        SPLIT_X_WORDS,
        0 if index_paged else index_max_seq,
        down_x_words(S, expert_inter, expert_mxfp4, native_fp4_mfma),
    )
    # Paged top-k: a 4096-bin histogram, then (key, position) candidates of
    # the threshold bin, both in the X region of the selecting CTA.
    SEL_ITEMS = 8
    SEL_CHUNK = THREADS * SEL_ITEMS
    SEL_BINS = 4096
    SEL_CAP = (X_WORDS - SEL_BINS) // 2
    assert not index_paged or SEL_CAP >= 2048
    MISC_OFF = X_WORDS
    MISC_WORDS = max(
        8 + S * (XQ_GROUPS if native_fp4_mfma else XQ_BLOCKS),
        S * MOE_SLOTS * (expert_inter // (32 if native_fp4_mfma else 128)),
        N_SPLIT + 2 + (4 if attention_ptpc else 0),
    )
    KEYS_OFF = MISC_OFF + MISC_WORDS
    DNW_OFF = KEYS_OFF + LDS_KEYS
    RED_OFF = DNW_OFF + S * MOE_SLOTS
    OUT_OFF = RED_OFF + RED_WORDS
    # UK writes its 128-row result straight from the reduction tile to q_lat;
    # all remaining stages need at most these compact output tiles.
    OUT_WORDS = max(S * ROW_TILE, S * 2 * UG_TILE)
    WORK_WORDS = OUT_OFF + OUT_WORDS
    assert WORK_WORDS <= 32768, "keep static LDS below the MI355X per-workgroup budget"

    base, first, acc = {}, {}, 0
    for name, n in stage_tasks(
        S, H, topk, with_indexer, index_max_seq, expert_mxfp4, inter=expert_inter
    ):
        first[name] = acc
        acc += n
    # CTA placement: split before uk, so every split tile lands on a CTA freed by
    # qkv_a (uk shares the q_b CTAs it waits on anyway)
    tasks = dict(
        stage_tasks(
            S, H, topk, with_indexer, index_max_seq, expert_mxfp4, inter=expert_inter
        )
    )
    acc = 0
    for name in (
        "qkv_a",
        "q_norm",
        "cache",
        "q_b",
        "split",
        "uk",
        "uv",
        "o",
        "router",
        "ug",
        "down",
    ):
        base[name] = acc % G
        acc += tasks[name]
    if with_indexer:
        # q_b occupies exactly half the grid.  Put the second half of index-Q on
        # the complementary CTAs while q_b CTAs reuse their normalized q_lora
        # tile for the first half, instead of serializing two index-Q tiles on
        # every q_b CTA.
        base["index_q"] = (base["q_b"] + N_QB) % G
        base["index_score"] = 101
        base["index_select"] = 100

    @fx.struct
    class Smem:
        work: fx.Array[fx.Float32, WORK_WORDS, 16]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def glm5_monokernel(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
        positions: Int64,
        slot_mapping: Int64,
        sparse_kv_indptr: Int64,
        kv_cache: Int64,
        pe_cache: Int64,
        indices: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_in: Int64,
        g_q: Int64,
        g_kv: Int64,
        g_post: Int64,
        w_qkv_a: Int64,
        s_qkv_a: Int64,
        w_q_b: Int64,
        s_q_b: Int64,
        w_uk: Int64,
        s_uk: Int64,
        w_uv: Int64,
        s_uv: Int64,
        w_o: Int64,
        s_o: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        timeline_buf: Int64,
        step: Int64,
        block_table: Int64,
        req_ids: Int64,
        out_indices: Int64,
        rank: Int32,
        layer: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(Smem).peek()
        xs = lds.work.ptr
        misc = xs + MISC_OFF
        keys = fx.recast_iter(fx.Int32, xs + KEYS_OFF)
        dnw = xs + DNW_OFF
        red = xs + RED_OFF
        outs = xs + OUT_OFF
        pl = xs  # score Q is dead before split probabilities are written
        attn_keys = keys
        ktile = xs + KT_OFF  # f32-typed views holding raw bf16 pairs
        petile = xs + PT_OFF
        v4f = fx.Vector.make_type(4, fx.Float32)

        r_h = _rsrc(h_in)
        # this launch's epoch: every mailbox tag must equal it.  ``step`` is a
        # device counter bumped once per decode step (graph friendly); ``layer``
        # makes it unique per layer within the step.
        step_value = _uniform(bo.buffer_load(_rsrc(step), 0, vec_width=1, dtype=T.i32))
        tag = step_value * launches_per_step + layer + 1
        peer_slot = (step_value * launches_per_step + layer) & 1
        pos0 = _uniform(bo.buffer_load(_rsrc(cur_pos), 0, vec_width=1, dtype=T.i32))
        r_peers = _rsrc(peers)
        # One wave sends to one peer, so retain only that wave's destination.
        pv = fx.Vector(
            bo.buffer_load(r_peers, fx.min(wave, W - 1) * 2, vec_width=2, dtype=T.i32)
        )
        peer_dst = (fx.Int64(_uniform(pv[1])) << 32) | fx.Int64(
            fx.Uint32(_uniform(pv[0]))
        )

        # ------------------------------------------------------------ helpers
        def ld_f32(r, i):
            return fx.Float32(bo.buffer_load(r, i, vec_width=1, dtype=T.f32))

        def ld_bf16(r, i):
            return fx.Float32(
                fx.BFloat16(bo.buffer_load(r, i, vec_width=1, dtype=T.bf16))
            )

        def row_index_bounds(s):
            begin = fx.Int32(
                bo.buffer_load(_rsrc(sparse_kv_indptr), s, vec_width=1, dtype=T.i32)
            )
            end = fx.Int32(
                bo.buffer_load(_rsrc(sparse_kv_indptr), s + 1, vec_width=1, dtype=T.i32)
            )
            return begin, end

        def row_active(s):
            if const_expr(use_atom_kv_cache):
                begin, end = row_index_bounds(s)
                return end > begin
            return True

        def row_position(s):
            if const_expr(use_atom_kv_cache):
                position = fx.Int32(
                    bo.buffer_load(_rsrc(positions), s * 2, vec_width=1, dtype=T.i32)
                )
                present = row_active(s) | (row_slot(s) >= 0)
                return present.select(position, fx.Int32(0))
            return pos0 + s

        def row_slot(s):
            if const_expr(use_atom_kv_cache):
                return fx.Int32(
                    bo.buffer_load(_rsrc(slot_mapping), s * 2, vec_width=1, dtype=T.i32)
                )
            return pos0 + s

        def row_writes_cache(s):
            return row_slot(s) >= 0

        def index_bound(s):
            return row_writes_cache(s).select(row_position(s) + 1, fx.Int32(0))

        def index_request(s):
            return fx.Int32(bo.buffer_load(_rsrc(req_ids), s, vec_width=1, dtype=T.i32))

        def index_block(req, key):
            return fx.Int32(
                bo.buffer_load(
                    _rsrc(block_table),
                    req * block_table_stride + key // index_block_size,
                    vec_width=1,
                    dtype=T.i32,
                )
            )

        def index_value_word(block, off, d):
            """Dword holding index-cache values d .. d + 3 (d % 4 == 0) of one token."""
            if const_expr(index_shuffled):
                byte = (off // 16) * (16 * INDEX_DIM) + (off % 16) * 16
                byte = byte + (d // 16) * 256 + d % 16
            else:
                byte = off * INDEX_DIM + d
            return block * (index_block_bytes // 4) + byte // 4

        def index_scale_word(block, off):
            return (
                block * (index_block_bytes // 4)
                + index_block_size * INDEX_DIM // 4
                + off
            )

        def lds_ld(ptr, i):
            return fx.ptr_load(ptr + i)

        def lds_st(ptr, i, v):
            fx.ptr_store(v, ptr + i)

        def bf16_pair(a, b):
            """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
            return (
                fx.Vector.from_elements([a, b], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Float32)[0]
            )

        def bf16_round(a):
            return fx.Float32(fx.Float32(a).to(fx.BFloat16))

        def index_arg(i):
            """Load one uniform pointer from the compact indexer parameter table.

            Fused-indexer launches carry this table in the otherwise independent
            timeline argument, keeping the no-indexer kernel ABI identical to the
            original layer.  The indices argument similarly carries index_cache.
            """
            pv = fx.Vector(
                bo.buffer_load(_rsrc(timeline_buf), i * 2, vec_width=2, dtype=T.i32)
            )
            return (fx.Int64(_uniform(pv[1])) << 32) | fx.Int64(
                fx.Uint32(_uniform(pv[0]))
            )

        # ---- tagged-pair mailboxes
        # TileRT lineage: payload + launch epoch is the progress protocol for
        # resident CTAs; the helpers below are the FlyDSL/ROCm adaptation.
        def mb(name):
            return scratch + fx.Int64(SC[name])

        def put(base_addr, i, v, cm=CM_DEV):
            """Pair i := (v, tag); ``v`` f32 (or int32 bits)."""
            bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
            bo.buffer_store(
                fx.Vector.from_elements([bits, tag], fx.Int32),
                _rsrc(base_addr),
                i * 2,
                cache_modifier=cm,
            )

        def put2(base_addr, i, v0, v1, cm=CM_DEV):
            """Pairs i, i+1 (i even) in one 16-byte store."""
            vec = fx.Vector.from_elements(
                [
                    fx.Float32(v0).bitcast(fx.Int32),
                    tag,
                    fx.Float32(v1).bitcast(fx.Int32),
                    tag,
                ],
                fx.Int32,
            )
            bo.buffer_store(vec, _rsrc(base_addr), i * 2, cache_modifier=cm)

        def put_bf(base_addr, i, vs, cm=CM_DEV):
            """Elements i .. i + len(vs) (2 or 4, i aligned) as packed bf16 pairs: pair
            i / 2 + j := (bf16(vs[2j]) | bf16(vs[2j + 1]) << 16, tag), one 8 / 16-byte store.
            """
            words = []
            for j in range_constexpr(len(vs) // 2):
                words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), tag]
            bo.buffer_store(
                fx.Vector.from_elements(words, fx.Int32),
                _rsrc(base_addr),
                i,
                cache_modifier=cm,
            )

        def bf2_f32(w):
            """Packed bf16 pair word -> (f32 low, f32 high)."""
            return (w << 16).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(
                fx.Float32
            )

        def _qptr(addr):
            return fx.inttoptr(
                fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                fx.Int64(addr),
            )

        def _ld_pair(addr, scope):
            """One (value, tag) pair as a single 64-bit relaxed atomic load: never hoisted,
            coherent at ``scope`` (agent -> sc1, system -> sc0 sc1)."""
            return fx.generic_load(
                _qptr(addr), memory_order=fx.AtomicOrdering.Monotonic, syncscope=scope
            )

        def poll(specs, scope="agent", batch=POLL_MAX):
            """Batched poll of mailbox pairs: ``specs`` = [(base_addr, pair index, npairs in {1, 2})].

            All pairs are loaded together with plain 8 / 16-byte coherent buffer loads
            (sc1 locally, sc0 sc1 for peer memory); while any tag is not this launch's
            the whole batch is re-loaded, so a batch costs one round trip after its
            last producer lands.  A side-effecting (compiler-opaque) asm statement in
            the retry loop keeps the loads from being hoisted.  Returns one list of
            Int32 value bits per spec."""
            if const_expr(len(specs) == 0):
                return []
            if const_expr(len(specs) > batch):  # bound live registers
                return poll(specs[:batch], scope, batch) + poll(
                    specs[batch:], scope, batch
                )
            cm = CM_DEV if const_expr(scope == "agent") else CM_SYS

            def load_all():
                words = []
                for b, i, n in specs:
                    w = fx.Vector(
                        bo.buffer_load(
                            _rsrc(b),
                            fx.Int32(i) * 2,
                            vec_width=2 * n,
                            dtype=T.i32,
                            cache_modifier=cm,
                        )
                    )
                    words += [w[e] for e in range(2 * n)]
                return fx.Vector.from_elements(words, fx.Int32)

            nw = sum(2 * n for _, _, n in specs)

            def pending(v):
                bad = v[1] != tag
                for e in range_constexpr(3, nw, 2):
                    bad = bad | (v[e] != tag)
                return bad

            v = load_all()
            while pending(v):
                spin_pause()
                v = load_all()
            outs_, e = [], 0
            for _, _, n in specs:
                outs_.append([v[e + 2 * q] for q in range(n)])
                e += 2 * n
            return outs_

        def hint_wait(n, addr_of, mark=None):
            """Consumers poll their payload directly (tight per-wave spins); a wave-0
            pre-poll of each producer's last pair only added a hop of latency."""
            if const_expr(mark is not None):
                stamp(mark[0], mark[1], 1)
            gpu.barrier()

        def pre_poll(n, addr_of):
            """Wave 0 spins on one small pair per producer (lane j -> producer j < n <= 64)
            before a large payload poll, so waiting CTAs do not flood memory."""
            if wave == 0:
                b, i = addr_of(fx.min(lane, n - 1))
                poll([(b, i, 1)])
            gpu.barrier()

        def get(base_addr, i):
            return poll([(base_addr, i, 1)])[0][0]

        def getf(base_addr, i):
            return get(base_addr, i).bitcast(fx.Float32)

        def getf_many(specs):
            """[(base, i)] single pairs -> list of f32."""
            return [
                v[0].bitcast(fx.Float32) for v in poll([(b, i, 1) for b, i in specs])
            ]

        def get2_many(specs):
            """[(base, i)] double pairs (i even) -> list of (f32, f32)."""
            return [
                (v[0].bitcast(fx.Float32), v[1].bitcast(fx.Float32))
                for v in poll([(b, i, 2) for b, i in specs])
            ]

        def get2(base_addr, i):
            return get2_many([(base_addr, i)])[0]

        def get_bf2_many(specs):
            """[(base, i)] packed bf16 elements i, i + 1 (i even) -> list of (f32, f32)."""
            return [bf2_f32(v[0]) for v in poll([(b, i // 2, 1) for b, i in specs])]

        # ---- wave reductions
        def wave_sum(v):
            for sh in range_constexpr(6):
                v = _xred(v, 32 >> sh, lambda a, b: a + b)
            return v

        def wave_max(v):
            for sh in range_constexpr(6):
                v = _xred(v, 32 >> sh, fx.max)
            return v

        def block_sums(vs):
            """Block-wide sums of several per-thread values with one LDS exchange."""
            ws = [wave_sum(v) for v in vs]
            if lane == 0:
                for i in range_constexpr(len(vs)):
                    lds_st(red, i * WAVES + wave, ws[i])
            gpu.barrier()
            tots = []
            for i in range_constexpr(len(vs)):
                t = lds_ld(red, i * WAVES)
                for w in range_constexpr(1, WAVES):
                    t = t + lds_ld(red, i * WAVES + w)
                tots.append(t)
            gpu.barrier()
            return tots

        def block_maxs(vs):
            """Block-wide maxima of several per-thread values with one LDS exchange."""
            ws = [wave_max(v) for v in vs]
            if lane == 0:
                for i in range_constexpr(len(vs)):
                    lds_st(red, i * WAVES + wave, ws[i])
            gpu.barrier()
            tots = []
            for i in range_constexpr(len(vs)):
                t = lds_ld(red, i * WAVES)
                for w in range_constexpr(1, WAVES):
                    t = fx.max(t, lds_ld(red, i * WAVES + w))
                tots.append(t)
            gpu.barrier()
            return tots

        def block_sum(v):
            w = wave_sum(v)
            if lane == 0:
                lds_st(red, wave, w)
            gpu.barrier()
            t = lds_ld(red, 0)
            for i in range_constexpr(1, WAVES):
                t = t + lds_ld(red, i)
            gpu.barrier()
            return t

        # ------------------------------------------------ MFMA GEMV machinery
        def unit_fp8(w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word, coef=None):
            """Issue one 64-k chunk of row group ``rg`` of a packed FP8 matrix; the
            bf16 activation chunk starts at LDS word ``b_word``."""
            wv = fx.Vector(
                bo.buffer_load(
                    w_rsrc, ((rg * NKC + kc) * 64 + lane) * 4, vec_width=4, dtype=T.i32
                )
            )
            s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // BK) + kc * 64 // BK)
            if const_expr(callable(coef)):  # factor known only after a later wait
                return ("fp8", [wv], lambda: s * coef(), b_word + (lane // 16) * 4)
            if const_expr(coef is not None):
                s = s * coef
            return ("fp8", [wv], s, b_word + (lane // 16) * 4)

        def unit_fp8x2(
            w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, coef=None, scale_rows=SCALE_BM
        ):
            """Issue both 64-k halves of one 128-k FP8 weight-scale block."""
            wv = [
                fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        ((rg * NKC + kc + h) * 64 + lane) * 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for h in range(2)
            ]
            s = ld_f32(s_rsrc, (rg * 16 // scale_rows) * (K // 128) + kc // 2)
            if const_expr(callable(coef)):
                return ("fp8x2", wv, lambda: s * coef(), b_word + (lane // 16) * 4)
            if const_expr(coef is not None):
                s = s * coef
            return ("fp8x2", wv, s, b_word + (lane // 16) * 4)

        def unit_f8f8(w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, coef, ln=None):
            """Issue one 128-k chunk (packed 64-k chunks kc, kc + 1; kc even) of row group
            ``rg`` against the FP8 activation of LDS words ``b_word`` + [0, 32) (``f8_word``
            order); ``coef()`` = activation block scale (times route weight).  ``ln``
            = the lane whose weights are loaded (default: own lane)."""
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        ((rg * NKC + kc + h) * 64 + ln) * 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for h in range(2)
            ]
            s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // 128) + kc // 2)
            return ("f8f8", wv, lambda: s * coef(), b_word + (lane // 16) * 4)

        def unit_ptpc(w_rsrc, rg, kc, NKC, b_word, coef=None, ln=None):
            """Issue one packed 128-K PTPC FP8 tile; output-channel scale is applied after reduction."""
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        ((rg * NKC + kc + h) * 64 + ln) * 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for h in range(2)
            ]
            return ("ptpc", wv, coef, b_word + (lane // 16) * 4)

        def unit_ptpc64(w_rsrc, rg, kc, NKC, b_word, coef=None, ln=None):
            """Issue one packed 64-K PTPC FP8 tile."""
            ln = lane if ln is None else ln
            weight = fx.Vector(
                bo.buffer_load(
                    w_rsrc,
                    ((rg * NKC + kc) * 64 + ln) * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            return ("ptpc64", [weight], coef, b_word + (lane // 16) * 4)

        def unit_mxfp4(w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln=None):
            """Issue one native packed 128-K MXFP4 tile and four E8M0 row scales."""
            ln = lane if ln is None else ln
            row = rg * 16 + ln % 16
            if const_expr(atom_experts):
                g4 = ln // 16
                raw = fx.Vector.from_elements(
                    [
                        fx.Int32(
                            bo.buffer_load(
                                w_rsrc,
                                (rg * (K // 64) + kc * 2 + sp // 2) * 128
                                + (sp % 2) * 64
                                + (row % 16) * 4
                                + g4,
                                vec_width=1,
                                dtype=T.i32,
                            )
                        )
                        for sp in range_constexpr(4)
                    ],
                    fx.Int32,
                )
                scales = [
                    (
                        fx.Int32(
                            bo.buffer_load(
                                s_rsrc,
                                atom_mxfp4_scale_index(row, kc * 4 + sp, K // 32),
                                vec_width=1,
                                dtype=T.i8,
                            )
                        )
                        << fx.Int32(23)
                    ).bitcast(fx.Float32)
                    for sp in range_constexpr(4)
                ]
            else:
                raw = fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        ((rg * (K // 128) + kc) * 64 + ln) * 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                packed_scale = fx.Int32(
                    bo.buffer_load(
                        s_rsrc, row * (K // 128) + kc, vec_width=1, dtype=T.i32
                    )
                )
                scales = [
                    (
                        (packed_scale.shrui(fx.Int32(sp * 8)) & fx.Int32(0xFF))
                        << fx.Int32(23)
                    ).bitcast(fx.Float32)
                    for sp in range_constexpr(4)
                ]
            return ("mxfp4", (raw, scales), coef, b_word + (lane // 16) * 4)

        def unit_mxfp4_bf16(w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln=None):
            fmt, weights, factor, _ = unit_mxfp4(
                w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln
            )
            return ("mxfp4_bf16", weights, factor, b_word + (lane // 16) * 4)

        def unit_native_mxfp4(
            w_rsrc, s_rsrc, rg, kc, K, b_word, b_scale_word, coef=None, ln=None
        ):
            """Issue ATOM's 16-byte/lane W4 tile directly on the scaled MFMA."""
            ln = lane if ln is None else ln
            raw = fx.Vector(
                bo.buffer_load(
                    w_rsrc,
                    (rg * (K // 64) + kc * 2) * 128 + ln * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            scale = fx.Int32(
                bo.buffer_load(
                    s_rsrc,
                    atom_mxfp4_scale_index(
                        rg * 16 + ln % 16, kc * 4 + ln // 16, K // 32
                    ),
                    vec_width=1,
                    dtype=T.i8,
                )
            )
            return ("native_mxfp4", (raw, scale, b_scale_word), coef, b_word)

        def lds_mxfp8(chunk_word):
            """One scaled-MFMA B operand from a row-major 128-value MXFP8 tile."""
            words = [
                fx.Vector(
                    fx.ptr_load(
                        xs + chunk_word + 16 * half + 4 * (lane // 16),
                        result_type=v4f,
                    )
                ).bitcast(fx.Int32)
                for half in range(2)
            ]
            return fx.Vector.from_elements(
                [words[half][e] for half in range(2) for e in range(4)],
                fx.Int32,
            )

        def unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln=None):
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        (((rg * NKC + kc) * 2 + sp) * 64 + ln) * 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for sp in range(2)
            ]
            return ("bf16", wv, None, b_word + (lane // 16) * 4)

        def unit_attention(
            w_rsrc,
            s_rsrc,
            rg,
            kc,
            NKC,
            K,
            BK,
            b_word,
            ln=None,
            scale_rows=SCALE_BM,
            coef=None,
        ):
            if const_expr(attention_bf16):
                return unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln)
            if const_expr(attention_ptpc):
                if const_expr(BK == 64):
                    return unit_ptpc64(w_rsrc, rg, kc, NKC, b_word, coef=coef, ln=ln)
                return unit_ptpc(w_rsrc, rg, kc, NKC, b_word, coef=coef, ln=ln)
            if const_expr(BK == 64):
                return unit_fp8(w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word)
            return unit_fp8x2(
                w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, scale_rows=scale_rows
            )

        def mma_units(acc, units):
            """acc[4] += coef * (W_chunk @ X_chunk) for every issued unit."""
            for fmt, wv, coef, bw in units:
                if const_expr(callable(coef)):
                    coef = coef()
                c = fx.Vector.filled(4, 0.0, fx.Float32)
                if const_expr(fmt in ("mxfp4", "mxfp4_bf16")):
                    raw, scales = wv
                    for sp in range_constexpr(4):
                        a = _mxfp4_to_bf16x8(raw[sp], scales[sp])
                        if const_expr(fmt == "mxfp4"):
                            wh, ws = sp // 2, sp % 2
                            bv = fx.Vector(
                                fx.ptr_load(xs + (bw + wh * 16), result_type=v4f)
                            ).bitcast(fx.Int32)
                            b = _fp8_to_bf16x8(bv[ws * 2], bv[ws * 2 + 1])
                        else:
                            b = fx.ptr_load(
                                xs + (bw + sp * 16), result_type=v4f
                            ).bitcast(fx.BFloat16)
                        c = fx.Vector(
                            rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c])
                        )
                if const_expr(fmt == "f8f8"):  # one FP8 x FP8 MFMA (E8M0 scales = 1)
                    a = fx.Vector.from_elements(
                        [wv[h][e] for h in range(2) for e in range(4)], fx.Int32
                    )
                    bv = [
                        fx.Vector(
                            fx.ptr_load(xs + (bw + h * 16), result_type=v4f)
                        ).bitcast(fx.Int32)
                        for h in range(2)
                    ]
                    b = fx.Vector.from_elements(
                        [bv[h][e] for h in range(2) for e in range(4)], fx.Int32
                    )
                    one = fx.Int32(127)
                    c = fx.Vector(
                        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.vec(4, T.f32), [a, b, c, 0, 0, 0, one, 0, one]
                        )
                    )
                if const_expr(fmt == "native_mxfp4"):
                    raw, weight_scale, activation_scale_word = wv
                    activation_scale = lds_ld(misc, activation_scale_word).bitcast(
                        fx.Int32
                    )
                    c = fx.Vector(
                        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.vec(4, T.f32),
                            [
                                raw,
                                lds_mxfp8(bw),
                                c,
                                4,
                                0,
                                0,
                                weight_scale,
                                0,
                                activation_scale,
                            ],
                        )
                    )
                if const_expr(fmt in ("ptpc", "ptpc64")):
                    halves64 = 2 if fmt == "ptpc" else 1
                    for half64 in range_constexpr(halves64):
                        a = wv[half64].bitcast(fx.Int64)
                        b = fx.Vector(
                            fx.ptr_load(xs + (bw + half64 * 16), result_type=v4f)
                        ).bitcast(fx.Int64)
                        for half32 in range_constexpr(2):
                            c = fx.Vector(
                                rocdl.mfma_f32_16x16x32_fp8_fp8(
                                    T.vec(4, T.f32),
                                    [a[half32], b[half32], c, 0, 0, 0],
                                )
                            )
                nsp = (
                    4
                    if fmt == "fp8x2"
                    else (
                        2
                        if fmt
                        not in (
                            "f8f8",
                            "ptpc",
                            "ptpc64",
                            "mxfp4",
                            "mxfp4_bf16",
                            "native_mxfp4",
                        )
                        else 0
                    )
                )
                for sp in range_constexpr(nsp):
                    if const_expr(fmt in ("fp8", "fp8x2")):
                        wh = sp // 2 if fmt == "fp8x2" else 0
                        ws = sp % 2 if fmt == "fp8x2" else sp
                        a = _fp8_to_bf16x8(wv[wh][ws * 2], wv[wh][ws * 2 + 1])
                    else:
                        a = wv[sp].bitcast(fx.BFloat16)
                    b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(
                        fx.BFloat16
                    )
                    c = fx.Vector(
                        rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c])
                    )
                if const_expr(coef is None):
                    acc = [acc[e] + c[e] for e in range(4)]
                else:
                    acc = [acc[e] + c[e] * coef for e in range(4)]
            return acc

        def run_units(make_unit, cpw, batch, pre=None):
            """Software pipelined: issue batch b+1's loads before computing batch b.
            ``pre`` = the already-issued first batch (prefetched before a wait)."""
            acc = [fx.Float32(0.0) for _ in range(4)]
            starts = list(range(0, cpw, batch))
            cur = (
                pre
                if pre is not None
                else [make_unit(c) for c in range(0, min(batch, cpw))]
            )
            for bi in range_constexpr(len(starts)):
                nxt = None
                if const_expr(bi + 1 < len(starts)):
                    n0 = starts[bi + 1]
                    nxt = [make_unit(c) for c in range(n0, min(n0 + batch, cpw))]
                acc = mma_units(acc, cur)
                cur = nxt
            return acc

        def reduce_rows(R, acc, emit, count=S):
            """Sum per-wave MFMA tiles; emit(row_local, local sample column, value)."""
            wpr = WAVES // R
            fx.ptr_store(
                fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4
            )
            gpu.barrier()
            n_out = R * 16 * count
            for i in range_constexpr((n_out + THREADS - 1) // THREADS):
                t = tid + i * THREADS
                if t < n_out:
                    rl = t % (R * 16)
                    n = t // (R * 16)
                    r = rl % 16
                    tot = fx.Float32(0.0)
                    for j in range_constexpr(wpr):
                        ww = (rl // 16) * wpr + j
                        tot = tot + lds_ld(
                            red, (ww * 64 + n + 16 * (r // 4)) * 4 + r % 4
                        )
                    emit(rl, n, tot)

        def emit_out(stride):
            def f(rl, n, v):
                lds_st(outs, n * stride + rl, v)

            return f

        def pick(values, index):
            value = values[0]
            for i in range_constexpr(1, len(values)):
                value = (index == i).select(values[i], value)
            return value

        def attention_b_word(element):
            return element // 4 if const_expr(attention_ptpc) else element // 2

        def emit_ptpc(stride, row_base, scale_rsrc, x_scales):
            def f(rl, n, v):
                lds_st(
                    outs,
                    n * stride + rl,
                    bf16_round(
                        v * pick(x_scales, n) * ld_f32(scale_rsrc, row_base + rl)
                    ),
                )

            return f

        def emit_ptpc_weight(stride, row_base, scale_rsrc):
            def f(rl, n, v):
                lds_st(
                    outs,
                    n * stride + rl,
                    bf16_round(v * ld_f32(scale_rsrc, row_base + rl)),
                )

            return f

        def emit_ptpc_scalar(stride, scale):
            def f(rl, n, v):
                lds_st(outs, n * stride + rl, bf16_round(v * scale))

            return f

        def stage_x_rmsnorm_ptpc(ld4s, n, gamma, loaded=None, count=S):
            """RMSNorm followed by production per-token FP8 quantization."""
            per = n // (4 * THREADS)
            ks = [(tid + i * THREADS) * 4 for i in range(per)]
            gs, vals = (
                loaded if loaded is not None else load_x_rmsnorm(ld4s, n, gamma, count)
            )
            sums = []
            for s in range_constexpr(count):
                ss = fx.Float32(0.0)
                for i in range_constexpr(per):
                    for value in vals[s * per + i]:
                        ss = ss + value * value
                sums.append(ss)
            totals = block_sums(sums)
            normed = []
            for s in range_constexpr(count):
                rstd = _rsq(totals[s] * (1.0 / n) + EPS)
                for i in range_constexpr(per):
                    normed.append(
                        [
                            vals[s * per + i][j] * rstd * gs[i][j]
                            for j in range_constexpr(4)
                        ]
                    )
            local_max = []
            for s in range_constexpr(count):
                amax = fx.Float32(0.0)
                for i in range_constexpr(per):
                    for value in normed[s * per + i]:
                        amax = fx.max(amax, fmath.absf(value))
                local_max.append(amax)
            maxima = block_maxs(local_max)
            scales = []
            for s in range_constexpr(count):
                scale = (maxima[s] == 0.0).select(
                    fx.Float32(1.0), maxima[s] * (1.0 / FP8_MAX)
                )
                scales.append(scale)
                reciprocal = 1.0 / scale
                for i in range_constexpr(per):
                    values = [
                        div_rn(value, scale, reciprocal)
                        for value in normed[s * per + i]
                    ]
                    lds_st(
                        xs,
                        (s * n + ks[i]) // 4,
                        fp8_pack4(values[0], values[1], values[2], values[3]).bitcast(
                            fx.Float32
                        ),
                    )
            return scales

        def stage_x_pairs_ptpc(name, count, n, src_of, group=None):
            """Quantize packed-BF16 mailbox rows with one scale per activation group."""
            group = n if group is None else group
            groups = (n + group - 1) // group
            scales = []
            for s in range_constexpr(count):
                for g in range_constexpr(groups):
                    width = min(group, n - g * group)
                    words = width // 4
                    per = (words + THREADS - 1) // THREADS
                    local_max = fx.Float32(0.0)
                    row_values = []
                    valids = []
                    for i in range_constexpr(per):
                        word = tid + i * THREADS
                        source_word = fx.min(word, words - 1)
                        element = s * n + g * group + source_word * 4
                        got = poll([(mb(name), src_of(element) // 2, 2)])[0]
                        first = bf2_f32(got[0])
                        second = bf2_f32(got[1])
                        values = [first[0], first[1], second[0], second[1]]
                        valid = word < words
                        row_values.append(values)
                        valids.append(valid)
                        for value in values:
                            local_max = fx.max(
                                local_max,
                                valid.select(fmath.absf(value), fx.Float32(0.0)),
                            )
                    maximum = block_maxs([local_max])[0]
                    scale = (maximum == 0.0).select(
                        fx.Float32(1.0), maximum * (1.0 / FP8_MAX)
                    )
                    scales.append(scale)
                    reciprocal = 1.0 / scale
                    for i in range_constexpr(per):
                        word = tid + i * THREADS
                        if valids[i]:
                            values = row_values[i]
                            quantized = [
                                div_rn(value, scale, reciprocal) for value in values
                            ]
                            lds_st(
                                xs,
                                (s * n + g * group + word * 4) // 4,
                                fp8_pack4(
                                    quantized[0],
                                    quantized[1],
                                    quantized[2],
                                    quantized[3],
                                ).bitcast(fx.Float32),
                            )
            return scales

        def stage_x_rmsnorm(ld4s, n, gamma, mark=None, loaded=None, count=S):
            """LDS bf16 X[s][0:n] = bf16(rmsnorm(x_s) * gamma) for every sample s, where
            ld4s([(s, k)]) -> [(x_s[k], .., x_s[k+3])] (one batched load); returns the rstds.
            ``loaded``: the (gamma, x) loads already issued by load_x_rmsnorm."""
            per = n // (4 * THREADS)
            ks = [(tid + i * THREADS) * 4 for i in range(per)]
            gs, vals = (
                loaded if loaded is not None else load_x_rmsnorm(ld4s, n, gamma, count)
            )
            sss = []
            for s in range_constexpr(count):
                ss = fx.Float32(0.0)
                for i in range_constexpr(per):
                    for a in vals[s * per + i]:
                        ss = ss + a * a
                sss.append(ss)
            if const_expr(mark is not None):
                stamp(mark[0], mark[1], 6)
            rstds = [_rsq(tot * (1.0 / n) + EPS) for tot in block_sums(sss)]
            if const_expr(mark is not None):
                stamp(mark[0], mark[1], 7)
            for s in range_constexpr(count):
                for i in range_constexpr(per):
                    a = vals[s * per + i]
                    for j in range_constexpr(2):
                        lds_st(
                            xs,
                            (s * n + ks[i]) // 2 + j,
                            bf16_pair(
                                a[2 * j] * rstds[s] * gs[i][2 * j],
                                a[2 * j + 1] * rstds[s] * gs[i][2 * j + 1],
                            ),
                        )
            return rstds

        def load_x_rmsnorm(ld4s, n, gamma, count=S):
            """The gamma loads (issued ahead of the wait), then ld4s -> (gammas, x values)."""
            rg_ = _rsrc(gamma)
            ks = [(tid + i * THREADS) * 4 for i in range(n // (4 * THREADS))]
            gs = []
            for k in ks:
                g = (
                    fx.Vector(bo.buffer_load(rg_, k // 2, vec_width=2, dtype=T.i32))
                    .bitcast(fx.BFloat16)
                    .to(fx.Float32)
                )
                gs.append([g[j] for j in range(4)])
            return gs, ld4s([(s, k) for s in range(count) for k in ks])

        def stage_x_pairs(name, n_total, src_of):
            """LDS bf16 X[k] = packed bf16 mailbox ``name`` element src_of(k) for k < n_total
            (src_of contiguous over aligned groups of 4): one 16-byte poll per 4 elements.
            """
            nq = n_total // 4
            full = nq // THREADS
            vals = poll(
                [
                    (mb(name), src_of((tid + i * THREADS) * 4) // 2, 2)
                    for i in range(full)
                ]
            )
            for i in range_constexpr(full):
                for j in range_constexpr(2):
                    lds_st(
                        xs, (tid + i * THREADS) * 2 + j, vals[i][j].bitcast(fx.Float32)
                    )
            if const_expr(nq % THREADS):
                w = tid + full * THREADS
                if w < nq:
                    v = poll([(mb(name), src_of(w * 4) // 2, 2)])[0]
                    for j in range_constexpr(2):
                        lds_st(xs, w * 2 + j, v[j].bitcast(fx.Float32))

        def quant_scaled(a0, a1):
            """Per-wave FP8 quant of a 128-block held as 2 f32 per lane -> (scaled q0, q1, scale)."""
            amax = wave_max(fx.max(fmath.absf(a0), fmath.absf(a1)))
            nz = amax > 0.0
            qs = nz.select(amax * (1.0 / FP8_MAX), fx.Float32(1.0))
            inv = nz.select(
                _rcp(amax) * FP8_MAX, fx.Float32(1.0)
            )  # hardware rcp, no IEEE divide
            q0 = fx.min(fx.max(a0 * inv, -FP8_MAX), FP8_MAX)
            q1 = fx.min(fx.max(a1 * inv, -FP8_MAX), FP8_MAX)
            return q0, q1, qs

        def quant_mxfp8(a0, a1):
            """Per-32 E4M3 values and E8M0 scale used by ATOM/AITER W4A8."""
            amax = fx.max(fmath.absf(a0), fmath.absf(a1))
            for offset in (1, 2, 4, 8):
                amax = _xred(amax, offset, fx.max)
            bits = (amax * fx.Float32(1.0 / FP8_MAX)).bitcast(fx.Int32)
            exponent = (bits >> 23) & fx.Int32(0xFF)
            exponent = exponent + ((bits & fx.Int32(0x7FFFFF)) != 0).select(
                fx.Int32(1), fx.Int32(0)
            )
            exponent = fx.max(fx.min(exponent, fx.Int32(254)), fx.Int32(1))
            inverse = _rcp((exponent << 23).bitcast(fx.Float32))
            q0 = fx.min(fx.max(a0 * inverse, -FP8_MAX), FP8_MAX)
            q1 = fx.min(fx.max(a1 * inverse, -FP8_MAX), FP8_MAX)
            return q0, q1, exponent

        def quant_block(a0, a1):
            """quant_scaled, values returned as the FP8-rounded f32s."""
            q0, q1, qs = quant_scaled(a0, a1)
            d0, d1 = _fp8_roundtrip(q0, q1)
            return d0, d1, qs

        def stage_xq(samples):
            """Poll the router's packed FP8 activation + block scales of ``samples``
            (sample list, or one runtime sample) into LDS, slot 0 for one runtime sample."""
            nxw = HIDDEN // 4 // THREADS
            nscale = XQ_GROUPS if native_fp4_mfma else XQ_BLOCKS
            got = poll(
                [
                    (mb("xq"), sx * (HIDDEN // 4) + tid + i * THREADS, 1)
                    for sx in samples
                    for i in range(nxw)
                ]
                + [
                    (mb("xqs"), sx * nscale + fx.min(tid, nscale - 1), 1)
                    for sx in samples
                ]
            )
            for j in range_constexpr(len(samples)):
                for i in range_constexpr(nxw):
                    wd = (
                        tid + i * THREADS
                        if const_expr(native_fp4_mfma)
                        else f8_word((tid + i * THREADS) * 4)
                    )
                    lds_st(
                        xs,
                        j * (HIDDEN // 4) + wd,
                        got[j * nxw + i][0].bitcast(fx.Float32),
                    )
                if tid < nscale:
                    lds_st(
                        misc,
                        8 + j * nscale + tid,
                        got[len(samples) * nxw + j][0].bitcast(fx.Float32),
                    )

        def st_f8(k, q0, q1):
            """LDS FP8 activation bytes k, k + 1 (k even, held by this lane; lane ^ 1 holds
            k ^ 2) in ``f8_word`` order.  Call from the whole wave."""
            w = (
                fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False))
                & 0xFFFF
            )
            nb = _xshfl(w, 1)
            if lane % 2 == 0:
                lds_st(xs, f8_word(k), (w | (nb << 16)).bitcast(fx.Float32))

        def st_mxfp8(k, q0, q1):
            """Store one row-major MXFP8 pair for the scaled MFMA."""
            w = (
                fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False))
                & 0xFFFF
            )
            nb = _xshfl(w, 1)
            if lane % 2 == 0:
                lds_st(xs, k // 4, (w | (nb << 16)).bitcast(fx.Float32))

        def st_ptpc(k, q0, q1):
            """Store one wave's PTPC FP8 pairs in the AITER GEMM's linear K order."""
            word = (
                fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False))
                & 0xFFFF
            )
            neighbor = _xshfl(word, 1)
            if lane % 2 == 0:
                lds_st(xs, k // 4, (word | (neighbor << 16)).bitcast(fx.Float32))

        def load_bias():
            """This lane's 4 expert biases (issue before the scores wait)."""
            if const_expr(dense_experts):
                return None
            return [ld_f32(_rsrc(bias), lane + i * 64) for i in range(N_EXPERTS // 64)]

        def route_top8(s, raws=None, bs=None):
            """Top-8 of sample s (call from one whole wave, after the router scores landed).

            Preserve all FP32 bits of (sigmoid + bias). Each round reduces the score
            first, then the expert ID only among exactly equal winners. Packing the
            ID into the score's low byte can change GLM's top-k on real checkpoints.
            Candidate i of this lane is expert lane + 64 i.
            Returns (expert id, route weight = raw score / sum of
            the 8 raw scores * ROUTE_SCALE) of pick ``lane`` in score order, valid in
            lanes < TOP_K."""
            if const_expr(bs is None):
                bs = load_bias()
            if const_expr(raws is None):
                raws = getf_many(
                    [
                        (mb("scores"), s * N_EXPERTS + lane + i * 64)
                        for i in range(N_EXPERTS // 64)
                    ]
                )
                stamp("ug", bid, 7)
            ks = []
            for i in range_constexpr(N_EXPERTS // 64):
                kb = (raws[i] + bs[i]).bitcast(fx.Int32)
                ok = (kb >= 0).select(kb ^ fx.Int32(-(2**31)), ~kb)
                ks.append(fx.Uint32(ok))
            ids = [lane + i * 64 for i in range(N_EXPERTS // 64)]
            # sort this lane's 4 keys descending; each round then takes the wave max of
            # the lane heads and shifts the winning lane's list (0 is below every key)
            for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2)):
                first = (ks[a] > ks[b]) | ((ks[a] == ks[b]) & (ids[a] < ids[b]))
                ka, kb, ia, ib = ks[a], ks[b], ids[a], ids[b]
                ks[a], ks[b] = first.select(ka, kb), first.select(kb, ka)
                ids[a], ids[b] = first.select(ia, ib), first.select(ib, ia)
            ks = [fx.Int32(k) for k in ks] + [fx.Int32(0)]
            ids = ids + [fx.Int32(N_EXPERTS)]
            mv = fx.Int32(0)  # lane k: the expert ID of pick k
            for k in range_constexpr(TOP_K):
                m = wave_umax_dpp(ks[0])
                winner = wave_umax_dpp((ks[0] == m).select(255 - ids[0], fx.Int32(0)))
                expert = 255 - winner
                hit = (ks[0] == m) & (ids[0] == expert)
                ks = [hit.select(ks[i + 1], ks[i]) for i in range(4)] + [ks[4]]
                ids = [hit.select(ids[i + 1], ids[i]) for i in range(4)] + [ids[4]]
                mv = write_lane_i32(expert, k, mv)
            e = mv
            src = (e % 64) * 4
            got = [bpermute_i32(src, r.bitcast(fx.Int32)) for r in raws]
            raw = got[0]
            for i in range_constexpr(1, N_EXPERTS // 64):
                raw = (e // 64 == i).select(got[i], raw)
            raw = (lane < TOP_K).select(raw.bitcast(fx.Float32), fx.Float32(0.0))
            tot = raw
            for off in (1, 2, 4):
                tot = _xred(tot, off, lambda a, b: a + b)
            return e, raw * (_rcp(tot) * ROUTE_SCALE)

        def peer_reduce(region, t, residual, out_fn, tile=ROW_TILE):
            """Push BF16 partials to every peer, then sum them in rank order.

            One wave owns each destination, allowing the peer stores to progress
            concurrently without retaining all peer pointers in every wave."""
            region_base = fx.Int64(SY[region]) + fx.Int64(peer_slot) * fx.Int64(
                SY["_part_stride"]
            )
            if const_expr(W > 1):
                if wave < W:
                    pair_count = S * tile // 2
                    for batch in range_constexpr((pair_count + 63) // 64):
                        pair = lane + batch * 64
                        if pair < pair_count:
                            si = pair // (tile // 2)
                            ri = (pair % (tile // 2)) * 2
                            put_bf(
                                peer_dst + region_base,
                                (rank * S + si) * HIDDEN + t * tile + ri,
                                [
                                    lds_ld(outs, si * tile + ri),
                                    lds_ld(outs, si * tile + ri + 1),
                                ],
                                CM_SYS,
                            )
                gpu.barrier()
            if tid < S * tile // 2:
                s = tid // (tile // 2)
                r = (tid % (tile // 2)) * 2
                row = t * tile + r
                if const_expr(callable(residual)):
                    r0, r1 = residual(s, row)
                v0 = lds_ld(outs, s * tile + r)
                v1 = lds_ld(outs, s * tile + r + 1)
                if const_expr(W == 1):  # no TP peers: the sum is the local value
                    parts = [(v0, v1)]
                    got = []
                    if const_expr(not callable(residual)):
                        got = poll([(residual, (s * HIDDEN + row) // 2, 1)])
                else:
                    own = sym + region_base
                    specs = [
                        (own, ((src * S + s) * HIDDEN + row) // 2, 1)
                        for src in range(W)
                    ]
                    if const_expr(not callable(residual)):  # packed bf16 pair
                        specs.append((residual, (s * HIDDEN + row) // 2, 1))
                    got = poll(specs, "one-as")
                    parts = [bf2_f32(v[0]) for v in got[:W]]
                    got = got[W:]
                if const_expr(not callable(residual)):
                    r0, r1 = bf2_f32(got[0][0])
                t0 = fx.Float32(0.0)
                t1 = fx.Float32(0.0)
                for src in range_constexpr(W):
                    t0 = t0 + parts[src][0]
                    t1 = t1 + parts[src][1]
                out_fn(s, row, r0 + t0, r1 + t1)

        def dcp_merge_latent(s, t):
            if const_expr(D == 1):
                return True
            region_base = fx.Int64(SY["dcp"]) + fx.Int64(peer_slot) * fx.Int64(
                SY["_dcp_part_stride"]
            )
            summary_base = dcp_summary_index(rank, s, t, 0, S, N_UV)
            if wave < D:
                for batch in range_constexpr((DCP_SUMMARY_PAIRS + 63) // 64):
                    item = lane + batch * 64
                    if item < DCP_SUMMARY_PAIRS:
                        value = fx.Float32(0.0)
                        if item < KV_LORA // 2:
                            value = lds_ld(xs, item)
                        elif item == KV_LORA // 2:
                            value = lds_ld(misc, N_SPLIT)
                        else:
                            value = lds_ld(misc, N_SPLIT + 1)
                        put(peer_dst + region_base, summary_base + item, value, CM_SYS)
            gpu.barrier()
            owner = dcp_uv_owner(t, L, rank)
            if owner:
                own = sym + region_base
                if wave == 0:
                    src = fx.min(lane, D - 1)
                    src_base = dcp_summary_index(src, s, t, 0, S, N_UV)
                    values = poll(
                        [
                            (own, src_base + KV_LORA // 2, 1),
                            (own, src_base + KV_LORA // 2 + 1, 1),
                        ],
                        "one-as",
                    )
                    valid = lane < D
                    m = valid.select(values[0][0].bitcast(fx.Float32), fx.Float32(NEG))
                    local_sum = valid.select(
                        values[1][0].bitcast(fx.Float32), fx.Float32(0.0)
                    )
                    m_all = wave_max(m)
                    z = local_sum * _exp(m - m_all)
                    den = wave_sum(z)
                    if valid:
                        lds_st(
                            misc,
                            lane,
                            z * (den > 0.0).select(_rcp(den), fx.Float32(0.0)),
                        )
                gpu.barrier()
                if tid < KV_LORA // 2:
                    parts = poll(
                        [
                            (
                                own,
                                dcp_summary_index(src, s, t, tid, S, N_UV),
                                1,
                            )
                            for src in range(D)
                        ],
                        "one-as",
                    )
                    o0, o1 = fx.Float32(0.0), fx.Float32(0.0)
                    for src in range_constexpr(D):
                        a0, a1 = bf2_f32(parts[src][0])
                        weight = lds_ld(misc, src)
                        o0, o1 = o0 + a0 * weight, o1 + a1 * weight
                    lds_st(xs, tid, bf16_pair(o0, o1))
                gpu.barrier()
            return owner

        def start(name):
            return (bid + (G - base[name])) & (G - 1)

        def stamp(name, t, which, lead=0):
            if const_expr(timeline):
                if tid == lead:
                    now = mem_realtime()
                    tl_addr = index_arg(7) if const_expr(with_indexer) else timeline_buf
                    fx.generic_store(
                        fx.inttoptr(
                            fx.PointerType.get(
                                fx.Int64.ir_type, fx.AddressSpace.Global, 8
                            ),
                            tl_addr + fx.Int64((first[name] + t) * TL_COLS + which) * 8,
                        ),
                        now,
                    )

        def n_sel(count=S):
            """This lane's local MFMA B column; inactive columns duplicate the last one."""
            return fx.min(lane % 16, count - 1)

        # ================================================= 1. q_a / kv_a GEMV
        # 1 row group x 96 chunks: 8 waves split K, 12 chunks each (all prefetched)
        r_wqa, r_sqa = _rsrc(w_qkv_a), _rsrc(s_qkv_a)
        QA_NKC = HIDDEN // 64
        QA_UNITS = QA_NKC // (attention_k_chunks_per_unit * WAVES)
        if const_expr(with_indexer):
            r_wik, r_sik = _rsrc(index_arg(0)), _rsrc(index_arg(1))
            r_wiw = _rsrc(index_arg(2))
        for t in range(start("qkv_a"), N_QKV_A, G):
            t = fx.Int32(t)
            stamp("qkv_a", t, 0)
            for sample_base in range_constexpr(0, S, SAMPLE_TILE):
                group_count = min(SAMPLE_TILE, S - sample_base)

                def u_qa(c):
                    kc = (wave * QA_UNITS + c) * attention_k_chunks_per_unit
                    return unit_attention(
                        r_wqa,
                        r_sqa,
                        t,
                        kc,
                        QA_NKC,
                        HIDDEN,
                        128,
                        attention_b_word(n_sel(group_count) * HIDDEN + kc * 64),
                    )

                def ld_h(sks):
                    res = []
                    for s, k in sks:
                        w = fx.Vector(
                            bo.buffer_load(
                                r_h,
                                ((sample_base + s) * HIDDEN + k) // 2,
                                vec_width=2,
                                dtype=T.i32,
                            )
                        )
                        v = w.bitcast(fx.BFloat16).to(fx.Float32)
                        res.append([v[j] for j in range(4)])
                    return res

                if const_expr(S <= SAMPLE_TILE):
                    h_ld = load_x_rmsnorm(ld_h, HIDDEN, g_in, group_count)
                    pre = [u_qa(c) for c in range(QA_UNITS)]
                    if const_expr(attention_ptpc):
                        qa_scales = stage_x_rmsnorm_ptpc(
                            ld_h, HIDDEN, g_in, loaded=h_ld, count=group_count
                        )
                    else:
                        stage_x_rmsnorm(
                            ld_h, HIDDEN, g_in, loaded=h_ld, count=group_count
                        )
                else:
                    if const_expr(attention_ptpc):
                        qa_scales = stage_x_rmsnorm_ptpc(
                            ld_h, HIDDEN, g_in, count=group_count
                        )
                    else:
                        stage_x_rmsnorm(ld_h, HIDDEN, g_in, count=group_count)
                    pre = [u_qa(c) for c in range(QA_UNITS)]
                gpu.barrier()
                stamp("qkv_a", t, 2)
                acc = run_units(u_qa, QA_UNITS, QA_UNITS, pre)
                if const_expr(attention_ptpc):
                    reduce_rows(
                        1,
                        acc,
                        emit_ptpc(
                            QKV_A_TILE,
                            t * QKV_A_TILE,
                            r_sqa,
                            qa_scales,
                        ),
                        group_count,
                    )
                else:
                    reduce_rows(1, acc, emit_out(QKV_A_TILE), group_count)
                stamp("qkv_a", t, 3)
                gpu.barrier()
                if tid < group_count * QKV_A_TILE:
                    s = sample_base + tid // QKV_A_TILE
                    row = t * QKV_A_TILE + tid % QKV_A_TILE
                    v = lds_ld(outs, tid)
                    if row < Q_LORA:
                        put(mb("q_a"), s * Q_LORA + row, v)
                    else:
                        put(mb("kv_a"), s * (KV_LORA + PE_DIM) + row - Q_LORA, v)

                if const_expr(with_indexer):
                    IW_CPW = QA_NKC // WAVES

                    def u_index_w(c):
                        kc = wave * IW_CPW + c
                        return unit_bf16(
                            r_wiw,
                            t,
                            kc,
                            QA_NKC,
                            (n_sel(group_count) * HIDDEN + kc * 64) // 2,
                        )

                    if t < INDEX_DIM // QKV_A_TILE:
                        if const_expr(index_k_bf16):

                            def u_index_k(c):
                                kc = wave * IW_CPW + c
                                return unit_bf16(
                                    r_wik,
                                    t,
                                    kc,
                                    QA_NKC,
                                    (n_sel(group_count) * HIDDEN + kc * 64) // 2,
                                )

                            ik_acc = run_units(u_index_k, IW_CPW, IW_CPW)
                        else:

                            def u_index_k(c):
                                kc = (wave * QA_UNITS + c) * 2
                                return unit_fp8x2(
                                    r_wik,
                                    r_sik,
                                    t,
                                    kc,
                                    QA_NKC,
                                    HIDDEN,
                                    (n_sel(group_count) * HIDDEN + kc * 64) // 2,
                                )

                            ik_acc = run_units(u_index_k, QA_UNITS, QA_UNITS)
                        iw_acc = [fx.Float32(0.0) for _ in range(4)]
                        if t < INDEX_HEADS // QKV_A_TILE:
                            iw_acc = run_units(u_index_w, IW_CPW, IW_CPW)
                        reduce_rows(1, ik_acc, emit_out(INDEX_TILE), group_count)
                        gpu.barrier()
                        if tid < group_count * INDEX_TILE:
                            s = sample_base + tid // INDEX_TILE
                            row = t * INDEX_TILE + tid % INDEX_TILE
                            put(mb("index_k"), s * INDEX_DIM + row, lds_ld(outs, tid))
                        if t < INDEX_HEADS // QKV_A_TILE:
                            reduce_rows(1, iw_acc, emit_out(INDEX_TILE), group_count)
                            gpu.barrier()
                            if tid < group_count * INDEX_TILE:
                                s = sample_base + tid // INDEX_TILE
                                row = t * INDEX_TILE + tid % INDEX_TILE
                                put(
                                    mb("index_w"),
                                    s * INDEX_HEADS + row,
                                    lds_ld(outs, tid),
                                )
            stamp("qkv_a", t, 4)

        def ld_qa(sks):
            v = get2_many(
                [(mb("q_a"), s * Q_LORA + k + j) for s, k in sks for j in (0, 2)]
            )
            return [list(v[2 * i]) + list(v[2 * i + 1]) for i in range(len(sks))]

        # ===================== 2. one q_a RMSNorm CTA per sample, shared downstream
        for s_norm in range(start("q_norm"), S, G):
            s_norm = fx.Int32(s_norm)
            stamp("q_norm", s_norm, 0)
            hint_wait(
                Q_LORA // QKV_A_TILE,
                lambda k: (
                    mb("q_a"),
                    s_norm * Q_LORA + k * QKV_A_TILE + QKV_A_TILE - 1,
                ),
                mark=("q_norm", s_norm),
            )

            def ld_qa_one(sks):
                return ld_qa([(s_norm, k) for _, k in sks])

            stage_x_rmsnorm(ld_qa_one, Q_LORA, g_q, count=1)
            stamp("q_norm", s_norm, 2)
            gpu.barrier()
            k = tid * 4
            w0 = lds_ld(xs, k // 2)
            w1 = lds_ld(xs, k // 2 + 1)
            a0, a1 = bf2_f32(w0.bitcast(fx.Int32))
            a2, a3 = bf2_f32(w1.bitcast(fx.Int32))
            put_bf(mb("q_an"), s_norm * Q_LORA + k, [a0, a1, a2, a3])
            stamp("q_norm", s_norm, 4)

        # ================ 3. KV RMSNorm + k_pe RoPE -> cache (+ this launch's rows)
        for t in range(start("cache"), 1, G):
            stamp("cache", t, 0)
            r_kv = _rsrc(kv_cache)
            r_pe = _rsrc(pe_cache)
            # gamma and the RoPE factors are issued ahead of the wait
            g = ld_bf16(_rsrc(g_kv), tid)
            tpe = tid % (PE_DIM // 2)
            cs = [
                ld_bf16(_rsrc(rope_cos), row_position(s) * (PE_DIM // 2) + tpe)
                for s in range(S)
            ]
            sns = [
                ld_bf16(_rsrc(rope_sin), row_position(s) * (PE_DIM // 2) + tpe)
                for s in range(S)
            ]
            hint_wait(
                (KV_LORA + PE_DIM) // QKV_A_TILE,
                lambda k: (
                    mb("kv_a"),
                    (S - 1) * (KV_LORA + PE_DIM) + k * QKV_A_TILE + QKV_A_TILE - 1,
                ),
                mark=("cache", t),
            )
            # every sample's kv latent and k_pe pair in one poll, one block reduction
            vs = getf_many(
                [(mb("kv_a"), s * (KV_LORA + PE_DIM) + tid) for s in range(S)]
            )
            pes = get2_many(
                [
                    (
                        mb("kv_a"),
                        s * (KV_LORA + PE_DIM) + KV_LORA + (tid % (PE_DIM // 2)) * 2,
                    )
                    for s in range(S)
                ]
            )
            stamp("cache", t, 2)
            ssq = block_sums([v * v for v in vs])
            for s in range_constexpr(S):
                pos = row_position(s)
                slot = row_slot(s)
                active = row_writes_cache(s)
                value = vs[s] * _rsq(ssq[s] * (1.0 / KV_LORA) + EPS) * g
                if const_expr(cache_fp8):
                    value = bf16_round(value)
                    kvn, _ = _fp8_roundtrip(value, value)
                else:
                    kvn = bf16_round(value)
                if const_expr(use_atom_kv_cache):
                    if active:
                        if const_expr(cache_fp8):
                            pair = fx.Int32(
                                rocdl.cvt_pk_fp8_f32(
                                    T.i32, kvn, _xshfl(kvn, 1), fx.Int32(0), False
                                )
                            ) & fx.Int32(0xFFFF)
                            upper_pair = bpermute_i32(
                                fp8_kv_upper_pair_lane(lane) * 4, pair
                            )
                            if lane % 4 == 0:
                                bo.buffer_store(
                                    pair | (upper_pair << 16),
                                    r_kv,
                                    slot * (QK_DIM // 4) + tid // 4,
                                )
                        else:
                            bo.buffer_store(
                                kvn.to(fx.BFloat16), r_kv, slot * QK_DIM + tid
                            )
                else:
                    bo.buffer_store(kvn.to(fx.BFloat16), r_kv, pos * KV_LORA + tid)
                put(mb("kvnew"), s * KV_LORA + tid, kvn)
                if tid < PE_DIM // 2:
                    x0, x1 = pes[s]
                    c, sn = cs[s], sns[s]
                    p0, p1 = x0 * c - x1 * sn, x0 * sn + x1 * c
                    if const_expr(cache_fp8):
                        p0, p1 = bf16_round(p0), bf16_round(p1)
                        p0, p1 = _fp8_roundtrip(p0, p1)
                    else:
                        p0, p1 = bf16_round(p0), bf16_round(p1)
                    if const_expr(use_atom_kv_cache):
                        if active:
                            if const_expr(cache_fp8):
                                pair = fx.Int32(
                                    rocdl.cvt_pk_fp8_f32(
                                        T.i32, p0, p1, fx.Int32(0), False
                                    )
                                ) & fx.Int32(0xFFFF)
                                upper_pair = bpermute_i32(
                                    fp8_pe_upper_pair_lane(lane) * 4, pair
                                )
                                if lane % 2 == 0:
                                    bo.buffer_store(
                                        pair | (upper_pair << 16),
                                        r_pe,
                                        slot * (QK_DIM // 4) + KV_LORA // 4 + tid // 2,
                                    )
                            else:
                                pe_offset = slot * QK_DIM + KV_LORA + tid * 2
                                bo.buffer_store(p0.to(fx.BFloat16), r_pe, pe_offset)
                                bo.buffer_store(p1.to(fx.BFloat16), r_pe, pe_offset + 1)
                    else:
                        pe_offset = pos * PE_DIM + tid * 2
                        bo.buffer_store(p0.to(fx.BFloat16), r_pe, pe_offset)
                        bo.buffer_store(p1.to(fx.BFloat16), r_pe, pe_offset + 1)
                    put2(mb("penew"), s * PE_DIM + tid * 2, p0, p1)

            if const_expr(with_indexer):
                # Index keys use LayerNorm (not RMSNorm), interleaved RoPE on
                # the first 64 dimensions, and BF16 cache storage. Hadamard is
                # omitted because applying the same orthogonal transform to Q
                # and K leaves their dot products unchanged.
                r_gik, r_bik = _rsrc(index_arg(5)), _rsrc(index_arg(6))
                r_index_cache = _rsrc(indices)
                for s in range_constexpr(S):
                    ik = getf(mb("index_k"), s * INDEX_DIM + fx.min(tid, INDEX_DIM - 1))
                    live = tid < INDEX_DIM
                    iv = live.select(ik, fx.Float32(0.0))
                    mean = block_sum(iv) * (1.0 / INDEX_DIM)
                    centered = live.select(ik - mean, fx.Float32(0.0))
                    rstd = _rsq(
                        block_sum(centered * centered) * (1.0 / INDEX_DIM) + 1.0e-6
                    )
                    if tid < INDEX_DIM // 2:
                        i0 = tid * 2
                        k0, k1 = get2(mb("index_k"), s * INDEX_DIM + i0)
                        v0 = (k0 - mean) * rstd * ld_f32(r_gik, i0) + ld_f32(r_bik, i0)
                        v1 = (k1 - mean) * rstd * ld_f32(r_gik, i0 + 1) + ld_f32(
                            r_bik, i0 + 1
                        )
                        if tid < PE_DIM // 2:
                            c, sn = cs[s], sns[s]
                            v0, v1 = v0 * c - v1 * sn, v0 * sn + v1 * c
                        if const_expr(index_paged):
                            # vLLM's ue8m0 quant: scale = 2^ceil(log2(max(amax, 1e-4) / 448))
                            amax = wave_max(fx.max(fmath.absf(v0), fmath.absf(v1)))
                            bits = div_rn(
                                fx.max(amax, fx.Float32(1.0e-4)),
                                fx.Float32(FP8_MAX),
                                fx.Float32(1.0 / FP8_MAX),
                            ).bitcast(fx.Int32)
                            exponent = (bits >> 23) & fx.Int32(0xFF)
                            exponent = exponent + (
                                (bits & fx.Int32(0x7FFFFF)) != 0
                            ).select(fx.Int32(1), fx.Int32(0))
                            k_scale = (exponent << 23).bitcast(fx.Float32)
                            k_inv = ((fx.Int32(254) - exponent) << 23).bitcast(
                                fx.Float32
                            )
                            q0, q1 = v0 * k_inv, v1 * k_inv
                            word = fp8_pack4(q0, q1, _xshfl(q0, 1), _xshfl(q1, 1))
                            slot = row_slot(s)
                            block = slot // index_block_size
                            off = slot % index_block_size
                            if row_writes_cache(s) & (tid % 2 == 0):
                                bo.buffer_store(
                                    word,
                                    r_index_cache,
                                    index_value_word(block, off, i0),
                                )
                            if row_writes_cache(s) & (tid == 0):
                                bo.buffer_store(
                                    k_scale, r_index_cache, index_scale_word(block, off)
                                )
                            d0, d1 = _fp8_roundtrip(q0, q1)
                            put_bf(
                                mb("index_k_new"),
                                s * INDEX_DIM + i0,
                                [d0 * k_scale, d1 * k_scale],
                            )
                        else:
                            bo.buffer_store(
                                fx.Vector.from_elements([v0, v1], fx.Float32).to(
                                    fx.BFloat16
                                ),
                                r_index_cache,
                                (pos0 + s) * INDEX_DIM + i0,
                            )
                            put_bf(mb("index_k_new"), s * INDEX_DIM + i0, [v0, v1])
                    gpu.barrier()
                    if tid == 0:
                        put(mb("index_ready"), s, fx.Int32(1))
            stamp("cache", t, 4)

        # =============================================== 4. normalized q_a -> q_b (+RoPE)
        r_wqb, r_sqb = _rsrc(w_q_b), _rsrc(s_q_b)
        QB_NKC = Q_LORA // 64
        QB_UNITS = QB_NKC // (attention_k_chunks_per_unit * WAVES)
        if const_expr(with_indexer):
            r_wiq, r_siq = _rsrc(index_arg(3)), _rsrc(index_arg(4))

        for t in range(start("q_b"), N_QB, G):
            t = fx.Int32(t)
            stamp("q_b", t, 0)

            def u_qb(c):
                kc = (wave * QB_UNITS + c) * attention_k_chunks_per_unit
                return unit_attention(
                    r_wqb,
                    r_sqb,
                    t,
                    kc,
                    QB_NKC,
                    Q_LORA,
                    128,
                    attention_b_word(n_sel() * Q_LORA + kc * 64),
                )

            pre = [u_qb(c) for c in range(QB_UNITS)]
            hint_wait(
                S,
                lambda s: (mb("q_an"), (s * Q_LORA + Q_LORA - 2) // 2),
                mark=("q_b", t),
            )
            if const_expr(attention_ptpc):
                qb_scales = stage_x_pairs_ptpc("q_an", S, Q_LORA, lambda k: k)
            else:
                stage_x_pairs("q_an", S * Q_LORA, lambda k: k)
            stamp("q_b", t, 2)
            gpu.barrier()
            acc = run_units(u_qb, QB_UNITS, QB_UNITS, pre)
            if const_expr(attention_ptpc):
                reduce_rows(
                    1,
                    acc,
                    emit_ptpc(Q_B_TILE, t * Q_B_TILE, r_sqb, qb_scales),
                )
            else:
                reduce_rows(1, acc, emit_out(Q_B_TILE))
            stamp("q_b", t, 3)
            gpu.barrier()
            head = t // QB_PER_HEAD
            hoff = (t % QB_PER_HEAD) * Q_B_TILE
            if hoff < NOPE_DIM:
                if tid < S * Q_B_TILE // 4:
                    s = tid // (Q_B_TILE // 4)
                    r = (tid % (Q_B_TILE // 4)) * 4
                    put_bf(
                        mb("q_nope"),
                        (s * H + head) * NOPE_DIM + hoff + r,
                        [lds_ld(outs, s * Q_B_TILE + r + j) for j in range(4)],
                    )
            else:
                if tid < S * Q_B_TILE // 2:
                    s = tid // (Q_B_TILE // 2)
                    pr = tid % (Q_B_TILE // 2)
                    i = hoff - NOPE_DIM + pr * 2
                    x0 = lds_ld(outs, s * Q_B_TILE + pr * 2)
                    x1 = lds_ld(outs, s * Q_B_TILE + pr * 2 + 1)
                    c = ld_bf16(
                        _rsrc(rope_cos), row_position(s) * (PE_DIM // 2) + i // 2
                    )
                    sn = ld_bf16(
                        _rsrc(rope_sin), row_position(s) * (PE_DIM // 2) + i // 2
                    )
                    put_bf(
                        mb("q_pe"),
                        (s * H + head) * PE_DIM + i,
                        [x0 * c - x1 * sn, x0 * sn + x1 * c],
                    )
            stamp("q_b", t, 4)

            if const_expr(with_indexer):
                # Reuse this CTA's normalized q_lora tile for one index-query
                # row tile.  Complementary CTAs compute the remaining half below.
                iq_t = t
                stamp("index_q", iq_t, 0)

                def u_index_q(c):
                    kc = (wave * QB_UNITS + c) * 2
                    return unit_fp8x2(
                        r_wiq,
                        r_siq,
                        iq_t,
                        kc,
                        QB_NKC,
                        Q_LORA,
                        (n_sel() * Q_LORA + kc * 64) // 2,
                    )

                iq_acc = run_units(u_index_q, QB_UNITS, QB_UNITS)
                reduce_rows(1, iq_acc, emit_out(INDEX_TILE))
                stamp("index_q", iq_t, 3)
                gpu.barrier()
                if tid < S * INDEX_TILE // 4:
                    s = tid // (INDEX_TILE // 4)
                    r = (tid % (INDEX_TILE // 4)) * 4
                    put_bf(
                        mb("index_q"),
                        s * INDEX_Q_ROWS + iq_t * INDEX_TILE + r,
                        [lds_ld(outs, s * INDEX_TILE + r + j) for j in range(4)],
                    )
                stamp("index_q", iq_t, 4)

        if const_expr(with_indexer):
            # The 128 CTAs without q_b work produce the other 128 index-query
            # tiles concurrently.  They reload q_a, but remove one full GEMV
            # from the q_b CTAs' serialized critical path.
            N_INDEX_Q_EXTRA = INDEX_Q_ROWS // INDEX_TILE - N_QB
            for tt in range(start("index_q"), N_INDEX_Q_EXTRA, G):
                tt = fx.Int32(tt)
                iq_t = N_QB + tt
                stamp("index_q", iq_t, 0)

                def u_index_q_extra(c):
                    kc = (wave * QB_UNITS + c) * 2
                    return unit_fp8x2(
                        r_wiq,
                        r_siq,
                        iq_t,
                        kc,
                        QB_NKC,
                        Q_LORA,
                        (n_sel() * Q_LORA + kc * 64) // 2,
                    )

                pre = [u_index_q_extra(c) for c in range(QB_UNITS)]
                hint_wait(
                    S,
                    lambda s: (mb("q_an"), (s * Q_LORA + Q_LORA - 2) // 2),
                    mark=("index_q", iq_t),
                )
                stage_x_pairs("q_an", S * Q_LORA, lambda k: k)
                stamp("index_q", iq_t, 2)
                gpu.barrier()
                iq_acc = run_units(u_index_q_extra, QB_UNITS, QB_UNITS, pre)
                reduce_rows(1, iq_acc, emit_out(INDEX_TILE))
                stamp("index_q", iq_t, 3)
                gpu.barrier()
                if tid < S * INDEX_TILE // 4:
                    s = tid // (INDEX_TILE // 4)
                    r = (tid % (INDEX_TILE // 4)) * 4
                    put_bf(
                        mb("index_q"),
                        s * INDEX_Q_ROWS + iq_t * INDEX_TILE + r,
                        [lds_ld(outs, s * INDEX_TILE + r + j) for j in range(4)],
                    )
                stamp("index_q", iq_t, 4)

        # ==================================== 4. absorbed query: q_lat = W_UK^T q_nope
        # 8 row groups (128 latent rows of one head) x 3 chunks: one row group per wave
        r_wuk, r_suk = _rsrc(w_uk), _rsrc(s_uk)
        UK_NKC = NOPE_DIM // 64
        for t in range(start("uk"), N_UK, G):
            t = fx.Int32(t)
            stamp("uk", t, 0)
            head = t // UK_PER_HEAD

            def u_uk(c):
                return unit_attention(
                    r_wuk,
                    r_suk,
                    t * WAVES + wave,
                    c,
                    UK_NKC,
                    NOPE_DIM,
                    64,
                    attention_b_word(n_sel() * NOPE_DIM + c * 64),
                    coef=(
                        (
                            lambda: pick(
                                [
                                    uk_scales[
                                        sample * ((NOPE_DIM + 127) // 128) + c // 2
                                    ]
                                    for sample in range(S)
                                ],
                                n_sel(),
                            )
                        )
                        if const_expr(attention_ptpc)
                        else None
                    ),
                )

            pre = [u_uk(c) for c in range(UK_NKC)]
            hint_wait(
                NOPE_DIM // Q_B_TILE,
                lambda k: (
                    mb("q_nope"),
                    ((S - 1) * H + head) * NOPE_DIM + k * Q_B_TILE + Q_B_TILE - 1,
                ),
                mark=("uk", t),
            )
            if const_expr(attention_ptpc):
                uk_scales = stage_x_pairs_ptpc(
                    "q_nope",
                    S,
                    NOPE_DIM,
                    lambda k: ((k // NOPE_DIM) * H + head) * NOPE_DIM + k % NOPE_DIM,
                    group=128,
                )
            else:
                stage_x_pairs(
                    "q_nope",
                    S * NOPE_DIM,
                    lambda k: ((k // NOPE_DIM) * H + head) * NOPE_DIM + k % NOPE_DIM,
                )
            stamp("uk", t, 2)
            gpu.barrier()
            acc = run_units(u_uk, UK_NKC, UK_NKC, pre)
            fx.ptr_store(
                fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4
            )
            stamp("uk", t, 3)
            gpu.barrier()
            if tid < S * UK_TILE // 4:
                k = tid * 4
                s = k // UK_TILE
                r0 = k % UK_TILE
                vals = []
                if const_expr(attention_ptpc):
                    ptpc_scale = ld_f32(r_suk, 0)
                for j in range_constexpr(4):
                    r = r0 + j
                    ww = r // 16
                    value = lds_ld(
                        red, (ww * 64 + s + 16 * ((r % 16) // 4)) * 4 + r % 4
                    )
                    if const_expr(attention_ptpc):
                        value = bf16_round(value * ptpc_scale)
                    vals.append(value)
                put_bf(
                    mb("q_lat"),
                    (s * H + head) * KV_LORA + (t % UK_PER_HEAD) * UK_TILE + r0,
                    vals,
                )
            stamp("uk", t, 4)

        # ====================== 4b. fused sparse index score + exact top-2048
        if const_expr(with_indexer and not index_paged):
            r_index_cache = _rsrc(indices)
            N_INDEX_SPLIT = index_max_seq // INDEX_KEYS_PER_TASK

            def load_index_q8(head, k):
                words = fx.Vector.from_elements(
                    [lds_ld(xs, (head * INDEX_DIM + k) // 2 + j) for j in range(4)],
                    fx.Float32,
                )
                return words.bitcast(fx.BFloat16)

            for tt in range(start("index_score"), S * N_INDEX_SPLIT, G):
                tt = fx.Int32(tt)
                s = tt // N_INDEX_SPLIT
                tile = tt % N_INDEX_SPLIT
                stamp("index_score", tt, 0)
                bound = pos0 + s + 1
                if tile * INDEX_KEYS_PER_TASK < bound:
                    get(mb("index_ready"), s)
                    if tid < INDEX_HEADS:
                        lds_st(keys, tid, get(mb("index_w"), s * INDEX_HEADS + tid))
                    # Every key group reuses the same 32x128 query.  Stage and RoPE
                    # it once per scoring CTA instead of polling and rotating it in
                    # each of the four key-group wave pairs.
                    for b in range_constexpr((INDEX_Q_ROWS // 2) // THREADS):
                        q_pair = tid + b * THREADS
                        q_elem = q_pair * 2
                        kq = q_elem % INDEX_DIM
                        q0, q1 = get_bf2_many(
                            [(mb("index_q"), s * INDEX_Q_ROWS + q_elem)]
                        )[0]
                        if kq < PE_DIM:
                            c = ld_bf16(
                                _rsrc(rope_cos), (pos0 + s) * (PE_DIM // 2) + kq // 2
                            )
                            sn = ld_bf16(
                                _rsrc(rope_sin), (pos0 + s) * (PE_DIM // 2) + kq // 2
                            )
                            q0, q1 = q0 * c - q1 * sn, q0 * sn + q1 * c
                        lds_st(xs, q_pair, bf16_pair(q0, q1))
                    gpu.barrier()
                    key_group = wave // 2
                    head_group = wave % 2
                    key_pos = tile * INDEX_KEYS_PER_TASK + key_group * 16 + lane % 16
                    safe_key = fx.min(key_pos, bound - 1)
                    head = head_group * 16 + lane % 16
                    score_frag = fx.Vector.filled(4, 0.0, fx.Float32)
                    for k32 in range_constexpr(INDEX_DIM // 32):
                        k = k32 * 32 + (lane // 16) * 8
                        qv = load_index_q8(head, k)
                        kv = fx.Vector(
                            bo.buffer_load(
                                r_index_cache,
                                (safe_key * INDEX_DIM + k) // 2,
                                vec_width=4,
                                dtype=T.i32,
                            )
                        ).bitcast(fx.BFloat16)
                        if safe_key >= pos0:
                            sn = safe_key - pos0
                            kv_pairs = get_bf2_many(
                                [
                                    (mb("index_k_new"), sn * INDEX_DIM + k + j * 2)
                                    for j in range(4)
                                ]
                            )
                            kv_values = []
                            for pair in kv_pairs:
                                kv_values += list(pair)
                            kv = fx.Vector.from_elements(kv_values, fx.Float32).to(
                                fx.BFloat16
                            )
                        score_frag = fx.Vector(
                            rocdl.mfma_f32_16x16x32_bf16(
                                T.vec(4, T.f32), [qv, kv, score_frag]
                            )
                        )
                    partial = fx.Float32(0.0)
                    for e in range_constexpr(4):
                        index_head = head_group * 16 + (lane // 16) * 4 + e
                        weight = lds_ld(keys, index_head).bitcast(fx.Float32)
                        partial = (
                            partial + fx.max(score_frag[e], fx.Float32(0.0)) * weight
                        )
                    partial = _xred(partial, 16, lambda a, b: a + b)
                    partial = _xred(partial, 32, lambda a, b: a + b)
                    if lane < 16:
                        lds_st(red, wave * 16 + lane, partial)
                    gpu.barrier()
                    if (wave % 2 == 0) & (lane < 16) & (key_pos < bound):
                        score = lds_ld(red, wave * 16 + lane) + lds_ld(
                            red, (wave + 1) * 16 + lane
                        )
                        put(mb("index_scores"), s * index_max_seq + key_pos, score)
                stamp("index_score", tt, 4)

            for s in range(start("index_select"), S, G):
                s = fx.Int32(s)
                stamp("index_select", s, 0)
                bound = pos0 + s + 1

                def select_digit(shift, prefix, remain):
                    digit = 255 - fx.min(tid, 255)
                    count = (tid < 256).select(lds_ld(keys, digit), fx.Int32(0))
                    inclusive = fx.coop.warp_inclusive_scan(
                        count, fx.ReductionOp.ADD, width=64
                    )
                    wave_total = read_lane_i32(inclusive, 63)
                    if (wave < 4) & (lane == 63):
                        lds_st(keys, 256 + wave, wave_total)
                    gpu.barrier()
                    before_wave = fx.Int32(0)
                    for w in range_constexpr(4):
                        before_wave = before_wave + (wave > w).select(
                            lds_ld(keys, 256 + w), fx.Int32(0)
                        )
                    above = before_wave + inclusive - count
                    hit = (tid < 256) & (above < remain) & (above + count >= remain)
                    gpu.barrier()
                    if hit:
                        lds_st(
                            keys, 256, fx.Int32(prefix | (fx.Uint32(digit) << shift))
                        )
                        lds_st(keys, 257, remain - above)
                    gpu.barrier()
                    return fx.Uint32(lds_ld(keys, 256)), lds_ld(keys, 257)

                # Transform scores to monotonic integer keys and build the high
                # byte histogram in the same pass, avoiding one extra 4-K LDS
                # scan before the remaining radix digits.
                if tid < 256:
                    lds_st(keys, tid, fx.Int32(0))
                gpu.barrier()
                for batch in range_constexpr((index_max_seq + THREADS - 1) // THREADS):
                    i = tid + batch * THREADS
                    if i < bound:
                        bits = getf(mb("index_scores"), s * index_max_seq + i).bitcast(
                            fx.Int32
                        )
                        key = (bits >= 0).select(bits ^ fx.Int32(-(2**31)), ~bits)
                        # The selector CTA no longer needs the large GEMV staging
                        # region, so reuse it for the 4-K radix keys instead of
                        # increasing the monokernel's LDS allocation.
                        lds_st(xs, i, key.bitcast(fx.Float32))
                        digit = fx.Int32((fx.Uint32(key) >> 24) & fx.Uint32(255))
                        fx.atomic_add(keys + digit, fx.Int32(1), syncscope="workgroup")
                gpu.barrier()
                prefix, remain = select_digit(
                    24, fx.Uint32(0), fx.min(fx.Int32(topk), bound)
                )
                prefix_mask = 255 << 24
                for shift in (16, 8, 0):
                    if tid < 256:
                        lds_st(keys, tid, fx.Int32(0))
                    gpu.barrier()
                    for batch in range_constexpr(
                        (index_max_seq + THREADS - 1) // THREADS
                    ):
                        i = tid + batch * THREADS
                        if i < bound:
                            key = fx.Uint32(lds_ld(xs, i).bitcast(fx.Int32))
                            if (key & fx.Uint32(prefix_mask)) == prefix:
                                digit = fx.Int32((key >> shift) & fx.Uint32(255))
                                fx.atomic_add(
                                    keys + digit, fx.Int32(1), syncscope="workgroup"
                                )
                    gpu.barrier()
                    prefix, remain = select_digit(shift, prefix, remain)
                    prefix_mask |= 255 << shift

                threshold = prefix
                r_index_out = _rsrc(mb("indices"))
                items = index_max_seq // THREADS
                item_indices = [tid * items + j for j in range_constexpr(items)]
                item_keys = [
                    fx.Uint32(lds_ld(xs, fx.min(i, bound - 1)).bitcast(fx.Int32))
                    for i in item_indices
                ]
                gt = [
                    (i < bound) & (key > threshold)
                    for i, key in zip(item_indices, item_keys)
                ]
                eq = [
                    (i < bound) & (key == threshold)
                    for i, key in zip(item_indices, item_keys)
                ]

                def scan_flags(flags):
                    """Thread-major exclusive offsets for eight flags without extra LDS.

                    The radix histogram already reserves keys[256:264] for wave
                    totals, so the scan can reuse those words and keep S=8 at the
                    96-KiB static-LDS boundary.
                    """
                    local = fx.Int32(0)
                    local_offsets = []
                    for flag in flags:
                        local_offsets.append(local)
                        local = local + flag.select(fx.Int32(1), fx.Int32(0))
                    inclusive = fx.coop.warp_inclusive_scan(
                        local, fx.ReductionOp.ADD, width=64
                    )
                    wave_total = read_lane_i32(inclusive, 63)
                    if lane == 63:
                        lds_st(keys, 256 + wave, wave_total)
                    gpu.barrier()
                    before_wave = fx.Int32(0)
                    total = fx.Int32(0)
                    for w in range_constexpr(WAVES):
                        wave_count = lds_ld(keys, 256 + w)
                        before_wave = before_wave + (wave > w).select(
                            wave_count, fx.Int32(0)
                        )
                        total = total + wave_count
                    thread_base = before_wave + inclusive - local
                    return [thread_base + off for off in local_offsets], total

                gt_offsets, out_gt = scan_flags(gt)
                gpu.barrier()
                eq_offsets, _ = scan_flags(eq)
                gpu.barrier()
                for j in range_constexpr(items):
                    if gt[j]:
                        bo.buffer_store(
                            fx.Int32(item_indices[j]),
                            r_index_out,
                            s * topk + gt_offsets[j],
                            cache_modifier=CM_DEV,
                        )
                    if eq[j] & (eq_offsets[j] < topk):
                        lds_st(red, eq_offsets[j], item_indices[j].bitcast(fx.Float32))
                gpu.barrier()
                need_eq = fx.min(fx.Int32(topk), bound) - out_gt
                for batch in range_constexpr((topk + THREADS - 1) // THREADS):
                    j = tid + batch * THREADS
                    if j >= bound:
                        bo.buffer_store(
                            fx.Int32(0),
                            r_index_out,
                            s * topk + j,
                            cache_modifier=CM_DEV,
                        )
                    if j < need_eq:
                        i = lds_ld(red, j).bitcast(fx.Int32)
                        bo.buffer_store(
                            i, r_index_out, s * topk + out_gt + j, cache_modifier=CM_DEV
                        )
                # Every lane contributed compact indices.  Make all of those
                # device-memory stores visible before lane 0 publishes the one
                # readiness tag consumed by the attention CTAs.
                fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope="agent")
                gpu.barrier()
                if tid == 0:
                    put(mb("indices_ready"), s, fx.Int32(1))
                stamp("index_select", s, 4)

        if const_expr(index_paged):
            r_index_cache = _rsrc(indices)
            r_out_indices = _rsrc(out_indices)
            r_index_scores = _rsrc(mb("index_scores"))
            r_index_out = _rsrc(mb("indices"))
            hist = fx.recast_iter(fx.Int32, xs)
            N_PAGED_SPLIT = index_max_seq // INDEX_KEYS_PER_TASK

            def load_index_q8(head, k):
                words = fx.Vector.from_elements(
                    [lds_ld(xs, (head * INDEX_DIM + k) // 2 + j) for j in range(4)],
                    fx.Float32,
                )
                return words.bitcast(fx.BFloat16)

            def radix_key(bits):
                return (bits >= 0).select(bits ^ fx.Int32(-(2**31)), ~bits)

            def stored_key(s, i):
                bits = fx.Int32(
                    bo.buffer_load(
                        r_index_scores,
                        (s * index_max_seq + i) * 2,
                        vec_width=1,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                return fx.Uint32(radix_key(bits))

            for s in range_constexpr(S):
                bound = index_bound(s)
                req = index_request(s)
                pos = bound - 1
                n_tiles = (bound + INDEX_KEYS_PER_TASK - 1) // INDEX_KEYS_PER_TASK
                first_cta = (101 + s * (G // S)) % G
                first_tile = (bid + (G - first_cta)) & (G - 1)
                # The row's rotated query and head weights stay in LDS for all
                # of this CTA's tiles.
                if first_tile < n_tiles:
                    get(mb("index_ready"), s)
                    if tid < INDEX_HEADS:
                        lds_st(keys, tid, get(mb("index_w"), s * INDEX_HEADS + tid))
                    for b in range_constexpr((INDEX_Q_ROWS // 2) // THREADS):
                        q_pair = tid + b * THREADS
                        q_elem = q_pair * 2
                        kq = q_elem % INDEX_DIM
                        q0, q1 = get_bf2_many(
                            [(mb("index_q"), s * INDEX_Q_ROWS + q_elem)]
                        )[0]
                        if kq < PE_DIM:
                            c = ld_bf16(_rsrc(rope_cos), pos * (PE_DIM // 2) + kq // 2)
                            sn = ld_bf16(_rsrc(rope_sin), pos * (PE_DIM // 2) + kq // 2)
                            q0, q1 = q0 * c - q1 * sn, q0 * sn + q1 * c
                        lds_st(xs, q_pair, bf16_pair(q0, q1))
                    gpu.barrier()
                for tile in range(first_tile, n_tiles, G):
                    tile = fx.Int32(tile)
                    stamp("index_score", s * N_PAGED_SPLIT + tile, 0)
                    key_group = wave // 2
                    head_group = wave % 2
                    key_pos = tile * INDEX_KEYS_PER_TASK + key_group * 16 + lane % 16
                    safe_key = fx.min(key_pos, bound - 1)
                    head = head_group * 16 + lane % 16
                    block = index_block(req, safe_key)
                    off = safe_key % index_block_size
                    is_new = safe_key == pos
                    k_scale = is_new.select(
                        fx.Float32(1.0),
                        ld_f32(r_index_cache, index_scale_word(block, off)),
                    )
                    score_frag = fx.Vector.filled(4, 0.0, fx.Float32)
                    for k32 in range_constexpr(INDEX_DIM // 32):
                        k = k32 * 32 + (lane // 16) * 8
                        qv = load_index_q8(head, k)
                        kw = fx.Vector(
                            bo.buffer_load(
                                r_index_cache,
                                index_value_word(block, off, k),
                                vec_width=2,
                                dtype=T.i32,
                            )
                        )
                        kv = _fp8_to_bf16x8(kw[0], kw[1])
                        if is_new:
                            kv_pairs = get_bf2_many(
                                [
                                    (mb("index_k_new"), s * INDEX_DIM + k + j * 2)
                                    for j in range(4)
                                ]
                            )
                            kv_values = []
                            for pair in kv_pairs:
                                kv_values += list(pair)
                            kv = fx.Vector.from_elements(kv_values, fx.Float32).to(
                                fx.BFloat16
                            )
                        score_frag = fx.Vector(
                            rocdl.mfma_f32_16x16x32_bf16(
                                T.vec(4, T.f32), [qv, kv, score_frag]
                            )
                        )
                    partial = fx.Float32(0.0)
                    for e in range_constexpr(4):
                        index_head = head_group * 16 + (lane // 16) * 4 + e
                        weight = lds_ld(keys, index_head).bitcast(fx.Float32)
                        partial = (
                            partial + fx.max(score_frag[e], fx.Float32(0.0)) * weight
                        )
                    partial = partial * k_scale
                    partial = _xred(partial, 16, lambda a, b: a + b)
                    partial = _xred(partial, 32, lambda a, b: a + b)
                    if lane < 16:
                        lds_st(red, wave * 16 + lane, partial)
                    gpu.barrier()
                    if (wave % 2 == 0) & (lane < 16) & (key_pos < bound):
                        score = lds_ld(red, wave * 16 + lane) + lds_ld(
                            red, (wave + 1) * 16 + lane
                        )
                        put(mb("index_scores"), s * index_max_seq + key_pos, score)
                    gpu.barrier()
                    stamp("index_score", s * N_PAGED_SPLIT + tile, 4)

            def select_digit(shift, prefix, remain):
                digit = 255 - fx.min(tid, 255)
                count = (tid < 256).select(lds_ld(keys, digit), fx.Int32(0))
                inclusive = fx.coop.warp_inclusive_scan(
                    count, fx.ReductionOp.ADD, width=64
                )
                wave_total = read_lane_i32(inclusive, 63)
                if (wave < 4) & (lane == 63):
                    lds_st(keys, 256 + wave, wave_total)
                gpu.barrier()
                before_wave = fx.Int32(0)
                for w in range_constexpr(4):
                    before_wave = before_wave + (wave > w).select(
                        lds_ld(keys, 256 + w), fx.Int32(0)
                    )
                above = before_wave + inclusive - count
                hit = (tid < 256) & (above < remain) & (above + count >= remain)
                gpu.barrier()
                if hit:
                    lds_st(keys, 256, fx.Int32(prefix | (fx.Uint32(digit) << shift)))
                    lds_st(keys, 257, remain - above)
                gpu.barrier()
                return fx.Uint32(lds_ld(keys, 256)), lds_ld(keys, 257)

            def scan_flags(flags):
                local = fx.Int32(0)
                local_offsets = []
                for flag in flags:
                    local_offsets.append(local)
                    local = local + flag.select(fx.Int32(1), fx.Int32(0))
                inclusive = fx.coop.warp_inclusive_scan(
                    local, fx.ReductionOp.ADD, width=64
                )
                wave_total = read_lane_i32(inclusive, 63)
                if lane == 63:
                    lds_st(keys, 256 + wave, wave_total)
                gpu.barrier()
                before_wave = fx.Int32(0)
                total = fx.Int32(0)
                for w in range_constexpr(WAVES):
                    wave_count = lds_ld(keys, 256 + w)
                    before_wave = before_wave + (wave > w).select(
                        wave_count, fx.Int32(0)
                    )
                    total = total + wave_count
                thread_base = before_wave + inclusive - local
                return [thread_base + off for off in local_offsets], total

            for s in range(start("index_select"), S, G):
                s = fx.Int32(s)
                stamp("index_select", s, 0)
                bound = index_bound(s)
                req = index_request(s)
                count = fx.min(fx.Int32(topk), bound)
                out_begin, _ = row_index_bounds(s)
                n_chunks = (bound + SEL_CHUNK - 1) // SEL_CHUNK

                def chunk_items(chunk):
                    return [
                        chunk * SEL_CHUNK + tid * SEL_ITEMS + j
                        for j in range_constexpr(SEL_ITEMS)
                    ]

                def publish(p, i):
                    slot = index_block(req, i) * index_block_size + i % index_block_size
                    bo.buffer_store(
                        slot, r_index_out, s * topk + p, cache_modifier=CM_DEV
                    )
                    bo.buffer_store(
                        slot, r_out_indices, out_begin + p, cache_modifier=CM_DEV
                    )

                def poll_keys(items):
                    vals = poll(
                        [
                            (
                                mb("index_scores"),
                                s * index_max_seq + fx.min(i, bound - 1),
                                1,
                            )
                            for i in items
                        ]
                    )
                    return [fx.Uint32(radix_key(v[0])) for v in vals]

                def load_keys(items):
                    """Keys of consecutive item pairs, two (value, tag) pairs per load."""
                    out = []
                    for j in range_constexpr(0, len(items), 2):
                        w = fx.Vector(
                            bo.buffer_load(
                                r_index_scores,
                                (
                                    s * index_max_seq
                                    + fx.min(items[j], index_max_seq - 2)
                                )
                                * 2,
                                vec_width=4,
                                dtype=T.i32,
                                cache_modifier=CM_DEV,
                            )
                        )
                        out += [fx.Uint32(radix_key(w[0])), fx.Uint32(radix_key(w[2]))]
                    return out

                def clear_hist():
                    for b in range_constexpr(SEL_BINS // THREADS):
                        lds_st(hist, tid + b * THREADS, fx.Int32(0))
                    gpu.barrier()

                def select_wide(shift, prefix, remain):
                    """Pick the 12-bit digit at ``shift`` holding the remain-th largest key.

                    Returns (prefix with that digit, keys still needed, keys in that bin).
                    """
                    per = SEL_BINS // THREADS
                    cs = [
                        lds_ld(hist, SEL_BINS - 1 - (tid * per + j)) for j in range(per)
                    ]
                    local = cs[0]
                    for c in cs[1:]:
                        local = local + c
                    inclusive = fx.coop.warp_inclusive_scan(
                        local, fx.ReductionOp.ADD, width=64
                    )
                    wave_total = read_lane_i32(inclusive, 63)
                    if lane == 63:
                        lds_st(keys, 256 + wave, wave_total)
                    gpu.barrier()
                    running = inclusive - local
                    for w in range_constexpr(WAVES):
                        running = running + (wave > w).select(
                            lds_ld(keys, 256 + w), fx.Int32(0)
                        )
                    for j in range_constexpr(per):
                        digit = SEL_BINS - 1 - (tid * per + j)
                        hit = (running < remain) & (running + cs[j] >= remain)
                        if hit:
                            lds_st(
                                keys,
                                280,
                                fx.Int32(prefix | (fx.Uint32(digit) << shift)),
                            )
                            lds_st(keys, 281, remain - running)
                            lds_st(keys, 282, cs[j])
                        running = running + cs[j]
                    gpu.barrier()
                    return (
                        fx.Uint32(lds_ld(keys, 280)),
                        lds_ld(keys, 281),
                        lds_ld(keys, 282),
                    )

                def scan_pair(first, second, parity):
                    """Exclusive block offsets of two flag lists in one scan.

                    Wave totals and running bases are double-buffered by ``parity``
                    (the chunk index), so consecutive chunks need one barrier each.
                    Returns (first offsets, second offsets, first total, second total),
                    offsets already include the running bases.
                    """
                    local = fx.Int32(0)
                    local_offsets = []
                    for a, b in zip(first, second):
                        local_offsets.append(local)
                        local = (
                            local
                            + a.select(fx.Int32(1), fx.Int32(0))
                            + b.select(fx.Int32(1 << 16), fx.Int32(0))
                        )
                    inclusive = fx.coop.warp_inclusive_scan(
                        local, fx.ReductionOp.ADD, width=64
                    )
                    slot = 256 + parity * WAVES
                    if lane == 63:
                        lds_st(keys, slot + wave, read_lane_i32(inclusive, 63))
                    gpu.barrier()
                    before = fx.Int32(0)
                    total = fx.Int32(0)
                    for w in range_constexpr(WAVES):
                        wave_count = lds_ld(keys, slot + w)
                        before = before + (wave > w).select(wave_count, fx.Int32(0))
                        total = total + wave_count
                    thread_base = before + inclusive - local
                    base_a = lds_ld(keys, 272 + parity * 2)
                    base_b = lds_ld(keys, 273 + parity * 2)
                    offs = [thread_base + off for off in local_offsets]
                    n_a = total & fx.Int32(0xFFFF)
                    n_b = total >> 16
                    if tid == 0:
                        lds_st(keys, 272 + (1 - parity) * 2, base_a + n_a)
                        lds_st(keys, 273 + (1 - parity) * 2, base_b + n_b)
                    return (
                        [base_a + (o & fx.Int32(0xFFFF)) for o in offs],
                        [base_b + (o >> 16) for o in offs],
                        base_a + n_a,
                        base_b + n_b,
                    )

                def set_bases(a, b):
                    if tid == 0:
                        lds_st(keys, 272, a)
                        lds_st(keys, 273, b)
                    gpu.barrier()

                def gather(chunk, valid, item_keys, values, threshold, out_gt, remain):
                    """Keys above ``threshold`` append to red from the running base; the
                    first ``remain`` ties fill red[out_gt:]."""
                    gt = [v & (k > threshold) for v, k in zip(valid, item_keys)]
                    eq = [v & (k == threshold) for v, k in zip(valid, item_keys)]
                    gt_offsets, eq_offsets, _, _ = scan_pair(gt, eq, chunk & 1)
                    for j in range_constexpr(len(values)):
                        if gt[j]:
                            lds_st(red, gt_offsets[j], values[j].bitcast(fx.Float32))
                        if eq[j] & (eq_offsets[j] < remain):
                            lds_st(
                                red,
                                out_gt + eq_offsets[j],
                                values[j].bitcast(fx.Float32),
                            )

                def publish_all():
                    gpu.barrier()
                    for batch in range_constexpr((topk + THREADS - 1) // THREADS):
                        j = tid + batch * THREADS
                        if j < count:
                            publish(j, lds_ld(red, j).bitcast(fx.Int32))

                def candidates(level, prefix, remain, n_cand):
                    """Keys above the ``level`` bin go to red, the bin's keys to LDS."""
                    set_bases(fx.Int32(0), fx.Int32(0))
                    for chunk in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                        chunk = fx.Int32(chunk)
                        items = chunk_items(chunk)
                        item_keys = load_keys(items)
                        high = [
                            (i < bound) & ((k >> level) > (prefix >> level))
                            for i, k in zip(items, item_keys)
                        ]
                        same = [
                            (i < bound) & ((k >> level) == (prefix >> level))
                            for i, k in zip(items, item_keys)
                        ]
                        win_offsets, cand_offsets, _, _ = scan_pair(
                            high, same, chunk & 1
                        )
                        for j in range_constexpr(SEL_ITEMS):
                            if high[j]:
                                lds_st(
                                    red, win_offsets[j], items[j].bitcast(fx.Float32)
                                )
                            c = SEL_BINS + cand_offsets[j] * 2
                            if same[j]:
                                lds_st(xs, c, item_keys[j].bitcast(fx.Float32))
                                lds_st(xs, c + 1, items[j].bitcast(fx.Float32))
                    gpu.barrier()
                    n_win = count - remain
                    stamp("index_select", s, 2)

                    def cand_pass(shift, mask, prefix, remain):
                        if tid < 256:
                            lds_st(keys, tid, fx.Int32(0))
                        gpu.barrier()
                        for c in range(tid, n_cand, fx.Int32(THREADS)):
                            key = fx.Uint32(
                                lds_ld(xs, SEL_BINS + fx.Int32(c) * 2).bitcast(fx.Int32)
                            )
                            if (key & fx.Uint32(mask)) == prefix:
                                digit = fx.Int32((key >> shift) & fx.Uint32(255))
                                fx.atomic_add(
                                    keys + digit, fx.Int32(1), syncscope="workgroup"
                                )
                        gpu.barrier()
                        return select_digit(shift, prefix, remain)

                    if const_expr(level == 20):
                        prefix, remain = cand_pass(12, 0xFFF00000, prefix, remain)
                        prefix, remain = cand_pass(4, 0xFFFFF000, prefix, remain)
                        prefix, remain = cand_pass(0, 0xFFFFFFF0, prefix, remain)
                    else:
                        prefix, remain = cand_pass(0, 0xFFFFFF00, prefix, remain)
                    set_bases(n_win, fx.Int32(0))
                    stamp("index_select", s, 3)
                    out_gt = count - remain
                    n_cand_chunks = (n_cand + SEL_CHUNK - 1) // SEL_CHUNK
                    for chunk in range(fx.Int32(0), n_cand_chunks, fx.Int32(1)):
                        chunk = fx.Int32(chunk)
                        idx = chunk_items(chunk)
                        safe = [fx.min(c, n_cand - 1) for c in idx]
                        gather(
                            chunk,
                            [c < n_cand for c in idx],
                            [
                                fx.Uint32(
                                    lds_ld(xs, SEL_BINS + c * 2).bitcast(fx.Int32)
                                )
                                for c in safe
                            ],
                            [
                                lds_ld(xs, SEL_BINS + c * 2 + 1).bitcast(fx.Int32)
                                for c in safe
                            ],
                            prefix,
                            out_gt,
                            remain,
                        )
                    stamp("index_select", s, 5)
                    publish_all()

                def full_radix():
                    """Byte-wise radix over every key; used when the threshold bin overflows LDS."""

                    def radix_pass(shift, prefix_mask, prefix, remain):
                        if tid < 256:
                            lds_st(keys, tid, fx.Int32(0))
                        gpu.barrier()
                        for chunk in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                            items = chunk_items(fx.Int32(chunk))
                            item_keys = load_keys(items)
                            for j in range_constexpr(SEL_ITEMS):
                                key = item_keys[j]
                                if (items[j] < bound) & (
                                    (key & fx.Uint32(prefix_mask)) == prefix
                                ):
                                    digit = fx.Int32((key >> shift) & fx.Uint32(255))
                                    fx.atomic_add(
                                        keys + digit, fx.Int32(1), syncscope="workgroup"
                                    )
                        gpu.barrier()
                        return select_digit(shift, prefix, remain)

                    prefix, remain = radix_pass(24, 0, fx.Uint32(0), count)
                    prefix, remain = radix_pass(16, 0xFF000000, prefix, remain)
                    prefix, remain = radix_pass(8, 0xFFFF0000, prefix, remain)
                    prefix, remain = radix_pass(0, 0xFFFFFF00, prefix, remain)
                    set_bases(fx.Int32(0), fx.Int32(0))
                    out_gt = count - remain
                    for chunk in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                        chunk = fx.Int32(chunk)
                        items = chunk_items(chunk)
                        gather(
                            chunk,
                            [i < bound for i in items],
                            load_keys(items),
                            items,
                            prefix,
                            out_gt,
                            remain,
                        )
                    publish_all()

                if bound > 0:
                    clear_hist()
                    for chunk in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                        items = chunk_items(fx.Int32(chunk))
                        item_keys = poll_keys(items)
                        for j in range_constexpr(SEL_ITEMS):
                            if items[j] < bound:
                                fx.atomic_add(
                                    hist + fx.Int32(item_keys[j] >> 20),
                                    fx.Int32(1),
                                    syncscope="workgroup",
                                )
                    gpu.barrier()
                    p12, r12, c12 = select_wide(20, fx.Uint32(0), count)
                    stamp("index_select", s, 1)
                    if c12 <= SEL_CAP:
                        candidates(20, p12, r12, c12)
                    if c12 > SEL_CAP:
                        clear_hist()
                        for chunk in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                            items = chunk_items(fx.Int32(chunk))
                            item_keys = load_keys(items)
                            for j in range_constexpr(SEL_ITEMS):
                                k = item_keys[j]
                                if (items[j] < bound) & ((k >> 20) == (p12 >> 20)):
                                    fx.atomic_add(
                                        hist + fx.Int32((k >> 8) & fx.Uint32(0xFFF)),
                                        fx.Int32(1),
                                        syncscope="workgroup",
                                    )
                        gpu.barrier()
                        p24, r24, c24 = select_wide(8, p12, r12)
                        if c24 <= SEL_CAP:
                            candidates(8, p24, r24, c24)
                        if c24 > SEL_CAP:
                            full_radix()
                fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope="agent")
                gpu.barrier()
                if tid == 0:
                    put(mb("indices_ready"), s, fx.Int32(1))
                stamp("index_select", s, 4)

        # ================================== 5. sparse MLA split: 32 keys x 8 heads
        r_kv = _rsrc(kv_cache)
        r_pe = _rsrc(pe_cache)
        r_idx = _rsrc(indices)
        KPW = SPLIT_KEYS // WAVES

        def split_keys(t, s):
            """(nkeys, sparse) of sample s; wave 0 writes this split's cache rows to LDS."""
            if const_expr(use_atom_kv_cache):
                index_base, index_end = row_index_bounds(s)
                nkeys = index_end - index_base
                sparse = index_end > index_base
            else:
                index_base = s * topk
                kv_len = pos0 + s + 1
                sparse = kv_len > topk
                nkeys = sparse.select(fx.Int32(topk), kv_len)
            if const_expr(with_indexer):
                if sparse:
                    if wave == 0:
                        if lane == 0:
                            get(mb("indices_ready"), s)
                        # Only wave 0 consumes the compact index payload; its
                        # divergent ready poll reconverges before this acquire.
                        fx.memory_fence(
                            ordering=fx.AtomicOrdering.Acquire, syncscope="agent"
                        )
            if wave == 0:
                if lane < SPLIT_KEYS:
                    k_pos = t * SPLIT_KEYS + lane
                    k_cl = (k_pos < nkeys).select(k_pos, 0)
                    if const_expr(with_indexer):
                        idx = k_cl
                        if sparse:
                            idx = fx.Int32(
                                bo.buffer_load(
                                    _rsrc(mb("indices")),
                                    s * topk + k_cl,
                                    vec_width=1,
                                    dtype=T.i32,
                                    cache_modifier=CM_DEV,
                                )
                            )
                        lds_st(attn_keys, lane, idx)
                    elif const_expr(use_atom_kv_cache):
                        idx = fx.Int32(0)
                        if sparse:
                            idx = fx.Int32(
                                bo.buffer_load(
                                    r_idx, index_base + k_cl, vec_width=1, dtype=T.i32
                                )
                            )
                        lds_st(attn_keys, lane, idx)
                    else:
                        idx = fx.Int32(
                            bo.buffer_load(
                                r_idx, index_base + k_cl, vec_width=1, dtype=T.i32
                            )
                        )
                        lds_st(attn_keys, lane, sparse.select(idx, k_cl))
            return nkeys, sparse

        def gather_old_kv():
            """Each wave copies its 8 keys' KV latent (1 KB) + k_pe (128 B) cache rows
            into the LDS tiles (rows of this launch are patched in by patch_new_kv)."""
            krows = [lds_ld(attn_keys, wave * KPW + jj) for jj in range(KPW)]
            for jj in range_constexpr(KPW):
                j = wave * KPW + jj
                if const_expr(cache_fp8):
                    row = krows[jj] * (QK_DIM // 4)
                    raw = fx.Vector(
                        bo.buffer_load(r_kv, row + lane * 2, vec_width=2, dtype=T.i32)
                    )
                    fx.ptr_store(
                        _fp8_to_bf16x8(raw[0], raw[1]).bitcast(fx.Float32),
                        ktile + (j * KS + lane * 4),
                    )
                    if lane < PE_DIM // 8:
                        pe_raw = fx.Vector(
                            bo.buffer_load(
                                r_pe,
                                row + KV_LORA // 4 + lane * 2,
                                vec_width=2,
                                dtype=T.i32,
                            )
                        )
                        fx.ptr_store(
                            _fp8_to_bf16x8(pe_raw[0], pe_raw[1]).bitcast(fx.Float32),
                            petile + (j * PS + lane * 4),
                        )
                else:
                    kv_row_words = (
                        QK_DIM // 2 if const_expr(use_atom_kv_cache) else KV_LORA // 2
                    )
                    kv8 = fx.Vector(
                        bo.buffer_load(
                            r_kv,
                            krows[jj] * kv_row_words + lane * 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    fx.ptr_store(kv8.bitcast(fx.Float32), ktile + (j * KS + lane * 4))
                    if lane < PE_DIM // 2:
                        pe_row = (
                            krows[jj] * (QK_DIM // 2) + KV_LORA // 2 + lane
                            if const_expr(use_atom_kv_cache)
                            else krows[jj] * (PE_DIM // 2) + lane
                        )
                        lds_st(petile, j * PS + lane, ld_f32(r_pe, pe_row))

        def patch_new_kv():
            """Rows appended by this launch come from the cache task's kvnew / penew pairs."""
            if const_expr(use_atom_kv_cache):
                new_active = [row_writes_cache(new_s) for new_s in range(S)]
                new_slots = [row_slot(new_s) for new_s in range(S)]
            for jj in range_constexpr(KPW):
                j = wave * KPW + jj
                kr = lds_ld(attn_keys, j)
                if const_expr(use_atom_kv_cache):
                    sn = fx.Int32(-1)
                    for new_s in range_constexpr(S):
                        match = new_active[new_s] & (kr == new_slots[new_s])
                        sn = match.select(fx.Int32(new_s), sn)
                    is_new = sn >= 0
                else:
                    is_new = kr >= pos0
                    sn = kr - pos0
                if is_new:
                    kvp = get2_many(
                        [
                            (mb("kvnew"), sn * KV_LORA + lane * 8 + m * 2)
                            for m in range(4)
                        ]
                    )
                    w = [bf16_pair(a0, a1) for a0, a1 in kvp]
                    fx.ptr_store(
                        fx.Vector.from_elements(w, fx.Float32),
                        ktile + (j * KS + lane * 4),
                    )
                    if lane < PE_DIM // 2:
                        a0, a1 = get2(mb("penew"), sn * PE_DIM + lane * 2)
                        lds_st(petile, j * PS + lane, bf16_pair(a0, a1))

        N_HEAD_GROUPS = H // WAVES
        for tt in range(start("split"), S * N_HEAD_GROUPS * N_SPLIT, G):
            tt = fx.Int32(tt)
            stamp("split", tt, 0)
            s = tt // (N_HEAD_GROUPS * N_SPLIT)
            head_group = (tt // N_SPLIT) % N_HEAD_GROUPS
            t = tt % N_SPLIT
            h = head_group * WAVES + wave
            nkeys, sparse = split_keys(t, s)
            gpu.barrier()
            gather_old_kv()  # before waiting for q: these rows are from earlier launches
            if const_expr(True):
                N_PE_T = PE_DIM // Q_B_TILE
                hint_wait(
                    N_UK + H * N_PE_T + 1,
                    lambda k: (
                        (k < N_UK).select(
                            fx.Int64(SC["q_lat"]),
                            (k < N_UK + H * N_PE_T).select(
                                fx.Int64(SC["q_pe"]), fx.Int64(SC["penew"])
                            ),
                        )
                        + scratch,
                        (k < N_UK).select(
                            (s * H + k // UK_PER_HEAD) * KV_LORA
                            + (k % UK_PER_HEAD) * UK_TILE
                            + UK_TILE
                            - 1,
                            (k < N_UK + H * N_PE_T).select(
                                (s * H + (k - N_UK) // N_PE_T) * PE_DIM
                                + ((k - N_UK) % N_PE_T) * Q_B_TILE
                                + Q_B_TILE
                                - 1,
                                s * PE_DIM + PE_DIM - 1,
                            ),
                        ),
                    ),
                    mark=("split", tt),
                )
            # q of all heads -> bf16 Q[h][576] (words h * 288 + d / 2): latent 512 then pe 64
            NQ = H * KV_LORA // 4 // THREADS
            tpe = fx.min(tid, H * PE_DIM // 4 - 1)
            qv = poll(
                [
                    (mb("q_lat"), (s * H * KV_LORA + (tid + i * THREADS) * 4) // 2, 2)
                    for i in range(NQ)
                ]
                + [(mb("q_pe"), (s * H * PE_DIM + tpe * 4) // 2, 2)]
            )
            for i in range_constexpr(NQ):
                w4 = tid + i * THREADS
                qw = (w4 // (KV_LORA // 4)) * QS + (w4 % (KV_LORA // 4)) * 2
                lds_st(xs, qw, qv[i][0].bitcast(fx.Float32))
                lds_st(xs, qw + 1, qv[i][1].bitcast(fx.Float32))
            if tid < H * PE_DIM // 4:
                hh = tid // (PE_DIM // 4)
                qw = hh * QS + KV_LORA // 2 + (tid % (PE_DIM // 4)) * 2
                lds_st(xs, qw, qv[NQ][0].bitcast(fx.Float32))
                lds_st(xs, qw + 1, qv[NQ][1].bitcast(fx.Float32))
            patch_new_kv()
            if const_expr(True):
                stamp("split", tt, 2)
            gpu.barrier()
            stamp("split", tt, 5)
            # scores = K Q^T on MFMA.  The 64-key tile maps the eight waves to
            # (four row groups, two K halves); the batch-8 32-key tile uses two
            # row groups and four waves per K half.  All waves subsequently own
            # one attention head for softmax and P@V.
            hn = head_group * WAVES + fx.min(lane % 16, WAVES - 1)
            rgk = wave % (SPLIT_KEYS // 16)
            c = fx.Vector.filled(4, 0.0, fx.Float32)
            for st in range_constexpr(QK_DIM // 32 // 2):
                kst = ((wave // (SPLIT_KEYS // 16)) % 2) * (QK_DIM // 32 // 2) + st
                key = rgk * 16 + lane % 16
                kw = (kst < KV_LORA // 32).select(
                    KT_OFF + key * KS + kst * 16,
                    PT_OFF + key * PS + (kst - KV_LORA // 32) * 16,
                )
                a = fx.ptr_load(xs + (kw + (lane // 16) * 4), result_type=v4f).bitcast(
                    fx.BFloat16
                )
                b = fx.ptr_load(
                    xs + (hn * QS + kst * 16 + (lane // 16) * 4), result_type=v4f
                ).bitcast(fx.BFloat16)
                c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
            if const_expr(SPLIT_KEYS == 64) or wave < 4:
                fx.ptr_store(c, red + (wave * 64 + lane) * 4)
            gpu.barrier()
            stamp("split", tt, 6)
            # split-local softmax: wave h, lane = key j (score = sum of the two K halves)
            kidx = t * SPLIT_KEYS + lane
            valid = (lane < SPLIT_KEYS) & (kidx < nkeys)
            r16 = lane % 16
            cl = split_score_column(wave, lane)
            key_rg = fx.min(lane // 16, SPLIT_KEYS // 16 - 1)
            half_stride = SPLIT_KEYS // 16
            raw = lds_ld(red, (key_rg * 64 + cl) * 4 + r16 % 4) + lds_ld(
                red, ((key_rg + half_stride) * 64 + cl) * 4 + r16 % 4
            )
            sc_v = valid.select(raw * scale, fx.Float32(NEG))
            m = wave_max(sc_v)
            p = valid.select(_exp(sc_v - m), fx.Float32(0.0))
            lsum = wave_sum(p)
            p_n = _xshfl(p, 1)
            if (lane < SPLIT_KEYS) & (lane % 2 == 0):
                lds_st(pl, h * (SPLIT_KEYS // 2) + lane // 2, bf16_pair(p, p_n))
            gpu.barrier()
            stamp("split", tt, 3)
            # O = P V on MFMA: heads M, keys K, latent dims N.  Each V word holds
            # a dim pair (even dim low), so one read feeds two MFMAs (even / odd dims):
            # each wave owns 2 groups of 32 dims.  V is read key-strided from the tile.
            for g in range_constexpr(KV_LORA // 32 // WAVES):
                dw = (
                    wave * (KV_LORA // 32 // WAVES) + g
                ) * 16 + lane % 16  # dim pair word
                c0 = fx.Vector.filled(4, 0.0, fx.Float32)
                c1 = fx.Vector.filled(4, 0.0, fx.Float32)
                for js in range_constexpr(SPLIT_KEYS // 32):
                    a = fx.ptr_load(
                        pl + (hn * (SPLIT_KEYS // 2) + js * 16 + (lane // 16) * 4),
                        result_type=v4f,
                    ).bitcast(fx.BFloat16)
                    ws = [
                        fx.ptr_load(
                            ktile + ((js * 32 + (lane // 16) * 8 + i) * KS + dw)
                        ).bitcast(fx.Int32)
                        for i in range(8)
                    ]
                    w_lo = [
                        (ws[2 * i] & 0xFFFF) | (ws[2 * i + 1] << 16) for i in range(4)
                    ]
                    w_hi = [
                        fx.Int32(fx.Uint32(ws[2 * i]) >> 16) | (ws[2 * i + 1] & -65536)
                        for i in range(4)
                    ]
                    b0 = fx.Vector.from_elements(w_lo, fx.Int32).bitcast(fx.BFloat16)
                    b1 = fx.Vector.from_elements(w_hi, fx.Int32).bitcast(fx.BFloat16)
                    c0 = fx.Vector(
                        rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b0, c0])
                    )
                    c1 = fx.Vector(
                        rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b1, c1])
                    )
                if lane < 32:
                    for e in range_constexpr(4):
                        hh = split_acc_head(head_group, lane // 16, e)
                        put_bf(
                            mb("sp_acc"),
                            ((s * N_SPLIT + t) * H + hh) * KV_LORA + dw * 2,
                            [c0[e], c1[e]],
                        )
            if lane == 0:  # written last: the merge's readiness hint
                put(mb("sp_m"), (s * N_SPLIT + t) * H + h, m)
                put(mb("sp_l"), (s * N_SPLIT + t) * H + h, lsum)
            stamp("split", tt, 4)

        # ========================== 6. split merge + W_UV: o = W_UV (softmax . KV)
        # 4 row groups x 8 chunks: 2 waves per row group, 4 chunks each
        r_wuv, r_suv = _rsrc(w_uv), _rsrc(s_uv)
        UV_NKC = KV_LORA // 64
        UV_R = UV_TILE // 16
        UV_WPR = WAVES // UV_R
        UV_UNITS = UV_NKC // (attention_k_chunks_per_unit * UV_WPR)
        for tt in range(start("uv"), S * N_UV, G):
            tt = fx.Int32(tt)
            stamp("uv", tt, 0)
            s = tt // N_UV  # sample
            t = tt % N_UV  # global 64-row tile
            local_t = dcp_local_uv_tile(t, L)
            head = t // (V_DIM // UV_TILE)

            def u_uv(c):
                kc = ((wave % UV_WPR) * UV_UNITS + c) * attention_k_chunks_per_unit
                return unit_attention(
                    r_wuv,
                    r_suv,
                    local_t * UV_R + wave // UV_WPR,
                    kc,
                    UV_NKC,
                    KV_LORA,
                    128,
                    attention_b_word(kc * 64),
                    scale_rows=uv_scale_rows,
                    coef=(
                        (lambda: lds_ld(misc, N_SPLIT + 2 + kc // 2))
                        if const_expr(attention_ptpc)
                        else None
                    ),
                )

            pre = [u_uv(c) for c in range(UV_UNITS)]
            hint_wait(
                N_SPLIT,
                lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head),
                mark=("uv", tt),
            )
            pre_poll(N_SPLIT, lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head))
            stamp("uv", tt, 5)
            # Four 16-split quarters keep each poll and its live payload bounded.
            # Every thread owns one (quarter, half-of-latent) pair and processes
            # its two latent-pair positions sequentially; red holds quarter sums.
            SPH = N_SPLIT // 4
            dp_lo = tid % (KV_LORA // 4)
            hf = tid // (KV_LORA // 4)
            spi = fx.min(lane, N_SPLIT - 1)
            ml = (s * N_SPLIT + spi) * H + head
            ml_got = poll([(mb("sp_m"), ml, 1), (mb("sp_l"), ml, 1)])
            if wave == 0:  # per-split weights exp(m - M) / L for this head -> misc[sp]
                ok_sp = lane < N_SPLIT
                m_sp = ok_sp.select(ml_got[0][0].bitcast(fx.Float32), fx.Float32(NEG))
                l_sp = ok_sp.select(ml_got[1][0].bitcast(fx.Float32), fx.Float32(0.0))
                m_all = wave_max(m_sp)
                w_sp = _exp(m_sp - m_all)
                den = wave_sum(l_sp * w_sp)
                if ok_sp:
                    lds_st(
                        misc,
                        lane,
                        w_sp * (den > 0.0).select(_rcp(den), fx.Float32(0.0)),
                    )
                if lane == 0:
                    lds_st(misc, N_SPLIT, m_all)
                    lds_st(misc, N_SPLIT + 1, den)
            stamp("uv", tt, 2)
            gpu.barrier()
            for dh in range_constexpr(2):
                dp = dp_lo + dh * (KV_LORA // 4)
                got = poll(
                    [
                        (
                            mb("sp_acc"),
                            ((s * N_SPLIT + hf * SPH + j) * H + head) * (KV_LORA // 2)
                            + dp,
                            1,
                        )
                        for j in range(SPH)
                    ]
                )
                o0 = fx.Float32(0.0)
                o1 = fx.Float32(0.0)
                for j in range_constexpr(SPH):
                    wj = lds_ld(misc, hf * SPH + j)
                    a0, a1 = bf2_f32(got[j][0])
                    o0 = o0 + a0 * wj
                    o1 = o1 + a1 * wj
                lds_st(red, (hf * (KV_LORA // 2) + dp) * 2, o0)
                lds_st(red, (hf * (KV_LORA // 2) + dp) * 2 + 1, o1)
            gpu.barrier()
            uv_valid = tid < KV_LORA // 2
            uv0, uv1 = fx.Float32(0.0), fx.Float32(0.0)
            if uv_valid:
                o0 = fx.Float32(0.0)
                o1 = fx.Float32(0.0)
                for q in range_constexpr(4):
                    o0 = o0 + lds_ld(red, (q * (KV_LORA // 2) + tid) * 2)
                    o1 = o1 + lds_ld(red, (q * (KV_LORA // 2) + tid) * 2 + 1)
                uv0, uv1 = bf16_round(o0), bf16_round(o1)
                if const_expr(not attention_ptpc):
                    lds_st(xs, tid, bf16_pair(uv0, uv1))
            if const_expr(attention_ptpc):
                uv_max = wave_max(
                    uv_valid.select(
                        fx.max(fmath.absf(uv0), fmath.absf(uv1)), fx.Float32(0.0)
                    )
                )
                if (lane == 0) & (wave < KV_LORA // 128):
                    lds_st(
                        misc,
                        N_SPLIT + 2 + wave,
                        (uv_max == 0.0).select(
                            fx.Float32(1.0), uv_max * (1.0 / FP8_MAX)
                        ),
                    )
            gpu.barrier()
            if const_expr(attention_ptpc):
                if uv_valid:
                    uv_scale = lds_ld(misc, N_SPLIT + 2 + tid // 64)
                    uv_rcp = 1.0 / uv_scale
                    st_ptpc(
                        tid * 2,
                        div_rn(uv0, uv_scale, uv_rcp),
                        div_rn(uv1, uv_scale, uv_rcp),
                    )
                gpu.barrier()
            owner = dcp_merge_latent(s, t)
            if owner:
                acc = run_units(u_uv, UV_UNITS, UV_UNITS, pre)
                if const_expr(attention_ptpc):
                    reduce_rows(
                        UV_R,
                        acc,
                        emit_ptpc_scalar(UV_TILE, ld_f32(r_suv, 0)),
                    )
                else:
                    reduce_rows(UV_R, acc, emit_out(UV_TILE))
                stamp("uv", tt, 3)
                gpu.barrier()
                if tid < UV_TILE // 4:
                    r = tid * 4
                    put_bf(
                        mb("o"),
                        s * O_K + local_t * UV_TILE + r,
                        [lds_ld(outs, r + j) for j in range(4)],
                    )
            stamp("uv", tt, 4)

        # ====================== 7. W_o + attention TP peer reduce + residual -> a
        # 2 row groups x 32 chunks: 4 waves per row group, 8 chunks each
        r_wo, r_so = _rsrc(w_o), _rsrc(s_o)
        O_NKC = O_K // 64
        O_R = ROW_TILE // 16
        O_WPR = WAVES // O_R
        O_UNITS = O_NKC // (attention_k_chunks_per_unit * O_WPR)
        for t in range(start("o"), N_ROW_TILES, G):
            t = fx.Int32(t)
            stamp("o", t, 0)

            def u_o(c):
                kc = ((wave % O_WPR) * O_UNITS + c) * attention_k_chunks_per_unit
                return unit_attention(
                    r_wo,
                    r_so,
                    t * O_R + wave // O_WPR,
                    kc,
                    O_NKC,
                    O_K,
                    128,
                    attention_b_word(n_sel() * O_K + kc * 64),
                )

            pre = [u_o(c) for c in range(O_UNITS)]
            hint_wait(
                S * N_UV,
                lambda k: (
                    mb("o"),
                    (k // N_UV) * O_K + (k % N_UV) * UV_TILE + UV_TILE - 1,
                ),
                mark=("o", t),
            )
            if const_expr(attention_ptpc):
                o_scales = stage_x_pairs_ptpc("o", S, O_K, lambda k: k)
            else:
                stage_x_pairs("o", S * O_K, lambda k: k)
            stamp("o", t, 2)
            gpu.barrier()
            acc = run_units(u_o, O_UNITS, O_UNITS, pre)
            if const_expr(attention_ptpc):
                reduce_rows(
                    O_R,
                    acc,
                    emit_ptpc(ROW_TILE, t * ROW_TILE, r_so, o_scales),
                )
            else:
                reduce_rows(O_R, acc, emit_out(ROW_TILE))
            stamp("o", t, 3)
            gpu.barrier()

            def resid_h(s, row):
                w = fx.Vector.from_elements(
                    [
                        fx.Int32(
                            bo.buffer_load(
                                r_h, (s * HIDDEN + row) // 2, vec_width=1, dtype=T.i32
                            )
                        )
                    ],
                    fx.Int32,
                )
                v = w.bitcast(fx.BFloat16).to(fx.Float32)
                return v[0], v[1]

            peer_reduce(
                "attn",
                t,
                resid_h,
                lambda s, row, v0, v1: put_bf(mb("a"), s * HIDDEN + row, [v0, v1]),
            )
            stamp("o", t, 4)

        # ====== 8. post-attn RMSNorm -> router scores + this task's FP8 activation blocks
        # One sample per CTA: 1 row group x 96 chunks (bf16), 8 waves split K.
        # S > 1 gets S times as many independent router CTAs instead of serializing
        # every sample's normalization and output columns inside one CTA.
        r_wr = _rsrc(w_r)
        R_NKC = HIDDEN // 64
        for tt in range(start("router"), S * N_ROUTER, G):
            tt = fx.Int32(tt)
            t = tt if const_expr(S == 1) else tt % N_ROUTER
            router_sample = fx.Int32(0) if const_expr(S == 1) else tt // N_ROUTER
            stamp("router", tt, 0)

            # K-fold: MFMA rows / B columns 0..7 take this wave's first K half, rows /
            # columns 8..15 the second, so every loaded weight row is distinct and the
            # whole K slice is prefetched; logit = C[r][n] + C[8 + r][8 + n]
            r_sub = t * ROUTER_TILE % 16  # this task's rows of the 16-row group
            r_ln = (lane & -16) | (r_sub + lane % ROUTER_TILE)
            R_CPW = R_NKC // WAVES // 2
            r_fold = (lane % 16) // ROUTER_TILE
            r_ns = fx.Int32(0)

            def u_r(c):
                kc = wave * (R_NKC // WAVES) + r_fold * R_CPW + c
                return unit_bf16(
                    r_wr,
                    t * ROUTER_TILE // 16,
                    kc,
                    R_NKC,
                    (r_ns * HIDDEN + kc * 64) // 2,
                    r_ln,
                )

            if const_expr(not dense_experts):
                pre = [u_r(c) for c in range(R_CPW)]
            hint_wait(
                N_ROW_TILES,
                lambda k: (
                    mb("a"),
                    router_sample * HIDDEN + k * ROW_TILE + ROW_TILE - 1,
                ),
                mark=("router", tt),
            )
            # this task's FP8 activation block inputs ride along with the staging loads:
            # wave w quantizes block w * N_ROUTER + t of this CTA's sample.
            r_gp = _rsrc(g_post)
            x_blk = wave * N_ROUTER + t
            x_s = router_sample
            x_ok = (wave < XQ_WAVES) & (x_blk < XQ_BLOCKS)
            xk = fx.min(x_blk, XQ_BLOCKS - 1) * 128 + lane * 2
            xg = (ld_bf16(r_gp, xk), ld_bf16(r_gp, xk + 1))
            xa = []

            def ld_a(sks):
                specs = [
                    (mb("a"), (router_sample * HIDDEN + k) // 2, 2) for s, k in sks
                ]
                specs.append((mb("a"), (x_s * HIDDEN + xk) // 2, 1))
                v = poll(specs, batch=len(specs))
                stamp("router", tt, 5, lead=THREADS - 64)
                xa.append(bf2_f32(v[-1][0]))
                return [list(bf2_f32(w[0])) + list(bf2_f32(w[1])) for w in v[:-1]]

            rstds = stage_x_rmsnorm(ld_a, HIDDEN, g_post, mark=("router", tt), count=1)
            stamp("router", tt, 2)
            # this task's FP8 activation blocks go out ahead of the gate GEMV
            if x_ok:
                x_rstd = rstds[0]
                a0, a1 = xa[0]
                if const_expr(native_fp4_mfma):
                    q0, q1, qs = quant_mxfp8(a0 * x_rstd * xg[0], a1 * x_rstd * xg[1])
                else:
                    q0, q1, qs = quant_scaled(a0 * x_rstd * xg[0], a1 * x_rstd * xg[1])
                w8 = (
                    fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False))
                    & 0xFFFF
                )
                w8n = _xshfl(w8, 1)
                if lane % 2 == 0:  # FP8 bytes k .. k + 3 in one tagged word
                    put(mb("xq"), (x_s * HIDDEN + xk) // 4, w8 | (w8n << 16))
                d0, d1 = _fp8_roundtrip(q0, q1)
                qscale = (
                    (qs << 23).bitcast(fx.Float32)
                    if const_expr(native_fp4_mfma)
                    else qs
                )
                bo.buffer_store(
                    fx.Vector.from_elements([d0 * qscale, d1 * qscale], fx.Float32),
                    _rsrc(mb("xqd")),
                    x_s * HIDDEN + xk,
                )
                if const_expr(native_fp4_mfma):
                    if lane % 16 == 0:
                        put(
                            mb("xqs"),
                            x_s * XQ_GROUPS + x_blk * 4 + lane // 16,
                            qs,
                        )
                else:
                    if lane == 0:
                        put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
            gpu.barrier()
            if const_expr(not dense_experts):
                acc = run_units(u_r, R_CPW, R_CPW, pre)
                fx.ptr_store(
                    fx.Vector.from_elements(acc, fx.Float32),
                    red + (wave * 64 + lane) * 4,
                )
                gpu.barrier()
                stamp("router", tt, 3)
                if tid < ROUTER_TILE:
                    r = tid % ROUTER_TILE
                    n = fx.Int32(0)
                    logit = fx.Float32(0.0)
                    for w in range_constexpr(WAVES):
                        for f in range_constexpr(2):
                            m = f * ROUTER_TILE + r
                            logit = logit + lds_ld(
                                red,
                                (w * 64 + f * ROUTER_TILE + n + 16 * (m // 4)) * 4
                                + m % 4,
                            )
                    put(
                        mb("scores"),
                        router_sample * N_EXPERTS + t * ROUTER_TILE + r,
                        _rcp(1.0 + _exp(-logit)),
                    )
            stamp("router", tt, 4)

        def dn_route(bs):
            """Expert-down routing: one whole wave per sample, in wave-sized batches."""
            for sample_batch in range_constexpr(sample_wave_batches(S)):
                route_sample = wave + sample_batch * WAVES
                if const_expr(dense_experts):
                    # slot j < dense_experts: slice j, weight 1; spare slots weigh 0
                    if (route_sample < S) & (lane < MOE_SLOTS):
                        q = route_sample * MOE_SLOTS + lane
                        used = lane < dense_experts
                        lds_st(keys, q, used.select(lane, fx.Int32(SHARED)))
                        lds_st(dnw, q, used.select(fx.Float32(1.0), fx.Float32(0.0)))
                elif route_sample < S:
                    e, w = route_top8(route_sample, bs=bs)
                    if (
                        lane < MOE_SLOTS
                    ):  # slot 0: the shared expert, then pick lane (slot lane + 1)
                        q = route_sample * MOE_SLOTS + (lane + 1) % MOE_SLOTS
                        lds_st(keys, q, (lane == TOP_K).select(fx.Int32(SHARED), e))
                        lds_st(dnw, q, (lane == TOP_K).select(fx.Float32(1.0), w))

        # ================================ 9. expert up/gate + SiLU
        # 2 row groups (16 gate + 16 up rows) x 96 chunks: 4 waves per group, 24 chunks each
        UG_NKC = HIDDEN // 64
        UG_CPW = UG_NKC // (WAVES // 2)
        UG_W_BYTES = 2 * expert_inter * HIDDEN // (2 if expert_mxfp4 else 1)
        UG_S_BYTES = (
            2 * expert_inter * (HIDDEN // 32)
            if expert_mxfp4
            else 2 * expert_inter // SCALE_BM * (HIDDEN // 128) * 4
        )

        def ug_units(e_sel, c, live=None):
            """Unit maker of up/gate tile c of expert e_sel; ``live`` False -> empty
            buffers (loads return 0 without memory traffic)."""
            if const_expr(live is None):
                r_wug = _rsrc(w_ug + fx.Int64(e_sel) * fx.Int64(UG_W_BYTES))
                r_sug = _rsrc(s_ug + fx.Int64(e_sel) * fx.Int64(UG_S_BYTES))
            else:
                r_wug = bo.create_buffer_resource_from_addr(
                    w_ug + fx.Int64(e_sel) * fx.Int64(UG_W_BYTES),
                    num_records_bytes=live.select(fx.Int32(UG_W_BYTES), fx.Int32(0)),
                )
                r_sug = bo.create_buffer_resource_from_addr(
                    s_ug + fx.Int64(e_sel) * fx.Int64(UG_S_BYTES),
                    num_records_bytes=live.select(fx.Int32(UG_S_BYTES), fx.Int32(0)),
                )
            gate_up = wave // (WAVES // 2)  # waves 0-3: gate rows, 4-7: up rows

            def u_ug(cc):  # cc: 128-k chunk of this wave
                kc = (wave % (WAVES // 2)) * UG_CPW + cc * 2
                return unit_f8f8(
                    r_wug,
                    r_sug,
                    gate_up * (expert_inter // 16) + c,
                    kc,
                    UG_NKC,
                    HIDDEN,
                    kc * 16,
                    lambda: _uniform_f32(lds_ld(misc, 8 + kc // 2)),
                )

            return u_ug

        def ug_finish(u, s_u, slot, c, e_sel, prob, u_ug, pre):
            """MFMA the staged activation (X, scales in misc[8:]) against the tile,
            SiLU(gate) * up -> mid."""
            acc = run_units(u_ug, UG_CPW // 2, UG_CPW // 2, pre)
            reduce_rows(2, acc, emit_out(UG_TILE * 2))
            stamp("ug", u, 3)
            gpu.barrier()
            if tid < UG_TILE // 2:
                r = tid * 2
                g0, g1 = lds_ld(outs, r), lds_ld(outs, r + 1)
                u0, u1 = lds_ld(outs, UG_TILE + r), lds_ld(outs, UG_TILE + r + 1)
                put2(
                    mb("mid"),
                    (s_u * MOE_SLOTS + slot) * expert_inter + c * UG_TILE + r,
                    g0 * _rcp(1.0 + _exp(-g0)) * u0,
                    g1 * _rcp(1.0 + _exp(-g1)) * u1,
                )
            if (c == 0) & (tid == 0):  # routing record (debug / tests)
                put(mb("sel"), s_u * MOE_SLOTS + slot, e_sel)
                put(mb("prob"), s_u * MOE_SLOTS + slot, prob())
            stamp("ug", u, 4)

        def ug_task(u):
            s_u = u // (MOE_SLOTS * I_PER_SLOT)
            return s_u, (u // I_PER_SLOT) % MOE_SLOTS, u % I_PER_SLOT

        if const_expr(S == 1):
            # one task per CTA: task u takes intermediates (u % 32) * 8 of routed slot
            # u // 32 (slot 8 for u < 32, which also take the shared expert's); the 8 gate
            # + 8 up rows are one MFMA row group and all waves split K
            UG8 = 8
            UG8_CPW = UG_NKC // WAVES
            for u in range(start("ug"), G, G):
                u = fx.Int32(u)
                stamp("ug", u, 0)
                s_u, c = fx.Int32(0), u % (expert_inter // UG8)
                has_sh = u < expert_inter // UG8
                slot = has_sh.select(
                    fx.Int32(MOE_SLOTS - 1), u // (expert_inter // UG8)
                )
                # the FP8 activation is computed here from the post-attention state (in
                # parallel with the router): RMSNorm, then per-128 quant with one wave per
                # block -> X[0] (fp8 values in bf16), block scales -> misc[8:]
                NB = XQ_BLOCKS // WAVES
                ks_ = [(wave + j * WAVES) * 128 + lane * 2 for j in range(NB)]
                r_gp = _rsrc(g_post)
                gps = [
                    (ld_bf16(r_gp, k), ld_bf16(r_gp, k + 1)) for k in ks_
                ]  # issued ahead of the wait
                bs = load_bias()
                w_rg = ((lane % 16) // 8) * (
                    expert_inter // 16
                ) + c // 2  # MFMA rows 0-7 gate, 8-15 up
                w_ln = (lane & -16) | ((c % 2) * 8 + lane % 8)
                s_rg = (lane // 32) * (
                    expert_inter // 16
                ) + c // 2  # this lane's output rows

                def u_ug8(
                    cc, e, live=None
                ):  # expert e's weights (loads return 0 unless live)
                    nw = (
                        None
                        if live is None
                        else live.select(fx.Int32(UG_W_BYTES), fx.Int32(0))
                    )
                    ns = (
                        None
                        if live is None
                        else live.select(fx.Int32(UG_S_BYTES), fx.Int32(0))
                    )
                    r_wug = bo.create_buffer_resource_from_addr(
                        w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES), num_records_bytes=nw
                    )
                    r_sug = bo.create_buffer_resource_from_addr(
                        s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES), num_records_bytes=ns
                    )
                    unit = wave * (UG8_CPW // 2) + cc
                    kc = unit * 2
                    if const_expr(expert_mxfp4):
                        if const_expr(native_fp4_mfma):
                            return unit_native_mxfp4(
                                r_wug,
                                r_sug,
                                w_rg,
                                unit,
                                HIDDEN,
                                unit * 32,
                                8 + unit * 4 + lane // 16,
                                ln=w_ln,
                            )
                        return unit_mxfp4(
                            r_wug,
                            r_sug,
                            w_rg,
                            unit,
                            HIDDEN,
                            unit * 32,
                            lambda: _uniform_f32(lds_ld(misc, 8 + unit)),
                            w_ln,
                        )
                    wv = [
                        fx.Vector(
                            bo.buffer_load(
                                r_wug,
                                ((w_rg * UG_NKC + kc + h) * 64 + w_ln) * 4,
                                vec_width=4,
                                dtype=T.i32,
                            )
                        )
                        for h in range(2)
                    ]
                    sc = ld_f32(
                        r_sug, (s_rg * 16 // SCALE_BM) * (HIDDEN // 128) + kc // 2
                    )
                    return (
                        "f8f8",
                        wv,
                        lambda: sc * _uniform_f32(lds_ld(misc, 8 + kc // 2)),
                        kc * 16 + (lane // 16) * 4,
                    )

                # the shared expert's weights do not depend on routing: prefetch them (the
                # later zero-weight MMAs of the other tasks are cheaper than a branch)
                pre = [
                    u_ug8(cc, fx.Int32(SHARED), has_sh) for cc in range(UG8_CPW // 2)
                ]
                hint_wait(
                    N_ROW_TILES,
                    lambda k: (mb("a"), s_u * HIDDEN + k * ROW_TILE + ROW_TILE - 1),
                    mark=("ug", u),
                )
                # the sum of squares takes the router's element partition and order
                # (stage_x_rmsnorm), so rstd -- and every FP8 rounding -- is bit-identical
                NQ4 = HIDDEN // (4 * THREADS)
                got = poll(
                    [
                        (mb("a"), (s_u * HIDDEN + (tid + i * THREADS) * 4) // 2, 2)
                        for i in range(NQ4)
                    ]
                    + [(mb("a"), (s_u * HIDDEN + k) // 2, 1) for k in ks_]
                )
                av = [bf2_f32(w[0]) for w in got[NQ4:]]
                ss = fx.Float32(0.0)
                for w in got[:NQ4]:
                    for a in list(bf2_f32(w[0])) + list(bf2_f32(w[1])):
                        ss = ss + a * a
                rstd = _rsq(block_sum(ss) * (1.0 / HIDDEN) + EPS)
                for j in range_constexpr(NB):
                    if const_expr(native_fp4_mfma):
                        q0, q1, qs = quant_mxfp8(
                            av[j][0] * rstd * gps[j][0],
                            av[j][1] * rstd * gps[j][1],
                        )
                        st_mxfp8(ks_[j], q0, q1)
                        if lane % 16 == 0:
                            lds_st(
                                misc,
                                8 + (wave + j * WAVES) * 4 + lane // 16,
                                qs.bitcast(fx.Float32),
                            )
                    else:
                        q0, q1, qs = quant_scaled(
                            av[j][0] * rstd * gps[j][0],
                            av[j][1] * rstd * gps[j][1],
                        )
                        st_f8(ks_[j], q0, q1)
                        if lane == 0:
                            lds_st(misc, 8 + wave + j * WAVES, qs)
                if wave == 0:
                    e, w = route_top8(s_u, bs=bs)
                    if lane == slot - 1:
                        lds_st(keys, 0, e)
                        lds_st(misc, 0, w)
                stamp("ug", u, 2)
                gpu.barrier()
                e_sel = _uniform(lds_ld(keys, 0))
                post = [u_ug8(cc, e_sel) for cc in range(UG8_CPW // 2)]
                reduce_rows(
                    1, mma_units([fx.Float32(0.0) for _ in range(4)], pre), emit_out(16)
                )
                gpu.barrier()
                reduce_rows(
                    1,
                    mma_units([fx.Float32(0.0) for _ in range(4)], post),
                    lambda rl, n, v: lds_st(outs, 16 + rl, v),
                )
                stamp("ug", u, 3)
                gpu.barrier()
                if (
                    tid < UG8
                ):  # threads 0-3: the shared expert's rows, 4-7: the routed slot's
                    r = (tid % (UG8 // 2)) * 2
                    o = (tid // (UG8 // 2)) * 16
                    g0, g1 = lds_ld(outs, o + r), lds_ld(outs, o + r + 1)
                    u0, u1 = lds_ld(outs, o + UG8 + r), lds_ld(outs, o + UG8 + r + 1)
                    if has_sh | (tid >= UG8 // 2):
                        put2(
                            mb("mid"),
                            (tid < UG8 // 2).select(fx.Int32(0), slot) * expert_inter
                            + c * UG8
                            + r,
                            g0 * _rcp(1.0 + _exp(-g0)) * u0,
                            g1 * _rcp(1.0 + _exp(-g1)) * u1,
                        )
                if (c == 0) & (tid == 0):  # routing record (debug / tests)
                    put(mb("sel"), slot, e_sel)
                    put(mb("prob"), slot, lds_ld(misc, 0))
                    if has_sh:
                        put(mb("sel"), 0, fx.Int32(SHARED))
                        put(mb("prob"), 0, fx.Float32(1.0))
                stamp("ug", u, 4)
        elif const_expr(S > 1):
            # One eight-intermediate tile per CTA and task round. Shared-expert weights
            # feed all sample columns of one MFMA, while routed-expert weights are
            # prefetched one sample ahead.  This avoids the segmented partial
            # tiles and mailbox reduction used by the older S=2/4 schedule.
            UG8 = 8
            UG8_UNITS = (HIDDEN // 128) // WAVES
            XW = HIDDEN // 4
            dn_route(load_bias())
            gpu.barrier()
            stage_xq(list(range(S)))
            gpu.barrier()
            u0 = start("ug")
            for task_round in range_constexpr(ug_task_rounds(expert_inter)):
                u = fx.Int32(u0 + task_round * G)
                if u < expert_inter:
                    c = u % (expert_inter // UG8)
                    has_sh = u < expert_inter // UG8
                    slot = has_sh.select(
                        fx.Int32(MOE_SLOTS - 1), u // (expert_inter // UG8)
                    )
                    w_rg = ((lane % 16) // 8) * (expert_inter // 16) + c // 2
                    w_ln = (lane & -16) | ((c % 2) * 8 + lane % 8)
                    s_rg = (lane // 32) * (expert_inter // 16) + c // 2

                    def ug8_units(e, sample, live=None):
                        if const_expr(live is None):
                            rw = _rsrc(w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES))
                            rs = _rsrc(s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES))
                        else:
                            rw = bo.create_buffer_resource_from_addr(
                                w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES),
                                num_records_bytes=live.select(
                                    fx.Int32(UG_W_BYTES), fx.Int32(0)
                                ),
                            )
                            rs = bo.create_buffer_resource_from_addr(
                                s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES),
                                num_records_bytes=live.select(
                                    fx.Int32(UG_S_BYTES), fx.Int32(0)
                                ),
                            )
                        sn = n_sel() if sample is None else fx.Int32(sample)
                        units = []
                        for cc in range_constexpr(UG8_UNITS):
                            unit = wave * UG8_UNITS + cc
                            kc = unit * 2
                            if const_expr(expert_mxfp4):
                                if const_expr(native_fp4_mfma):
                                    units.append(
                                        unit_native_mxfp4(
                                            rw,
                                            rs,
                                            w_rg,
                                            unit,
                                            HIDDEN,
                                            sn * XW + unit * 32,
                                            8 + sn * XQ_GROUPS + unit * 4 + lane // 16,
                                            ln=w_ln,
                                        )
                                    )
                                else:
                                    units.append(
                                        unit_mxfp4(
                                            rw,
                                            rs,
                                            w_rg,
                                            unit,
                                            HIDDEN,
                                            sn * XW + unit * 32,
                                            lambda unit=unit, sn=sn: lds_ld(
                                                misc, 8 + sn * XQ_BLOCKS + unit
                                            ),
                                            w_ln,
                                        )
                                    )
                                continue
                            wv = [
                                fx.Vector(
                                    bo.buffer_load(
                                        rw,
                                        ((w_rg * UG_NKC + kc + j) * 64 + w_ln) * 4,
                                        vec_width=4,
                                        dtype=T.i32,
                                    )
                                )
                                for j in range(2)
                            ]
                            sc = ld_f32(
                                rs, (s_rg * 16 // SCALE_BM) * (HIDDEN // 128) + kc // 2
                            )
                            units.append(
                                (
                                    "f8f8",
                                    wv,
                                    lambda sc=sc, kc=kc, sn=sn: (
                                        sc * lds_ld(misc, 8 + sn * XQ_BLOCKS + kc // 2)
                                    ),
                                    sn * XW + kc * 16 + (lane // 16) * 4,
                                )
                            )
                        return units

                    def ug8_emit(sample, shared):
                        if tid < (S if shared else 1) * UG8 // 2:
                            n = tid // (UG8 // 2)
                            r = (tid % (UG8 // 2)) * 2
                            g0, g1 = (
                                lds_ld(outs, n * 16 + r),
                                lds_ld(outs, n * 16 + r + 1),
                            )
                            v0 = lds_ld(outs, n * 16 + UG8 + r)
                            v1 = lds_ld(outs, n * 16 + UG8 + r + 1)
                            sn = n if shared else fx.Int32(sample)
                            sl = fx.Int32(0) if shared else slot
                            put2(
                                mb("mid"),
                                (sn * MOE_SLOTS + sl) * expert_inter + c * UG8 + r,
                                g0 * _rcp(1.0 + _exp(-g0)) * v0,
                                g1 * _rcp(1.0 + _exp(-g1)) * v1,
                            )
                        if (c == 0) & (tid < S if shared else tid == 0):
                            sn = tid if shared else fx.Int32(sample)
                            sl = fx.Int32(0) if shared else slot
                            put(
                                mb("sel"),
                                sn * MOE_SLOTS + sl,
                                lds_ld(keys, sn * MOE_SLOTS + sl),
                            )
                            put(
                                mb("prob"),
                                sn * MOE_SLOTS + sl,
                                lds_ld(dnw, sn * MOE_SLOTS + sl),
                            )

                    shared_pre = ug8_units(fx.Int32(SHARED), None, has_sh)
                    cur = ug8_units(_uniform(lds_ld(keys, slot)), 0)
                    if has_sh:
                        reduce_rows(
                            1,
                            mma_units([fx.Float32(0.0) for _ in range(4)], shared_pre),
                            emit_out(16),
                        )
                        gpu.barrier()
                        ug8_emit(0, True)
                    for sample in range_constexpr(S):
                        stamp("ug", sample * G + u, 0)
                        pre = cur
                        if const_expr(sample + 1 < S):
                            cur = ug8_units(
                                _uniform(lds_ld(keys, (sample + 1) * MOE_SLOTS + slot)),
                                sample + 1,
                            )
                        reduce_rows(
                            1,
                            mma_units([fx.Float32(0.0) for _ in range(4)], pre),
                            emit_out(16),
                        )
                        gpu.barrier()
                        ug8_emit(sample, False)
                        stamp("ug", sample * G + u, 4)
                    if const_expr(task_round + 1 < ug_task_rounds(expert_inter)):
                        gpu.barrier()

        elif const_expr(ug_split(S, expert_inter) is not None):
            # S = 2, 4 (the router already quantized every sample's activation): job 0 is
            # this CTA's K-segment of a leftover tile (partial sums -> ugp; the segment-0
            # CTA sums them after its own tiles), jobs 1..NF whole tiles.  Tile x: expert
            # slot x // 16 (0 = the shared expert with MFMA column n = sample n, then
            # sample-major routed slots), intermediates (x % 16) * 16.  Job k + 1 is
            # routed and its weights are in flight while job k computes.
            NF, SEG = ug_split(S, expert_inter)
            KB = XQ_BLOCKS // SEG  # 128-k blocks per segment and row group
            XW = HIDDEN // 4  # LDS words of one sample's FP8 activation
            gu_row = (wave // (WAVES // 2)) * (
                expert_inter // 16
            )  # waves 0-3: gate rows, 4-7: up rows
            wq = wave % (WAVES // 2)
            seg = bid % SEG
            x_seg = NF * G + bid // SEG

            def ug_job(k):
                x = fx.Int32(x_seg if k == 0 else bid + (k - 1) * G)
                es = x // I_PER_SLOT
                c = x % I_PER_SLOT
                shared = es == 0
                s_u = fx.max(es - 1, 0) // TOP_K
                slot = shared.select(fx.Int32(0), (es - 1) % TOP_K + 1)
                e_sel = _uniform(
                    lds_ld(keys, s_u * MOE_SLOTS + slot)
                )  # routed once by dn_route
                prob = lds_ld(dnw, s_u * MOE_SLOTS + slot)
                bsel = shared.select(n_sel(), s_u)  # this lane's activation sample
                if const_expr(k == 0):  # waves wq < KB each own one 128-k block
                    kbs = [seg * KB + fx.min(wq, KB - 1)]
                    nrec = (wq < KB).select(fx.Int32(UG_W_BYTES), fx.Int32(0))
                else:
                    kbs = [wq * (UG_CPW // 2) + cc for cc in range(UG_CPW // 2)]
                    nrec = fx.Int32(UG_W_BYTES)
                r_w = bo.create_buffer_resource_from_addr(
                    w_ug + fx.Int64(e_sel) * fx.Int64(UG_W_BYTES),
                    num_records_bytes=nrec,
                )
                r_s = _rsrc(s_ug + fx.Int64(e_sel) * fx.Int64(UG_S_BYTES))

                def mk(kb):
                    return unit_f8f8(
                        r_w,
                        r_s,
                        gu_row + c,
                        kb * 2,
                        UG_NKC,
                        HIDDEN,
                        bsel * XW + kb * 32,
                        lambda: lds_ld(misc, 8 + bsel * XQ_BLOCKS + kb),
                    )

                return (shared, s_u, slot, c, e_sel, prob, [mk(kb) for kb in kbs])

            def ug_mid(shared, s_u, slot, c, e_sel, prob):
                """Outs (per column: 16 gate then 16 up sums) -> SiLU(gate) * up -> mid of
                sample s_u, or of every sample n (column n) for the shared expert."""
                if tid < S * UG_TILE // 2:
                    n = tid // (UG_TILE // 2)
                    r = (tid % (UG_TILE // 2)) * 2
                    if shared | (n == 0):
                        g0, g1 = (
                            lds_ld(outs, n * 2 * UG_TILE + r),
                            lds_ld(outs, n * 2 * UG_TILE + r + 1),
                        )
                        v0 = lds_ld(outs, n * 2 * UG_TILE + UG_TILE + r)
                        v1 = lds_ld(outs, n * 2 * UG_TILE + UG_TILE + r + 1)
                        put2(
                            mb("mid"),
                            (shared.select(n, s_u) * MOE_SLOTS + slot) * expert_inter
                            + c * UG_TILE
                            + r,
                            g0 * _rcp(1.0 + _exp(-g0)) * v0,
                            g1 * _rcp(1.0 + _exp(-g1)) * v1,
                        )
                if (c == 0) & (tid < S):  # routing record (debug / tests)
                    if shared | (tid == 0):
                        put(
                            mb("sel"), shared.select(tid, s_u) * MOE_SLOTS + slot, e_sel
                        )
                        put(
                            mb("prob"), shared.select(tid, s_u) * MOE_SLOTS + slot, prob
                        )

            stamp("ug", bid, 5)
            dn_route(load_bias())  # every sample's top-8, in wave-sized batches
            gpu.barrier()
            stamp("ug", bid, 6)
            cur = ug_job(0)
            stamp("ug", bid, 0)
            stage_xq(list(range(S)))
            stamp("ug", bid, 2)
            gpu.barrier()
            job0 = cur[:6]
            for k in range_constexpr(NF + 1):
                shared, s_u, slot, c, e_sel, prob, pre = cur
                if const_expr(k > 0):
                    stamp("ug", k * G + bid, 0)
                if const_expr(k < NF):
                    cur = ug_job(k + 1)
                acc = mma_units([fx.Float32(0.0) for _ in range(4)], pre)
                reduce_rows(2, acc, emit_out(UG_TILE * 2))
                stamp("ug", k * G + bid, 3)
                gpu.barrier()
                if const_expr(k == 0):
                    if tid < S * 2 * UG_TILE:
                        put(
                            mb("ugp"),
                            ((x_seg - NF * G) * SEG + seg) * S * 2 * UG_TILE + tid,
                            lds_ld(outs, tid),
                        )
                else:
                    ug_mid(shared, s_u, slot, c, e_sel, prob)
                stamp("ug", k * G + bid, 4)
            if seg == 0:  # sum the leftover tile's K-segments
                gpu.barrier()
                if tid < S * 2 * UG_TILE:
                    parts = getf_many(
                        [
                            (
                                (mb("ugp")),
                                ((x_seg - NF * G) * SEG + j) * S * 2 * UG_TILE + tid,
                            )
                            for j in range(SEG)
                        ]
                    )
                    tot_p = parts[0]
                    for j in range_constexpr(1, SEG):
                        tot_p = tot_p + parts[j]
                    lds_st(outs, tid, tot_p)
                gpu.barrier()
                ug_mid(*job0)
        else:
            # S > 1 (the router already quantized every sample's activation): this CTA's
            # tasks are software pipelined -- task k+1 is routed (by every wave on its
            # own) and its weights are in flight while task k computes
            UG_NT = (N_UG + G - 1) // G
            NSC = N_EXPERTS // 64
            u0 = start("ug")

            def ug_prep(k):
                u = fx.Int32(u0 + k * G)
                live = u < N_UG
                s_u, slot, c = ug_task(fx.min(u, N_UG - 1))
                bs = load_bias()
                raws = getf_many(
                    [
                        (mb("scores"), s_u * N_EXPERTS + lane + i * 64)
                        for i in range(NSC)
                    ]
                )
                e, w = route_top8(s_u, raws, bs)
                i_pk = fx.max(slot - 1, 0)
                e_sel = _uniform(
                    (slot == 0).select(fx.Int32(SHARED), read_lane_i32(e, i_pk))
                )
                prob = (slot == 0).select(
                    fx.Float32(1.0),
                    read_lane_i32(w.bitcast(fx.Int32), i_pk).bitcast(fx.Float32),
                )
                u_ug = ug_units(e_sel, c, live)
                return (
                    u,
                    live,
                    s_u,
                    slot,
                    c,
                    e_sel,
                    prob,
                    u_ug,
                    [u_ug(cc) for cc in range(UG_CPW // 2)],
                )

            cur = ug_prep(0)
            for k in range_constexpr(UG_NT):
                u, live, s_u, slot, c, e_sel, prob, u_ug, pre = cur
                if live:
                    stamp("ug", u, 0)
                    stage_xq([s_u])
                    stamp("ug", u, 2)
                gpu.barrier()
                if const_expr(k + 1 < UG_NT):
                    cur = ug_prep(k + 1)
                if live:
                    ug_finish(u, s_u, slot, c, e_sel, lambda: prob, u_ug, pre)

        # ======== 10. mid FP8 quant + expert down + route weighting + MoE TP reduce
        # 2 row groups x (sample tile * 9 slots * 4) chunks: 4 waves per group.
        # S=8 is evaluated as two four-sample groups so its FP8 mid tile and
        # in-flight weight batches fit comfortably in LDS/VGPRs.
        DN_NKC = expert_inter // 64
        DN_R = (
            (DN_TILE + 15) // 16
        )  # 16-row groups touched by a tile (24-row tiles start at row 0 or 8 of one)
        DN_WPR = WAVES // DN_R
        DN_BATCH = down_prefetch_batch(S, native_fp4_mfma)
        DN_W_BYTES = HIDDEN * expert_inter // (2 if expert_mxfp4 else 1)
        DN_S_BYTES = (
            HIDDEN * (expert_inter // 32)
            if expert_mxfp4
            else HIDDEN // SCALE_BM * (expert_inter // 128) * 4
        )
        for t in range(start("down"), N_DN_TILES, G):
            t = fx.Int32(t)
            stamp("down", t, 0)
            if const_expr(S == 1):  # S > 1 routed before up/gate
                dn_route(load_bias())
            gpu.barrier()
            gu = wave // DN_WPR
            dn_rg = t * DN_TILE // 16
            dn_off = t * DN_TILE % 16
            # this lane's row, as a tile row; rows outside the tile load their lane ^ 8 twin
            # (same cache lines) and are dropped in the output
            dn_lr = gu * 16 + lane % 16 - dn_off
            dn_ln = ((dn_lr >= 0) & (dn_lr < DN_TILE)).select(lane, lane ^ 8)

            DN_NU = S * MOE_SLOTS * DN_NKC // 2
            DN_UPW = (DN_NU + DN_WPR - 1) // DN_WPR
            DN_BLK = S * MOE_SLOTS * expert_inter // 128

            def u_dn(cc):  # cc: 128-k chunk of this wave
                qu = (wave % DN_WPR) * DN_UPW + cc
                live = qu < DN_NU
                unit = fx.min(qu, DN_NU - 1)
                q = unit * 2  # 64-k chunk index over (s, slot, kc)
                s_q = q // (MOE_SLOTS * DN_NKC)
                slot_q = (q // DN_NKC) % MOE_SLOTS
                kc = q % DN_NKC
                e = _uniform(lds_ld(keys, s_q * MOE_SLOTS + slot_q))
                wb = bo.create_buffer_resource_from_addr(
                    w_dn + fx.Int64(e) * fx.Int64(DN_W_BYTES),
                    num_records_bytes=(
                        None
                        if DN_NU % DN_WPR == 0
                        else live.select(fx.Int32(DN_W_BYTES), fx.Int32(0))
                    ),
                )
                sb = bo.create_buffer_resource_from_addr(
                    s_dn + fx.Int64(e) * fx.Int64(DN_S_BYTES),
                    num_records_bytes=(
                        None
                        if DN_NU % DN_WPR == 0
                        else live.select(fx.Int32(DN_S_BYTES), fx.Int32(0))
                    ),
                )

                def coef():  # mid block scale * route weight, only in this sample's column
                    return (lane % 16 == s_q).select(
                        _uniform_f32(lds_ld(misc, q // 2)), fx.Float32(0.0)
                    )

                if const_expr(expert_mxfp4):

                    def bf16_coef():
                        return (lane % 16 == s_q).select(
                            _uniform_f32(lds_ld(dnw, s_q * MOE_SLOTS + slot_q)),
                            fx.Float32(0.0),
                        )

                    if const_expr(native_fp4_mfma):
                        return unit_native_mxfp4(
                            wb,
                            sb,
                            dn_rg + gu,
                            unit % (expert_inter // 128),
                            expert_inter,
                            unit * 32,
                            unit * 4 + lane // 16,
                            bf16_coef,
                            dn_ln,
                        )
                    return unit_mxfp4_bf16(
                        wb,
                        sb,
                        dn_rg + gu,
                        unit % (expert_inter // 128),
                        expert_inter,
                        unit * 64,
                        bf16_coef,
                        dn_ln,
                    )
                return unit_f8f8(
                    wb, sb, dn_rg + gu, kc, DN_NKC, expert_inter, q * 16, coef, dn_ln
                )

            if const_expr(S <= 4):
                pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
            hint_wait(
                N_UG,
                lambda k: (
                    mb("mid"),
                    (
                        k // (MOE_SLOTS * I_PER_SLOT) * MOE_SLOTS
                        + (k // I_PER_SLOT) % MOE_SLOTS
                    )
                    * expert_inter
                    + (k % I_PER_SLOT) * UG_TILE
                    + UG_TILE
                    - 1,
                ),
                mark=("down", t),
            )
            mids = get2_many(
                [
                    (mb("mid"), fx.min(wave + b * WAVES, DN_BLK - 1) * 128 + lane * 2)
                    for b in range((DN_BLK + WAVES - 1) // WAVES)
                ]
            )
            stamp("down", t, 2)
            for b in range_constexpr((DN_BLK + WAVES - 1) // WAVES):
                blk = wave + b * WAVES
                if blk < DN_BLK:
                    if const_expr(expert_mxfp4):
                        if const_expr(native_fp4_mfma):
                            q0, q1, qs = quant_mxfp8(mids[b][0], mids[b][1])
                            st_mxfp8(blk * 128 + lane * 2, q0, q1)
                            if lane % 16 == 0:
                                lds_st(
                                    misc,
                                    blk * 4 + lane // 16,
                                    qs.bitcast(fx.Float32),
                                )
                        else:
                            lds_st(
                                xs,
                                blk * 64 + lane,
                                bf16_pair(mids[b][0], mids[b][1]),
                            )
                    else:
                        q0, q1, qs = quant_scaled(mids[b][0], mids[b][1])
                        st_f8(blk * 128 + lane * 2, q0, q1)
                        if lane == 0:
                            lds_st(
                                misc,
                                blk,
                                qs * lds_ld(dnw, blk // (expert_inter // 128)),
                            )
            gpu.barrier()
            if const_expr(S > 4):
                pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
            acc = run_units(u_dn, DN_UPW, DN_BATCH, pre)

            def emit_dn(rl, n, v):
                if (rl >= dn_off) & (rl < dn_off + DN_TILE):
                    lds_st(outs, n * DN_TILE + rl - dn_off, v)

            reduce_rows(DN_R, acc, emit_dn)
            stamp("down", t, 3)
            gpu.barrier()

            def store_x(s, row, v0, v1):
                bo.buffer_store(
                    fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16),
                    _rsrc(x_out),
                    s * HIDDEN + row,
                )

            def residual_a(s, row):
                return bf2_f32(get(mb("a"), (s * HIDDEN + row) // 2))

            peer_reduce("ffn", t, residual_a, store_x, tile=DN_TILE)
            gpu.barrier()
            stamp("down", t, 4)

    @flyc.jit
    def launch(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
        positions: Int64,
        slot_mapping: Int64,
        sparse_kv_indptr: Int64,
        kv_cache: Int64,
        pe_cache: Int64,
        indices: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_in: Int64,
        g_q: Int64,
        g_kv: Int64,
        g_post: Int64,
        w_qkv_a: Int64,
        s_qkv_a: Int64,
        w_q_b: Int64,
        s_q_b: Int64,
        w_uk: Int64,
        s_uk: Int64,
        w_uv: Int64,
        s_uv: Int64,
        w_o: Int64,
        s_o: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        timeline_buf: Int64,
        step: Int64,
        block_table: Int64,
        req_ids: Int64,
        out_indices: Int64,
        rank: Int32,
        layer: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        glm5_monokernel(
            h_in,
            x_out,
            cur_pos,
            positions,
            slot_mapping,
            sparse_kv_indptr,
            kv_cache,
            pe_cache,
            indices,
            rope_cos,
            rope_sin,
            g_in,
            g_q,
            g_kv,
            g_post,
            w_qkv_a,
            s_qkv_a,
            w_q_b,
            s_q_b,
            w_uk,
            s_uk,
            w_uv,
            s_uv,
            w_o,
            s_o,
            w_r,
            bias,
            w_ug,
            s_ug,
            w_dn,
            s_dn,
            scratch,
            sym,
            peers,
            timeline_buf,
            step,
            block_table,
            req_ids,
            out_indices,
            rank,
            layer,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    return launch
