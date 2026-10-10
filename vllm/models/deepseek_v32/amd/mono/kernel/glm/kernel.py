# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025 FlyDSL Project Contributors
# mypy: ignore-errors
#
# This file contains code copied from FlyDSL (ROCm/FlyDSL PR #1204 at 21a3d1ee), as vendored by
# ROCm/ATOM PR #2435 (head 45e4b55d, atom/model_ops/monokernel/glm/kernel.py). The original source code was
# licensed under the Apache License 2.0 and included the following copyright notice:
# Copyright (c) 2025 FlyDSL Project Contributors
# Modified by the vLLM project contributors (Apache-2.0 sec. 4(b)): import paths rewritten to this package;
#   bounded mailbox polls with per-stage error words, early-out and a cross-rank expiry flag, per-step
#   mailbox epochs (launches_per_step / advance), the in-kernel paged indexer (index-K cache write,
#   scoring, top-k, CSR), row-parallel / batched index-score stages; build-time stage options (hoisted
#   cache-stage loads, 64-key split tasks), a 3-pass 11/11/10-bit radix select and index-K/W
#   projections on idle CTAs; the weight / KV-cache formats, DCP, single-request indexer and stage
#   timeline vLLM never builds removed.

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
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T
from aiter.ops.flydsl.kernels import buffer_ops as bo
from vllm.models.deepseek_v32.amd.mono.envs import fault_kernel
from vllm.models.deepseek_v32.amd.mono.kernel.config import (
    AttentionWeight,
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
)
from vllm.models.deepseek_v32.amd.mono.kernel.glm.layout import (
    BLOCKS,
    INDEX_DIM,
    INDEX_HEADS,
    INDEX_KEYS_PER_TASK,
    INDEX_Q_ROWS,
    INDEX_TILE,
    N_QKV_A,
    POLL_STAGES,
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
    XQ_WAVES,
    dn_tile,
    down_x_words,
    layout,
    sample_wave_batches,
    sparse_keys_per_task,
    split_acc_head,
    split_score_column,
    stage_tasks,
    ug_split,
    ug_task_rounds,
)
from vllm.models.deepseek_v32.amd.mono.kernel.layout import CM_DEV, CM_SYS, LAYER_SLOTS, NEG, POLL_MAX, THREADS
from vllm.models.deepseek_v32.amd.mono.kernel.ops import (
    bpermute_i32,
    div_rn,
    f8_word,
    fp8_pack4,
    mem_realtime,
    read_lane_i32,
    spin_pause,
    wave_umax_dpp,
    write_lane_i32,
    exp as _exp,
    fp8_roundtrip as _fp8_roundtrip,
    fp8_to_bf16x8 as _fp8_to_bf16x8,
    mxfp4_to_bf16x8 as _mxfp4_to_bf16x8,
    rcp as _rcp,
    rsq as _rsq,
    rsrc as _rsrc,
    uniform as _uniform,
    uniform_f32 as _uniform_f32,
    xred as _xred,
    xshfl as _xshfl,
)


def _ue8m0_scale(amax):
    """vLLM ``_fp8_ue8m0_quantize`` scale, exactly: 2^ceil(log2(div_rn(max(amax, 1e-4), 448))) and its reciprocal
    (both powers of two, so ``v * inv`` == the IEEE ``v / scale``). Exponent arithmetic on the bits instead of
    log2 / exp2 (x >= 1e-4 / 448 is a positive normal float). [vLLM uses tl.math.log2: identical unless its
    approximation rounds log2 of a value just above a power of two down to the integer -- not observed.]"""
    x = div_rn(fx.max(fx.Float32(amax), fx.Float32(1.0e-4)), fx.Float32(448.0), fx.Float32(1.0 / 448.0))
    bits = x.bitcast(fx.Int32)
    e = (bits >> 23) & fx.Int32(255)
    e = e + ((bits & fx.Int32(0x7FFFFF)) != 0).select(fx.Int32(1), fx.Int32(0))
    scale = fx.Int32(e << 23).bitcast(fx.Float32)
    inv = fx.Int32((fx.Int32(254) - e) << 23).bitcast(fx.Float32)
    return scale, inv


def build_glm5_monokernel(
    S: int = 1,
    heads: int = 8,
    npes: int = 8,
    topk: int = 2048,
    launches_per_step: int = 1,
    with_indexer: bool = False,
    index_max_seq: int = 4096,
    attention_weight: AttentionWeight | str = AttentionWeight.FP8_BLOCK128,
    inter: int = INTER,
    scale: float = SOFTMAX_SCALE,
    poll_limit: int | None = None,
    poll_early_out: bool = False,
    index_q_fp8: bool = True,
    cache_hoist: bool = False,
    split_keys64: bool = False,
    select_radix11: bool = False,
    index_proj_spread: bool = False,
):
    """Return the ``@flyc.jit`` launcher for one rank's whole layer.

    Defaults reproduce the vendored FlyDSL kernel; vLLM's tuned defaults are in
    ``LiveConfig`` (mono/live.py).

    ``with_indexer``: the fused indexer on vLLM's paged FP8 index cache (FP8 attention only).
    Index-K is ue8m0-FP8 quantized into the uint8 [blocks, 16, 132] SHUFFLE cache at the
    row's slot, scoring reads it through the decode block table, and the exact top-2048 is
    written as the physical-slot CSR at ``sparse_kv_indptr`` offsets.
    ``index_q_fp8``: ue8m0 FP8 index q per head, as vLLM's ``fused_q``.
    ``cache_hoist`` / ``split_keys64`` / ``select_radix11`` / ``index_proj_spread``:
    hoisted cache-stage loads, 64-key split tasks, an 11/11/10-bit radix select, and the
    index-K/W projections on CTAs idle during qkv_a.

    ``poll_limit`` (back-port of FlyDSL #1214) bounds the
    re-polls of every mailbox wait.  An expired wait sets its stage's
    ``poll_err`` scratch word and continues, so a protocol defect finishes the
    launch with wrong values instead of hanging the GPU. After the first expiry the
    rank's ``poll_abort`` word holds ``step + 1`` and every later wait of that step
    returns at its first retry -- only with ``poll_early_out=True`` (it adds one device-scope load
    per retry iteration; the default generates exactly the bounded loop of before).
    """
    assert poll_limit is None or poll_limit > 0
    assert heads % WAVES == 0, "split attention maps one local head to each wave"
    attention_bf16 = AttentionWeight(attention_weight) is AttentionWeight.BF16
    attention_k_chunks_per_unit = 1 if attention_bf16 else 2
    SPLIT_KEYS = 64 if split_keys64 else sparse_keys_per_task(S, heads)
    assert topk % SPLIT_KEYS == 0 and 1 <= S <= 12
    assert 1 <= launches_per_step <= LAYER_SLOTS
    # the index-K cache stage maps one wave per row (two row groups)
    assert not with_indexer or (topk == 2048 and index_max_seq % THREADS == 0 and S <= 2 * WAVES)
    assert not (attention_bf16 and with_indexer), "BF16 attention uses the external indexer"
    # 3 passes, 2048-bin histogram in `red`, item keys kept in registers
    assert not select_radix11 or with_indexer, "select_radix11: the fused select"
    # the index-K / index-W projections as their own 10 tasks on CTAs idle during qkv_a
    assert not index_proj_spread or (with_indexer and S <= 8), "index_proj_spread: the fused indexer"
    H = heads
    # Test only: MONO_FAULT_KERNEL="rank=R,step=N,us=D" stalls rank R for D us at the start of the first layer
    # launch of kernel step N, so the other ranks' bounded waits expire. Unset: no code is generated.
    FAULT = fault_kernel()
    W = npes
    expert_inter = inter
    G = BLOCKS
    SC, SY = layout(
        S, H, W, topk, with_indexer, index_max_seq, inter=expert_inter, split_keys=SPLIT_KEYS if split_keys64 else None
    )
    N_SPLIT = topk // SPLIT_KEYS
    QB_ROWS = H * (NOPE_DIM + PE_DIM)
    N_QB = QB_ROWS // Q_B_TILE
    assert not with_indexer or (INDEX_Q_ROWS // INDEX_TILE == G and N_QB * 2 == G)
    QB_PER_HEAD = (NOPE_DIM + PE_DIM) // Q_B_TILE
    N_UK = H * KV_LORA // UK_TILE
    UK_PER_HEAD = KV_LORA // UK_TILE
    N_UV = H * V_DIM // UV_TILE
    O_K = H * V_DIM
    QK_DIM = KV_LORA + PE_DIM
    # split LDS: bf16 q of all heads, then the KV latent / k_pe tiles (bf16 pairs); row
    # strides are padded by 4 words so the MFMA operand rows spread over the banks
    QS = QK_DIM // 2 + 4
    KS = KV_LORA // 2 + 4
    PS = PE_DIM // 2 + 4
    KT_OFF = H * QS
    PT_OFF = KT_OFF + SPLIT_KEYS * KS
    # the input projection runs four samples at a time (48 KiB activation tile at S=8)
    SAMPLE_TILE = min(S, 4)
    DN_TILE = dn_tile(S)
    N_DN_TILES = HIDDEN // DN_TILE
    RED_WORDS = WAVES * 64 * 4
    LDS_KEYS = max(264 if with_indexer else 0, SPLIT_KEYS, S * MOE_SLOTS)

    # One phase-overlaid LDS arena: the X region (input projection, MoE activations, sparse
    # attention) first, then metadata, reductions and outputs, which are live together in GEMVs.
    SPLIT_X_WORDS = PT_OFF + SPLIT_KEYS * PS
    X_WORDS = max(
        SAMPLE_TILE * HIDDEN // 2, S * HIDDEN // 4, SPLIT_X_WORDS, index_max_seq, down_x_words(S, expert_inter)
    )
    MISC_OFF = X_WORDS
    MISC_WORDS = max(8 + S * XQ_BLOCKS, S * MOE_SLOTS * (expert_inter // 128), N_SPLIT + 2)
    KEYS_OFF = MISC_OFF + MISC_WORDS
    DNW_OFF = KEYS_OFF + LDS_KEYS
    RED_OFF = DNW_OFF + S * MOE_SLOTS
    OUT_OFF = RED_OFF + RED_WORDS
    # UK writes its 128-row result straight from the reduction tile to q_lat;
    # all remaining stages need at most these compact output tiles.
    OUT_WORDS = max(S * ROW_TILE, S * 2 * UG_TILE)
    WORK_WORDS = OUT_OFF + OUT_WORDS
    assert WORK_WORDS <= 32768, "keep static LDS below the MI355X per-workgroup budget"

    base = {}
    # CTA placement: split before uk, so every split tile lands on a CTA freed by
    # qkv_a (uk shares the q_b CTAs it waits on anyway)
    tasks = dict(
        stage_tasks(
            S, H, topk, with_indexer, index_max_seq, inter=expert_inter, split_keys=SPLIT_KEYS if split_keys64 else None
        )
    )
    acc = 0
    for name in ("qkv_a", "q_norm", "cache", "q_b", "split", "uk", "uv", "o", "router", "ug", "down"):
        base[name] = acc % G
        acc += tasks[name]
    if with_indexer:
        # q_b fills half the grid: the second half of index-Q runs on the other CTAs
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
        if const_expr(FAULT is not None):
            if (rank == FAULT[0]) & (step_value == FAULT[1]) & (layer == 0):
                t_fault = mem_realtime()
                while mem_realtime() - t_fault < fx.Int64(FAULT[2] * 100):  # 100 MHz counter
                    spin_pause()
        peer_slot = (step_value * launches_per_step + layer) & 1

        def ld_u64(r, i):
            """Uniform 64-bit pointer i of a table."""
            pv = fx.Vector(bo.buffer_load(r, i * 2, vec_width=2, dtype=T.i32))
            return (fx.Int64(_uniform(pv[1])) << 32) | fx.Int64(fx.Uint32(_uniform(pv[0])))

        # One wave sends to one peer, so retain only that wave's destination.
        peer_dst = ld_u64(_rsrc(peers), fx.min(wave, W - 1))

        # ------------------------------------------------------------ helpers
        def ld_f32(r, i):
            return fx.Float32(bo.buffer_load(r, i, vec_width=1, dtype=T.f32))

        def ld_bf16(r, i):
            return fx.Float32(fx.BFloat16(bo.buffer_load(r, i, vec_width=1, dtype=T.bf16)))

        def row_index_bounds(s):
            begin = fx.Int32(bo.buffer_load(_rsrc(sparse_kv_indptr), s, vec_width=1, dtype=T.i32))
            end = fx.Int32(bo.buffer_load(_rsrc(sparse_kv_indptr), s + 1, vec_width=1, dtype=T.i32))
            return begin, end

        def row_active(s):
            begin, end = row_index_bounds(s)
            return end > begin

        def row_position(s):
            position = fx.Int32(bo.buffer_load(_rsrc(positions), s * 2, vec_width=1, dtype=T.i32))
            present = row_active(s) | (row_slot(s) >= 0)
            return present.select(position, fx.Int32(0))

        def row_slot(s):
            return fx.Int32(bo.buffer_load(_rsrc(slot_mapping), s * 2, vec_width=1, dtype=T.i32))

        def row_writes_cache(s):
            return row_slot(s) >= 0

        def lds_ld(ptr, i):
            return fx.ptr_load(ptr + i)

        def lds_st(ptr, i, v):
            fx.ptr_store(v, ptr + i)

        def bf16_pair(a, b):
            """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
            return fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Float32)[0]

        def bf16_round(a):
            return fx.Float32(fx.Float32(a).to(fx.BFloat16))

        def index_arg(i):
            """Entry i of the fused indexer's parameter table (carried in the timeline_buf argument)."""
            return ld_u64(_rsrc(timeline_buf), i)

        # ---- tagged-pair mailboxes
        # TileRT lineage: payload + launch epoch is the progress protocol for
        # resident CTAs; the helpers below are the FlyDSL/ROCm adaptation.
        stage_now = ["qkv_a"]  # trace-time stage name, names an expired poll

        def stage(name):
            stage_now[0] = name

        def mb(name):
            return scratch + fx.Int64(SC[name])

        def put(base_addr, i, v, cm=CM_DEV):
            """Pair i := (v, tag); ``v`` f32 (or int32 bits)."""
            bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
            bo.buffer_store(fx.Vector.from_elements([bits, tag], fx.Int32), _rsrc(base_addr), i * 2, cache_modifier=cm)

        def put2(base_addr, i, v0, v1, cm=CM_DEV):
            """Pairs i, i+1 (i even) in one 16-byte store."""
            vec = fx.Vector.from_elements(
                [fx.Float32(v0).bitcast(fx.Int32), tag, fx.Float32(v1).bitcast(fx.Int32), tag], fx.Int32
            )
            bo.buffer_store(vec, _rsrc(base_addr), i * 2, cache_modifier=cm)

        def put_bf(base_addr, i, vs, cm=CM_DEV):
            """Elements i .. i + len(vs) (2 or 4, i aligned) as packed bf16 pairs: pair
            i / 2 + j := (bf16(vs[2j]) | bf16(vs[2j + 1]) << 16, tag), one 8 / 16-byte store.
            """
            words = []
            for j in range_constexpr(len(vs) // 2):
                words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), tag]
            bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), _rsrc(base_addr), i, cache_modifier=cm)

        def bf2_f32(w):
            """Packed bf16 pair word -> (f32 low, f32 high)."""
            return (w << 16).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(fx.Float32)

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
                return poll(specs[:batch], scope, batch) + poll(specs[batch:], scope, batch)
            cm = CM_DEV if const_expr(scope == "agent") else CM_SYS

            def load_all():
                words = []
                for b, i, n in specs:
                    w = fx.Vector(
                        bo.buffer_load(_rsrc(b), fx.Int32(i) * 2, vec_width=2 * n, dtype=T.i32, cache_modifier=cm)
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
            if const_expr(poll_limit is None):
                while pending(v):
                    spin_pause()
                    v = load_all()
            else:
                # Early-out: the first expired wait on this rank stores ``step + 1`` into the
                # ``poll_abort`` word; every later wait of the same step on this rank (any CTA,
                # any launch sharing this step value) sees it on its first retry and stops
                # spinning, so a dead or stalled peer costs ~one ``poll_limit`` per step instead
                # of one per wait. Results are garbage either way; ``poll_err`` flags the stage
                # and the host fail-stop raises. The fast path (no retry) adds no load.
                retries = fx.Int32(0)
                if const_expr(poll_early_out):
                    abort_mark = step_value + 1
                    r_abort = _rsrc(mb("poll_abort"))
                    ab = fx.Int32(0)
                    while pending(v) & (retries < poll_limit) & (ab != abort_mark):
                        spin_pause()
                        v = load_all()
                        retries = retries + 1
                        ab = fx.Int32(bo.buffer_load(r_abort, 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
                else:
                    while pending(v) & (retries < poll_limit):
                        spin_pause()
                        v = load_all()
                        retries = retries + 1
                if pending(v):
                    r_err = _rsrc(mb("poll_err"))
                    bo.buffer_store(fx.Int32(1), r_err, POLL_STAGES.index(stage_now[0]))
                    if const_expr(W > 1):
                        # Flag the expiry on rank 0 before posting anything derived from it; the
                        # host-written address keeps the fast path's live registers unchanged.
                        a = fx.Vector(
                            bo.buffer_load(
                                r_err, (SC["poll_xrank_addr"] - SC["poll_err"]) // 4, vec_width=2, dtype=T.i32
                            )
                        )
                        flag = (fx.Int64(a[1]) << 32) | fx.Int64(fx.Uint32(a[0]))
                        fx.generic_store(
                            fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4), flag),
                            fx.Int32(1),
                            memory_order=fx.AtomicOrdering.Monotonic,
                        )
                        fx.memory_fence(ordering=fx.AtomicOrdering.Release)
                    if const_expr(poll_early_out):
                        bo.buffer_store(abort_mark, r_abort, 0, cache_modifier=CM_DEV)
            outs_, e = [], 0
            for _, _, n in specs:
                outs_.append([v[e + 2 * q] for q in range(n)])
                e += 2 * n
            return outs_

        def pre_poll(n, addr_of):
            """Wave 0 spins on one small pair per producer (lane j -> producer j < n <= 64)
            before a large payload poll, so waiting CTAs do not flood memory."""
            if wave == 0:
                b, i = addr_of(fx.min(lane, n - 1))
                poll([(b, i, 1)])
            gpu.barrier()

        def get(base_addr, i):
            return poll([(base_addr, i, 1)])[0][0]

        def getf_many(specs):
            """[(base, i)] single pairs -> list of f32."""
            return [v[0].bitcast(fx.Float32) for v in poll([(b, i, 1) for b, i in specs])]

        def get2_many(specs):
            """[(base, i)] double pairs (i even) -> list of (f32, f32)."""
            return [(v[0].bitcast(fx.Float32), v[1].bitcast(fx.Float32)) for v in poll([(b, i, 2) for b, i in specs])]

        def get2(base_addr, i):
            return get2_many([(base_addr, i)])[0]

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
        def unit_fp8(w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word):
            """Issue one 64-k chunk of row group ``rg`` of a packed FP8 matrix; the
            bf16 activation chunk starts at LDS word ``b_word``."""
            wv = fx.Vector(bo.buffer_load(w_rsrc, ((rg * NKC + kc) * 64 + lane) * 4, vec_width=4, dtype=T.i32))
            s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // BK) + kc * 64 // BK)
            return ("fp8", [wv], s, b_word + (lane // 16) * 4)

        def unit_fp8x2(w_rsrc, s_rsrc, rg, kc, NKC, K, b_word):
            """Issue both 64-k halves of one 128-k FP8 weight-scale block."""
            wv = [
                fx.Vector(bo.buffer_load(w_rsrc, ((rg * NKC + kc + h) * 64 + lane) * 4, vec_width=4, dtype=T.i32))
                for h in range(2)
            ]
            s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // 128) + kc // 2)
            return ("fp8x2", wv, s, b_word + (lane // 16) * 4)

        def unit_mxfp4(w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln=None):
            """Issue one native packed 128-K MXFP4 tile and four E8M0 row scales."""
            ln = lane if ln is None else ln
            row = rg * 16 + ln % 16
            raw = fx.Vector(bo.buffer_load(w_rsrc, ((rg * (K // 128) + kc) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
            packed_scale = fx.Int32(bo.buffer_load(s_rsrc, row * (K // 128) + kc, vec_width=1, dtype=T.i32))
            scales = [
                ((packed_scale.shrui(fx.Int32(sp * 8)) & fx.Int32(0xFF)) << fx.Int32(23)).bitcast(fx.Float32)
                for sp in range_constexpr(4)
            ]
            return ("mxfp4", (raw, scales), coef, b_word + (lane // 16) * 4)

        def unit_mxfp4_bf16(w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln=None):
            """unit_mxfp4 against a BF16 (not FP8) activation."""
            return ("mxfp4_bf16",) + unit_mxfp4(w_rsrc, s_rsrc, rg, kc, K, b_word, coef, ln)[1:]

        def unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln=None):
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(bo.buffer_load(w_rsrc, (((rg * NKC + kc) * 2 + sp) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
                for sp in range(2)
            ]
            return ("bf16", wv, None, b_word + (lane // 16) * 4)

        def unit_attention(w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word):
            if const_expr(attention_bf16):
                return unit_bf16(w_rsrc, rg, kc, NKC, b_word)
            if const_expr(BK == 64):
                return unit_fp8(w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word)
            return unit_fp8x2(w_rsrc, s_rsrc, rg, kc, NKC, K, b_word)

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
                            bv = fx.Vector(fx.ptr_load(xs + (bw + wh * 16), result_type=v4f)).bitcast(fx.Int32)
                            b = _fp8_to_bf16x8(bv[ws * 2], bv[ws * 2 + 1])
                        else:
                            b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                        c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                nsp = {"fp8x2": 4, "fp8": 2, "bf16": 2}.get(fmt, 0)
                for sp in range_constexpr(nsp):
                    if const_expr(fmt in ("fp8", "fp8x2")):
                        wh = sp // 2 if fmt == "fp8x2" else 0
                        ws = sp % 2 if fmt == "fp8x2" else sp
                        a = _fp8_to_bf16x8(wv[wh][ws * 2], wv[wh][ws * 2 + 1])
                    else:
                        a = wv[sp].bitcast(fx.BFloat16)
                    b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                    c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
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
            cur = pre if pre is not None else [make_unit(c) for c in range(0, min(batch, cpw))]
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
            fx.ptr_store(fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4)
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
                        tot = tot + lds_ld(red, (ww * 64 + n + 16 * (r // 4)) * 4 + r % 4)
                    emit(rl, n, tot)

        def emit_out(stride):
            def f(rl, n, v):
                lds_st(outs, n * stride + rl, v)

            return f

        def stage_x_rmsnorm(ld4s, n, gamma, loaded=None, count=S):
            """LDS bf16 X[s][0:n] = bf16(rmsnorm(x_s) * gamma) for every sample s, where
            ld4s([(s, k)]) -> [(x_s[k], .., x_s[k+3])] (one batched load); returns the rstds.
            ``loaded``: the (gamma, x) loads already issued by load_x_rmsnorm."""
            per = n // (4 * THREADS)
            ks = [(tid + i * THREADS) * 4 for i in range(per)]
            gs, vals = loaded if loaded is not None else load_x_rmsnorm(ld4s, n, gamma, count)
            sss = []
            for s in range_constexpr(count):
                ss = fx.Float32(0.0)
                for i in range_constexpr(per):
                    for a in vals[s * per + i]:
                        ss = ss + a * a
                sss.append(ss)
            rstds = [_rsq(tot * (1.0 / n) + EPS) for tot in block_sums(sss)]
            for s in range_constexpr(count):
                for i in range_constexpr(per):
                    a = vals[s * per + i]
                    for j in range_constexpr(2):
                        lds_st(
                            xs,
                            (s * n + ks[i]) // 2 + j,
                            bf16_pair(a[2 * j] * rstds[s] * gs[i][2 * j], a[2 * j + 1] * rstds[s] * gs[i][2 * j + 1]),
                        )
            return rstds

        def load_x_rmsnorm(ld4s, n, gamma, count=S):
            """The gamma loads (issued ahead of the wait), then ld4s -> (gammas, x values)."""
            rg_ = _rsrc(gamma)
            ks = [(tid + i * THREADS) * 4 for i in range(n // (4 * THREADS))]
            gs = []
            for k in ks:
                g = fx.Vector(bo.buffer_load(rg_, k // 2, vec_width=2, dtype=T.i32)).bitcast(fx.BFloat16).to(fx.Float32)
                gs.append([g[j] for j in range(4)])
            return gs, ld4s([(s, k) for s in range(count) for k in ks])

        def stage_x_pairs(name, n_total, src_of):
            """LDS bf16 X[k] = packed bf16 mailbox ``name`` element src_of(k) for k < n_total
            (src_of contiguous over aligned groups of 4): one 16-byte poll per 4 elements.
            """
            nq = n_total // 4
            full = nq // THREADS
            vals = poll([(mb(name), src_of((tid + i * THREADS) * 4) // 2, 2) for i in range(full)])
            for i in range_constexpr(full):
                for j in range_constexpr(2):
                    lds_st(xs, (tid + i * THREADS) * 2 + j, vals[i][j].bitcast(fx.Float32))
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
            # hardware rcp, no IEEE divide
            inv = nz.select(_rcp(amax) * FP8_MAX, fx.Float32(1.0))
            q0 = fx.min(fx.max(a0 * inv, -FP8_MAX), FP8_MAX)
            q1 = fx.min(fx.max(a1 * inv, -FP8_MAX), FP8_MAX)
            return q0, q1, qs

        def stage_xq(samples):
            """Poll the router's packed FP8 activation + block scales of ``samples``
            (sample list, or one runtime sample) into LDS words s * HIDDEN / 4 (``f8_word``
            order; slot 0 for a single runtime sample) and misc[8 + s * XQ_BLOCKS:]."""
            nxw = HIDDEN // 4 // THREADS
            got = poll(
                [(mb("xq"), sx * (HIDDEN // 4) + tid + i * THREADS, 1) for sx in samples for i in range(nxw)]
                + [(mb("xqs"), sx * XQ_BLOCKS + fx.min(tid, XQ_BLOCKS - 1), 1) for sx in samples]
            )
            for j in range_constexpr(len(samples)):
                for i in range_constexpr(nxw):
                    wd = f8_word((tid + i * THREADS) * 4)
                    lds_st(xs, j * (HIDDEN // 4) + wd, got[j * nxw + i][0].bitcast(fx.Float32))
                if tid < XQ_BLOCKS:
                    lds_st(misc, 8 + j * XQ_BLOCKS + tid, got[len(samples) * nxw + j][0].bitcast(fx.Float32))

        def st_f8(k, q0, q1):
            """LDS FP8 activation bytes k, k + 1 (k even, held by this lane; lane ^ 1 holds
            k ^ 2) in ``f8_word`` order.  Call from the whole wave."""
            w = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
            nb = _xshfl(w, 1)
            if lane % 2 == 0:
                lds_st(xs, f8_word(k), (w | (nb << 16)).bitcast(fx.Float32))

        def load_bias():
            """This lane's 4 expert biases (issue before the scores wait)."""
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
                raws = getf_many([(mb("scores"), s * N_EXPERTS + lane + i * 64) for i in range(N_EXPERTS // 64)])
                stage("ug")
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
            region_base = fx.Int64(SY[region]) + fx.Int64(peer_slot) * fx.Int64(SY["_part_stride"])
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
                                [lds_ld(outs, si * tile + ri), lds_ld(outs, si * tile + ri + 1)],
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
                    specs = [(own, ((src * S + s) * HIDDEN + row) // 2, 1) for src in range(W)]
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

        def start(name):
            return (bid + (G - base[name])) & (G - 1)

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

        def stage_h(sample_base, group_count, prefetch_units):
            """The qkv_a stages' activation staging (RMSNorm of h rows sample_base ..) into xs, with ``prefetch_units``
            issued ahead of the norm when S fits one sample tile (as qkv_a does)."""

            def ld_h(sks):
                res = []
                for s_, k in sks:
                    w = fx.Vector(bo.buffer_load(r_h, ((sample_base + s_) * HIDDEN + k) // 2, vec_width=2, dtype=T.i32))
                    v = w.bitcast(fx.BFloat16).to(fx.Float32)
                    res.append([v[j] for j in range(4)])
                return res

            if const_expr(S <= SAMPLE_TILE):
                h_ld = load_x_rmsnorm(ld_h, HIDDEN, g_in, group_count)
                pre = prefetch_units()
                stage_x_rmsnorm(ld_h, HIDDEN, g_in, loaded=h_ld, count=group_count)
            else:
                stage_x_rmsnorm(ld_h, HIDDEN, g_in, count=group_count)
                pre = prefetch_units()
            gpu.barrier()
            return pre

        if const_expr(with_indexer and index_proj_spread):
            # CTAs (165 + S) .. + 9 run nothing before their q_b tiles (which wait for q_norm): index-K
            # tiles 0..7 (FP8) and index-W tiles 0..1 (BF16) go there, in parallel with qkv_a
            IDXP0 = (N_QKV_A + S + 1) % G
            jp = (bid + (G - IDXP0)) & (G - 1)
            if jp < INDEX_DIM // QKV_A_TILE + INDEX_HEADS // QKV_A_TILE:
                IW_CPW_ = QA_NKC // WAVES
                for sample_base in range_constexpr(0, S, SAMPLE_TILE):
                    group_count = min(SAMPLE_TILE, S - sample_base)
                    if jp < INDEX_DIM // QKV_A_TILE:

                        def u_ik(c, jp=jp, group_count=group_count):
                            kc = (wave * QA_UNITS + c) * 2
                            return unit_fp8x2(
                                r_wik, r_sik, jp, kc, QA_NKC, HIDDEN, (n_sel(group_count) * HIDDEN + kc * 64) // 2
                            )

                        pre = stage_h(sample_base, group_count, lambda: [u_ik(c) for c in range(QA_UNITS)])
                        acc = run_units(u_ik, QA_UNITS, QA_UNITS, pre)
                        reduce_rows(1, acc, emit_out(INDEX_TILE), group_count)
                        gpu.barrier()
                        if tid < group_count * INDEX_TILE:
                            s = sample_base + tid // INDEX_TILE
                            row = jp * INDEX_TILE + tid % INDEX_TILE
                            put(mb("index_k"), s * INDEX_DIM + row, lds_ld(outs, tid))
                    else:
                        tw = jp - INDEX_DIM // QKV_A_TILE

                        def u_iw(c, tw=tw, group_count=group_count):
                            kc = wave * IW_CPW_ + c
                            return unit_bf16(r_wiw, tw, kc, QA_NKC, (n_sel(group_count) * HIDDEN + kc * 64) // 2)

                        pre = stage_h(sample_base, group_count, lambda: [u_iw(c) for c in range(IW_CPW_)])
                        acc = run_units(u_iw, IW_CPW_, IW_CPW_, pre)
                        reduce_rows(1, acc, emit_out(INDEX_TILE), group_count)
                        gpu.barrier()
                        if tid < group_count * INDEX_TILE:
                            s = sample_base + tid // INDEX_TILE
                            row = tw * INDEX_TILE + tid % INDEX_TILE
                            put(mb("index_w"), s * INDEX_HEADS + row, lds_ld(outs, tid))
                    gpu.barrier()

        for t in range(start("qkv_a"), N_QKV_A, G):
            t = fx.Int32(t)
            stage("qkv_a")
            for sample_base in range_constexpr(0, S, SAMPLE_TILE):
                group_count = min(SAMPLE_TILE, S - sample_base)

                def u_qa(c):
                    kc = (wave * QA_UNITS + c) * attention_k_chunks_per_unit
                    return unit_attention(
                        r_wqa, r_sqa, t, kc, QA_NKC, HIDDEN, 128, (n_sel(group_count) * HIDDEN + kc * 64) // 2
                    )

                pre = stage_h(sample_base, group_count, lambda: [u_qa(c) for c in range(QA_UNITS)])
                acc = run_units(u_qa, QA_UNITS, QA_UNITS, pre)
                reduce_rows(1, acc, emit_out(QKV_A_TILE), group_count)
                gpu.barrier()
                if tid < group_count * QKV_A_TILE:
                    s = sample_base + tid // QKV_A_TILE
                    row = t * QKV_A_TILE + tid % QKV_A_TILE
                    v = lds_ld(outs, tid)
                    if row < Q_LORA:
                        put(mb("q_a"), s * Q_LORA + row, v)
                    else:
                        put(mb("kv_a"), s * (KV_LORA + PE_DIM) + row - Q_LORA, v)

                if const_expr(with_indexer and not index_proj_spread):
                    IW_CPW = QA_NKC // WAVES

                    def u_index_w(c):
                        kc = wave * IW_CPW + c
                        return unit_bf16(r_wiw, t, kc, QA_NKC, (n_sel(group_count) * HIDDEN + kc * 64) // 2)

                    if t < INDEX_DIM // QKV_A_TILE:

                        def u_index_k(c):
                            kc = (wave * QA_UNITS + c) * 2
                            return unit_fp8x2(
                                r_wik, r_sik, t, kc, QA_NKC, HIDDEN, (n_sel(group_count) * HIDDEN + kc * 64) // 2
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
                                put(mb("index_w"), s * INDEX_HEADS + row, lds_ld(outs, tid))

        def ld_qa(sks):
            v = get2_many([(mb("q_a"), s * Q_LORA + k + j) for s, k in sks for j in (0, 2)])
            return [list(v[2 * i]) + list(v[2 * i + 1]) for i in range(len(sks))]

        # ===================== 2. one q_a RMSNorm CTA per sample, shared downstream
        for s_norm in range(start("q_norm"), S, G):
            s_norm = fx.Int32(s_norm)
            stage("q_norm")
            gpu.barrier()

            def ld_qa_one(sks):
                return ld_qa([(s_norm, k) for _, k in sks])

            stage_x_rmsnorm(ld_qa_one, Q_LORA, g_q, count=1)
            gpu.barrier()
            k = tid * 4
            w0 = lds_ld(xs, k // 2)
            w1 = lds_ld(xs, k // 2 + 1)
            a0, a1 = bf2_f32(w0.bitcast(fx.Int32))
            a2, a3 = bf2_f32(w1.bitcast(fx.Int32))
            put_bf(mb("q_an"), s_norm * Q_LORA + k, [a0, a1, a2, a3])

        # ================ 3. KV RMSNorm + k_pe RoPE -> cache (+ this launch's rows)
        for t in range(start("cache"), 1, G):
            stage("cache")
            r_kv = _rsrc(kv_cache)
            r_pe = _rsrc(pe_cache)
            # gamma and the RoPE factors are issued ahead of the wait
            g = ld_bf16(_rsrc(g_kv), tid)
            t_pe = tid % (PE_DIM // 2)
            if const_expr(cache_hoist):
                # every row's position / slot loaded once, ahead of the kv_a poll (not per row behind the
                # previous row's stores)
                h_pos = [row_position(s) for s in range(S)]
                h_slot = [row_slot(s) for s in range(S)]
            cs = [
                ld_bf16(
                    _rsrc(rope_cos), (h_pos[s] if const_expr(cache_hoist) else row_position(s)) * (PE_DIM // 2) + t_pe
                )
                for s in range(S)
            ]
            sns = [
                ld_bf16(
                    _rsrc(rope_sin), (h_pos[s] if const_expr(cache_hoist) else row_position(s)) * (PE_DIM // 2) + t_pe
                )
                for s in range(S)
            ]
            gpu.barrier()
            # every sample's kv latent and k_pe pair in one poll, one block reduction
            vs = getf_many([(mb("kv_a"), s * (KV_LORA + PE_DIM) + tid) for s in range(S)])
            pes = get2_many(
                [(mb("kv_a"), s * (KV_LORA + PE_DIM) + KV_LORA + (tid % (PE_DIM // 2)) * 2) for s in range(S)]
            )
            ssq = block_sums([v * v for v in vs])
            for s in range_constexpr(S):
                if const_expr(cache_hoist):
                    slot = h_slot[s]
                    active = slot >= 0
                else:
                    row_position(s)  # unused, but dropping it reallocates registers of this build
                    slot = row_slot(s)
                    active = row_writes_cache(s)
                kvn = bf16_round(vs[s] * _rsq(ssq[s] * (1.0 / KV_LORA) + EPS) * g)
                if active:
                    bo.buffer_store(kvn.to(fx.BFloat16), r_kv, slot * QK_DIM + tid)
                put(mb("kvnew"), s * KV_LORA + tid, kvn)
                if tid < PE_DIM // 2:
                    x0, x1 = pes[s]
                    c, sn = cs[s], sns[s]
                    p0, p1 = x0 * c - x1 * sn, x0 * sn + x1 * c
                    p0, p1 = bf16_round(p0), bf16_round(p1)
                    if active:
                        pe_offset = slot * QK_DIM + KV_LORA + tid * 2
                        bo.buffer_store(p0.to(fx.BFloat16), r_pe, pe_offset)
                        bo.buffer_store(p1.to(fx.BFloat16), r_pe, pe_offset + 1)
                    put2(mb("penew"), s * PE_DIM + tid * 2, p0, p1)

            if const_expr(with_indexer):
                # vLLM's index-K (fused_norm_rope): LayerNorm (eps 1e-6), RoPE on dims [0, 64), ue8m0 FP8 into the
                # SHUFFLE index cache at the row's MLA slot; values / scale also to index_k_new / index_k_scale.
                # Wave w = row w (lane l: dims 2l, 2l + 1); loads not needing the mailbox go before its one poll.
                r_gik, r_bik = _rsrc(index_arg(5)), _rsrc(index_arg(6))
                r_icache = _rsrc(index_arg(8))
                iblk_stride = fx.Int32(index_arg(11))
                rp_i0 = lane * 2
                rp_g0, rp_g1 = ld_f32(r_gik, rp_i0), ld_f32(r_gik, rp_i0 + 1)
                rp_bb0, rp_bb1 = ld_f32(r_bik, rp_i0), ld_f32(r_bik, rp_i0 + 1)
                for rp_g in range_constexpr((S + WAVES - 1) // WAVES):  # S > 8: a second row group
                    if wave + rp_g * WAVES < S:
                        rp_row = wave + rp_g * WAVES
                        rp_slot = row_slot(rp_row)
                        rp_rpos = row_position(rp_row)
                        rp_rc = ld_bf16(_rsrc(rope_cos), rp_rpos * (PE_DIM // 2) + lane % (PE_DIM // 2))
                        rp_rs = ld_bf16(_rsrc(rope_sin), rp_rpos * (PE_DIM // 2) + lane % (PE_DIM // 2))
                        rp_k0, rp_k1 = get2(mb("index_k"), rp_row * INDEX_DIM + rp_i0)
                        # bf16-round the projection as vLLM's wk_weights_proj GEMM output is bf16
                        rp_k0, rp_k1 = bf16_round(rp_k0), bf16_round(rp_k1)
                        rp_mean = wave_sum(rp_k0 + rp_k1) * (1.0 / INDEX_DIM)
                        rp_d0, rp_d1 = rp_k0 - rp_mean, rp_k1 - rp_mean
                        rp_rstd = _rsq(wave_sum(rp_d0 * rp_d0 + rp_d1 * rp_d1) * (1.0 / INDEX_DIM) + 1.0e-6)
                        rp_v0 = rp_d0 * rp_rstd * rp_g0 + rp_bb0
                        rp_v1 = rp_d1 * rp_rstd * rp_g1 + rp_bb1
                        if lane < PE_DIM // 2:
                            rp_v0, rp_v1 = rp_v0 * rp_rc - rp_v1 * rp_rs, rp_v0 * rp_rs + rp_v1 * rp_rc
                        rp_amax = wave_max(fx.max(fx.max(rp_v0, -rp_v0), fx.max(rp_v1, -rp_v1)))
                        rp_sc, rp_inv_sc = _ue8m0_scale(rp_amax)
                        rp_q0, rp_q1 = _fp8_roundtrip(rp_v0 * rp_inv_sc, rp_v1 * rp_inv_sc)
                        put_bf(mb("index_k_new"), rp_row * INDEX_DIM + rp_i0, [rp_q0, rp_q1])
                        rp_n0, rp_n1 = _xshfl(rp_q0, 1), _xshfl(rp_q1, 1)
                        rp_blk = rp_slot // 16
                        rp_off = rp_slot % 16
                        if (rp_slot >= 0) & (lane % 2 == 0):
                            rp_byte = rp_blk * iblk_stride + rp_off * 16 + (rp_i0 // 16) * 256 + rp_i0 % 16
                            bo.buffer_store(
                                fp8_pack4(rp_q0, rp_q1, rp_n0, rp_n1), r_icache, rp_byte // 4, cache_modifier=CM_DEV
                            )
                        if (rp_slot >= 0) & (lane == 0):
                            bo.buffer_store(
                                rp_sc,
                                r_icache,
                                (rp_blk * iblk_stride + 16 * INDEX_DIM + rp_off * 4) // 4,
                                cache_modifier=CM_DEV,
                            )
                            put(mb("index_k_scale"), rp_row, rp_sc)
                gpu.barrier()
                if tid < S:
                    put(mb("index_ready"), tid, fx.Int32(1))

        # =============================================== 4. normalized q_a -> q_b (+RoPE)
        r_wqb, r_sqb = _rsrc(w_q_b), _rsrc(s_q_b)
        QB_NKC = Q_LORA // 64
        QB_UNITS = QB_NKC // (attention_k_chunks_per_unit * WAVES)
        if const_expr(with_indexer):
            r_wiq, r_siq = _rsrc(index_arg(3)), _rsrc(index_arg(4))

        def u_index_q(iq_t):
            def mk(c):
                kc = (wave * QB_UNITS + c) * 2
                return unit_fp8x2(r_wiq, r_siq, iq_t, kc, QB_NKC, Q_LORA, (n_sel() * Q_LORA + kc * 64) // 2)

            return mk

        def index_q_out(iq_t, acc):
            reduce_rows(1, acc, emit_out(INDEX_TILE))
            gpu.barrier()
            if tid < S * INDEX_TILE // 4:
                s = tid // (INDEX_TILE // 4)
                r = (tid % (INDEX_TILE // 4)) * 4
                put_bf(
                    mb("index_q"),
                    s * INDEX_Q_ROWS + iq_t * INDEX_TILE + r,
                    [lds_ld(outs, s * INDEX_TILE + r + j) for j in range(4)],
                )

        for t in range(start("q_b"), N_QB, G):
            t = fx.Int32(t)
            stage("q_b")

            def u_qb(c):
                kc = (wave * QB_UNITS + c) * attention_k_chunks_per_unit
                return unit_attention(r_wqb, r_sqb, t, kc, QB_NKC, Q_LORA, 128, (n_sel() * Q_LORA + kc * 64) // 2)

            pre = [u_qb(c) for c in range(QB_UNITS)]
            gpu.barrier()
            stage_x_pairs("q_an", S * Q_LORA, lambda k: k)
            gpu.barrier()
            acc = run_units(u_qb, QB_UNITS, QB_UNITS, pre)
            reduce_rows(1, acc, emit_out(Q_B_TILE))
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
                    c = ld_bf16(_rsrc(rope_cos), row_position(s) * (PE_DIM // 2) + i // 2)
                    sn = ld_bf16(_rsrc(rope_sin), row_position(s) * (PE_DIM // 2) + i // 2)
                    put_bf(mb("q_pe"), (s * H + head) * PE_DIM + i, [x0 * c - x1 * sn, x0 * sn + x1 * c])

            if const_expr(with_indexer):
                # this CTA's normalized q_lora tile also feeds one index-query tile; the
                # complementary CTAs compute the other half below
                stage("index_q")
                index_q_out(t, run_units(u_index_q(t), QB_UNITS, QB_UNITS))

        if const_expr(with_indexer):
            # The 128 CTAs without q_b work produce the other 128 index-query
            # tiles concurrently.  They reload q_a, but remove one full GEMV
            # from the q_b CTAs' serialized critical path.
            N_INDEX_Q_EXTRA = INDEX_Q_ROWS // INDEX_TILE - N_QB
            for tt in range(start("index_q"), N_INDEX_Q_EXTRA, G):
                tt = fx.Int32(tt)
                iq_t = N_QB + tt
                stage("index_q")
                mk = u_index_q(iq_t)
                pre = [mk(c) for c in range(QB_UNITS)]
                gpu.barrier()
                stage_x_pairs("q_an", S * Q_LORA, lambda k: k)
                gpu.barrier()
                index_q_out(iq_t, run_units(mk, QB_UNITS, QB_UNITS, pre))

        # ==================================== 4. absorbed query: q_lat = W_UK^T q_nope
        def _uk_section():
            # 8 row groups (128 latent rows of one head) x 3 chunks: one row group per wave
            r_wuk, r_suk = _rsrc(w_uk), _rsrc(s_uk)
            UK_NKC = NOPE_DIM // 64
            for t in range(start("uk"), N_UK, G):
                t = fx.Int32(t)
                stage("uk")
                head = t // UK_PER_HEAD

                def u_uk(c):
                    return unit_attention(
                        r_wuk, r_suk, t * WAVES + wave, c, UK_NKC, NOPE_DIM, 64, (n_sel() * NOPE_DIM + c * 64) // 2
                    )

                pre = [u_uk(c) for c in range(UK_NKC)]
                gpu.barrier()
                stage_x_pairs("q_nope", S * NOPE_DIM, lambda k: ((k // NOPE_DIM) * H + head) * NOPE_DIM + k % NOPE_DIM)
                gpu.barrier()
                acc = run_units(u_uk, UK_NKC, UK_NKC, pre)
                fx.ptr_store(fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4)
                gpu.barrier()
                if tid < S * UK_TILE // 4:
                    k = tid * 4
                    s = k // UK_TILE
                    r0 = k % UK_TILE
                    vals = []
                    for j in range_constexpr(4):
                        r = r0 + j
                        ww = r // 16
                        vals.append(lds_ld(red, (ww * 64 + s + 16 * ((r % 16) // 4)) * 4 + r % 4))
                    put_bf(mb("q_lat"), (s * H + head) * KV_LORA + (t % UK_PER_HEAD) * UK_TILE + r0, vals)

        _uk_section()

        # ================================== 5. sparse MLA split: 32 keys x 8 heads
        def _section5():
            r_kv = _rsrc(kv_cache)
            r_pe = _rsrc(pe_cache)
            r_idx = _rsrc(indices)
            KPW = SPLIT_KEYS // WAVES

            def split_keys(t, s):
                """(nkeys, sparse) of sample s; wave 0 writes this split's cache rows to LDS."""
                index_base, index_end = row_index_bounds(s)
                nkeys = index_end - index_base
                sparse = index_end > index_base
                if const_expr(with_indexer):
                    if sparse:
                        if wave == 0:
                            if lane == 0:
                                get(mb("indices_ready"), s)
                            # Only wave 0 consumes the compact index payload; its
                            # divergent ready poll reconverges before this acquire.
                            fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope="agent")
                if wave == 0:
                    if lane < SPLIT_KEYS:
                        k_pos = t * SPLIT_KEYS + lane
                        k_cl = (k_pos < nkeys).select(k_pos, 0)
                        # fused indexer: the select stage (another CTA) wrote this row's CSR: device scope
                        idx = fx.Int32(0)
                        if sparse:
                            idx = fx.Int32(
                                bo.buffer_load(
                                    r_idx,
                                    index_base + k_cl,
                                    vec_width=1,
                                    dtype=T.i32,
                                    cache_modifier=CM_DEV if with_indexer else 0,
                                )
                            )
                        lds_st(attn_keys, lane, idx)
                return nkeys, sparse

            def gather_old_kv():
                """Each wave copies its 8 keys' KV latent (1 KB) + k_pe (128 B) cache rows
                into the LDS tiles (rows of this launch are patched in by patch_new_kv)."""
                krows = [lds_ld(attn_keys, wave * KPW + jj) for jj in range(KPW)]
                for jj in range_constexpr(KPW):
                    j = wave * KPW + jj
                    kv8 = fx.Vector(
                        bo.buffer_load(r_kv, krows[jj] * (QK_DIM // 2) + lane * 4, vec_width=4, dtype=T.i32)
                    )
                    fx.ptr_store(kv8.bitcast(fx.Float32), ktile + (j * KS + lane * 4))
                    if lane < PE_DIM // 2:
                        pe_row = krows[jj] * (QK_DIM // 2) + KV_LORA // 2 + lane
                        lds_st(petile, j * PS + lane, ld_f32(r_pe, pe_row))

            def patch_new_kv():
                """Rows appended by this launch come from the cache task's kvnew / penew pairs."""
                new_active = [row_writes_cache(new_s) for new_s in range(S)]
                new_slots = [row_slot(new_s) for new_s in range(S)]
                for jj in range_constexpr(KPW):
                    j = wave * KPW + jj
                    kr = lds_ld(attn_keys, j)
                    sn = fx.Int32(-1)
                    for new_s in range_constexpr(S):
                        match = new_active[new_s] & (kr == new_slots[new_s])
                        sn = match.select(fx.Int32(new_s), sn)
                    if sn >= 0:
                        kvp = get2_many([(mb("kvnew"), sn * KV_LORA + lane * 8 + m * 2) for m in range(4)])
                        w = [bf16_pair(a0, a1) for a0, a1 in kvp]
                        fx.ptr_store(fx.Vector.from_elements(w, fx.Float32), ktile + (j * KS + lane * 4))
                        if lane < PE_DIM // 2:
                            a0, a1 = get2(mb("penew"), sn * PE_DIM + lane * 2)
                            lds_st(petile, j * PS + lane, bf16_pair(a0, a1))

            N_HEAD_GROUPS = H // WAVES
            for tt in range(start("split"), S * N_HEAD_GROUPS * N_SPLIT, G):
                tt = fx.Int32(tt)
                stage("split")
                s = tt // (N_HEAD_GROUPS * N_SPLIT)
                head_group = (tt // N_SPLIT) % N_HEAD_GROUPS
                t = tt % N_SPLIT
                h = head_group * WAVES + wave

                def stage_q():
                    gpu.barrier()
                    # q of all heads -> bf16 Q[h][576] (words h * 288 + d / 2): latent 512 then pe 64
                    NQ = H * KV_LORA // 4 // THREADS
                    t_pe = fx.min(tid, H * PE_DIM // 4 - 1)
                    qv = poll(
                        [(mb("q_lat"), (s * H * KV_LORA + (tid + i * THREADS) * 4) // 2, 2) for i in range(NQ)]
                        + [(mb("q_pe"), (s * H * PE_DIM + t_pe * 4) // 2, 2)]
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

                nkeys, sparse = split_keys(t, s)
                gpu.barrier()
                gather_old_kv()  # before waiting for q: these rows are from earlier launches
                stage_q()
                patch_new_kv()
                gpu.barrier()
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
                        KT_OFF + key * KS + kst * 16, PT_OFF + key * PS + (kst - KV_LORA // 32) * 16
                    )
                    a = fx.ptr_load(xs + (kw + (lane // 16) * 4), result_type=v4f).bitcast(fx.BFloat16)
                    b = fx.ptr_load(xs + (hn * QS + kst * 16 + (lane // 16) * 4), result_type=v4f).bitcast(fx.BFloat16)
                    c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                if const_expr(SPLIT_KEYS == 64) or wave < 4:
                    fx.ptr_store(c, red + (wave * 64 + lane) * 4)
                gpu.barrier()
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
                # O = P V on MFMA: heads M, keys K, latent dims N.  Each V word holds
                # a dim pair (even dim low), so one read feeds two MFMAs (even / odd dims):
                # each wave owns 2 groups of 32 dims.  V is read key-strided from the tile.
                for g in range_constexpr(KV_LORA // 32 // WAVES):
                    # dim pair word
                    dw = (wave * (KV_LORA // 32 // WAVES) + g) * 16 + lane % 16
                    c0 = fx.Vector.filled(4, 0.0, fx.Float32)
                    c1 = fx.Vector.filled(4, 0.0, fx.Float32)
                    for js in range_constexpr(SPLIT_KEYS // 32):
                        a = fx.ptr_load(
                            pl + (hn * (SPLIT_KEYS // 2) + js * 16 + (lane // 16) * 4), result_type=v4f
                        ).bitcast(fx.BFloat16)
                        ws = [
                            fx.ptr_load(ktile + ((js * 32 + (lane // 16) * 8 + i) * KS + dw)).bitcast(fx.Int32)
                            for i in range(8)
                        ]
                        w_lo = [(ws[2 * i] & 0xFFFF) | (ws[2 * i + 1] << 16) for i in range(4)]
                        w_hi = [fx.Int32(fx.Uint32(ws[2 * i]) >> 16) | (ws[2 * i + 1] & -65536) for i in range(4)]
                        b0 = fx.Vector.from_elements(w_lo, fx.Int32).bitcast(fx.BFloat16)
                        b1 = fx.Vector.from_elements(w_hi, fx.Int32).bitcast(fx.BFloat16)
                        c0 = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b0, c0]))
                        c1 = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b1, c1]))
                    if lane < 32:
                        for e in range_constexpr(4):
                            hh = split_acc_head(head_group, lane // 16, e)
                            put_bf(mb("sp_acc"), ((s * N_SPLIT + t) * H + hh) * KV_LORA + dw * 2, [c0[e], c1[e]])
                if lane == 0:  # written last: the merge's readiness hint
                    put(mb("sp_m"), (s * N_SPLIT + t) * H + h, m)
                    put(mb("sp_l"), (s * N_SPLIT + t) * H + h, lsum)

        # ====================== 4b. fused sparse index score + exact top-2048
        if const_expr(with_indexer):
            r_index_cache = _rsrc(index_arg(8))
            N_INDEX_SPLIT = index_max_seq // INDEX_KEYS_PER_TASK
            r_ibt = _rsrc(index_arg(9))
            ibt_stride = fx.Int32(index_arg(10))
            iblk_stride = fx.Int32(index_arg(11))
            # vLLM's head weight scale (fused_q: softmax_scale * n_head^-0.5); a positive constant, so it changes
            # logit magnitudes only, never the ranking
            IDX_W_SCALE = (INDEX_DIM**-0.5) * (INDEX_HEADS**-0.5)

            def row_bound(s):
                """Keys row s scores: its context (position + 1); inactive rows 0."""
                b, e = row_index_bounds(s)
                return (e > b).select(row_position(s) + 1, fx.Int32(0))

            def load_index_q8(head, k):
                words = fx.Vector.from_elements(
                    [lds_ld(xs, (head * INDEX_DIM + k) // 2 + j) for j in range(4)], fx.Float32
                )
                return words.bitcast(fx.BFloat16)

            def score_task_batched(s, tile0, bound):
                """64-key tile ``tile0`` of row s with its memory round trips collapsed: (1) block-table entries +
                RoPE tables, (2) the keys' FP8 bytes + scales, all issued before (3) ONE poll of this thread's 4
                index-q pairs and a head weight; (4) only the tile holding the row's current key polls that key's
                mailbox values. No index_ready wait: older keys come from earlier launches' cache writes, the
                current key from its own tagged mailbox."""
                NB_Q = (INDEX_Q_ROWS // 2) // THREADS
                key_group = wave // 2
                head_group = wave % 2
                head = head_group * 16 + lane % 16
                rp = row_position(s)
                cur = bound - 1
                key_pos = tile0 * INDEX_KEYS_PER_TASK + key_group * 16 + lane % 16
                safe_key = fx.min(key_pos, bound - 1)
                k_blk = fx.Int32(bo.buffer_load(r_ibt, s * ibt_stride + safe_key // 16, vec_width=1, dtype=T.i32))
                rcs, rsn = [], []
                for b in range_constexpr(NB_Q):
                    kq = ((tid + b * THREADS) * 2) % INDEX_DIM
                    kqc = fx.min(kq, PE_DIM - 2) // 2
                    rcs.append(ld_bf16(_rsrc(rope_cos), rp * (PE_DIM // 2) + kqc))
                    rsn.append(ld_bf16(_rsrc(rope_sin), rp * (PE_DIM // 2) + kqc))
                k_base = k_blk * iblk_stride + (safe_key % 16) * 16
                kw = []
                for k32 in range_constexpr(INDEX_DIM // 32):
                    k = k32 * 32 + (lane // 16) * 8
                    kw.append(
                        fx.Vector(
                            bo.buffer_load(
                                r_index_cache, (k_base + (k // 16) * 256 + k % 16) // 4, vec_width=2, dtype=T.i32
                            )
                        )
                    )
                k_sc = fx.Float32(
                    bo.buffer_load(
                        r_index_cache,
                        (k_blk * iblk_stride + 16 * INDEX_DIM + (safe_key % 16) * 4) // 4,
                        vec_width=1,
                        dtype=T.f32,
                    )
                )
                specs = [
                    (mb("index_q"), (s * INDEX_Q_ROWS + (tid + b * THREADS) * 2) // 2, 1) for b in range_constexpr(NB_Q)
                ]
                specs.append((mb("index_w"), s * INDEX_HEADS + tid % INDEX_HEADS, 1))
                got = poll(specs)
                if tid < INDEX_HEADS:
                    w_raw = bf16_round(got[NB_Q][0].bitcast(fx.Float32))  # vLLM's weights: a bf16 GEMM output
                    lds_st(keys, tid, w_raw.bitcast(fx.Int32))
                for b in range_constexpr(NB_Q):
                    q_pair = tid + b * THREADS
                    kq = (q_pair * 2) % INDEX_DIM
                    q0, q1 = bf2_f32(got[b][0])
                    if kq < PE_DIM:
                        q0, q1 = q0 * rcs[b] - q1 * rsn[b], q0 * rsn[b] + q1 * rcs[b]
                    if const_expr(index_q_fp8):
                        qmax = wave_max(fx.max(fx.max(q0, -q0), fx.max(q1, -q1)))
                        q_sc, q_inv = _ue8m0_scale(qmax)
                        q0, q1 = _fp8_roundtrip(q0 * q_inv, q1 * q_inv)
                        if lane == 0:
                            lds_st(keys, INDEX_HEADS + q_pair // (INDEX_DIM // 2), q_sc.bitcast(fx.Int32))
                    lds_st(xs, q_pair, bf16_pair(q0, q1))
                # the row's current key (position bound - 1) in this tile: poll its mailbox values (CTA-uniform
                # condition; every lane polls row s's entries, which are produced, and keeps them only if new)
                kn = fx.Vector.filled(INDEX_DIM // 8, 0, fx.Int32)
                ksn = fx.Float32(0.0)
                lo = tile0 * INDEX_KEYS_PER_TASK
                has_new = (cur >= lo) & (cur < lo + INDEX_KEYS_PER_TASK)
                nrow = s
                if has_new:
                    nspecs = [
                        (mb("index_k_new"), (nrow * INDEX_DIM + k32 * 32 + (lane // 16) * 8 + 2 * j) // 2, 1)
                        for k32 in range(INDEX_DIM // 32)
                        for j in range(4)
                    ]
                    nspecs.append((mb("index_k_scale"), nrow, 1))
                    ng = poll(nspecs)
                    kn = fx.Vector.from_elements([ng[i][0] for i in range(INDEX_DIM // 8)], fx.Int32)
                    ksn = ng[INDEX_DIM // 8][0].bitcast(fx.Float32)
                gpu.barrier()
                new_key = safe_key == cur
                score_frag = fx.Vector.filled(4, 0.0, fx.Float32)
                for k32 in range_constexpr(INDEX_DIM // 32):
                    k = k32 * 32 + (lane // 16) * 8
                    qv = load_index_q8(head, k)
                    kv_c = _fp8_to_bf16x8(kw[k32][0], kw[k32][1]).bitcast(fx.Int32)
                    vals = []
                    for j in range_constexpr(4):  # mailbox: packed bf16 pairs, the cache's own byte order
                        vals.append(new_key.select(kn[k32 * 4 + j], kv_c[j]))
                    kv = fx.Vector.from_elements(vals, fx.Int32).bitcast(fx.BFloat16)
                    score_frag = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [qv, kv, score_frag]))
                partial = fx.Float32(0.0)
                for e in range_constexpr(4):
                    index_head = head_group * 16 + (lane // 16) * 4 + e
                    weight = lds_ld(keys, index_head).bitcast(fx.Float32)
                    if const_expr(index_q_fp8):
                        weight = weight * lds_ld(keys, INDEX_HEADS + index_head).bitcast(fx.Float32)
                    partial = partial + fx.max(score_frag[e], fx.Float32(0.0)) * weight
                partial = _xred(partial, 16, lambda a, b: a + b)
                partial = _xred(partial, 32, lambda a, b: a + b)
                if lane < 16:
                    lds_st(red, wave * 16 + lane, partial)
                gpu.barrier()
                if (wave % 2 == 0) & (lane < 16) & (key_pos < bound):
                    score = lds_ld(red, wave * 16 + lane) + lds_ld(red, (wave + 1) * 16 + lane)
                    ksc = (key_pos == cur).select(ksn, k_sc)
                    put(mb("index_scores"), s * index_max_seq + key_pos, score * (ksc * IDX_W_SCALE))

            N_SCORE_TASKS = S * N_INDEX_SPLIT
            for tt in range(start("index_score"), N_SCORE_TASKS, G):
                tt = fx.Int32(tt)
                s = tt // N_INDEX_SPLIT
                tile = tt % N_INDEX_SPLIT
                stage("index_score")
                bound = row_bound(s)
                # rows with context <= topk select every key (identity, select stage): nothing to score
                if (tile * INDEX_KEYS_PER_TASK < bound) & (bound > topk):
                    score_task_batched(s, tile, bound)

            for s in range(start("index_select"), S, G):
                s = fx.Int32(s)
                stage("index_select")
                bound = row_bound(s)

                def select_digit(shift, prefix, remain):
                    digit = 255 - fx.min(tid, 255)
                    count = (tid < 256).select(lds_ld(keys, digit), fx.Int32(0))
                    inclusive = fx.coop.warp_inclusive_scan(count, fx.ReductionOp.ADD, width=64)
                    wave_total = read_lane_i32(inclusive, 63)
                    if (wave < 4) & (lane == 63):
                        lds_st(keys, 256 + wave, wave_total)
                    gpu.barrier()
                    before_wave = fx.Int32(0)
                    for w in range_constexpr(4):
                        before_wave = before_wave + (wave > w).select(lds_ld(keys, 256 + w), fx.Int32(0))
                    above = before_wave + inclusive - count
                    hit = (tid < 256) & (above < remain) & (above + count >= remain)
                    gpu.barrier()
                    if hit:
                        lds_st(keys, 256, fx.Int32(prefix | (fx.Uint32(digit) << shift)))
                        lds_st(keys, 257, remain - above)
                    gpu.barrier()
                    return fx.Uint32(lds_ld(keys, 256)), lds_ld(keys, 257)

                # rows are requests; CSR row = [i_base, i_base + min(bound, topk)), physical slots
                i_base, _ = row_index_bounds(s)

                def slot_of(i):
                    blk = fx.Int32(bo.buffer_load(r_ibt, s * ibt_stride + i // 16, vec_width=1, dtype=T.i32))
                    return blk * 16 + i % 16

                # every global read of this stage is issued as one batch at clamped addresses
                # (no branch between the loads), so each phase costs one memory round trip, not one per item
                if bound <= topk:
                    # every key selected (sampler.cu rowLen <= topK): identity, no radix passes
                    n_id = (topk + THREADS - 1) // THREADS
                    id_slots = [slot_of(fx.min(tid + b_ * THREADS, bound - 1)) for b_ in range_constexpr(n_id)]
                    for batch in range_constexpr(n_id):
                        j = tid + batch * THREADS
                        if j < bound:
                            bo.buffer_store(id_slots[batch], _rsrc(indices), i_base + j, cache_modifier=CM_DEV)
                else:
                    if const_expr(select_radix11):
                        hist = fx.recast_iter(fx.Int32, red)  # 2048 bins (RED_WORDS = 2048)
                        n_sc = (index_max_seq + THREADS - 1) // THREADS

                        def zero_hist():
                            for j_ in range_constexpr(2048 // THREADS):
                                lds_st(hist, tid + j_ * THREADS, fx.Int32(0))

                        def select_bins(shift, nbits, prefix, remain):
                            """Bins in descending digit order, ``per`` consecutive bins per thread: the bin
                            where the running count from the top reaches ``remain`` -> (prefix | digit << shift,
                            remain - count above that bin)."""
                            nbins = 1 << nbits
                            per = nbins // THREADS
                            digs = [nbins - 1 - (tid * per + j_) for j_ in range(per)]
                            cnts = [lds_ld(hist, d_) for d_ in digs]
                            local = cnts[0]
                            for c_ in cnts[1:]:
                                local = local + c_
                            inclusive = fx.coop.warp_inclusive_scan(local, fx.ReductionOp.ADD, width=64)
                            if lane == 63:
                                lds_st(keys, 256 + wave, read_lane_i32(inclusive, 63))
                            gpu.barrier()
                            before_wave = fx.Int32(0)
                            for w in range_constexpr(WAVES):
                                before_wave = before_wave + (wave > w).select(lds_ld(keys, 256 + w), fx.Int32(0))
                            run = before_wave + inclusive - local
                            hit = fx.Int32(0)
                            hd = fx.Int32(0)
                            hr = fx.Int32(0)
                            for j_ in range_constexpr(per):
                                h_ = (run < remain) & (run + cnts[j_] >= remain)
                                hd = h_.select(fx.Int32(digs[j_]), hd)
                                hr = h_.select(remain - run, hr)
                                hit = h_.select(fx.Int32(1), hit)
                                run = run + cnts[j_]
                            gpu.barrier()
                            if hit != 0:
                                lds_st(keys, 256, fx.Int32(prefix | (fx.Uint32(hd) << shift)))
                                lds_st(keys, 257, hr)
                            gpu.barrier()
                            return fx.Uint32(lds_ld(keys, 256)), lds_ld(keys, 257)

                        zero_hist()
                        gpu.barrier()
                        sc_vals = getf_many(
                            [
                                (mb("index_scores"), s * index_max_seq + fx.min(tid + b_ * THREADS, bound - 1))
                                for b_ in range_constexpr(n_sc)
                            ]
                        )
                        kreg = []
                        for batch in range_constexpr(n_sc):
                            i = tid + batch * THREADS
                            bits = sc_vals[batch].bitcast(fx.Int32)
                            key = fx.Uint32((bits >= 0).select(bits ^ fx.Int32(-(2**31)), ~bits))
                            kreg.append(key)
                            if i < bound:
                                lds_st(xs, i, key.bitcast(fx.Float32))  # for the thread-major compaction
                                fx.atomic_add(hist + fx.Int32(key >> fx.Uint32(21)), fx.Int32(1), syncscope="workgroup")
                        gpu.barrier()
                        prefix, remain = select_bins(21, 11, fx.Uint32(0), fx.min(fx.Int32(topk), bound))
                        for shift, nbits, hi in ((10, 11, 21), (0, 10, 10)):
                            zero_hist()
                            gpu.barrier()
                            for batch in range_constexpr(n_sc):
                                i = tid + batch * THREADS
                                key = kreg[batch]
                                if (i < bound) & ((key >> fx.Uint32(hi)) == (prefix >> fx.Uint32(hi))):
                                    fx.atomic_add(
                                        hist + fx.Int32((key >> fx.Uint32(shift)) & fx.Uint32((1 << nbits) - 1)),
                                        fx.Int32(1),
                                        syncscope="workgroup",
                                    )
                            gpu.barrier()
                            prefix, remain = select_bins(shift, nbits, prefix, remain)
                    else:
                        if tid < 256:
                            lds_st(keys, tid, fx.Int32(0))
                        gpu.barrier()
                        n_sc = (index_max_seq + THREADS - 1) // THREADS
                        sc_vals = getf_many(
                            [
                                (mb("index_scores"), s * index_max_seq + fx.min(tid + b_ * THREADS, bound - 1))
                                for b_ in range_constexpr(n_sc)
                            ]
                        )
                        for batch in range_constexpr(n_sc):
                            i = tid + batch * THREADS
                            if i < bound:
                                bits = sc_vals[batch].bitcast(fx.Int32)
                                key = (bits >= 0).select(bits ^ fx.Int32(-(2**31)), ~bits)
                                # The selector CTA no longer needs the large GEMV staging
                                # region, so reuse it for the 4-K radix keys instead of
                                # increasing the monokernel's LDS allocation.
                                lds_st(xs, i, key.bitcast(fx.Float32))
                                digit = fx.Int32((fx.Uint32(key) >> 24) & fx.Uint32(255))
                                fx.atomic_add(keys + digit, fx.Int32(1), syncscope="workgroup")
                        gpu.barrier()
                        prefix, remain = select_digit(24, fx.Uint32(0), fx.min(fx.Int32(topk), bound))
                        prefix_mask = 255 << 24
                        for shift in (16, 8, 0):
                            if tid < 256:
                                lds_st(keys, tid, fx.Int32(0))
                            gpu.barrier()
                            for batch in range_constexpr((index_max_seq + THREADS - 1) // THREADS):
                                i = tid + batch * THREADS
                                if i < bound:
                                    key = fx.Uint32(lds_ld(xs, i).bitcast(fx.Int32))
                                    if (key & fx.Uint32(prefix_mask)) == prefix:
                                        digit = fx.Int32((key >> shift) & fx.Uint32(255))
                                        fx.atomic_add(keys + digit, fx.Int32(1), syncscope="workgroup")
                            gpu.barrier()
                            prefix, remain = select_digit(shift, prefix, remain)
                            prefix_mask |= 255 << shift

                    threshold = prefix
                    # select_digit returns keys[256] / keys[257] (prefix, remain) read AFTER its last barrier,
                    # and scan_flags' lane 63 of wave w stores keys[256 + w]: without this barrier a wave that
                    # runs ahead (wave 0 / 1) overwrites the threshold / remain before a lagging wave read them
                    # -> that wave compacts against a garbage threshold (wrong CSR, stores past the row).
                    # Pre-existing in FlyDSL's index_select.
                    gpu.barrier()
                    r_index_out = _rsrc(indices)  # the global physical-slot CSR
                    items = index_max_seq // THREADS
                    item_indices = [tid * items + j for j in range_constexpr(items)]
                    item_keys = [fx.Uint32(lds_ld(xs, fx.min(i, bound - 1)).bitcast(fx.Int32)) for i in item_indices]
                    gt = [(i < bound) & (key > threshold) for i, key in zip(item_indices, item_keys)]
                    eq = [(i < bound) & (key == threshold) for i, key in zip(item_indices, item_keys)]

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
                        inclusive = fx.coop.warp_inclusive_scan(local, fx.ReductionOp.ADD, width=64)
                        wave_total = read_lane_i32(inclusive, 63)
                        if lane == 63:
                            lds_st(keys, 256 + wave, wave_total)
                        gpu.barrier()
                        before_wave = fx.Int32(0)
                        total = fx.Int32(0)
                        for w in range_constexpr(WAVES):
                            wave_count = lds_ld(keys, 256 + w)
                            before_wave = before_wave + (wave > w).select(wave_count, fx.Int32(0))
                            total = total + wave_count
                        thread_base = before_wave + inclusive - local
                        return [thread_base + off for off in local_offsets], total

                    gt_offsets, out_gt = scan_flags(gt)
                    gpu.barrier()
                    eq_offsets, _ = scan_flags(eq)
                    gpu.barrier()
                    item_slots = [slot_of(fx.min(i, bound - 1)) for i in item_indices]
                    for j in range_constexpr(items):
                        if gt[j] & (gt_offsets[j] < topk):  # defensive: never past the row's CSR range
                            bo.buffer_store(item_slots[j], r_index_out, i_base + gt_offsets[j], cache_modifier=CM_DEV)
                        if eq[j] & (eq_offsets[j] < topk):
                            lds_st(red, eq_offsets[j], item_indices[j].bitcast(fx.Float32))
                    gpu.barrier()
                    need_eq = fx.min(fx.Int32(topk), bound) - out_gt
                    n_eq = (topk + THREADS - 1) // THREADS
                    # tie positions from LDS (entries past need_eq are stale: clamp them into the row)
                    eq_pos = [
                        fx.max(
                            fx.min(lds_ld(red, fx.min(tid + b_ * THREADS, topk - 1)).bitcast(fx.Int32), bound - 1),
                            fx.Int32(0),
                        )
                        for b_ in range_constexpr(n_eq)
                    ]
                    eq_slots = [slot_of(p_) for p_ in eq_pos]
                    for batch in range_constexpr(n_eq):
                        j = tid + batch * THREADS
                        if j < need_eq:
                            bo.buffer_store(eq_slots[batch], r_index_out, i_base + out_gt + j, cache_modifier=CM_DEV)
                # Every lane contributed compact indices.  Make all of those
                # device-memory stores visible before lane 0 publishes the one
                # readiness tag consumed by the attention CTAs.
                fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope="agent")
                gpu.barrier()
                if tid == 0:
                    put(mb("indices_ready"), s, fx.Int32(1))

        _section5()

        # ========================== 6. split merge + W_UV: o = W_UV (softmax . KV)
        # 4 row groups x 8 chunks: 2 waves per row group, 4 chunks each
        r_wuv, r_suv = _rsrc(w_uv), _rsrc(s_uv)
        UV_NKC = KV_LORA // 64
        UV_R = UV_TILE // 16
        UV_WPR = WAVES // UV_R
        UV_UNITS = UV_NKC // (attention_k_chunks_per_unit * UV_WPR)
        for tt in range(start("uv"), S * N_UV, G):
            tt = fx.Int32(tt)
            stage("uv")
            s = tt // N_UV  # sample
            t = tt % N_UV  # global 64-row tile
            head = t // (V_DIM // UV_TILE)

            def u_uv(c):
                kc = ((wave % UV_WPR) * UV_UNITS + c) * attention_k_chunks_per_unit
                return unit_attention(r_wuv, r_suv, t * UV_R + wave // UV_WPR, kc, UV_NKC, KV_LORA, 128, (kc * 64) // 2)

            pre = [u_uv(c) for c in range(UV_UNITS)]
            gpu.barrier()
            pre_poll(N_SPLIT, lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head))
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
                    lds_st(misc, lane, w_sp * (den > 0.0).select(_rcp(den), fx.Float32(0.0)))
                if lane == 0:
                    lds_st(misc, N_SPLIT, m_all)
                    lds_st(misc, N_SPLIT + 1, den)
            gpu.barrier()
            for dh in range_constexpr(2):
                dp = dp_lo + dh * (KV_LORA // 4)
                got = poll(
                    [
                        (mb("sp_acc"), ((s * N_SPLIT + hf * SPH + j) * H + head) * (KV_LORA // 2) + dp, 1)
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
            if tid < KV_LORA // 2:
                o0 = fx.Float32(0.0)
                o1 = fx.Float32(0.0)
                for q in range_constexpr(4):
                    o0 = o0 + lds_ld(red, (q * (KV_LORA // 2) + tid) * 2)
                    o1 = o1 + lds_ld(red, (q * (KV_LORA // 2) + tid) * 2 + 1)
                lds_st(xs, tid, bf16_pair(bf16_round(o0), bf16_round(o1)))
            gpu.barrier()
            acc = run_units(u_uv, UV_UNITS, UV_UNITS, pre)
            reduce_rows(UV_R, acc, emit_out(UV_TILE))
            gpu.barrier()
            if tid < UV_TILE // 4:
                r = tid * 4
                put_bf(mb("o"), s * O_K + t * UV_TILE + r, [lds_ld(outs, r + j) for j in range(4)])

        # ====================== 7. W_o + attention TP peer reduce + residual -> a
        # 2 row groups x 32 chunks: 4 waves per row group, 8 chunks each
        r_wo, r_so = _rsrc(w_o), _rsrc(s_o)
        O_NKC = O_K // 64
        O_R = ROW_TILE // 16
        O_WPR = WAVES // O_R
        O_UNITS = O_NKC // (attention_k_chunks_per_unit * O_WPR)
        for t in range(start("o"), N_ROW_TILES, G):
            t = fx.Int32(t)
            stage("o")

            def u_o(c):
                kc = ((wave % O_WPR) * O_UNITS + c) * attention_k_chunks_per_unit
                return unit_attention(
                    r_wo, r_so, t * O_R + wave // O_WPR, kc, O_NKC, O_K, 128, (n_sel() * O_K + kc * 64) // 2
                )

            pre = [u_o(c) for c in range(O_UNITS)]
            gpu.barrier()
            stage_x_pairs("o", S * O_K, lambda k: k)
            gpu.barrier()
            acc = run_units(u_o, O_UNITS, O_UNITS, pre)
            reduce_rows(O_R, acc, emit_out(ROW_TILE))
            gpu.barrier()

            def resid_h(s, row):
                w = fx.Vector.from_elements(
                    [fx.Int32(bo.buffer_load(r_h, (s * HIDDEN + row) // 2, vec_width=1, dtype=T.i32))], fx.Int32
                )
                v = w.bitcast(fx.BFloat16).to(fx.Float32)
                return v[0], v[1]

            peer_reduce("attn", t, resid_h, lambda s, row, v0, v1: put_bf(mb("a"), s * HIDDEN + row, [v0, v1]))

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
            stage("router")

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
                return unit_bf16(r_wr, t * ROUTER_TILE // 16, kc, R_NKC, (r_ns * HIDDEN + kc * 64) // 2, r_ln)

            pre = [u_r(c) for c in range(R_CPW)]
            gpu.barrier()
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
                specs = [(mb("a"), (router_sample * HIDDEN + k) // 2, 2) for s, k in sks]
                specs.append((mb("a"), (x_s * HIDDEN + xk) // 2, 1))
                v = poll(specs, batch=len(specs))
                xa.append(bf2_f32(v[-1][0]))
                return [list(bf2_f32(w[0])) + list(bf2_f32(w[1])) for w in v[:-1]]

            rstds = stage_x_rmsnorm(ld_a, HIDDEN, g_post, count=1)
            # this task's FP8 activation blocks go out ahead of the gate GEMV
            if x_ok:
                x_rstd = rstds[0]
                a0, a1 = xa[0]
                q0, q1, qs = quant_scaled(a0 * x_rstd * xg[0], a1 * x_rstd * xg[1])
                w8 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
                w8n = _xshfl(w8, 1)
                if lane % 2 == 0:  # FP8 bytes k .. k + 3 in one tagged word
                    put(mb("xq"), (x_s * HIDDEN + xk) // 4, w8 | (w8n << 16))
                d0, d1 = _fp8_roundtrip(q0, q1)
                bo.buffer_store(
                    fx.Vector.from_elements([d0 * qs, d1 * qs], fx.Float32), _rsrc(mb("xqd")), x_s * HIDDEN + xk
                )
                if lane == 0:
                    put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
            gpu.barrier()
            acc = run_units(u_r, R_CPW, R_CPW, pre)
            fx.ptr_store(fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4)
            gpu.barrier()
            if tid < ROUTER_TILE:
                r = tid % ROUTER_TILE
                n = fx.Int32(0)
                logit = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    for f in range_constexpr(2):
                        m = f * ROUTER_TILE + r
                        logit = logit + lds_ld(red, (w * 64 + f * ROUTER_TILE + n + 16 * (m // 4)) * 4 + m % 4)
                put(mb("scores"), router_sample * N_EXPERTS + t * ROUTER_TILE + r, _rcp(1.0 + _exp(-logit)))

        def dn_route(bs):
            """Expert-down routing: one whole wave per sample, in wave-sized batches."""
            for sample_batch in range_constexpr(sample_wave_batches(S)):
                route_sample = wave + sample_batch * WAVES
                if route_sample < S:
                    e, w = route_top8(route_sample, bs=bs)
                    # slot 0: the shared expert, then pick lane (slot lane + 1)
                    if lane < MOE_SLOTS:
                        q = route_sample * MOE_SLOTS + (lane + 1) % MOE_SLOTS
                        lds_st(keys, q, (lane == TOP_K).select(fx.Int32(SHARED_EXPERT), e))
                        lds_st(dnw, q, (lane == TOP_K).select(fx.Float32(1.0), w))

        # ================================ 9. expert up/gate + SiLU
        # 2 row groups (16 gate + 16 up rows) x 96 chunks: 4 waves per group, 24 chunks each
        UG_NKC = HIDDEN // 64
        UG_W_BYTES = expert_inter * HIDDEN
        UG_S_BYTES = 2 * expert_inter * (HIDDEN // 32)

        if const_expr(S == 1):
            # one task per CTA: task u takes intermediates (u % 32) * 8 of routed slot
            # u // 32 (slot 8 for u < 32, which also take the shared expert's); the 8 gate
            # + 8 up rows are one MFMA row group and all waves split K
            UG8 = 8
            UG8_CPW = UG_NKC // WAVES
            for u in range(start("ug"), G, G):
                u = fx.Int32(u)
                stage("ug")
                s_u, c = fx.Int32(0), u % (expert_inter // UG8)
                has_sh = u < expert_inter // UG8
                slot = has_sh.select(fx.Int32(MOE_SLOTS - 1), u // (expert_inter // UG8))
                # the FP8 activation is computed here from the post-attention state (in
                # parallel with the router): RMSNorm, then per-128 quant with one wave per
                # block -> X[0] (fp8 values in bf16), block scales -> misc[8:]
                NB = XQ_BLOCKS // WAVES
                ks_ = [(wave + j * WAVES) * 128 + lane * 2 for j in range(NB)]
                r_gp = _rsrc(g_post)
                # issued ahead of the wait
                gps = [(ld_bf16(r_gp, k), ld_bf16(r_gp, k + 1)) for k in ks_]
                bs = load_bias()
                # MFMA rows 0-7 gate, 8-15 up
                w_rg = ((lane % 16) // 8) * (expert_inter // 16) + c // 2
                w_ln = (lane & -16) | ((c % 2) * 8 + lane % 8)

                # expert e's weights (loads return 0 unless live)
                def u_ug8(cc, e, live=None):
                    nw = None if live is None else live.select(fx.Int32(UG_W_BYTES), fx.Int32(0))
                    ns = None if live is None else live.select(fx.Int32(UG_S_BYTES), fx.Int32(0))
                    r_wug = bo.create_buffer_resource_from_addr(
                        w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES), num_records_bytes=nw
                    )
                    r_sug = bo.create_buffer_resource_from_addr(
                        s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES), num_records_bytes=ns
                    )
                    unit = wave * (UG8_CPW // 2) + cc
                    return unit_mxfp4(
                        r_wug, r_sug, w_rg, unit, HIDDEN, unit * 32, lambda: _uniform_f32(lds_ld(misc, 8 + unit)), w_ln
                    )

                # the shared expert's weights do not depend on routing: prefetch them (the
                # later zero-weight MMAs of the other tasks are cheaper than a branch)
                pre = [u_ug8(cc, fx.Int32(SHARED_EXPERT), has_sh) for cc in range(UG8_CPW // 2)]
                gpu.barrier()
                # the sum of squares takes the router's element partition and order
                # (stage_x_rmsnorm), so rstd -- and every FP8 rounding -- is bit-identical
                NQ4 = HIDDEN // (4 * THREADS)
                got = poll(
                    [(mb("a"), (s_u * HIDDEN + (tid + i * THREADS) * 4) // 2, 2) for i in range(NQ4)]
                    + [(mb("a"), (s_u * HIDDEN + k) // 2, 1) for k in ks_]
                )
                av = [bf2_f32(w[0]) for w in got[NQ4:]]
                ss = fx.Float32(0.0)
                for w in got[:NQ4]:
                    for a in list(bf2_f32(w[0])) + list(bf2_f32(w[1])):
                        ss = ss + a * a
                rstd = _rsq(block_sum(ss) * (1.0 / HIDDEN) + EPS)
                for j in range_constexpr(NB):
                    q0, q1, qs = quant_scaled(av[j][0] * rstd * gps[j][0], av[j][1] * rstd * gps[j][1])
                    st_f8(ks_[j], q0, q1)
                    if lane == 0:
                        lds_st(misc, 8 + wave + j * WAVES, qs)
                if wave == 0:
                    e, w = route_top8(s_u, bs=bs)
                    if lane == slot - 1:
                        lds_st(keys, 0, e)
                        lds_st(misc, 0, w)
                gpu.barrier()
                e_sel = _uniform(lds_ld(keys, 0))
                post = [u_ug8(cc, e_sel) for cc in range(UG8_CPW // 2)]
                reduce_rows(1, mma_units([fx.Float32(0.0) for _ in range(4)], pre), emit_out(16))
                gpu.barrier()
                reduce_rows(
                    1, mma_units([fx.Float32(0.0) for _ in range(4)], post), lambda rl, n, v: lds_st(outs, 16 + rl, v)
                )
                gpu.barrier()
                # threads 0-3: the shared expert's rows, 4-7: the routed slot's
                if tid < UG8:
                    r = (tid % (UG8 // 2)) * 2
                    o = (tid // (UG8 // 2)) * 16
                    g0, g1 = lds_ld(outs, o + r), lds_ld(outs, o + r + 1)
                    u0, u1 = lds_ld(outs, o + UG8 + r), lds_ld(outs, o + UG8 + r + 1)
                    if has_sh | (tid >= UG8 // 2):
                        put2(
                            mb("mid"),
                            (tid < UG8 // 2).select(fx.Int32(0), slot) * expert_inter + c * UG8 + r,
                            g0 * _rcp(1.0 + _exp(-g0)) * u0,
                            g1 * _rcp(1.0 + _exp(-g1)) * u1,
                        )
                if (c == 0) & (tid == 0):  # routing record (debug / tests)
                    put(mb("sel"), slot, e_sel)
                    put(mb("prob"), slot, lds_ld(misc, 0))
                    if has_sh:
                        put(mb("sel"), 0, fx.Int32(SHARED_EXPERT))
                        put(mb("prob"), 0, fx.Float32(1.0))
        else:
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
                c = u % (expert_inter // UG8)
                has_sh = u < expert_inter // UG8
                slot = has_sh.select(fx.Int32(MOE_SLOTS - 1), u // (expert_inter // UG8))
                w_rg = ((lane % 16) // 8) * (expert_inter // 16) + c // 2
                w_ln = (lane & -16) | ((c % 2) * 8 + lane % 8)

                def ug8_units(e, sample, live=None):
                    if const_expr(live is None):
                        rw = _rsrc(w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES))
                        rs = _rsrc(s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES))
                    else:
                        rw = bo.create_buffer_resource_from_addr(
                            w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES),
                            num_records_bytes=live.select(fx.Int32(UG_W_BYTES), fx.Int32(0)),
                        )
                        rs = bo.create_buffer_resource_from_addr(
                            s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES),
                            num_records_bytes=live.select(fx.Int32(UG_S_BYTES), fx.Int32(0)),
                        )
                    sn = n_sel() if sample is None else fx.Int32(sample)
                    units = []
                    for cc in range_constexpr(UG8_UNITS):
                        unit = wave * UG8_UNITS + cc
                        units.append(
                            unit_mxfp4(
                                rw,
                                rs,
                                w_rg,
                                unit,
                                HIDDEN,
                                sn * XW + unit * 32,
                                lambda unit=unit, sn=sn: lds_ld(misc, 8 + sn * XQ_BLOCKS + unit),
                                w_ln,
                            )
                        )
                    return units

                def ug8_emit(sample, shared):
                    if tid < (S if shared else 1) * UG8 // 2:
                        n = tid // (UG8 // 2)
                        r = (tid % (UG8 // 2)) * 2
                        g0, g1 = lds_ld(outs, n * 16 + r), lds_ld(outs, n * 16 + r + 1)
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
                        put(mb("sel"), sn * MOE_SLOTS + sl, lds_ld(keys, sn * MOE_SLOTS + sl))
                        put(mb("prob"), sn * MOE_SLOTS + sl, lds_ld(dnw, sn * MOE_SLOTS + sl))

                shared_pre = ug8_units(fx.Int32(SHARED_EXPERT), None, has_sh)
                cur = ug8_units(_uniform(lds_ld(keys, slot)), 0)
                if has_sh:
                    reduce_rows(1, mma_units([fx.Float32(0.0) for _ in range(4)], shared_pre), emit_out(16))
                    gpu.barrier()
                    ug8_emit(0, True)
                for sample in range_constexpr(S):
                    stage("ug")
                    pre = cur
                    if const_expr(sample + 1 < S):
                        cur = ug8_units(_uniform(lds_ld(keys, (sample + 1) * MOE_SLOTS + slot)), sample + 1)
                    reduce_rows(1, mma_units([fx.Float32(0.0) for _ in range(4)], pre), emit_out(16))
                    gpu.barrier()
                    ug8_emit(sample, False)
                if const_expr(task_round + 1 < ug_task_rounds(expert_inter)):
                    gpu.barrier()

        # ======== 10. mid FP8 quant + expert down + route weighting + MoE TP reduce
        # 2 row groups x (sample tile * 9 slots * 4) chunks: 4 waves per group.
        # S=8 is evaluated as two four-sample groups so its FP8 mid tile and
        # in-flight weight batches fit comfortably in LDS/VGPRs.
        DN_NKC = expert_inter // 64
        # 16-row groups touched by a tile (24-row tiles start at row 0 or 8 of one)
        DN_R = (DN_TILE + 15) // 16
        DN_WPR = WAVES // DN_R
        # units per software-pipelined down batch; rounds = ceil(units per wave / batch)
        DN_BATCH = 4 if S > 4 else 9
        DN_W_BYTES = HIDDEN * expert_inter // 2
        DN_S_BYTES = HIDDEN * (expert_inter // 32)
        for t in range(start("down"), N_DN_TILES, G):
            t = fx.Int32(t)
            stage("down")
            # else routed before up/gate
            if const_expr(ug_split(S, expert_inter) is None):
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
                e = _uniform(lds_ld(keys, s_q * MOE_SLOTS + slot_q))
                wb = bo.create_buffer_resource_from_addr(
                    w_dn + fx.Int64(e) * fx.Int64(DN_W_BYTES),
                    num_records_bytes=(None if DN_NU % DN_WPR == 0 else live.select(fx.Int32(DN_W_BYTES), fx.Int32(0))),
                )
                sb = bo.create_buffer_resource_from_addr(
                    s_dn + fx.Int64(e) * fx.Int64(DN_S_BYTES),
                    num_records_bytes=(None if DN_NU % DN_WPR == 0 else live.select(fx.Int32(DN_S_BYTES), fx.Int32(0))),
                )

                def coef():  # route weight, only in this sample's column
                    return (lane % 16 == s_q).select(
                        _uniform_f32(lds_ld(dnw, s_q * MOE_SLOTS + slot_q)), fx.Float32(0.0)
                    )

                return unit_mxfp4_bf16(
                    wb, sb, dn_rg + gu, unit % (expert_inter // 128), expert_inter, unit * 64, coef, dn_ln
                )

            if const_expr(S <= 4):
                pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
            stage("down")  # dn_route left "ug"
            gpu.barrier()
            mids = get2_many(
                [
                    (mb("mid"), fx.min(wave + b * WAVES, DN_BLK - 1) * 128 + lane * 2)
                    for b in range((DN_BLK + WAVES - 1) // WAVES)
                ]
            )
            for b in range_constexpr((DN_BLK + WAVES - 1) // WAVES):
                blk = wave + b * WAVES
                if blk < DN_BLK:
                    lds_st(xs, blk * 64 + lane, bf16_pair(mids[b][0], mids[b][1]))
            gpu.barrier()
            if const_expr(S > 4):
                pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
            acc = run_units(u_dn, DN_UPW, DN_BATCH, pre)

            def emit_dn(rl, n, v):
                if (rl >= dn_off) & (rl < dn_off + DN_TILE):
                    lds_st(outs, n * DN_TILE + rl - dn_off, v)

            reduce_rows(DN_R, acc, emit_dn)
            gpu.barrier()

            def store_x(s, row, v0, v1):
                bo.buffer_store(
                    fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16), _rsrc(x_out), s * HIDDEN + row
                )

            def residual_a(s, row):
                return bf2_f32(get(mb("a"), (s * HIDDEN + row) // 2))

            peer_reduce("ffn", t, residual_a, store_x, tile=DN_TILE)
            gpu.barrier()

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
            rank,
            layer,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    return launch
