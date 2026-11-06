# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark paged decode attention of the 8-token verify step, fused with the
per-head output RMS norm (no weight) and the sigmoid gate, as one CuTe kernel.

head_dim 64, 8 query heads per KV head, BF16 paged cache [pages, HKV, 16, 128]
(K in columns 0:64, V in 64:128), 8 query rows per request (the last 8 cache
entries of the request), any context length L >= 8, optionally a sliding
window. One launch covers the rank's KV heads (grid y); the heads are
independent.

Two variants, picked on the host from the deployment's maximum length:

- One cluster (max_seq_len <= 2048, no window): a cluster of 16 CTAs of 4
  warps owns 16 query-head rows of one request and one KV head. Warp w of CTA
  r attends pages 4 r + w + 64 i (the next page's registers load while one is
  attended) with an online softmax: mma.sync m16n8k16 BF16 -> FP32 for Q K^T
  and P V, the softmax weights rounded to BF16, V transposed in registers by
  movmatrix (head dims permuted: logical (j, 2 t + e) = 16 t + 2 j + e). The
  warps merge in shared memory, each CTA pushes its merged rows to the CTA
  that owns them over distributed shared memory (st.async, mbarrier
  transaction bytes), and the owner merges the 16 CTAs in rank order, rounds
  o / l to BF16 and applies the RMS norm and the gate.
- Split (longer contexts or a window): a request within 8 rounds of the
  cluster's warps (512 pages from row 0's first visible key) takes the path
  above in z slice 0, the other slices exit at once; a longer one spreads its
  pages over 4 num_splits clusters, each CTA streaming its pages through an
  8-page cp.async ring, merges in the cluster over distributed shared memory
  and across clusters through a global workspace in cluster order (an arrival
  counter per row: the last arriver merges, then resets the counter to zero).
  The launch shape depends on the deployment only, so a captured graph serves
  every step. With window W, row t sees keys [L - 8 + t - W + 1, L - 8 + t].

Every FP32 operation is explicit PTX in the fast-math forms (subnormals
flushed, ex2.approx, rcp.approx, rsqrt.approx) in a fixed order, so the result
depends neither on the launch's timing nor on the device. Not bitwise to the
stock decode + rmsnorm_sigmoid_gate pair: the softmax weights are rounded to
BF16 before P V and the partials merge in FP32.
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, const_expr
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from ..common.host import assert_not_capturing, fp32
from ..common.primitives import (
    add_ftz_f32,
    atomic_add1_u32_address,
    atomic_exch_zero_u32_address,
    bf16x2_to_f32x2,
    cluster_arrive_relaxed_aligned,
    cluster_wait_aligned,
    cp_async_cg_16,
    cp_async_commit,
    cp_async_wait,
    ex2_approx_ftz,
    fence_mbarrier_init_cluster,
    fma_ftz_f32,
    load_global_cg_f32_address,
    load_global_cg_f32x2_address,
    load_global_nc_s32_or,
    load_global_nc_u32x4_address,
    load_global_u32_address,
    load_shared_f32_address,
    load_shared_f32_or,
    load_shared_f32x2_address,
    load_shared_f32x4_address,
    load_shared_u32x4_unpacked,
    map_shared_address_to_peer,
    max_ftz_f32,
    mbarrier_arrive_expect_tx_address,
    mbarrier_init_address,
    mbarrier_try_wait_phase0_cluster,
    membar_gl,
    mma_m16n8k16_bf16,
    movmatrix_trans_b16,
    mul_ftz_f32,
    pack_bf16x2_rn,
    rcp_approx_ftz,
    round_bf16,
    rsqrt_approx_ftz,
    scale_or_neg_inf,
    shuffle_sync_idx_u32,
    store_async_cluster_f32x2,
    store_async_cluster_f32x4,
    store_global_f32_address,
    store_global_f32x2_address,
    store_global_u32_address,
    store_shared_f32_address,
    store_shared_f32x4_address,
    sub_ftz_f32,
)

HEAD_DIM = 64
PAGE = 16  # cache slots per page
SLOT_ELEMS = 2 * HEAD_DIM  # BF16 per cache slot: K then V
HEADS = 8  # query heads per KV head
QUERY_ROWS = 8  # tokens per request (1 + 7 drafts)
ROWS = 16  # query-head rows per cluster
CLUSTER = 16  # CTAs per cluster, above the portable 8
WARPS = 4
THREADS = 32 * WARPS
GROUPS = QUERY_ROWS * HEADS // ROWS  # clusters per (request, KV head) and z slice
CLUSTER_WARPS = CLUSTER * WARPS
MAX_SPLITS = 64
# Contexts one cluster takes in the one-cluster kernel (two rounds of its warps).
SHORT_MAX_LEN = 2 * CLUSTER_WARPS * PAGE
# The split kernel's short-path limit in rounds of a cluster's warps (512
# pages, 8192 tokens); a 2048-token window (at most 130 pages) stays below it.
SHORT_ROUNDS = 8
# Pages of the long-context path per CTA at least (below, fewer clusters).
MIN_CTA_PAGES = 4
# The last arriver's cross-cluster merge of a row's partials: every load at
# once ("all"), the m of each at once then a loop ("m", up to MERGE_M_MAX
# partials), or a loop ("loop"). split_forms picks the merge form and whether
# the short-path loop is rolled from the KV-head count and the z slices; the
# forms change the schedule only, not the operations or their order.
MERGE_ALL_MAX = 8
MERGE_M_MAX = 16


# The long-context path's online-softmax step (pages) and the steps its page
# ring holds (one in flight while one is attended).
GROUP_PAGES = 4
RING_GROUPS = 2
RING_STAGES = GROUP_PAGES * RING_GROUPS
PAGE_BYTES = PAGE * SLOT_ELEMS * 2
# One exchanged partial row: 64 o, m, l and 2 pad floats; the 66 floats
# before the pad are pushed.
SLOT = 68
SLOT_BYTES = 4 * SLOT
PUSH_BYTES = 4 * (HEAD_DIM + 2)
# The layer's softmax scale (head_dim ** -0.5 * qk_scale_factor) and RMS-norm
# eps: constants of the one Muse Spark config served (the runtime refuses a
# layer with other values).
SM_SCALE = 0.6841258108
EPS = 1e-5
LOG2E = 1.4426950408889634
NEG_BIG = -1e30
NEG_INF = float("-inf")

# Shared memory (bytes): every CTA's warp partials [WARPS][ROWS][SLOT], row
# maxima and sums [WARPS][ROWS], the partials pushed to its own rows
# [CLUSTER][SLOT] and the mbarrier counting them; the long-context path's own
# mbarrier, page ring [RING_STAGES][PAGE][SLOT_ELEMS] and pushed partials
# [CLUSTER][WARPS][SLOT].
S_O = 0
S_M = S_O + WARPS * ROWS * SLOT_BYTES
S_L = S_M + WARPS * ROWS * 4
S_RECV = S_L + WARPS * ROWS * 4
S_MBAR = S_RECV + CLUSTER * SLOT_BYTES
S_BAR = S_MBAR + 8
SMEM_ONE_CLUSTER = S_BAR
S_RING = (S_BAR + 8 + 127) // 128 * 128
S_LONG_RECV = S_RING + RING_STAGES * PAGE_BYTES
SMEM_SPLIT = S_LONG_RECV + CLUSTER * WARPS * SLOT_BYTES

assert QUERY_ROWS * HEADS == GROUPS * ROWS and CLUSTER == ROWS
assert WARPS * 32 * 2 * 16 == PAGE_BYTES  # two 16-byte copies a thread per page


# The kernel's scale of the scores, in log2 units, and eps (FP32 as the
# kernel reads them).
SCALE_LOG2 = fp32(fp32(SM_SCALE) * fp32(LOG2E))
EPS_F32 = fp32(EPS)


def _qf(qr) -> list:
    """The four A fragments of the queries, from a lane's 16-byte loads of
    rows g (qr[0], qr[1]) and g + 8 (qr[2], qr[3])."""
    w0 = (*qr[0], *qr[1])
    w1 = (*qr[2], *qr[3])
    return [(w0[2 * s], w1[2 * s], w0[2 * s + 1], w1[2 * s + 1]) for s in range(4)]


def _store_qf(frag: cute.Tensor, qf) -> None:
    for s in range(4):
        for i in range(4):
            frag[4 * s + i] = qf[s][i]


def _read_qf(frag: cute.Tensor) -> list:
    return [tuple(frag[4 * s + i] for i in range(4)) for s in range(4)]


def _load_rows(q_base, q_stride, token0, head0, token1, head1, t):
    """A lane's query dims [16 t, 16 t + 16) of two query-head rows."""
    lane_dims = Int64(16 * t)
    q0 = q_base + (Int64(token0) * q_stride + Int64(head0 * HEAD_DIM) + lane_dims) * 2
    q1 = q_base + (Int64(token1) * q_stride + Int64(head1 * HEAD_DIM) + lane_dims) * 2
    return (
        load_global_nc_u32x4_address(q0),
        load_global_nc_u32x4_address(q0 + 16),
        load_global_nc_u32x4_address(q1),
        load_global_nc_u32x4_address(q1 + 16),
    )


def _load_page(regs: cute.Tensor, kv_base, page_stride, entry, g, t) -> None:
    """One KV page as a lane holds it: words 8 j .. 8 j + 7 the key slot
    8 j + g, words 16 + 8 j .. the value slot 8 j + g, head dims
    [16 t, 16 t + 16)."""
    base = kv_base + (Int64(entry) * page_stride + Int64(16 * t)) * 2
    for j in range(2):
        slot = base + Int64((8 * j + g) * SLOT_ELEMS * 2)
        for half, offset in enumerate((0, 16)):
            words = load_global_nc_u32x4_address(slot + offset)
            for w in range(4):
                regs[8 * j + 4 * half + w] = words[w]
            words = load_global_nc_u32x4_address(slot + 2 * HEAD_DIM + offset)
            for w in range(4):
                regs[16 + 8 * j + 4 * half + w] = words[w]


def _scores(kw, qf) -> list:
    """Q K^T of a lane's rows g, g + 8 and keys 2 t, 2 t + 1 of eight slots
    (kw: their key words): (g, k0), (g, k1), (g + 8, k0), (g + 8, k1)."""
    zero = (Float32(0.0),) * 4
    sa = mma_m16n8k16_bf16(qf[0], kw[0], kw[1], zero, ordered=True)
    sb = mma_m16n8k16_bf16(qf[2], kw[4], kw[5], zero, ordered=True)
    sa = mma_m16n8k16_bf16(qf[1], kw[2], kw[3], sa, ordered=True)
    sb = mma_m16n8k16_bf16(qf[3], kw[6], kw[7], sb, ordered=True)
    return [add_ftz_f32(sa[e], sb[e]) for e in range(4)]


def _online_step(s0: list, s1: list, state, o: cute.Tensor):
    """One online-softmax step of rows g (scores s0) and g + 8 (s1): rescales
    o, returns the weights and the new (m0, m1, l0, l1)."""
    m0, m1, l0, l1 = state
    mx0 = Float32(NEG_INF)
    mx1 = Float32(NEG_INF)
    for x0, x1 in zip(s0, s1):
        mx0 = max_ftz_f32(mx0, x0)
        mx1 = max_ftz_f32(mx1, x1)
    for offset in (1, 2):
        mx0 = max_ftz_f32(mx0, cute.arch.shuffle_sync_bfly(mx0, offset=offset))
        mx1 = max_ftz_f32(mx1, cute.arch.shuffle_sync_bfly(mx1, offset=offset))
    mn0 = max_ftz_f32(m0, mx0)
    mn1 = max_ftz_f32(m1, mx1)
    a0 = ex2_approx_ftz(sub_ftz_f32(m0, mn0))
    a1 = ex2_approx_ftz(sub_ftz_f32(m1, mn1))
    ps0 = Float32(0.0)
    ps1 = Float32(0.0)
    p0, p1 = [], []
    for x0, x1 in zip(s0, s1):
        p0.append(ex2_approx_ftz(sub_ftz_f32(x0, mn0)))
        p1.append(ex2_approx_ftz(sub_ftz_f32(x1, mn1)))
        ps0 = add_ftz_f32(ps0, p0[-1])
        ps1 = add_ftz_f32(ps1, p1[-1])
    l0 = fma_ftz_f32(l0, a0, ps0)
    l1 = fma_ftz_f32(l1, a1, ps1)
    for j in range(8):
        o[4 * j] = mul_ftz_f32(a0, o[4 * j])
        o[4 * j + 1] = mul_ftz_f32(a0, o[4 * j + 1])
        o[4 * j + 2] = mul_ftz_f32(a1, o[4 * j + 2])
        o[4 * j + 3] = mul_ftz_f32(a1, o[4 * j + 3])
    return p0, p1, (mn0, mn1, l0, l1)


def _pv(o: cute.Tensor, p0: list, p1: list, vlo, vhi) -> None:
    """O += P V for one page: p0 / p1 the BF16-rounded weights of rows g and
    g + 8 (keys 2 t, 2 t + 1, 8 + 2 t, 9 + 2 t), vlo / vhi the value words of
    slots g and 8 + g."""
    pa = (
        pack_bf16x2_rn(p0[0], p0[1]),
        pack_bf16x2_rn(p1[0], p1[1]),
        pack_bf16x2_rn(p0[2], p0[3]),
        pack_bf16x2_rn(p1[2], p1[3]),
    )
    for j in range(8):
        acc = mma_m16n8k16_bf16(
            pa,
            movmatrix_trans_b16(vlo[j]),
            movmatrix_trans_b16(vhi[j]),
            tuple(o[4 * j + i] for i in range(4)),
            ordered=True,
        )
        for i in range(4):
            o[4 * j + i] = acc[i]


def _attend_page(cur, qf, pos, lim0, lim1, low0, low1, sl2, state, o, windowed):
    """One page (16 keys) of online-softmax attention of the warp's rows g and
    g + 8 from registers; pos is the lane's first key, a row sees keys
    low <= key <= lim."""
    s0, s1 = [], []
    for j in range(2):
        s = _scores([cur[8 * j + w] for w in range(8)], qf)
        for e in range(2):
            key = pos + 8 * j + e
            s0.append(
                scale_or_neg_inf(s[e], sl2, key, lim0, low0 if windowed else None)
            )
            s1.append(
                scale_or_neg_inf(s[2 + e], sl2, key, lim1, low1 if windowed else None)
            )
    p0, p1, state = _online_step(s0, s1, state, o)
    _pv(o, p0, p1, [cur[16 + w] for w in range(8)], [cur[24 + w] for w in range(8)])
    return state


def _stash_partial(so, sm, sl, g, t, o, m0, m1, l0, l1) -> None:
    """A warp's softmax state (lane: dims [16 t, 16 t + 16) of rows g and
    g + 8) as rows of logical dims at ``so``, with the row maxima at ``sm``
    and sums at ``sl``."""
    for offset in (1, 2):
        l0 = add_ftz_f32(l0, cute.arch.shuffle_sync_bfly(l0, offset=offset))
        l1 = add_ftz_f32(l1, cute.arch.shuffle_sync_bfly(l1, offset=offset))
    r0 = so + g * SLOT_BYTES + 64 * t
    r1 = r0 + 8 * SLOT_BYTES
    for q in range(4):
        store_shared_f32x4_address(
            r0 + 16 * q, (o[8 * q], o[8 * q + 1], o[8 * q + 4], o[8 * q + 5])
        )
        store_shared_f32x4_address(
            r1 + 16 * q, (o[8 * q + 2], o[8 * q + 3], o[8 * q + 6], o[8 * q + 7])
        )
    first = Int32(t == 0)
    store_shared_f32_address(sm + 4 * g, m0, first)
    store_shared_f32_address(sl + 4 * g, l0, first)
    store_shared_f32_address(sm + 4 * (g + 8), m1, first)
    store_shared_f32_address(sl + 4 * (g + 8), l1, first)


def _merge_warps_and_push(smem, mbar, tid, rank) -> None:
    """Merge the CTA's four warp partials (8 lanes a row, lane i8 dims
    [4 i8, 4 i8 + 4) and [32 + 4 i8, ...)) and push row r to CTA r."""
    row = tid >> 3
    i8 = tid & 7
    ms = [load_shared_f32_address(smem + S_M + 64 * w + 4 * row) for w in range(WARPS)]
    big = Float32(NEG_BIG)
    m = big
    for w in range(WARPS):
        m = max_ftz_f32(m, ms[w])
    ls = Float32(0.0)
    xa = [Float32(0.0)] * 4
    xb = [Float32(0.0)] * 4
    for w in range(WARPS):
        sc = ex2_approx_ftz(sub_ftz_f32(ms[w], m))
        ls = fma_ftz_f32(sc, load_shared_f32_address(smem + S_L + 64 * w + 4 * row), ls)
        src = smem + S_O + w * ROWS * SLOT_BYTES + row * SLOT_BYTES + 16 * i8
        a = load_shared_f32x4_address(src)
        c = load_shared_f32x4_address(src + 128)
        xa = [fma_ftz_f32(sc, a[i], xa[i]) for i in range(4)]
        xb = [fma_ftz_f32(sc, c[i], xb[i]) for i in range(4)]
    dst = map_shared_address_to_peer(smem + S_RECV + rank * SLOT_BYTES + 16 * i8, row)
    remote_bar = map_shared_address_to_peer(mbar, row)
    store_async_cluster_f32x4(dst, xa, remote_bar)
    store_async_cluster_f32x4(dst + 128, xb, remote_bar)
    store_async_cluster_f32x2(
        dst + 256, m, ls, remote_bar, Int32(i8 == 0), clobber_memory=True
    )


def _finish_row(ox, oy, ls, gate2, eps, out_row, lane) -> None:
    """The epilogue of one row (lane: dims 2 lane, 2 lane + 1): o / l rounded
    to BF16, RMS norm without weight, sigmoid gate, BF16 store."""
    gx, gy = bf16x2_to_f32x2(gate2)
    neg_log2e = Float32(-LOG2E)
    one = Float32(1.0)
    sgx = rcp_approx_ftz(add_ftz_f32(ex2_approx_ftz(mul_ftz_f32(gx, neg_log2e)), one))
    sgy = rcp_approx_ftz(add_ftz_f32(ex2_approx_ftz(mul_ftz_f32(gy, neg_log2e)), one))
    inv = rcp_approx_ftz(ls)
    ox = round_bf16(mul_ftz_f32(ox, inv))
    oy = round_bf16(mul_ftz_f32(inv, oy))
    ss = fma_ftz_f32(ox, ox, mul_ftz_f32(oy, oy))
    for offset in (16, 8, 4, 2, 1):
        ss = add_ftz_f32(ss, cute.arch.shuffle_sync_bfly(ss, offset=offset))
    rinv = rsqrt_approx_ftz(fma_ftz_f32(ss, Float32(1.0 / HEAD_DIM), eps))
    cute.arch.griddepcontrol_launch_dependents()
    x = mul_ftz_f32(mul_ftz_f32(ox, rinv), sgx)
    y = mul_ftz_f32(sgy, mul_ftz_f32(oy, rinv))
    store_global_u32_address(out_row + 4 * Int64(lane), pack_bf16x2_rn(x, y))


@cute.jit
def _wait_phase0(mbar: Int32) -> None:
    """Spin until the mbarrier's phase 0 completes (cluster-scope acquire),
    one bounded wait per iteration."""
    done = Int32(0)
    while done == Int32(0):
        done = mbarrier_try_wait_phase0_cluster(mbar)


@cute.jit
def _merge_cluster_one(recv: Int32, n_active: Int32, lane: Int32):
    """The owner's merge of its row's CLUSTER partials at ``recv`` in rank
    order (ranks below ``n_active`` hold pages): (l, o[2 lane], o[2 lane + 1])."""
    big = Float32(NEG_BIG)
    mk = [
        load_shared_f32_or(recv + k * SLOT_BYTES + 256, Int32(k < n_active), big)
        for k in range(CLUSTER)
    ]
    m = big
    for k in cutlass.range_constexpr(CLUSTER):
        m = max_ftz_f32(m, mk[k])
    ls = Float32(0.0)
    ox = Float32(0.0)
    oy = Float32(0.0)
    for k in cutlass.range_constexpr(CLUSTER):
        if k < n_active:
            sc = ex2_approx_ftz(sub_ftz_f32(mk[k], m))
            ls = fma_ftz_f32(
                sc, load_shared_f32_address(recv + k * SLOT_BYTES + 260), ls
            )
            vx, vy = load_shared_f32x2_address(recv + k * SLOT_BYTES + 8 * lane)
            ox = fma_ftz_f32(sc, vx, ox)
            oy = fma_ftz_f32(sc, vy, oy)
    return ls, ox, oy


@cute.jit
def _page_round(
    cur: cute.Tensor,
    nxt: cute.Tensor,
    pg: Int32,
    qf_frag: cute.Tensor,
    num_pages: Int32,
    first_page: Int32,
    gw: Int32,
    e_second: Int32,
    bt_row: Int64,
    kv_base: Int64,
    kv_page_stride: Int64,
    g: Int32,
    t: Int32,
    lim0: Int32,
    lim1: Int32,
    low0: Int32,
    low1: Int32,
    sl2: Float32,
    m0: Float32,
    m1: Float32,
    l0: Float32,
    l1: Float32,
    o: cute.Tensor,
    windowed: cutlass.Constexpr[bool],
):
    """One round of the warp's page loop: page pg (in ``cur``) attended while
    page pg + 64 loads into ``nxt``, which then becomes ``cur``."""
    pn = pg + CLUSTER_WARPS
    if pn < num_pages:
        entry = load_global_nc_s32_or(
            bt_row + Int64(first_page + pn) * 4, Int32(pg != gw), e_second
        )
        _load_page(nxt, kv_base, kv_page_stride, entry, g, t)
    state = _attend_page(
        cur,
        _read_qf(qf_frag),
        (first_page + pg) * PAGE + 2 * t,
        lim0,
        lim1,
        low0,
        low1,
        sl2,
        (m0, m1, l0, l1),
        o,
        windowed,
    )
    for i in cutlass.range_constexpr(32):
        cur[i] = nxt[i]
    return state


@cute.jit
def _short_path(
    q_base: Int64,
    q_stride: Int64,
    gate_base: Int64,
    gate_stride: Int64,
    kv_base: Int64,
    kv_page_stride: Int64,
    bt_row: Int64,
    out_base: Int64,
    out_stride: Int64,
    sl2: Float32,
    eps: Float32,
    window: cutlass.Constexpr[int],
    smem: Int32,
    seq_len: Int32,
    b: Int32,
    rg: Int32,
    rank: Int32,
    warp: Int32,
    lane: Int32,
    tid: Int32,
    first_page: Int32,
    num_pages: Int32,
    e_first: Int32,
    e_second: Int32,
    split: cutlass.Constexpr[bool],
    rolled: cutlass.Constexpr[bool],
    rounds: cutlass.Constexpr[int],
) -> None:
    """The one-cluster computation of the cluster's 16 rows: pages
    ``first_page`` + [0, num_pages) of the request (their first two
    block-table entries of this warp read: e_first, e_second), at most two
    rounds of the cluster's warps in the one-cluster kernel, SHORT_ROUNDS in
    the split kernel (``rounds`` > 0: a window's at most ``rounds`` rounds,
    straight-line; else a loop the compiler unrolls, or a rolled loop in the
    one-cluster kernel or with ``rolled``)."""
    windowed = window > 0
    g = lane >> 2
    t = lane & 3
    gw = rank * WARPS + warp
    r0 = rg * ROWS + g
    r1 = r0 + 8
    token0 = b * QUERY_ROWS + (r0 >> 3)
    token1 = b * QUERY_ROWS + (r1 >> 3)
    qf_frag = cute.make_rmem_tensor(cute.make_layout((16,)), Uint32)
    _store_qf(
        qf_frag, _qf(_load_rows(q_base, q_stride, token0, r0 & 7, token1, r1 & 7, t))
    )
    # The owned row: warp 0 of CTA r owns row r of the cluster's 16.
    owned = rg * ROWS + warp * CLUSTER + rank
    out_token = Int64(b * QUERY_ROWS + (owned >> 3))
    lane_cols = Int64(HEAD_DIM * (owned & 7) + 2 * lane)
    gate2 = load_global_u32_address(
        gate_base + (out_token * gate_stride + lane_cols) * 2,
        Int32(warp == Int32(0)),
        ordered=True,
    )

    mbar = smem + S_MBAR
    if tid == Int32(0):
        mbarrier_init_address(mbar, 1, clobber_memory=True)
        fence_mbarrier_init_cluster()
    cluster_arrive_relaxed_aligned()

    n_active = cutlass.min(Int32(CLUSTER), (num_pages + (WARPS - 1)) >> 2)
    if tid == Int32(0):
        # The pushes of the active CTAs complete this many bytes (the
        # transaction count may go negative before this arrive).
        mbarrier_arrive_expect_tx_address(
            mbar, n_active * PUSH_BYTES, clobber_memory=True
        )
    lim0 = seq_len - QUERY_ROWS + (r0 >> 3)
    lim1 = seq_len - QUERY_ROWS + (r1 >> 3)
    # The first visible key of rows g and g + 8 (the window's; 0 without).
    low0 = Int32(0)
    low1 = Int32(0)
    if const_expr(windowed):
        low0 = lim0 - window + 1
        low1 = lim1 - window + 1

    o = cute.make_rmem_tensor(cute.make_layout((32,)), Float32)
    for i in cutlass.range_constexpr(32):
        o[i] = Float32(0.0)
    cur = cute.make_rmem_tensor(cute.make_layout((32,)), Uint32)
    nxt = cute.make_rmem_tensor(cute.make_layout((32,)), Uint32)
    m0 = Float32(NEG_BIG)
    m1 = Float32(NEG_BIG)
    l0 = Float32(0.0)
    l1 = Float32(0.0)
    if gw < num_pages:
        _load_page(cur, kv_base, kv_page_stride, e_first, g, t)
    round_args = (
        qf_frag,
        num_pages,
        first_page,
        gw,
        e_second,
        bt_row,
        kv_base,
        kv_page_stride,
        g,
        t,
        lim0,
        lim1,
        low0,
        low1,
        sl2,
    )
    # Pages gw + 64 i, the next page loading while one is attended.
    if const_expr(split and rounds > 0):
        bufs = (cur, nxt)
        for r in cutlass.range_constexpr(rounds):
            pg = gw + r * CLUSTER_WARPS
            if pg < num_pages:
                if const_expr(r + 1 < rounds):
                    pn = pg + CLUSTER_WARPS
                    if pn < num_pages:
                        entry = e_second
                        if const_expr(r > 0):
                            entry = load_global_nc_s32_or(
                                bt_row + Int64(first_page + pn) * 4, Int32(1), Int32(0)
                            )
                        _load_page(
                            bufs[1 - r % 2], kv_base, kv_page_stride, entry, g, t
                        )
                m0, m1, l0, l1 = _attend_page(
                    bufs[r % 2],
                    _read_qf(qf_frag),
                    (first_page + pg) * PAGE + 2 * t,
                    lim0,
                    lim1,
                    low0,
                    low1,
                    sl2,
                    (m0, m1, l0, l1),
                    o,
                    windowed,
                )
    elif const_expr(split and not rolled):
        # Up to SHORT_ROUNDS rounds: a loop the compiler unrolls.
        pg = gw
        it = Int32(0)
        while pg < num_pages:
            pn = pg + CLUSTER_WARPS
            if pn < num_pages:
                entry = load_global_nc_s32_or(
                    bt_row + Int64(first_page + pn) * 4,
                    Int32(it != Int32(0)),
                    e_second,
                )
                _load_page(nxt, kv_base, kv_page_stride, entry, g, t)
            m0, m1, l0, l1 = _attend_page(
                cur,
                _read_qf(qf_frag),
                (first_page + pg) * PAGE + 2 * t,
                lim0,
                lim1,
                low0,
                low1,
                sl2,
                (m0, m1, l0, l1),
                o,
                windowed,
            )
            for i in cutlass.range_constexpr(32):
                cur[i] = nxt[i]
            pg = pn
            it = it + 1
    else:
        # A rolled loop.
        for pg in cutlass.range(gw, num_pages, CLUSTER_WARPS, unroll=1):
            m0, m1, l0, l1 = _page_round(
                cur, nxt, pg, *round_args, m0, m1, l0, l1, o, windowed
            )

    if rank < n_active:
        so = smem + S_O + warp * ROWS * SLOT_BYTES
        _stash_partial(
            so, smem + S_M + 64 * warp, smem + S_L + 64 * warp, g, t, o, m0, m1, l0, l1
        )
        cute.arch.barrier()
        # The peers' mbarriers are initialized.
        cluster_wait_aligned()
        _merge_warps_and_push(smem, mbar, tid, rank)
    else:
        cluster_wait_aligned()

    if warp == Int32(0):
        _wait_phase0(mbar)
        ls, ox, oy = _merge_cluster_one(smem + S_RECV, n_active, lane)
        out_row = (
            out_base + (out_token * out_stride + Int64(HEAD_DIM * (owned & 7))) * 2
        )
        _finish_row(ox, oy, ls, gate2, eps, out_row, lane)


def _issue_step(ring, step, count, first, bt_row, kv_base, kv_page_stride, tid):
    """Copy the pages of ring step ``step`` (pages first + 4 step + [0, 4) of
    this CTA's ``count``; none past it) into ring slots 4 (step % 2) + [0, 4):
    thread i the 16-byte chunks i and i + 128 of each page. The block-table
    entries load first, then the copies issue (traced inline)."""
    pages = [step * GROUP_PAGES + pi for pi in range(GROUP_PAGES)]
    entries = [
        load_global_nc_s32_or(bt_row + Int64(first + i) * 4, Int32(i < count), Int32(0))
        for i in pages
    ]
    for pi, (i, entry) in enumerate(zip(pages, entries)):
        src = kv_base + Int64(entry) * kv_page_stride * 2
        dst = ring + ((step & (RING_GROUPS - 1)) * GROUP_PAGES + pi) * PAGE_BYTES
        keep = Int32(i < count)
        cp_async_cg_16(dst + tid * 16, src + Int64(tid * 16), keep)
        cp_async_cg_16(
            dst + (tid + THREADS) * 16, src + Int64((tid + THREADS) * 16), keep
        )


def _ring_scores(s: cute.Tensor, pi: int, page, pos, g, t, qf, lims, sl2) -> None:
    lim0, lim1, low0, low1 = lims
    for j in range(2):
        key_words = page + ((8 * j + g) * SLOT_ELEMS + 16 * t) * 2
        sc = _scores(
            (
                *load_shared_u32x4_unpacked(key_words),
                *load_shared_u32x4_unpacked(key_words + 16),
            ),
            qf,
        )
        for e in range(2):
            key = pos + 8 * j + e
            s[8 * pi + 4 * j + e] = scale_or_neg_inf(sc[e], sl2, key, lim0, low0)
            s[8 * pi + 4 * j + 2 + e] = scale_or_neg_inf(
                sc[2 + e], sl2, key, lim1, low1
            )


def _ring_pv(o: cute.Tensor, s: cute.Tensor, pi: int, page, g, t) -> None:
    v0 = page + (g * SLOT_ELEMS + HEAD_DIM + 16 * t) * 2
    v1 = v0 + 8 * SLOT_ELEMS * 2
    vlo = (*load_shared_u32x4_unpacked(v0), *load_shared_u32x4_unpacked(v0 + 16))
    vhi = (*load_shared_u32x4_unpacked(v1), *load_shared_u32x4_unpacked(v1 + 16))
    p0 = [s[8 * pi + 4 * j + e] for j in range(2) for e in range(2)]
    p1 = [s[8 * pi + 4 * j + 2 + e] for j in range(2) for e in range(2)]
    _pv(o, p0, p1, vlo, vhi)


@cute.jit
def _attend_group(
    group: Int32,
    npages: Int32,
    page0: Int32,
    qf_frag: cute.Tensor,
    g: Int32,
    t: Int32,
    lim0: Int32,
    lim1: Int32,
    low0: Int32,
    low1: Int32,
    sl2: Float32,
    m0: Float32,
    m1: Float32,
    l0: Float32,
    l1: Float32,
    o: cute.Tensor,
    s: cute.Tensor,
    windowed: cutlass.Constexpr[bool],
):
    """``_attend_page`` for up to GROUP_PAGES pages staged in the ring (the
    slots from shared address ``group``, pages page0 + p) in one
    online-softmax step."""
    qf = _read_qf(qf_frag)
    for i in cutlass.range_constexpr(GROUP_PAGES * 8):
        s[i] = Float32(NEG_INF)
    for pi in cutlass.range_constexpr(GROUP_PAGES):
        if pi < npages:
            page = group + pi * PAGE_BYTES
            _ring_scores(
                s,
                pi,
                page,
                (page0 + pi) * PAGE + 2 * t,
                g,
                t,
                qf,
                (lim0, lim1, low0 if windowed else None, low1 if windowed else None),
                sl2,
            )
    order = [
        (pi, j, e) for pi in range(GROUP_PAGES) for j in range(2) for e in range(2)
    ]
    p0, p1, state = _online_step(
        [s[8 * pi + 4 * j + e] for pi, j, e in order],
        [s[8 * pi + 4 * j + 2 + e] for pi, j, e in order],
        (m0, m1, l0, l1),
        o,
    )
    for k, (pi, j, e) in enumerate(order):
        s[8 * pi + 4 * j + e] = p0[k]
        s[8 * pi + 4 * j + 2 + e] = p1[k]
    for pi in cutlass.range_constexpr(GROUP_PAGES):
        if pi < npages:
            _ring_pv(o, s, pi, group + pi * PAGE_BYTES, g, t)
    return state


def _push_rows(smem, mbar, warp, lane, rank) -> None:
    """The long path's push: warp w's row r (8 lanes a row, 4 rows a lane
    group) to CTA r."""
    i8 = lane & 7
    so = smem + S_O + warp * ROWS * SLOT_BYTES
    for k in range(4):
        r = (lane >> 3) + 4 * k
        src = so + r * SLOT_BYTES + 16 * i8
        dst = map_shared_address_to_peer(
            smem + S_LONG_RECV + (rank * WARPS + warp) * SLOT_BYTES + 16 * i8, r
        )
        remote_bar = map_shared_address_to_peer(mbar, r)
        store_async_cluster_f32x4(dst, load_shared_f32x4_address(src), remote_bar)
        store_async_cluster_f32x4(
            dst + 128, load_shared_f32x4_address(src + 128), remote_bar
        )
        store_async_cluster_f32x2(
            dst + 256,
            load_shared_f32_address(smem + S_M + 64 * warp + 4 * r),
            load_shared_f32_address(smem + S_L + 64 * warp + 4 * r),
            remote_bar,
            Int32(i8 == 0),
            clobber_memory=True,
        )


@cute.jit
def _merge_cluster_long(recv: Int32, n_active: Int32, lane: Int32):
    """The owner's merge of its row's partials from the cluster's first
    ``n_active`` CTAs at ``recv`` (stride WARPS slots) in rank order:
    (m, l, o[2 lane], o[2 lane + 1])."""
    m = Float32(NEG_BIG)
    for k in range(n_active):
        m = max_ftz_f32(
            m, load_shared_f32_address(recv + k * (WARPS * SLOT_BYTES) + 256)
        )
    ls = Float32(0.0)
    ox = Float32(0.0)
    oy = Float32(0.0)
    for k in range(n_active):
        row = recv + k * (WARPS * SLOT_BYTES)
        sc = ex2_approx_ftz(sub_ftz_f32(load_shared_f32_address(row + 256), m))
        ls = fma_ftz_f32(sc, load_shared_f32_address(row + 260), ls)
        vx, vy = load_shared_f32x2_address(row + 8 * lane)
        ox = fma_ftz_f32(sc, vx, ox)
        oy = fma_ftz_f32(sc, vy, oy)
    return m, ls, ox, oy


@cute.jit
def _merge_rows(row_partials: Int64, active: Int32, lane: Int32):
    """(l, o[2 lane], o[2 lane + 1]) of a row's ``active`` cross-cluster
    partials, merged in cluster order."""
    mz = Float32(NEG_BIG)
    for z in range(active):
        mz = max_ftz_f32(
            mz,
            load_global_cg_f32_address(row_partials + Int64(z) * SLOT_BYTES + 256),
        )
    rl = Float32(0.0)
    rx = Float32(0.0)
    ry = Float32(0.0)
    for z in range(active):
        part = row_partials + Int64(z) * SLOT_BYTES
        sc = ex2_approx_ftz(sub_ftz_f32(load_global_cg_f32_address(part + 256), mz))
        rl = fma_ftz_f32(sc, load_global_cg_f32_address(part + 260), rl)
        vx, vy = load_global_cg_f32x2_address(part + 8 * Int64(lane))
        rx = fma_ftz_f32(sc, vx, rx)
        ry = fma_ftz_f32(sc, vy, ry)
    return rl, rx, ry


@cute.jit
def _merge_rows_m(
    row_partials: Int64,
    slots: cutlass.Constexpr[int],
    active: Int32,
    lane: Int32,
):
    """``_merge_rows`` with the m of each of the ``slots`` partials loaded at
    once (those from ``active`` on read partial active - 1 and are not
    merged)."""
    ms = []
    for z in cutlass.range_constexpr(slots):
        part = row_partials + Int64(cutlass.min(Int32(z), active - 1)) * SLOT_BYTES
        ms.append(load_global_cg_f32_address(part + 256))
    mz = Float32(NEG_BIG)
    for z in cutlass.range_constexpr(slots):
        if Int32(z) < active:
            mz = max_ftz_f32(mz, ms[z])
    rl = Float32(0.0)
    rx = Float32(0.0)
    ry = Float32(0.0)
    for z in range(active):
        part = row_partials + Int64(z) * SLOT_BYTES
        sc = ex2_approx_ftz(sub_ftz_f32(load_global_cg_f32_address(part + 256), mz))
        rl = fma_ftz_f32(sc, load_global_cg_f32_address(part + 260), rl)
        vx, vy = load_global_cg_f32x2_address(part + 8 * Int64(lane))
        rx = fma_ftz_f32(sc, vx, rx)
        ry = fma_ftz_f32(sc, vy, ry)
    return rl, rx, ry


@cute.jit
def _merge_rows_batched(
    row_partials: Int64,
    slots: cutlass.Constexpr[int],
    active: Int32,
    lane: Int32,
):
    """``_merge_rows`` with every partial's loads issued at once: (m, l) and
    the o pair of each of the ``slots`` partials (those from ``active`` on
    read partial active - 1 and are not merged)."""
    ml = []
    vo = []
    for z in cutlass.range_constexpr(slots):
        part = row_partials + Int64(cutlass.min(Int32(z), active - 1)) * SLOT_BYTES
        ml.append(load_global_cg_f32x2_address(part + 256))
        vo.append(load_global_cg_f32x2_address(part + 8 * Int64(lane)))
    mz = Float32(NEG_BIG)
    for z in cutlass.range_constexpr(slots):
        if Int32(z) < active:
            mz = max_ftz_f32(mz, ml[z][0])
    rl = Float32(0.0)
    rx = Float32(0.0)
    ry = Float32(0.0)
    for z in cutlass.range_constexpr(slots):
        if Int32(z) < active:
            sc = ex2_approx_ftz(sub_ftz_f32(ml[z][0], mz))
            rl = fma_ftz_f32(sc, ml[z][1], rl)
            rx = fma_ftz_f32(sc, vo[z][0], rx)
            ry = fma_ftz_f32(sc, vo[z][1], ry)
    return rl, rx, ry


@cute.jit
def _merge_partials(
    counters: Int64,
    partials: Int64,
    row_id: Int64,
    slots: cutlass.Constexpr[int],
    split: Int32,
    active: Int32,
    lane: Int32,
    m: Float32,
    ls: Float32,
    ox: Float32,
    oy: Float32,
    merge: cutlass.Constexpr[str],
):
    """Publish this cluster's (o, m, l) of a row, one of ``active``; the warp
    that completes the row's count returns (1, l, o[2 lane], o[2 lane + 1]),
    the merge of every partial in cluster order (whichever warp is last,
    loading them as ``merge`` says), after resetting the counter for the next
    launch; the others return 0."""
    row_partials = partials + row_id * (slots * SLOT_BYTES)
    mine = row_partials + Int64(split) * SLOT_BYTES
    store_global_f32x2_address(mine + 8 * Int64(lane), ox, oy)
    first = Int32(lane == Int32(0))
    store_global_f32_address(mine + 256, m, first)
    store_global_f32_address(mine + 260, ls, first)
    membar_gl()
    cute.arch.sync_warp()
    counter = counters + row_id * 4
    arrived = Uint32(0)
    if lane == Int32(0):
        arrived = atomic_add1_u32_address(counter)
    arrived = shuffle_sync_idx_u32(arrived, Int32(0))
    last = Int32(arrived == Uint32(active - 1))
    rl = ls
    rx = ox
    ry = oy
    if last != Int32(0):
        membar_gl()
        if const_expr(merge == "all"):
            rl, rx, ry = _merge_rows_batched(row_partials, slots, active, lane)
        elif const_expr(merge == "m"):
            rl, rx, ry = _merge_rows_m(row_partials, slots, active, lane)
        else:
            rl, rx, ry = _merge_rows(row_partials, active, lane)
        if lane == Int32(0):
            atomic_exch_zero_u32_address(counter)
    return last, rl, rx, ry


@cute.jit
def _long_context(
    q_base: Int64,
    q_stride: Int64,
    gate_base: Int64,
    gate_stride: Int64,
    kv_base: Int64,
    kv_page_stride: Int64,
    bt_row: Int64,
    out_base: Int64,
    out_stride: Int64,
    sl2: Float32,
    eps: Float32,
    window: cutlass.Constexpr[int],
    counters: Int64,
    partials: Int64,
    smem: Int32,
    seq_len: Int32,
    first: Int32,
    n: Int32,
    b: Int32,
    rg: Int32,
    rank: Int32,
    warp: Int32,
    lane: Int32,
    tid: Int32,
    kvh: Int32,
    kv_heads: Int32,
    z: Int32,
    num_z: cutlass.Constexpr[int],
    merge: cutlass.Constexpr[str],
) -> None:
    """A request of ``n`` pages from page ``first`` (more than SHORT_ROUNDS
    rounds of a cluster's warps): the GROUPS * num_z clusters of the
    (request, KV head) each take an equal contiguous page range; CTA r streams
    its sixteenth of the range through the ring and warp w attends the 16 rows
    of group w, so each page is read once per KV head. CTA r owns row
    16 w + r of every group w, gathers the cluster's partials of it over
    distributed shared memory and merges the clusters through the workspace."""
    slots = num_z * GROUPS
    split = z * GROUPS + rg
    pages_per_cluster = CLUSTER * MIN_CTA_PAGES
    active = cutlass.min(Int32(slots), (n + pages_per_cluster - 1) // pages_per_cluster)
    per = Int32(Uint32(n + active - 1) // Uint32(active))
    active = Int32(Uint32(n + per - 1) // Uint32(per))
    if split < active:  # the whole cluster, before any barrier
        _long_cluster(
            q_base,
            q_stride,
            gate_base,
            gate_stride,
            kv_base,
            kv_page_stride,
            bt_row,
            out_base,
            out_stride,
            sl2,
            eps,
            window,
            counters,
            partials,
            smem,
            seq_len,
            first,
            n,
            b,
            rank,
            warp,
            lane,
            tid,
            kvh,
            kv_heads,
            slots,
            split,
            active,
            per,
            merge,
        )


@cute.jit
def _long_cluster(
    q_base: Int64,
    q_stride: Int64,
    gate_base: Int64,
    gate_stride: Int64,
    kv_base: Int64,
    kv_page_stride: Int64,
    bt_row: Int64,
    out_base: Int64,
    out_stride: Int64,
    sl2: Float32,
    eps: Float32,
    window: cutlass.Constexpr[int],
    counters: Int64,
    partials: Int64,
    smem: Int32,
    seq_len: Int32,
    first: Int32,
    n: Int32,
    b: Int32,
    rank: Int32,
    warp: Int32,
    lane: Int32,
    tid: Int32,
    kvh: Int32,
    kv_heads: Int32,
    slots: cutlass.Constexpr[int],
    split: Int32,
    active: Int32,
    per: Int32,
    merge: cutlass.Constexpr[str],
) -> None:
    """Cluster ``split`` of ``active`` of a long request: pages
    first + split * per + [0, per) (clipped to the request)."""
    g = lane >> 2
    t = lane & 3
    sbeg = first + split * per
    sn = cutlass.min(per, first + n - sbeg)
    cper = (sn + CLUSTER - 1) >> 4
    n_active = Int32(Uint32(sn + cper - 1) // Uint32(cper))  # CTAs with pages
    cbeg = sbeg + rank * cper
    cnt = Int32(0)
    if rank < n_active:
        cnt = cutlass.min(cper, sbeg + sn - cbeg)

    r0 = warp * ROWS + g
    r1 = r0 + 8
    token0 = b * QUERY_ROWS + (r0 >> 3)
    token1 = b * QUERY_ROWS + (r1 >> 3)
    qr = _load_rows(q_base, q_stride, token0, r0 & 7, token1, r1 & 7, t)
    owned = warp * ROWS + rank
    out_token = Int64(b * QUERY_ROWS + (owned >> 3))
    lane_cols = Int64(HEAD_DIM * (owned & 7) + 2 * lane)
    gate2 = load_global_u32_address(
        gate_base + (out_token * gate_stride + lane_cols) * 2, ordered=True
    )

    steps = (cnt + GROUP_PAGES - 1) >> 2
    ring = smem + S_RING
    _issue_step(ring, Int32(0), cnt, cbeg, bt_row, kv_base, kv_page_stride, tid)
    cp_async_commit()

    mbar = smem + S_BAR
    if tid == Int32(0):
        mbarrier_init_address(mbar, 1, clobber_memory=True)
        fence_mbarrier_init_cluster()
    cluster_arrive_relaxed_aligned()
    qf_frag = cute.make_rmem_tensor(cute.make_layout((16,)), Uint32)
    _store_qf(qf_frag, _qf(qr))
    if tid == Int32(0):
        mbarrier_arrive_expect_tx_address(
            mbar, n_active * (WARPS * PUSH_BYTES), clobber_memory=True
        )
    lim0 = seq_len - QUERY_ROWS + (r0 >> 3)
    lim1 = seq_len - QUERY_ROWS + (r1 >> 3)
    low0 = Int32(0)
    low1 = Int32(0)
    if const_expr(window > 0):
        low0 = lim0 - window + 1
        low1 = lim1 - window + 1

    o = cute.make_rmem_tensor(cute.make_layout((32,)), Float32)
    for i in cutlass.range_constexpr(32):
        o[i] = Float32(0.0)
    s = cute.make_rmem_tensor(cute.make_layout((GROUP_PAGES * 8,)), Float32)
    m0 = Float32(NEG_BIG)
    m1 = Float32(NEG_BIG)
    l0 = Float32(0.0)
    l1 = Float32(0.0)
    for k in range(steps):
        # Step k landed and is visible to every thread, and every warp is
        # done with step k - 1, whose slots are refilled.
        cp_async_wait(RING_GROUPS - 2)
        cute.arch.barrier()
        if k + RING_GROUPS - 1 < steps:
            _issue_step(
                ring,
                k + RING_GROUPS - 1,
                cnt,
                cbeg,
                bt_row,
                kv_base,
                kv_page_stride,
                tid,
            )
        cp_async_commit()
        m0, m1, l0, l1 = _attend_group(
            ring + (k & (RING_GROUPS - 1)) * (GROUP_PAGES * PAGE_BYTES),
            cutlass.min(Int32(GROUP_PAGES), cnt - k * GROUP_PAGES),
            cbeg + k * GROUP_PAGES,
            qf_frag,
            g,
            t,
            lim0,
            lim1,
            low0,
            low1,
            sl2,
            m0,
            m1,
            l0,
            l1,
            o,
            s,
            window > 0,
        )
    cp_async_wait(0)

    if cnt > Int32(0):
        _stash_partial(
            smem + S_O + warp * ROWS * SLOT_BYTES,
            smem + S_M + 64 * warp,
            smem + S_L + 64 * warp,
            g,
            t,
            o,
            m0,
            m1,
            l0,
            l1,
        )
    cute.arch.sync_warp()
    cluster_wait_aligned()
    if cnt > Int32(0):
        _push_rows(smem, mbar, warp, lane, rank)

    _wait_phase0(mbar)
    m, ls, ox, oy = _merge_cluster_long(
        smem + S_LONG_RECV + warp * SLOT_BYTES, n_active, lane
    )
    done = Int32(1)
    if active > Int32(1):
        row_id = (Int64(b) * Int64(kv_heads) + Int64(kvh)) * (
            QUERY_ROWS * HEADS
        ) + Int64(owned)
        done, ls, ox, oy = _merge_partials(
            counters,
            partials,
            row_id,
            slots,
            split,
            active,
            lane,
            m,
            ls,
            ox,
            oy,
            merge,
        )
    if done != Int32(0):
        out_row = (
            out_base + (out_token * out_stride + Int64(HEAD_DIM * (owned & 7))) * 2
        )
        _finish_row(ox, oy, ls, gate2, eps, out_row, lane)


class _Attn2Kernel:
    """Grid (16 * 4 R, kv_heads, num_splits), cluster (16, 1, 1), 128 threads:
    cluster b * 4 + rg of KV head blockIdx.y computes rows 16 rg + [0, 16) of
    request b's 64 (token = row // 8, head = row % 8). The variant, PDL, the
    KV heads, the z slices and the window are compile-time; shapes, strides
    and addresses are read at the call."""

    def __init__(
        self,
        split: bool,
        enable_pdl: bool,
        kv_heads: int,
        num_splits: int,
        window: int,
    ) -> None:
        self.split, self.enable_pdl = split, enable_pdl
        self.kv_heads, self.num_splits, self.window = kv_heads, num_splits, window
        self.merge, self.rolled = split_forms(kv_heads, num_splits)
        # A window of at most SHORT_ROUNDS rounds of pages leaves the
        # long-context path unreachable: the kernel is the short path alone.
        self.long_path = split and window_pages(window) > SHORT_ROUNDS * CLUSTER_WARPS
        self.rounds = 0
        if split and not self.long_path:
            self.rounds = -(-window_pages(window) // CLUSTER_WARPS)
        self.smem_bytes = SMEM_SPLIT if self.long_path else SMEM_ONE_CLUSTER
        # Three split-kernel CTAs fit an SM (registers capped accordingly), so
        # the 16-CTA clusters of a long request (or of several requests) are
        # resident together.
        self.min_ctas_per_sm = 3 if split else 1

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        gate: cute.Tensor,
        kv: cute.Tensor,
        bt: cute.Tensor,
        seq_lens: cute.Tensor,
        out: cute.Tensor,
        workspace: cute.Tensor | None,
        stream: cuda.CUstream,
    ) -> None:
        rows = Int64(q.shape[0])
        counters = Int64(0)
        partials = Int64(0)
        if const_expr(self.split):
            assert workspace is not None
            counters = workspace.iterator.toint()
            # The uint32 counters, one per row, 16-byte padded, then the
            # partials.
            partials = counters + (
                ((rows * (HEADS * self.kv_heads * 4) + 15) >> 4) << 4
            )
        self.kernel(
            q.iterator.toint(),
            Int64(q.stride[0]),
            gate.iterator.toint(),
            Int64(gate.stride[0]),
            kv.iterator.toint(),
            Int64(kv.stride[0]),
            bt.iterator.toint(),
            Int64(bt.stride[0]),
            Int32(bt.shape[1]),
            seq_lens.iterator.toint(),
            out.iterator.toint(),
            Int64(out.stride[0]),
            Float32(SCALE_LOG2),
            Float32(EPS_F32),
            counters,
            partials,
        ).launch(
            grid=(
                Int32(rows >> 3) * (CLUSTER * GROUPS),
                self.kv_heads,
                self.num_splits,
            ),
            block=(THREADS, 1, 1),
            cluster=(CLUSTER, 1, 1),
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=self.enable_pdl,
            min_blocks_per_mp=self.min_ctas_per_sm,
        )

    @cute.kernel
    def kernel(
        self,
        q: Int64,
        q_stride: Int64,
        gate: Int64,
        gate_stride: Int64,
        kv: Int64,
        kv_page_stride: Int64,
        bt: Int64,
        bt_stride: Int64,
        max_pages: Int32,
        seq_lens: Int64,
        out: Int64,
        out_stride: Int64,
        scale_log2: Float32,
        eps: Float32,
        counters: Int64,
        partials: Int64,
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        bx, kvh, z = cute.arch.block_idx()
        lane = tid & 31
        warp = tid >> 5
        rank = cute.arch.block_idx_in_cluster()
        cluster_id = bx >> 4
        b = cluster_id >> 2
        rg = cluster_id & (GROUPS - 1)
        smem = Int32(cute.arch.get_dyn_smem(Uint8, alignment=1024).toint())
        # The KV head's query, gate and output heads and its cache rows.
        heads_bytes = Int64(kvh) * (HEADS * HEAD_DIM * 2)
        q_base = q + heads_bytes
        gate_base = gate + heads_bytes
        out_base = out + heads_bytes
        kv_base = kv + Int64(kvh) * (PAGE * SLOT_ELEMS * 2)

        cute.arch.griddepcontrol_wait()

        # Every load that does not depend on the length first: the length and
        # this warp's first two block-table entries (dereferenced below only
        # for pages of the request).
        seq_len = load_global_nc_s32_or(seq_lens + Int64(b) * 4, Int32(1), Int32(0))
        gw = rank * WARPS + warp
        bt_row = bt + Int64(b) * bt_stride * 4
        e_first = load_global_nc_s32_or(
            bt_row + Int64(gw) * 4, Int32(gw < max_pages), Int32(0)
        )
        e_second = load_global_nc_s32_or(
            bt_row + Int64(gw + CLUSTER_WARPS) * 4,
            Int32(gw + CLUSTER_WARPS < max_pages),
            Int32(0),
        )
        args = (
            q_base,
            q_stride,
            gate_base,
            gate_stride,
            kv_base,
            kv_page_stride,
            bt_row,
            out_base,
            out_stride,
            scale_log2,
            eps,
            self.window,
        )
        if const_expr(self.split):
            # The pages from row 0's first visible key to the last.
            last = (seq_len + PAGE - 1) >> 4
            first_page = Int32(0)
            if const_expr(self.window > 0):
                first_page = (
                    cutlass.max(Int32(0), seq_len - QUERY_ROWS - self.window + 1) >> 4
                )
            pages = last - first_page
            long_request = False
            if const_expr(self.long_path):
                long_request = pages > SHORT_ROUNDS * CLUSTER_WARPS
            if long_request:
                _long_context(
                    *args,
                    counters,
                    partials,
                    smem,
                    seq_len,
                    first_page,
                    pages,
                    b,
                    rg,
                    rank,
                    warp,
                    lane,
                    tid,
                    kvh,
                    Int32(self.kv_heads),
                    z,
                    self.num_splits,
                    self.merge,
                )
            elif z == Int32(0):
                # The entries above are those of pages from 0 (first_page is
                # the constant 0 without a window).
                if first_page != Int32(0):
                    e_first = load_global_nc_s32_or(
                        bt_row + Int64(first_page + gw) * 4, Int32(gw < pages), Int32(0)
                    )
                    e_second = load_global_nc_s32_or(
                        bt_row + Int64(first_page + gw + CLUSTER_WARPS) * 4,
                        Int32(gw + CLUSTER_WARPS < pages),
                        Int32(0),
                    )
                _short_path(
                    *args,
                    smem,
                    seq_len,
                    b,
                    rg,
                    rank,
                    warp,
                    lane,
                    tid,
                    first_page,
                    pages,
                    e_first,
                    e_second,
                    True,
                    self.rolled,
                    self.rounds,
                )
        else:
            _short_path(
                *args,
                smem,
                seq_len,
                b,
                rg,
                rank,
                warp,
                lane,
                tid,
                Int32(0),
                (seq_len + PAGE - 1) >> 4,
                e_first,
                e_second,
                False,
                False,
                0,
            )


@functools.cache
def compile_attn2(
    split: bool,
    enable_pdl: bool,
    kv_heads: int,
    num_splits: int,
    window: int,
    options: str = "",
):
    """cute.compile one variant of the kernel for ``kv_heads`` KV heads,
    ``num_splits`` z slices (1 for the one-cluster kernel) and ``window`` (0:
    none), with these extra compile ``options``, before CUDA graph capture.
    The result takes q, gate, kv_cache, block_table, seq_lens, out and the
    workspace (None for the one-cluster kernel), checks their dtypes, ranks,
    the static extents, unit strides, row-stride divisibility and alignment,
    and launches on the current stream."""
    assert_not_capturing("the attn2 kernel")
    rows = cute.sym_int32()
    width = kv_heads * HEADS * HEAD_DIM
    head_stride = PAGE * SLOT_ELEMS if kv_heads > 1 else cute.sym_int64()
    workspace = None
    if split:
        workspace = make_fake_tensor(Uint8, (cute.sym_int32(),), (1,), assumed_align=16)
    return cute.compile(
        _Attn2Kernel(split, enable_pdl, kv_heads, num_splits, window),
        make_fake_tensor(
            BFloat16,
            (rows, width),
            (cute.sym_int64(divisibility=8), 1),
            assumed_align=16,
        ),
        make_fake_tensor(
            BFloat16,
            (rows, width),
            (cute.sym_int64(divisibility=2), 1),
            assumed_align=4,
        ),
        make_fake_tensor(
            BFloat16,
            (cute.sym_int32(), kv_heads, PAGE, SLOT_ELEMS),
            (cute.sym_int64(divisibility=8), head_stride, SLOT_ELEMS, 1),
            assumed_align=16,
        ),
        make_fake_tensor(
            Int32,
            (cute.sym_int32(), cute.sym_int32()),
            (cute.sym_int64(), 1),
            assumed_align=4,
        ),
        make_fake_tensor(
            Int32, (cute.sym_int32(),), (cute.sym_int64(),), assumed_align=4
        ),
        make_fake_tensor(
            BFloat16,
            (rows, width),
            (cute.sym_int64(divisibility=2), 1),
            assumed_align=4,
        ),
        workspace,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi {options}".strip(),
    )


def split_kernel(max_seq_len: int, window: int) -> bool:
    """Whether a deployment of ``max_seq_len`` launches the split kernel."""
    return max_seq_len > SHORT_MAX_LEN or window > 0


def window_pages(window: int) -> int:
    """The most pages from row 0's first visible key to a request's last
    with ``window`` (0: no window, no bound): ceil(L / 16) - (L - 7 - W) // 16
    at most."""
    return (window + QUERY_ROWS + 2 * PAGE - 3) // PAGE if window > 0 else 1 << 30


def split_forms(kv_heads: int, num_splits: int) -> tuple[str, bool]:
    """(merge form, rolled short-path loop) of the split kernel."""
    slots = num_splits * GROUPS
    rolled = kv_heads == 1
    if kv_heads >= 4:
        return "loop", rolled
    if slots <= MERGE_ALL_MAX:
        return "all", rolled
    return ("m" if slots <= MERGE_M_MAX else "loop"), rolled


def counters_bytes(rows: int) -> int:
    """The workspace's uint32 counters, one per row, 16-byte padded."""
    return (rows * 4 + 15) // 16 * 16


def workspace_bytes(rows: int, num_splits: int) -> int:
    """Workspace bytes of a split launch over ``rows`` (request, KV head,
    query-head) rows: the counters, then a partial of SLOT floats per row and
    cluster of its (request, KV head)."""
    return counters_bytes(rows) + rows * num_splits * GROUPS * SLOT_BYTES


def prepare(
    max_seq_len: int,
    window: int,
    kv_heads: int,
    num_splits: int,
    enable_pdl: bool = True,
) -> None:
    """Compile the variant a deployment's layer launches (build time)."""
    split = split_kernel(max_seq_len, window)
    compile_attn2(split, bool(enable_pdl), kv_heads, num_splits if split else 1, window)


_INT32_MAX = 2**31 - 1


def _refusal(message: str) -> ValueError:
    return ValueError(f"muse_spark attn2: {message}")


def _check(
    q,
    kv_cache,
    block_table,
    seq_lens,
    gate,
    out,
    kv_heads,
    max_seq_len,
    window,
    num_splits,
    workspace,
) -> bool:
    """Raise ValueError naming the first condition the call fails, of those
    the kernel needs beyond the compiled call's own checks (dtypes, ranks,
    extents, unit strides, row strides, alignment); returns whether the split
    kernel runs."""
    if not 1 <= kv_heads <= 65535:
        raise _refusal(f"kv_heads = {kv_heads} is outside 1..65535")
    if not QUERY_ROWS <= max_seq_len <= _INT32_MAX:
        raise _refusal(f"max_seq_len = {max_seq_len} is outside 8..{_INT32_MAX}")
    if not 0 <= window <= _INT32_MAX:
        raise _refusal(f"sliding_window = {window} is outside 0..{_INT32_MAX}")
    if not 1 <= num_splits <= MAX_SPLITS:
        raise _refusal(f"num_splits = {num_splits} is outside 1..{MAX_SPLITS}")
    rows = q.shape[0] if q.dim() == 2 else 0
    if not (rows >= QUERY_ROWS and rows % QUERY_ROWS == 0):
        raise _refusal("q must be a BF16 [8 R, 512 kv_heads] tensor with dense rows")
    num_reqs = rows // QUERY_ROWS
    if num_reqs * CLUSTER * GROUPS > _INT32_MAX:
        raise _refusal("too many requests for one launch")
    if not (
        kv_cache.dim() == 4
        and kv_cache.shape[1] == kv_heads
        and kv_cache.stride(0) >= kv_heads * PAGE * SLOT_ELEMS
    ):
        raise _refusal(
            f"kv_cache must be a BF16 [pages, {kv_heads}, {PAGE}, {SLOT_ELEMS}] "
            "tensor of dense 16-byte aligned [heads, 16, 128] pages"
        )
    if not (
        block_table.dim() == 2
        and block_table.shape[0] >= num_reqs
        and block_table.shape[1] >= 1
    ):
        raise _refusal(
            f"block_table must be an int32 [>= {num_reqs}, pages] tensor with "
            "dense rows"
        )
    if not (
        seq_lens.dim() == 1
        and seq_lens.shape[0] >= num_reqs
        and (seq_lens.shape[0] == 1 or seq_lens.stride(0) == 1)
    ):
        raise _refusal(
            f"seq_lens must be a dense int32 vector of at least {num_reqs} entries"
        )
    device = q.get_device()
    if not (
        device >= 0
        and kv_cache.get_device() == device
        and block_table.get_device() == device
        and seq_lens.get_device() == device
        and gate.get_device() == device
        and out.get_device() == device
    ):
        raise _refusal("all tensors must be on q's CUDA device")
    split = split_kernel(max_seq_len, window)
    if split:
        if workspace is None:
            raise _refusal(
                f"max_seq_len > {SHORT_MAX_LEN} or a sliding window needs a workspace"
            )
        need = workspace_bytes(rows * HEADS * kv_heads, num_splits)
        if not (
            workspace.dim() == 1
            and workspace.shape[0] >= need
            and workspace.get_device() == device
        ):
            raise _refusal(
                f"workspace must be a dense uint8 vector of at least {need} "
                "bytes, 16-byte aligned, on q's device"
            )
    return split


def attn2(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor,
    kv_heads: int,
    enable_pdl: bool = True,
    max_seq_len: int = SHORT_MAX_LEN,
    window: int = 0,
    num_splits: int = 1,
    workspace: torch.Tensor | None = None,
) -> None:
    """``out [8 R, 512 kv_heads] = sigmoid(gate) * rmsnorm(attention(q,
    kv_cache))`` per query head for R = q.shape[0] / 8 requests whose 8 query
    rows are the last 8 of their seq_lens cache entries.

    q, gate and out hold the heads of each KV head contiguously (8 x 64
    columns per KV head); their rows may be strided (q and gate are column
    views of the attention front's output, ``split_qkv``). kv_cache is the
    layer's BF16 [pages, kv_heads, 16, 128] cache. Lengths must be
    8..max_seq_len and covered by the first ceil(L / 16) block_table entries
    (device values, not checked here). max_seq_len is the deployment's maximum
    length: above 2048, or with ``window`` > 0 (row t of a request sees keys
    [L - 8 + t - window + 1, L - 8 + t]), the split kernel runs over
    ``num_splits`` z slices and needs ``workspace``, uint8 of at least
    ``workspace_bytes`` whose counters are zero before the first call (every
    call leaves them zero). Launched on the current stream of the inputs'
    device (the compiled call switches to q's device and back), with PDL when
    ``enable_pdl``. Raises ValueError on inputs outside this contract (some
    through the compiled call's own checks); allocates nothing, so a call is
    graph-capturable once ``prepare`` has compiled its variant."""
    split = _check(
        q,
        kv_cache,
        block_table,
        seq_lens,
        gate,
        out,
        kv_heads,
        max_seq_len,
        window,
        num_splits,
        workspace,
    )
    kernel = compile_attn2(
        split, bool(enable_pdl), kv_heads, num_splits if split else 1, window
    )
    kernel(q, gate, kv_cache, block_table, seq_lens, out, workspace if split else None)
