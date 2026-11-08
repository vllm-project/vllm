# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent fused replay verify for large steps (KdaFusedTC): KdaFused's math
(deferred RecoverSSM, see ``fused``) with the state dot products and the F_B gate
projection on the tensor cores, in head tiles.

KdaFusedWS holds an item's state in the registers of an 8-warp scan group and forms
the 2q dot products with FP32 FMAs over shared-memory broadcasts, one item at a
time, and its prep reloads the head's F_B rows and conv taps for every item: at
large steps it runs at about 2.5x the state traffic. Here:

- the CTAs walk **head tiles** (``TILE`` requests of one head, tiles strided over
  the CTAs), each CTA's item lookups (state slot, record slot, pending count)
  resolved once into shared memory; the loader warp brings each tile's F_B rows
  [128, 128] into one of two buffers by TMA, and the prep warpgroups (a tile's
  items alternating between them, thread c = channel c) load each item's conv
  taps, dt_bias and beta into registers an item ahead;
- per item, one prep thread issues the gate projection D[feature, token] = F_B
  f_a^T (BF16, FP32 accumulation) into tensor memory, so each thread reads its own
  channel's raw gates; the rest of the prep (conv, SiLU, norms, decays, probes,
  replay keys, pair sums) is KdaFusedWS's;
- the state streams in key quarters (32 keys of the 128 rows, one 128-byte swizzle
  atom: buffer q holds quarter q, refilled with the next item's once its MMAs and
  TMA store read it); the **row group**
  (4 warps, thread v = state row v) replays each quarter in place in FP32
  (KdaFusedWS's arithmetic: the same checkpoint bits), stores the residual
  S - tf32(S) into tensor memory and one thread issues the quarter's TF32 MMAs
  D = S X + S X_lo + S_lo X (X = [Gamma_t k_t, Gamma_t q_t]; S X into one
  accumulator per quarter, the residual terms into a fifth); the quarter goes
  back by TMA store;
- thread v sums row v's accumulators, solves the q x q system, writes the
  corrections and runs the gated RMSNorm.

The tensor cores truncate FP32 operands to TF32; the three-way split with split
accumulators keeps the dot products within about 5e-7 of the largest (FP32 FMAs
1.4e-7)."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T as IR

from .fused import a_index, b_index, pair_count
from .fused_ws import KdaFusedWS
from .helpers import (
    fence_async_shared,
    load_shared_f32x4,
    mul_rn,
    reciprocal,
    sqrt_approx,
    store_bulk,
    sync_threads,
    triton_exp,
    triton_sigmoid,
    wait_bulk_reads,
)
from .primitives import (
    alloc,
    asm,
    at,
    copy_g2s,
    fma,
    fma2,
    scatter_reduce,
    warp_sum,
    zeros,
)

K = 128
ROW_THREADS = 128
PREP_THREADS = 128
THREADS = ROW_THREADS + 2 * PREP_THREADS + 64  # rows, two prep groups, TMA, MMA
QUARTER_BYTES = K * 32 * 4  # 32 keys of the 128 rows: one swizzle atom
FB_BYTES = K * K * 2
# X operand [X; X_lo] (N = 32: probe n < 16, its residual at 16 + n), key k at float
# (k // 4) * XG + 4 n + k % 4; the 4 pad floats per key group keep the prep
# threads' scalar stores conflict free.
PROBES = 16  # 2q <= 16 probes; the gate's N: q <= 16 tokens
XG = 4 * 2 * PROBES + 4
X_FLOATS = K // 4 * XG
FA_BYTES = PROBES * K * 2  # F_A rows of an item, 128-byte swizzled K-major
# Tensor memory: quarter q's S [X; X_lo] at 32 q, S_lo X at 128, the gate of prep
# group w at 144 + 16 w, S_lo of quarter q at 256 + 32 q.
SMALL_COL, GATE_COL, LO_COL = 128, 144, 256
TMEM_COLS = 512
SMEM_LIMIT = 227 * 1024
META = 8
ROW_BAR, LOADER_WARP, MMA_WARP = 3, 12, 13
TILE = 2  # requests per head tile


def idesc(n, bf16=False):
    """M = 128, FP32 accumulator, K-major A and B: TF32 (kind::tf32) or BF16."""
    ab = (1 << 7) | (1 << 10) if bf16 else (2 << 7) | (2 << 10)
    return (1 << 4) | ab | ((n >> 3) << 17) | ((K >> 4) << 24)


def sw128(dtype, rows, cols):
    atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, dtype)
    return cute.tile_to_shape(atom, (rows, cols), (0, 1))


def pieces(count):
    """Power-of-two piece sizes covering ``count`` values for scatter_reduce, with
    the fewest shuffles (a piece of 2^k values takes 2^k + 4 - k)."""
    best: list[tuple[int, tuple[int, ...]]] = [(0, ())]
    for n in range(1, count + 1):
        options = []
        for k in range(6):
            cost, sizes = best[max(n - (1 << k), 0)]
            options.append((cost + (1 << k) + 4 - k, (*sizes, 1 << k)))
        best.append(min(options))
    return best[count][1]


@cute.jit
def piece_sums(values, lane, warp, sums):
    """Warp sums of ``values`` (each summed over the lanes in the same order as one
    scatter_reduce) into ``sums[i, warp]``."""
    start = 0
    for size in pieces(len(values)):
        part = [values[start + i] if start + i < len(values) else Float32(0)
                for i in range(size)]  # fmt: skip
        total, selected = scatter_reduce(part, lane, size)
        if lane % (32 // size) == 0:
            sums[start + selected, warp] = total
        start += size


def bits_float(bits):
    return Float32(llvm.bitcast(IR.f32(), Uint32(bits).ir_value()))


def float_bits(value):
    return Uint32(llvm.bitcast(IR.i32(), Float32(value).ir_value()))


@cute.jit
def residual(value):
    """``value`` minus its TF32 truncation (how the tensor cores read FP32)."""
    return value - bits_float(float_bits(value) & Uint32(0xFFFFE000))


@asm(Int32)
def try_wait_suspend(barrier, phase):
    """mbarrier.try_wait with a suspend hint (1 ms)."""
    code = "{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
    code += "p,[$1],$2,1000000; selp.u32 $0,1,0,p; }"
    return code, "=r,r,r,~{memory}", (barrier, Int32(phase))


@cute.jit
def wait(barrier, phase=0):
    done = try_wait_suspend(barrier, phase)
    while done == 0:
        done = try_wait_suspend(barrier, phase)


@cute.jit
def wait_group(barrier, phase, warp, bar_id, threads: cutlass.Constexpr):
    """A warpgroup's wait: warp 0 polls, the others block on the named barrier."""
    if warp == 0:
        wait(barrier, phase)
    sync_threads(bar_id, threads)


@asm()
def store_shared_f32x4(pointer, values):
    code = "st.shared.v4.f32 [$0], {$1, $2, $3, $4};"
    return code, "r,f,f,f,f,~{memory}", (pointer, *values)


@asm()
def tmem_alloc(dst):
    code = f"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], {TMEM_COLS};"
    return code, "r,~{memory}", (dst,)


@asm()
def tmem_relinquish():
    return "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;", "", ()


@asm()
def tmem_dealloc(address):
    code = f"tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, {TMEM_COLS};"
    return code, "r", (Uint32(address),)


@asm()
def tc_fence_before():
    return "tcgen05.fence::before_thread_sync;", "~{memory}", ()


@asm()
def tc_fence_after():
    return "tcgen05.fence::after_thread_sync;", "~{memory}", ()


@asm()
def umma_ss(d, a_desc, b_desc, accumulate, *, n, bf16=False):
    kind = "f16" if bf16 else "tf32"
    code = "{ .reg .pred p; setp.ne.b32 p, $3, 0; "
    code += f"tcgen05.mma.cta_group::1.kind::{kind} [$0], $1, $2, "
    code += f"{idesc(n, bf16)}, p; }}"
    return code, "r,l,l,r", (Uint32(d), a_desc, b_desc, Int32(accumulate))


@asm()
def umma_ts(d, a_tmem, b_desc, accumulate):
    code = "{ .reg .pred p; setp.ne.b32 p, $3, 0; "
    code += "tcgen05.mma.cta_group::1.kind::tf32 [$0], [$1], $2, "
    code += f"{idesc(PROBES)}, p; }}"
    return code, "r,r,l,r", (Uint32(d), Uint32(a_tmem), b_desc, Int32(accumulate))


@asm()
def umma_commit(barrier):
    code = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];"
    return code, "r,~{memory}", (barrier,)


@asm((Float32,) * 16)
def tmem_load16(address):
    regs = ",".join(f"${i}" for i in range(16))
    code = f"tcgen05.ld.sync.aligned.32x32b.x16.b32 {{{regs}}}, [$16]; "
    code += "tcgen05.wait::ld.sync.aligned;"
    return code, ",".join(["=f"] * 16 + ["r"]), (Uint32(address),)


@asm()
def tmem_store16(address, values):
    regs = ",".join(f"${i + 1}" for i in range(16))
    code = f"tcgen05.st.sync.aligned.32x32b.x16.b32 [$0], {{{regs}}};"
    return code, ",".join(["r"] + ["f"] * 16), (Uint32(address), *values)


@asm()
def tmem_wait_st():
    return "tcgen05.wait::st.sync.aligned;", "~{memory}", ()


@asm()
def wait_bulk_reads_but_one():
    """Every bulk store group but the last committed has read its source."""
    return "cp.async.bulk.wait_group.read 1;", "~{memory}", ()


@cute.jit
def smem_desc(address, lead_bytes: cutlass.Constexpr, stride_bytes: cutlass.Constexpr):
    """No-swizzle K-major UMMA shared memory descriptor (SM100 version bits)."""
    static = ((lead_bytes >> 4) << 16) | ((stride_bytes >> 4) << 32) | (1 << 46)
    return Uint64((address >> 4) & 0x3FFF) | Uint64(static)


@cute.jit
def sw128_desc(address):
    """128-byte swizzled K-major descriptor (8-row groups 1 KB apart)."""
    static = (1 << 16) | ((1024 >> 4) << 32) | (1 << 46) | (2 << 61)
    return Uint64((address >> 4) & 0x3FFF) | Uint64(static)


def operand_bytes(tokens, rq):
    """One operand block: X, X_lo, replay keys and D_n, old corrections, v, beta,
    pair sums, meta."""
    p2 = 1 << (pair_count(tokens) - 1).bit_length()
    floats = X_FLOATS + (2 * rq + 1 + tokens) * K + 8 + p2
    return floats * 4 + META * 4


def staging_bytes(tokens, rq, conv_dim_first, window, conv_size):
    rows = 3 * tokens * K * 2  # x (3 parts)
    conv = 3 * K * window * conv_size if conv_dim_first else 9 * K * conv_size
    records = rq * 2 * K * 2 + rq * K * 4  # old key / gate, old corrections
    return rows + FA_BYTES + conv + records


def cta_items(heads, requests, ctas):
    """Items of the CTA with the most."""
    tiles = -(-requests // TILE) * heads
    return -(-tiles // ctas) * TILE


def shared_bytes(tokens, rq, conv_dim_first, window, conv_size, items):
    """With ``items`` per CTA (their lookup table)."""
    staging = staging_bytes(tokens, rq, conv_dim_first, window, conv_size)
    sums = 2 * 64 * 4 * 4 + 2 * 32 * 4 * 4 + 8 * 4 * 4 + 2 * tokens * K * 2
    quarters_fb = 4 * QUARTER_BYTES + 2 * FB_BYTES
    fixed = quarters_fb + 2 * operand_bytes(tokens, rq) + 2 * staging + sums + 2048
    return fixed + items * 3 * 4


class KdaFusedTC(KdaFusedWS):
    """KdaFused's operands and semantics on ``ctas`` persistent CTAs."""

    def __init__(self, heads, tokens, gate_mode, has_bias, conv_dim_first, window,
                 ctas):  # fmt: skip
        assert 2 <= 2 * tokens <= PROBES and gate_mode in (0, 1)
        super().__init__(heads, tokens, gate_mode, has_bias, conv_dim_first, window,
                         ctas)  # fmt: skip

    @cute.jit
    def __call__(self, *args):
        # The state as ((V, H, slots), K) rows with its head and slot strides: one
        # TMA box per key quarter of a (slot, head) page, in and out; F_B as
        # (H K, K) rows, one box per head.
        state, f_b = args[9], args[16]
        rows = cute.make_tensor(
            state.iterator,
            cute.make_layout(
                ((K, state.shape[1], state.shape[0]), K),
                stride=((K, state.stride[1], state.stride[0]), 1),
            ),
        )
        quarter = sw128(Float32, K, 32)
        load_atom, load_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), rows, quarter, (K, 32)
        )
        store_atom, store_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), rows, quarter, (K, 32)
        )
        fb_atom, fb_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), f_b, sw128(BFloat16, K, K), (K, K)
        )
        self.kernel(args[:-1], load_atom, load_tensor, store_atom, store_tensor,
                    fb_atom, fb_tensor).launch(
            grid=(self.ctas, 1, 1),
            block=(THREADS, 1, 1),
            stream=args[-1],
            use_pdl=True,
            min_blocks_per_mp=1,
        )  # fmt: skip

    # ---------------------------------------------------------------- helpers
    @cute.jit
    def tile_count(self, args):
        """This CTA's head tiles (tile t: requests TILE (t // H) ..., head t % H)."""
        requests = args[10].shape[0] // self.tokens
        tiles = -(-requests // TILE) * self.heads
        cta, _, _ = cute.arch.block_idx()
        return (Int32(tiles) - Int32(cta) + self.ctas - 1) // self.ctas

    @cute.jit
    def item_ref(self, table, j):
        """Item j of this CTA: (head, request, live, slot, record slot, pending n,
        window offset); a request past the batch (a short last tile) is not live."""
        cta, _, _ = cute.arch.block_idx()
        tile = Int32(cta) + (j // TILE) * self.ctas
        request = (tile // self.heads) * TILE + j % TILE
        slot, rec, n = table[j, 0], table[j, 1], table[j, 2]
        offset = Int32(0)
        if n > 0:
            offset = n - 1
        return tile % self.heads, request, slot > 0, slot, rec, n, offset

    @cute.jit
    def tma_box(self, atom, tensor, buf, layout, tile: cutlass.Constexpr, coord,
                barrier, store: cutlass.Constexpr):  # fmt: skip
        """Warp-wide: one TMA box between ``buf`` and ``tensor`` tile ``coord``."""
        s_buf = cute.make_tensor(buf.iterator, layout)
        g_buf = cute.local_tile(tensor, tile, coord)
        t_smem, t_gmem = cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(s_buf, 0, 2),
            cute.group_modes(g_buf, 0, 2),
        )
        if cutlass.const_expr(store):
            cute.copy(atom, t_smem, t_gmem)
        else:
            cute.copy(atom, t_gmem, t_smem, tma_bar_ptr=barrier)

    @cute.jit
    def page_ref(self, table, j):
        """Item j's (head, state slot, live): its page."""
        cta, _, _ = cute.arch.block_idx()
        head = (Int32(cta) + (j // TILE) * self.ctas) % self.heads
        slot = table[j, 0]
        return head, slot, slot > 0

    @cute.jit
    def load_quarter(self, args, tma, buf, page, quarter, barrier, lane):
        """Warp-wide: key quarter ``quarter`` of a page (head, slot, live); none if
        not live."""
        head, slot, live = page
        if lane == 0:
            expected = Int32(0)
            if live:
                expected = Int32(QUARTER_BYTES)
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, expected)
        cute.arch.sync_warp()
        if live:
            coord = (slot * self.heads + head, quarter)
            self.tma_box(tma[0], tma[1], buf, sw128(Float32, K, 32), (K, 32), coord,
                         barrier, False)  # fmt: skip

    @cute.jit
    def item_constants(self, args, c, j, coeff, conv_bias):
        """Channel c's conv taps and bias (into ``coeff``, ``conv_bias``), dt_bias
        and A_log of item j's head, and the raw beta of its token c % q (any item:
        a request past the batch reads the last one's)."""
        weights, bias, beta = args[1], args[2], args[5]
        a_log, dt_bias = args[6], args[7]
        dim: cutlass.Constexpr = self.heads * K
        requests: cutlass.Constexpr = args[10].shape[0] // self.tokens
        cta, _, _ = cute.arch.block_idx()
        tile = Int32(cta) + (j // TILE) * self.ctas
        head = tile % self.heads
        request = cutlass.min((tile // self.heads) * TILE + j % TILE, requests - 1)
        for part in cutlass.range_constexpr(3):
            for i in cutlass.range_constexpr(4):
                coeff[part, i] = Float32(weights[part, i, head * K + c])
            if cutlass.const_expr(self.has_bias):
                conv_bias[part] = Float32(bias[part * dim + head * K + c])
        raw_beta = Float32(beta[request * self.tokens + c % self.tokens, head])
        return Float32(dt_bias[head * K + c]), Float32(a_log[head]), raw_beta

    @cute.jit
    def stage(self, args, staging, table, p, j):
        """cp.async item j's projection rows, F_A rows (into the gate's B operand),
        conv window and old records."""
        x, f_a = args[0], args[4]
        convs, xs, fas, _, _, kgs, us = staging
        T: cutlass.Constexpr = self.tokens
        dim: cutlass.Constexpr = self.heads * K
        head, request, live, slot, rec, n, offset = self.item_ref(table, j)
        token = request * T
        if live:
            self.stage_conv(args, convs, p, head, slot, offset)
            granule, kind = p % 16, p // 16
            if kind < 3:
                for t in cutlass.range_constexpr(T):
                    source = at(x, token + t, kind * dim + head * K + 8 * granule)
                    copy_g2s(at(xs, t, kind, 8 * granule), source)
            # F_A token t, keys 8 g ...: 128-byte swizzled K-major (16-byte chunk
            # g % 8 of row t at chunk g % 8 ^ t % 8, the two K atoms 2 KB apart).
            if p < 16 * T:
                t, g = p // 16, p % 16
                offset_bytes = (g // 8) * 2048 + (t // 8) * 1024 + (t % 8) * 128
                offset_bytes += ((g % 8) ^ (t % 8)) * 16
                copy_g2s(fas.iterator + offset_bytes // 2, at(f_a, token + t, 8 * g))
            if rec >= 0:
                self.stage_records(args, kgs, us, p, head, rec)

    # ----------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(
        self,
        args,
        load_atom: cute.CopyAtom,
        load_tensor: cute.Tensor,
        store_atom: cute.CopyAtom,
        store_tensor: cute.Tensor,
        fb_atom: cute.CopyAtom,
        fb_tensor: cute.Tensor,
    ):
        tma = (load_atom, load_tensor, store_atom, store_tensor, fb_atom, fb_tensor)
        conv_state = args[3]
        T: cutlass.Constexpr = self.tokens
        RQ: cutlass.Constexpr = args[18].shape[2]
        P2: cutlass.Constexpr = 1 << (pair_count(T) - 1).bit_length()
        tid, _, _ = cute.arch.thread_idx()
        tiles = self.tile_count(args)
        count = tiles * TILE  # this CTA's items

        smem = cutlass.utils.SmemAllocator()
        quarters = alloc(smem, Float32, (4, K * 32), align=1024)
        fbs = alloc(smem, BFloat16, (2, K * K), align=1024)
        fas = alloc(smem, BFloat16, (2, PROBES * K), align=1024)
        conv_type: cutlass.Constexpr = conv_state.element_type
        xc_s = alloc(smem, Float32, (2, X_FLOATS), align=128)
        kp_s = alloc(smem, Float32, (2, RQ, K))
        dn_s = alloc(smem, Float32, (2, K))
        u_s = alloc(smem, Float32, (2, RQ, K))
        v_s = alloc(smem, Float32, (2, T, K))
        beta_s = alloc(smem, Float32, (2, 8))
        pr_s = alloc(smem, Float32, (2, P2))
        meta = alloc(smem, Int32, (2, META))
        if cutlass.const_expr(self.conv_dim_first):
            convs = alloc(smem, conv_type, (2, 3, K * self.window), align=16)
        else:
            convs = alloc(smem, conv_type, (2, 3, 3 * K), align=16)
        xs = alloc(smem, BFloat16, (2, T, 3, K), align=16)
        kgs = alloc(smem, BFloat16, (2, RQ, 2 * K), align=16)
        us = alloc(smem, Float32, (2, RQ, K), align=16)
        sums = alloc(smem, Float32, (2, 32, 4))
        psums = alloc(smem, Float32, (2, 64, 4))
        out_sums = alloc(smem, Float32, (8, 4))
        gate_s = alloc(smem, BFloat16, (2, self.tokens, K), align=16)
        requests: cutlass.Constexpr = args[10].shape[0] // T
        table = alloc(smem, Int32, (cta_items(self.heads, requests, self.ctas), 3))
        bars = alloc(smem, cutlass.Int64, (24,), align=8)
        tmem_base = alloc(smem, Uint32, (1,), align=4)
        b0 = bars.iterator
        # ops full / done per block, gate done per prep group, F_B full / free per
        # buffer, quarter full / replayed / MMAs done per quarter buffer, the
        # item's MMAs.
        barriers = {
            "ops_full": b0, "done": b0 + 2, "gate": b0 + 4, "fb_full": b0 + 6,
            "fb_free": b0 + 8, "q_full": b0 + 10, "q_mma": b0 + 14, "mma": b0 + 18,
            "q_ready": b0 + 19,
        }  # fmt: skip

        if tid == 0:
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(barriers["ops_full"] + i, PREP_THREADS)
                cute.arch.mbarrier_init(barriers["done"] + i, ROW_THREADS)
                cute.arch.mbarrier_init(barriers["gate"] + i, 1)
                cute.arch.mbarrier_init(barriers["fb_full"] + i, 1)
                cute.arch.mbarrier_init(barriers["fb_free"] + i, 2)
            for q in cutlass.range_constexpr(4):
                cute.arch.mbarrier_init(barriers["q_full"] + q, 1)
                cute.arch.mbarrier_init(barriers["q_mma"] + q, 1)
                cute.arch.mbarrier_init(barriers["q_ready"] + q, ROW_THREADS)
            cute.arch.mbarrier_init(barriers["mma"], 1)
            cute.arch.mbarrier_init_fence()
        if tid < 32:
            tmem_alloc(tmem_base.iterator)
            tmem_relinquish()
        tc_fence_before()
        cute.arch.barrier()
        tc_fence_after()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        # Each item's state slot, record slot and pending n (0 past the batch).
        slots, record_rows, pending = args[8], args[20], args[21]
        cta, _, _ = cute.arch.block_idx()
        for j in cutlass.range(tid, count, THREADS, unroll=1):
            tile = Int32(cta) + (j // TILE) * self.ctas
            request = (tile // self.heads) * TILE + j % TILE
            slot, rec, n = Int32(0), Int32(-1), Int32(0)
            if request < requests:
                slot = Int32(slots[request, 0])
                rec = Int32(record_rows[request])
                if rec >= 0:
                    n = cutlass.max(cutlass.min(Int32(pending[rec]), RQ), 0)
            table[j, 0] = slot
            table[j, 1] = rec
            table[j, 2] = n
        cute.arch.barrier()
        ops = (xc_s, kp_s, dn_s, u_s, v_s, beta_s, pr_s, meta)
        tmem = tmem_base[0]

        warp = tid // 32
        if warp == LOADER_WARP:
            self.loader(args, tma, fbs, quarters, meta, table, tid % 32, tiles,
                        barriers)  # fmt: skip
        elif warp == MMA_WARP:
            self.mma_role(tid % 32, count, quarters, ops, tmem, barriers)
        elif tid >= ROW_THREADS:
            w = (tid - ROW_THREADS) // PREP_THREADS
            p = (tid - ROW_THREADS) % PREP_THREADS
            staging = (
                convs[w, None, None], xs[w, None, None, None], fas[w, None],
                sums[w, None, None], psums[w, None, None], kgs[w, None, None],
                us[w, None, None],
            )  # fmt: skip
            self.prep(args, w, p, count, staging, ops, fbs, table, tmem, barriers)
        else:
            self.rows(args, tma, tid, count, quarters, ops, (out_sums, gate_s), tmem,
                      barriers)  # fmt: skip
        tc_fence_before()
        cute.arch.barrier()
        if tid < 32:
            tmem_dealloc(tmem)

    # ------------------------------------------------------------ loader role
    @cute.jit
    def load_fb(self, tma, fbs, k, lane, barriers):
        """Warp-wide: tile k's F_B rows into buffer k % 2."""
        cta, _, _ = cute.arch.block_idx()
        fb_full = barriers["fb_full"] + k % 2
        head = (Int32(cta) + k * self.ctas) % self.heads
        if lane == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(fb_full, FB_BYTES)
        cute.arch.sync_warp()
        self.tma_box(tma[4], tma[5], fbs[k % 2, None], sw128(BFloat16, K, K), (K, K),
                     (head, 0), fb_full, False)  # fmt: skip

    @cute.jit
    def loader(self, args, tma, fbs, quarters, meta, table, lane, tiles, barriers):
        """The CTA's TMA traffic: each tile's F_B rows (tile k + 2's once both prep
        groups released tile k's), and per item quarter q: the replayed quarter
        back (none for n = 0), then once its MMAs and the store read it, the next
        item's quarter q into the same buffer."""
        q_full, q_mma = barriers["q_full"], barriers["q_mma"]
        q_ready, fb_free = barriers["q_ready"], barriers["fb_free"]
        count = tiles * TILE
        for k in cutlass.range_constexpr(2):
            if k < tiles:
                self.load_fb(tma, fbs, Int32(k), lane, barriers)
        page = self.page_ref(table, Int32(0))
        if count > 0:
            for q in cutlass.range_constexpr(4):
                self.load_quarter(args, tma, quarters[q, None], page, q, q_full + q,
                                  lane)  # fmt: skip
        for j in cutlass.range(count, unroll=1):
            # The next item's page; the store decision from the item's operand block
            # (read after the rows took it on).
            page_next = self.page_ref(table, cutlass.min(j + 1, count - 1))
            head, slot, live, n = Int32(0), Int32(0), Int32(0), Int32(0)
            for q in cutlass.range_constexpr(4):
                buf = quarters[q, None]
                wait(q_ready + q, j % 2)
                if cutlass.const_expr(q == 0):
                    b = j % 2
                    live, head, n, slot = meta[b, 0], meta[b, 2], meta[b, 4], meta[b, 6]
                if (live != 0) & (n > 0):
                    coord = (slot * self.heads + head, q)
                    self.tma_box(tma[2], tma[3], buf, sw128(Float32, K, 32), (K, 32),
                                 coord, None, True)  # fmt: skip
                cute.arch.cp_async_bulk_commit_group()
                # The next item's quarters: the previous buffer's a quarter after its
                # store (which drains meanwhile), the last one's once its store read
                # it (the rows' tail runs meanwhile).
                if j + 1 < count:
                    if cutlass.const_expr(q > 0):
                        wait(q_mma + q - 1, j % 2)
                        wait_bulk_reads_but_one()
                        self.load_quarter(args, tma, quarters[q - 1, None], page_next,
                                          q - 1, q_full + q - 1, lane)  # fmt: skip
                    if cutlass.const_expr(q == 3):
                        wait(q_mma + 3, j % 2)
                        wait_bulk_reads()
                        self.load_quarter(args, tma, buf, page_next, 3, q_full + 3,
                                          lane)  # fmt: skip
            k = j // TILE
            if (j % TILE == TILE - 1) & (k + 2 < tiles):
                wait(fb_free + k % 2, (k // 2) % 2)
                self.load_fb(tma, fbs, k + 2, lane, barriers)
        wait_bulk_reads()

    # --------------------------------------------------------------- MMA role
    @cute.jit
    def mma_role(self, lane, count, quarters, ops, tmem, barriers):
        """Per replayed quarter: S [X; X_lo] into the quarter's accumulator and
        S_lo X into the shared one (one thread issues)."""
        xc_s, meta = ops[0], ops[-1]
        ops_full, q_ready = barriers["ops_full"], barriers["q_ready"]
        q_mma, mma_done = barriers["q_mma"], barriers["mma"]
        for j in cutlass.range(count, unroll=1):
            b = j % 2
            wait(ops_full + b, (j // 2) % 2)
            live = meta[b, 0] != 0
            x_address = Uint32(xc_s[b, None].iterator.toint())
            for quarter in cutlass.range_constexpr(4):
                wait(q_ready + quarter, j % 2)
                if live:
                    tc_fence_after()
                    if lane == 0:
                        buf_address = Uint32(quarters[quarter, None].iterator.toint())
                        for kq in cutlass.range_constexpr(4):
                            kk = 4 * quarter + kq  # keys 8 kk ... 8 kk + 7
                            a_desc = sw128_desc(buf_address + kq * 32)
                            x_desc = smem_desc(x_address + kk * 2 * XG * 4, XG * 4, 128)
                            acc = tmem + 32 * quarter
                            umma_ss(acc, a_desc, x_desc, kq, n=2 * PROBES)
                            lo = tmem + LO_COL + 32 * quarter + 8 * kq
                            umma_ts(tmem + SMALL_COL, lo, x_desc, kk)
                        umma_commit(q_mma + quarter)
                        if cutlass.const_expr(quarter == 3):
                            umma_commit(mma_done)
                else:
                    if lane == 0:
                        cute.arch.mbarrier_arrive(q_mma + quarter)

    # --------------------------------------------------------------- row role
    @cute.jit
    def rows(self, args, tma, v, count, quarters, ops, row_smem, tmem, barriers):
        xc_s, kp_s, dn_s, u_s, v_s, beta_s, pr_s, meta = ops
        out_sums, gate_s = row_smem
        output, gate, eps, corrections = args[10], args[11], args[15], args[18]
        T: cutlass.Constexpr = self.tokens
        RQ: cutlass.Constexpr = corrections.shape[2]
        ops_full, done = barriers["ops_full"], barriers["done"]
        q_full, q_ready = barriers["q_full"], barriers["q_ready"]
        mma_done = barriers["mma"]
        warp, lane = v // 32, v % 32
        lane_base = (warp * 32) << 16
        swizzle = v % 8
        nw = args[12][v]
        mma_phase = Int32(0)
        for j in cutlass.range(count, unroll=1):
            b = j % 2
            wait(ops_full + b, (j // 2) % 2)
            live = meta[b, 0] != 0
            head, token, rec = meta[b, 2], meta[b, 3], meta[b, 5]
            u = cute.make_rmem_tensor(RQ, Float32)
            if live:
                # The output gates, first used by the norm.
                if v < 16 * T:
                    t, g = v // 16, v % 16
                    copy_g2s(at(gate_s, b, t, 8 * g),
                             at(gate, token + t, head * K + 8 * g))  # fmt: skip
                cute.arch.cp_async_commit_group()
                for t in cutlass.range_constexpr(RQ):
                    u[t] = u_s[b, t, v]
            for quarter in cutlass.range(4, unroll=1):
                buf = quarters[quarter, None]
                wait(q_full + quarter, j % 2)
                if live:
                    tc_fence_after()
                    lo = tmem + LO_COL + 32 * quarter
                    # S_n = S_0 D_n + sum_t u_t (kn_t P_t) (KdaFusedWS's arithmetic)
                    # in place, its residual into tensor memory.
                    for half in cutlass.range_constexpr(2):
                        pointers, rows_now, decays = [], [], []
                        final = [[Float32(0)] * 4 for _ in range(4)]
                        for gi in cutlass.range_constexpr(4):
                            chunk = 4 * half + gi
                            pointer = buf.iterator + 32 * v + 4 * (chunk ^ swizzle)
                            pointers.append(pointer)
                            rows_now.append(load_shared_f32x4(pointer))
                            key = 32 * quarter + 4 * chunk
                            decays.append(load_shared_f32x4(at(dn_s, b, key)))
                        for t in cutlass.range_constexpr(RQ):
                            for gi in cutlass.range_constexpr(4):
                                key = 32 * quarter + 16 * half + 4 * gi
                                k4 = load_shared_f32x4(at(kp_s, b, t, key))
                                for pp in cutlass.range_constexpr(2):
                                    final[gi][2 * pp], final[gi][2 * pp + 1] = fma2(
                                        (u[t], u[t]),
                                        (k4[2 * pp], k4[2 * pp + 1]),
                                        (final[gi][2 * pp], final[gi][2 * pp + 1]),
                                    )
                        lows = [Float32(0)] * 16
                        for gi in cutlass.range_constexpr(4):
                            s4, d4 = rows_now[gi], decays[gi]
                            new = [Float32(0)] * 4
                            for pp in cutlass.range_constexpr(2):
                                new[2 * pp], new[2 * pp + 1] = fma2(
                                    (s4[2 * pp], s4[2 * pp + 1]),
                                    (d4[2 * pp], d4[2 * pp + 1]),
                                    (final[gi][2 * pp], final[gi][2 * pp + 1]),
                                )
                            store_shared_f32x4(pointers[gi], new)
                            for e in cutlass.range_constexpr(4):
                                lows[4 * gi + e] = residual(new[e])
                        tmem_store16(lo + lane_base + 16 * half, lows)
                    tmem_wait_st()
                    fence_async_shared()
                    tc_fence_before()
                cute.arch.mbarrier_arrive(q_ready + quarter)  # to the MMA and TMA

            if live:
                # Per row: the dot products, the q x q substitution, the outputs.
                wait(mma_done, mma_phase)
                mma_phase = mma_phase ^ 1
                tc_fence_after()
                dots = [Float32(0)] * (2 * T)
                for half in cutlass.range_constexpr(2):  # S X, then the residual terms
                    parts = [tmem_load16(tmem + lane_base + 32 * q + 16 * half)
                             for q in range(4)]  # fmt: skip
                    for jj in cutlass.range_constexpr(2 * T):
                        low = parts[0][jj] + parts[1][jj]
                        total = low + (parts[2][jj] + parts[3][jj])
                        dots[jj] = total if half == 0 else dots[jj] + total
                small = tmem_load16(tmem + lane_base + SMALL_COL)
                for jj in cutlass.range_constexpr(2 * T):
                    dots[jj] = dots[jj] + small[jj]
                us_new = zeros(T, Float32)
                outs = zeros(T, Float32)
                for t in cutlass.range_constexpr(T):
                    total = dots[t]
                    for s in cutlass.range_constexpr(t):
                        total = fma(pr_s[b, a_index(t, s)], us_new[s], total)
                    us_new[t] = (v_s[b, t, v] - total) * beta_s[b, t]
                for t in cutlass.range_constexpr(T):
                    o = dots[T + t]
                    for s in cutlass.range_constexpr(t + 1):
                        o = fma(pr_s[b, b_index(T, t, s)], us_new[s], o)
                    outs[t] = Float32(BFloat16(o))
                    if rec >= 0:
                        corrections[rec, head, t, v] = us_new[t]
                # The gated RMSNorm (KdaFusedWS's sums: warp w holds rows 32 w ...).
                for t in cutlass.range_constexpr(T):
                    total = warp_sum(outs[t] * outs[t])
                    if lane == 0:
                        out_sums[t, warp] = total
                cute.arch.cp_async_wait_group(0)
                sync_threads(ROW_BAR, ROW_THREADS)
                for t in cutlass.range_constexpr(T):
                    sumsq = self.quad_sum(out_sums[t, None])
                    variance = sumsq * (1.0 / 128) + eps
                    rstd = reciprocal(sqrt_approx(variance))
                    sigmoid_gate = triton_sigmoid(Float32(gate_s[b, t, v]))
                    result = outs[t] * rstd * nw * sigmoid_gate
                    output[token + t, head * K + v] = BFloat16(result)
            tc_fence_before()
            cute.arch.mbarrier_arrive(done + b)

    # -------------------------------------------------------------- prep role
    @cute.jit
    def prep(self, args, w, p, count, staging, ops, fbs, table, tmem, barriers):
        x, weights, bias, conv_state, f_a, beta, a_log, dt_bias, slots = args[:9]
        state, output, gate, norm_weight, lower_bound, scale, eps, f_b = args[9:17]
        key_gates = args[19]
        xc_s, kp_s, dn_s, u_s, v_s, beta_s, pr_s, meta = ops
        convs, xs, fas, sums, psums, kgs, us = staging
        ops_full, done = barriers["ops_full"] + w, barriers["done"] + w
        gate_done = barriers["gate"] + w
        fb_full, fb_free = barriers["fb_full"], barriers["fb_free"]
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        RQ: cutlass.Constexpr = args[18].shape[2]
        NP: cutlass.Constexpr = pair_count(T)
        P2: cutlass.Constexpr = 1 << (NP - 1).bit_length()
        dim: cutlass.Constexpr = H * K
        lane, warp = p % 32, cute.arch.make_warp_uniform(p // 32)
        c = p
        bar = 1 + w
        blk = w
        conv_type: cutlass.Constexpr = conv_state.element_type
        xg = (c // 4) * XG + c % 4  # this channel's X column
        gate_cols = tmem + GATE_COL + 16 * w + ((warp * 32) << 16)
        fa_address = Uint32(fas.iterator.toint())
        requests = output.shape[0] // T

        if w < count:
            self.stage(args, staging, table, p, Int32(w))
        cute.arch.cp_async_commit_group()
        coeff = cute.make_rmem_tensor((3, 4), Float32)
        conv_bias = cute.make_rmem_tensor(3, Float32)
        dt_c, log_a, raw_beta = self.item_constants(args, c, Int32(w), coeff, conv_bias)
        gate_phase = Int32(0)

        for j in cutlass.range(w, count, 2, unroll=1):
            head, request, live, slot, rec, n, offset = self.item_ref(table, j)
            token = request * T
            k = j // TILE  # the head tile
            if j % TILE < 2:  # its F_B rows
                wait_group(fb_full + k % 2, (k // 2) % 2, warp, bar, PREP_THREADS)
            last_of_tile = j % TILE >= TILE - 2
            a = triton_exp(log_a)
            cute.arch.cp_async_wait_group(0)
            fence_async_shared()  # F_A rows, for the tensor cores
            sync_threads(bar, PREP_THREADS)
            values = cute.make_rmem_tensor((3, T), Float32)
            u_old = cute.make_rmem_tensor(RQ, Float32)
            keys = cute.make_rmem_tensor(RQ, Float32)
            probes = cute.make_rmem_tensor(2 * T, Float32)
            ds = cute.make_rmem_tensor(T, Float32)
            old_keys = cute.make_rmem_tensor(RQ, Float32)
            old_decays = cute.make_rmem_tensor(RQ, Float32)
            beta_value, fd = triton_sigmoid(raw_beta), Float32(1)
            if live:
                # The gate projection D[feature, token] = F_B f_a^T.
                if warp == 0:
                    with cute.arch.elect_one():
                        fb_address = Uint32(fbs[k % 2, None].iterator.toint())
                        for kk in cutlass.range_constexpr(K // 16):
                            atom_k, step = kk // 4, (kk % 4) * 32
                            a_desc = sw128_desc(fb_address + atom_k * 16384 + step)
                            b_desc = sw128_desc(fa_address + atom_k * 2048 + step)
                            gate_acc = tmem + GATE_COL + 16 * w
                            umma_ss(gate_acc, a_desc, b_desc, kk, n=PROBES, bf16=True)
                        umma_commit(gate_done)

                # Convolution + SiLU of q, k, v (thread c: channel c of each part).
                conv_page = conv_state[Int64(slot), None, None]
                for part in cutlass.range_constexpr(3):
                    channel = part * dim + head * K + c
                    taps = cute.make_rmem_tensor(3, conv_type)
                    for t in cutlass.range_constexpr(3):
                        if cutlass.const_expr(self.conv_dim_first):
                            taps[t] = convs[part, c * self.window + offset + t]
                        else:
                            taps[t] = convs[part, t * K + c]
                    inputs = [xs[t, part, c] for t in range(T)]
                    for t in cutlass.range_constexpr(T):
                        accum = Float32(0)
                        if cutlass.const_expr(self.has_bias):
                            accum = conv_bias[part]
                        for i in cutlass.range_constexpr(4):
                            tap = t + i
                            term = taps[tap] if tap < 3 else inputs[tap - 3]
                            accum = fma(Float32(term), coeff[part, i], accum)
                        values[part, t] = self.silu(accum)
                    if cutlass.const_expr(self.conv_dim_first):
                        row = c * self.window
                        convs[part, row] = conv_type(taps[1])
                        convs[part, row + 1] = conv_type(taps[2])
                        for t in cutlass.range_constexpr(T):
                            convs[part, row + 2 + t] = conv_type(inputs[t])
                    else:
                        conv_page[channel, 0] = conv_type(taps[1])
                        conv_page[channel, 1] = conv_type(taps[2])
                        for t in cutlass.range_constexpr(T):
                            conv_page[channel, 2 + t] = conv_type(inputs[t])
                if cutlass.const_expr(self.conv_dim_first):
                    fence_async_shared()
                # The old records (staged) are read; the new key records go out.
                old_gates = cute.make_rmem_tensor(RQ, Float32)
                for t in cutlass.range_constexpr(RQ):
                    old_keys[t] = Float32(0)
                    old_gates[t] = Float32(0)
                    u_old[t] = Float32(0)
                    if t < n:
                        old_keys[t] = Float32(kgs[t, c])
                        old_gates[t] = Float32(kgs[t, K + c])
                        u_old[t] = us[t, c]
                if rec >= 0:
                    for t in cutlass.range_constexpr(T):
                        key_gates[rec, head, t, c] = BFloat16(values[1, t])

                # q, k and old key norms (one reduction).
                squares = [Float32(0) for _ in range(2 * T + RQ)]
                for t in cutlass.range_constexpr(T):
                    squares[2 * t] = values[0, t] * values[0, t]
                    squares[2 * t + 1] = values[1, t] * values[1, t]
                for t in cutlass.range_constexpr(RQ):
                    squares[2 * T + t] = old_keys[t] * old_keys[t]
                piece_sums(squares, lane, warp, sums)
                for t in cutlass.range_constexpr(RQ):
                    old_decays[t] = self.decay(a, old_gates[t], dt_c, lower_bound)

                # This channel's raw gates (BF16) and decays.
                wait_group(gate_done, gate_phase, warp, bar, PREP_THREADS)
                gate_phase = gate_phase ^ 1
                tc_fence_after()
                acc = tmem_load16(gate_cols)
                for t in cutlass.range_constexpr(T):
                    raw_g = Float32(BFloat16(acc[t]))
                    ds[t] = self.decay(a, raw_g, dt_c, lower_bound)
                    if rec >= 0:
                        key_gates[rec, head, t, K + c] = BFloat16(raw_g)
                tc_fence_before()
                if last_of_tile:  # noqa: SIM102
                    if p == 0:
                        cute.arch.mbarrier_arrive(fb_free + k % 2)
                if cutlass.const_expr(self.conv_dim_first):  # noqa: SIM102
                    if p == 0:
                        size: cutlass.Constexpr = K * self.window * conv_type.width // 8
                        for part in cutlass.range_constexpr(3):
                            channel0 = part * dim + head * K
                            store_bulk(
                                at(conv_page, channel0, 0), at(convs, part, 0), size
                            )
            else:
                if last_of_tile:  # noqa: SIM102
                    if p == 0:
                        cute.arch.mbarrier_arrive(fb_free + k % 2)

            if live:
                # Normalized q, k; the probes X (over this channel's decays).
                qs = cute.make_rmem_tensor(T, Float32)
                ks = cute.make_rmem_tensor(T, Float32)
                gamma = Float32(1)
                for t in cutlass.range_constexpr(T):
                    qn = self.quad_sum(sums[2 * t, None]) + 1.0e-6
                    kn = self.quad_sum(sums[2 * t + 1, None]) + 1.0e-6
                    fq = reciprocal(sqrt_approx(qn))
                    fk = reciprocal(sqrt_approx(kn))
                    qs[t] = mul_rn(mul_rn(values[0, t], fq), scale)
                    ks[t] = mul_rn(values[1, t], fk)
                    gamma = gamma * ds[t]
                    probes[t] = gamma * ks[t]
                    probes[T + t] = gamma * qs[t]
                # The replay keys (zero past n) and D_n.
                for r in cutlass.range_constexpr(RQ):
                    t = RQ - 1 - r
                    weighted = Float32(0)
                    if t < n:
                        total = self.quad_sum(sums[2 * T + t, None])
                        kn_old = old_keys[t] * cute.math.rsqrt(total + Float32(1e-6))
                        weighted = kn_old * fd
                        fd = fd * old_decays[t]
                    keys[t] = weighted
                # The pair products.
                products = [Float32(0) for _ in range(NP)]
                for s in cutlass.range_constexpr(T):
                    p_st = Float32(1)
                    for t in cutlass.range_constexpr(s, T):
                        if cutlass.const_expr(t > s):
                            p_st = p_st * ds[t]
                        weighted = ks[s] * p_st
                        if cutlass.const_expr(t > s):
                            products[a_index(t, s)] = weighted * ks[t]
                        products[b_index(T, t, s)] = weighted * qs[t]
                piece_sums(products, lane, warp, psums)

            # The staging is read (the conv window out): item j + 2's goes in now,
            # with its constants.
            if cutlass.const_expr(self.conv_dim_first):  # noqa: SIM102
                if p == 0:
                    wait_bulk_reads()
            sync_threads(bar, PREP_THREADS)
            if j + 2 < count:
                self.stage(args, staging, table, p, j + 2)
            cute.arch.cp_async_commit_group()
            constants = self.item_constants(args, c, j + 2, coeff, conv_bias)
            dt_c, log_a, raw_beta = constants

            # This item's operand block, once the rows released item j - 2's.
            if j >= 2:  # noqa: SIM102
                if warp == 0:
                    wait(done, ((j - 2) // 2) % 2)
            sync_threads(bar, PREP_THREADS)
            if live:
                pair_sum = Float32(0)
                if p < NP:
                    pair_sum = self.quad_sum(psums[p, None])
                for t in cutlass.range_constexpr(T):
                    v_s[blk, t, c] = values[2, t]
                for t in cutlass.range_constexpr(RQ):
                    u_s[blk, t, c] = u_old[t]
                    kp_s[blk, t, c] = keys[t]
                dn_s[blk, c] = fd
                for jj in cutlass.range_constexpr(2 * T):
                    xc_s[blk, xg + 4 * jj] = probes[jj]
                    xc_s[blk, xg + 4 * (PROBES + jj)] = residual(probes[jj])
                fence_async_shared()  # the probes, for the tensor cores
                if p < T:
                    beta_s[blk, p] = beta_value
                if p < P2:
                    pr_s[blk, p] = pair_sum
                if p == 0:
                    meta[blk, 0] = 1
                    meta[blk, 1] = request
                    meta[blk, 2] = head
                    meta[blk, 3] = token
                    meta[blk, 4] = n
                    meta[blk, 5] = rec
                    meta[blk, 6] = slot
            else:
                if p == 0:
                    meta[blk, 0] = 0
                if request < requests:  # a dead slot: zero outputs
                    for t in cutlass.range_constexpr(T):
                        output[token + t, head * K + c] = BFloat16(0)
            cute.arch.mbarrier_arrive(ops_full)
