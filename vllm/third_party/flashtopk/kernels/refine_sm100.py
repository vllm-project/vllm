# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by the FlashTopK authors (2026): the epilogue and TMEM read-out of scan_sm100.py (derived from
# FlashInfer's selective-logits candidate kernel) behind a gather warpgroup that compacts the pruning pass's
# surviving keys into tiles by cp.async, one MMA per compact tile.
"""SM100 FlashTopK refine in CuTe DSL (FP8 or BF16, 32 heads), the second kernel of key pruning: one CTA per 8-row
tile and KV split scores in full the keys prune_sm100.py's mask keeps for the tile, and leaves select the
candidates the scan would.

  * the gather warpgroup counts the survivors of each 128-key block of the CTA's range into a prefix sum in shared
    memory; gather warp w then fills compact tile w with survivors 128t .. 128t + 127 of every tile t = w (mod 4):
    each lane finds its keys' blocks by binary search and their bits with fns, and copies the K rows (cp.async, 16
    bytes at a time, 128-byte swizzle, the layout the MMA reads) with each key's index and kv scale;
  * the UMMA warp gives each compact tile one M=128, N=256 MMA with every head of the tile's rows, into one of two
    TMEM stages;
  * math warpgroup w takes the stage w tiles: the scan's arithmetic (scan_sm100.py's head sums), the row's raw gate
    and window, and the candidate buffer (every row's slots reserved in one round trip).

The gate stays at the seed. Warp roles (512 threads): 0-7 math, 8-11 gather, 12 Q loader, 13 UMMA.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, tcgen05

from .common import (
    abs_f32,
    atomic_add_global_i32,
    cp_async_4,
    cp_async_16,
    find_nth_set,
    fp32_to_i32_bits,
    load_global_v4_b32,
    loose_raw_gate,
    select_i32,
    store_global_b32,
)
from .scan_sm100 import _RND_RN, NUM_W_IN_REG, i32_bits_to_fp32, named_barrier_sync_unaligned

BLOCK_KV = 128
BLOCK_Q = 8
NUM_HEADS = 32
HEAD_DIM = 128
NUM_TILES = 4
MAX_BLOCKS = 16384  # KV blocks a CTA's range may span: its survivor prefix is in shared memory
MAX_BLOCKS_BF16 = 4096  # BF16 compact tiles are twice as large: a shorter prefix keeps shared memory under 227 KB
NUM_EPI_SUBTILES = 4


def _swizzled(slot, chunk):
    """Byte offset of 16-byte chunk `chunk` of row `slot` in a 128-row K-major tile with the 128-byte swizzle: the
    tile is 128-byte column blocks of all 128 rows (one for FP8 rows, two for BF16), each row's chunks XOR-ed with
    its row mod 8."""
    return (chunk // 8) * (BLOCK_KV * 128) + slot * 128 + (((slot % 8) ^ (chunk % 8)) * 16)


def max_blocks(bf16: bool) -> int:
    return MAX_BLOCKS_BF16 if bf16 else MAX_BLOCKS


class RefineSm100:
    def __init__(self, *, bf16: bool = False):
        self.bf16 = bf16
        self.elem = cutlass.BFloat16 if bf16 else cutlass.Float8E4M3FN
        self.row_bytes = HEAD_DIM * (2 if bf16 else 1)
        self.max_blocks = max_blocks(bf16)
        self.N = BLOCK_Q * NUM_HEADS
        self.threads_per_cta = 512
        self.num_math_threads = 256
        self.math_wg_registers = 216
        self.other_registers = 40
        self.acc_dtype = Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.mma_tiler_mn = (BLOCK_KV, self.N)

    def _setup_mma(self, dtype):
        major = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            dtype, dtype, major, major, self.acc_dtype, self.cta_group, self.mma_tiler_mn
        )
        k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], k * (HEAD_DIM // k))
        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,))
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, dtype, NUM_TILES)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, dtype, 1)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tiled_mma.make_fragment_C(acc_shape))
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,  # [seq_len_kv, 128] uint8 (FP8 E4M3 bits) or bf16, gathered by cp.async
        q: cute.Tensor,  # [num_q * 32, 128, 1] uint8 or bf16
        weights: cute.Tensor,  # [num_q, 32] f32
        kv_scales: cute.Tensor,  # [>= seq_len_kv] f32
        mask: cute.Tensor,  # [num_q_tiles * mask_blocks * 4] i32: the pruning pass's, one 8-row tile a plane
        row_start: cute.Tensor,
        row_end: cute.Tensor,
        origin: cute.Tensor,
        inv_delta: cute.Tensor,
        th_bucket: cute.Tensor,
        cand_val: cute.Tensor,
        cand_idx: cute.Tensor,
        cand_cnt: cute.Tensor,
        num_q: Int32,
        seq_len_kv: Int32,
        mask_blocks: Int32,
        num_splits: Int32,
        stream: cuda.CUstream,
    ):
        b = q if cutlass.const_expr(self.bf16) else cute.recast_tensor(q, cutlass.Float8E4M3FN)
        tiled_mma = self._setup_mma(b.element_type)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B((1, 1), tiled_mma.thr_id),
            b, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape,
        )
        w_layout = cute.make_layout((BLOCK_Q, NUM_HEADS), stride=(NUM_HEADS, 1))
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), weights, w_layout, (BLOCK_Q, NUM_HEADS)
        )
        self.num_q_tma_bytes = cute.size_in_bytes(b.element_type, b_smem_layout) + self.N * 4

        @cute.struct
        class SharedStorage:
            # q, count, full_tile[4], empty_tile[4], full_tmem[2], empty_tmem[2]
            mbar: cute.struct.MemRange[cutlass.Int64, 2 + 2 * NUM_TILES + 4]
            tmem_holding_buf: Int32

        num_tiles = (num_q + BLOCK_Q - 1) // BLOCK_Q
        self.kernel(
            tiled_mma, tma_atom_b, tma_tensor_b, tma_atom_w, tma_tensor_w, kv, kv_scales, mask, row_start, row_end,
            origin, inv_delta, th_bucket, cand_val, cand_idx, cand_cnt, num_q, seq_len_kv, mask_blocks, num_splits,
            self.a_smem_layout_staged, self.b_smem_layout_staged, SharedStorage,
        ).launch(grid=(num_tiles, num_splits, 1), block=[self.threads_per_cta, 1, 1], min_blocks_per_mp=1, stream=stream)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_w: cute.CopyAtom,
        mW: cute.Tensor,
        mKV: cute.Tensor,
        mScales: cute.Tensor,
        mMask: cute.Tensor,
        mRowStart: cute.Tensor,
        mRowEnd: cute.Tensor,
        mOrigin: cute.Tensor,
        mInvDelta: cute.Tensor,
        mThBucket: cute.Tensor,
        mCandVal: cute.Tensor,
        mCandIdx: cute.Tensor,
        mCandCnt: cute.Tensor,
        num_q: Int32,
        seq_len_kv: Int32,
        mask_blocks: Int32,
        num_splits: Int32,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        lane = cute.arch.lane_idx()
        tile = gdimx - 1 - bidx
        query_start = tile * BLOCK_Q

        total_blocks = (seq_len_kv + BLOCK_KV - 1) // BLOCK_KV
        blocks_per_split = (total_blocks + num_splits - 1) // num_splits
        split_lo = bidy * blocks_per_split * BLOCK_KV
        split_hi = min((bidy + 1) * blocks_per_split * BLOCK_KV, seq_len_kv)
        range_lo = seq_len_kv
        range_hi = Int32(0)
        for r in cutlass.range_constexpr(BLOCK_Q):
            row = query_start + r
            s = seq_len_kv
            e = Int32(0)
            if row < num_q:
                s = max(min(mRowStart[row], seq_len_kv), split_lo)
                e = min(min(mRowEnd[row], seq_len_kv), split_hi)
            nonempty = e > s
            range_lo = min(range_lo, select_i32(nonempty, s, seq_len_kv))
            range_hi = max(range_hi, select_i32(nonempty, e, 0))
        has_work = range_hi > range_lo
        kv_start = range_lo // BLOCK_KV * BLOCK_KV
        nb = Int32(0)
        if has_work:
            nb = (range_hi - kv_start + BLOCK_KV - 1) // BLOCK_KV
        # A CTA whose rows have no keys in its split leaves before any setup (adaptive pruning launches the path it
        # did not take with empty windows).
        if has_work:
            smem = utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            mbar = storage.mbar.data_ptr()
            q_bar, count_bar = mbar, mbar + 1
            full_tile, empty_tile = mbar + 2, mbar + 2 + NUM_TILES
            full_tmem, empty_tmem = mbar + 2 + 2 * NUM_TILES, mbar + 4 + 2 * NUM_TILES
            sQ = smem.allocate_tensor(
                element_type=self.elem, layout=b_smem_layout_staged.outer, byte_alignment=1024,
                swizzle=b_smem_layout_staged.inner,
            )
            sTile = smem.allocate_tensor(
                element_type=self.elem, layout=a_smem_layout_staged.outer, byte_alignment=1024,
                swizzle=a_smem_layout_staged.inner,
            )
            sMetaIdx = smem.allocate_tensor(Int32, cute.make_layout((NUM_TILES, BLOCK_KV), stride=(BLOCK_KV, 1)), 16)
            sMetaScale = smem.allocate_tensor(Float32, cute.make_layout((NUM_TILES, BLOCK_KV), stride=(BLOCK_KV, 1)), 16)
            sPrefix = smem.allocate_tensor(Int32, cute.make_layout((self.max_blocks + 1,)), 16)
            sW = smem.allocate_tensor(
                Float32, cute.make_layout((BLOCK_Q, NUM_HEADS, 1), stride=(NUM_HEADS, 1, BLOCK_Q * NUM_HEADS)), 128
            )
            sWarpWeights = smem.allocate_tensor(
                Float32, cute.make_layout((8, BLOCK_Q, NUM_HEADS), stride=(self.N + 4, NUM_HEADS, 1)), 16
            )
            sGate = smem.allocate_tensor(Float32, cute.make_layout((BLOCK_Q,)), 16)
            sWin = smem.allocate_tensor(Int32, cute.make_layout((BLOCK_Q, 2), stride=(2, 1)), 16)
            # The survivor count, then the gather warps' totals.
            sCounts = smem.allocate_tensor(Int32, cute.make_layout((5,)), 16)

            if tidx == 0:
                cute.arch.mbarrier_init(q_bar, 1)
                cute.arch.mbarrier_init(count_bar, 1)
                for i in cutlass.range_constexpr(NUM_TILES):
                    cute.arch.mbarrier_init(full_tile + i, 1)  # its gather warp, once the copies have landed
                    cute.arch.mbarrier_init(empty_tile + i, 5)  # the MMA's commit and the epilogue's four warps
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(full_tmem + i, 1)
                    cute.arch.mbarrier_init(empty_tmem + i, 4)
                cute.arch.mbarrier_init_fence()
            if tidx < BLOCK_Q:
                row_q = query_start + tidx
                row = min(row_q, num_q - 1)
                sGate[tidx] = loose_raw_gate(mInvDelta[row], mOrigin[row], mThBucket[row])
                s = seq_len_kv
                e = Int32(0)
                if row_q < num_q:
                    s = max(min(mRowStart[row_q], seq_len_kv), split_lo)
                    e = min(min(mRowEnd[row_q], seq_len_kv), split_hi)
                sWin[(tidx, 0)] = s
                sWin[(tidx, 1)] = max(e - s, Int32(0))
            cute.arch.sync_threads()

            tmem = utils.TmemAllocator(
                storage.tmem_holding_buf, barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=288),
                allocator_warp_id=0, is_two_cta=False,
            )
            thr_mma = tiled_mma.get_slice(0)
            tCrA = tiled_mma.make_fragment_A(sTile)
            tCrB = tiled_mma.make_fragment_B(sQ)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))

            if warp_idx == 12:
                cute.arch.setmaxregister_decrease(self.other_registers)
                if has_work:
                    gB = cute.local_tile(
                        cute.domain_offset((query_start * NUM_HEADS, 0, 0), mB_nkl),
                        cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None),
                    )
                    tBsB, tBgB = cpasync.tma_partition(
                        tma_atom_b, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3),
                        cute.group_modes(thr_mma.partition_B(gB), 0, 3),
                    )
                    gW = cute.local_tile(cute.domain_offset((query_start, 0), mW), (BLOCK_Q, NUM_HEADS), (None, None))
                    tWsW, tWgW = cpasync.tma_partition(
                        tma_atom_w, 0, cute.make_layout(1), cute.group_modes(sW, 0, 2), cute.group_modes(gW, 0, 2)
                    )
                    if lane == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(q_bar, self.num_q_tma_bytes)
                    cute.arch.sync_warp()
                    cute.copy(tma_atom_b, tBgB[(None, 0, 0, 0)], tBsB[(None, 0)], tma_bar_ptr=q_bar)
                    cute.copy(tma_atom_w, tWgW[(None, 0, 0)], tWsW[(None, 0)], tma_bar_ptr=q_bar)
            elif warp_idx == 13:
                cute.arch.setmaxregister_decrease(self.other_registers)
                named_barrier_sync_unaligned(288)
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                if has_work:
                    cute.arch.mbarrier_wait(count_bar, 0)
                    tiles = (sCounts[0] + BLOCK_KV - 1) // BLOCK_KV
                    cute.arch.mbarrier_wait(q_bar, 0)
                    for t in cutlass.range(tiles, unroll=1):
                        s = t % NUM_TILES
                        p = t % 2
                        cute.arch.mbarrier_wait(full_tile + s, (t // NUM_TILES) % 2)
                        cute.arch.mbarrier_wait(empty_tmem + p, (t // 2) % 2 ^ 1)
                        acc = cute.make_tensor(tmem_ptr + p * self.num_tmem_alloc_cols, tCtAcc_fake.layout)[(None, None, None, 0)]
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
                            cute.gemm(tiled_mma, acc, tCrA[None, None, k, s], tCrB[None, None, k, 0], acc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        with cute.arch.elect_one():
                            tcgen05.commit(full_tmem + p)
                            tcgen05.commit(empty_tile + s)
            elif warp_idx >= 8:
                cute.arch.setmaxregister_decrease(self.other_registers)
                if (warp_idx < 12) & has_work:
                    self._gather(
                        tidx - self.num_math_threads, warp_idx - 8, lane, nb, kv_start, tile, mask_blocks, mMask, mKV,
                        mScales, sPrefix, sTile, sMetaIdx, sMetaScale, sCounts, count_bar, full_tile, empty_tile,
                    )
            else:
                self._run_math(
                    has_work, tidx, lane, warp_idx, tmem, tCtAcc_fake, sW, sWarpWeights, sMetaIdx, sMetaScale, sGate,
                    sWin, sCounts, q_bar, count_bar, full_tmem, empty_tmem, empty_tile, query_start, mCandVal, mCandIdx,
                    mCandCnt,
                )

    @cute.jit
    def _gather(
        self, gt, gw, lane, nb, kv_start, tile, mask_blocks, mMask, mKV, mScales, sPrefix, sTile, sMetaIdx, sMetaScale,
        sCounts, count_bar, full_tile, empty_tile,
    ):
        """Survivors per block into an exclusive prefix, then compact tiles t = gw (mod 4)."""
        mask_base = mMask.iterator.toint() + (cutlass.Int64(tile) * mask_blocks + kv_start // BLOCK_KV) * 16
        # Each thread a run of consecutive blocks (four loads in flight), a warpgroup scan of the run totals, then the
        # run again.
        run = (nb + 127) // 128
        b_lo = min(gt * run, nb)
        b_hi = min(b_lo + run, nb)
        local = Int32(0)
        for blk in cutlass.range(b_lo, b_hi, 4, unroll=1):
            words = [load_global_v4_b32(mask_base + cutlass.Int64(min(blk + u, b_hi - 1)) * 16) for u in range(4)]
            for u in cutlass.range_constexpr(4):
                c = Int32(0)
                for word in words[u]:
                    c = c + Int32(cute.arch.popc(word))
                in_run = blk + u < b_hi
                if in_run:
                    sPrefix[blk + u] = c
                local = local + select_i32(in_run, c, 0)
        incl = local
        for off in cutlass.range_constexpr(5):
            # An up-shuffle's clamp is the lowest source lane: 0 (the default, 31, suits idx shuffles).
            up = cute.arch.shuffle_sync_up(incl, 1 << off, mask_and_clamp=0)
            if lane >= (1 << off):
                incl = incl + up
        if lane == 31:
            sCounts[1 + gw] = incl
        cute.arch.barrier(barrier_id=3, number_of_threads=128)
        base = Int32(0)
        total = Int32(0)
        for i in cutlass.range_constexpr(4):
            v = sCounts[1 + i]
            base = base + select_i32(i < gw, v, 0)
            total = total + v
        running = base + incl - local
        for blk in cutlass.range(b_lo, b_hi, unroll=1):
            c = sPrefix[blk]
            sPrefix[blk] = running
            running = running + c
        if gt == 0:
            sPrefix[nb] = total
            sCounts[0] = total
        cute.arch.barrier(barrier_id=3, number_of_threads=128)
        if gt == 0:
            cute.arch.mbarrier_arrive(count_bar)

        # Slot 32j + lane of compact tile t holds survivor 128t + 32j + lane. The next tile's keys are located while
        # the stage is still busy with this one.
        tiles = (total + BLOCK_KV - 1) // BLOCK_KV
        tile_bytes = BLOCK_KV * self.row_bytes
        smem_base = sTile.iterator.toint()
        scale_smem = sMetaScale.iterator.toint()
        kv_base = mKV.iterator.toint()
        scale_base = mScales.iterator.toint()
        i0, i1, i2, i3 = Int32(0), Int32(0), Int32(0), Int32(0)
        if gw < tiles:
            i0, i1, i2, i3 = self._locate(gw, lane, nb, kv_start, total, mask_base, sPrefix)
        for t in cutlass.range(gw, tiles, NUM_TILES, unroll=1):
            s = t % NUM_TILES
            cute.arch.mbarrier_wait(empty_tile + s, (t // NUM_TILES) % 2 ^ 1)
            keys = (i0, i1, i2, i3)
            for j in cutlass.range_constexpr(4):
                slot = j * 32 + lane
                if t * BLOCK_KV + slot < total:
                    src = kv_base + cutlass.Int64(keys[j]) * self.row_bytes
                    dst = smem_base + s * tile_bytes
                    for chunk in cutlass.range_constexpr(self.row_bytes // 16):
                        cp_async_16(dst + _swizzled(slot, chunk), src + chunk * 16)
                    sMetaIdx[(s, slot)] = keys[j]
                    cp_async_4(scale_smem + (s * BLOCK_KV + slot) * 4, scale_base + cutlass.Int64(keys[j]) * 4)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(full_tile + s)
            if t + NUM_TILES < tiles:
                i0, i1, i2, i3 = self._locate(t + NUM_TILES, lane, nb, kv_start, total, mask_base, sPrefix)

    @cute.jit
    def _locate(self, t, lane, nb, kv_start, total, mask_base, sPrefix):
        """The keys of tile t's slots 32j + lane: the last block whose prefix is at most the survivor's rank (ranks
        past the last survivor clamped to it), then the rank's bit in the block's mask. The four searches interleave."""
        g0 = min(t * BLOCK_KV + lane, total - 1)
        g1 = min(t * BLOCK_KV + 32 + lane, total - 1)
        g2 = min(t * BLOCK_KV + 64 + lane, total - 1)
        g3 = min(t * BLOCK_KV + 96 + lane, total - 1)
        b0, b1, b2, b3 = Int32(0), Int32(0), Int32(0), Int32(0)
        length = nb
        while length > 1:
            half = length // 2
            b0 = select_i32(sPrefix[b0 + half] <= g0, b0 + half, b0)
            b1 = select_i32(sPrefix[b1 + half] <= g1, b1 + half, b1)
            b2 = select_i32(sPrefix[b2 + half] <= g2, b2 + half, b2)
            b3 = select_i32(sPrefix[b3 + half] <= g3, b3 + half, b3)
            length = length - half
        blks = (b0, b1, b2, b3)
        words = [load_global_v4_b32(mask_base + cutlass.Int64(b) * 16) for b in blks]
        keys = []
        for j, g in enumerate((g0, g1, g2, g3)):
            r = g - sPrefix[blks[j]]
            w = words[j]
            part = Int32(0)
            word = w[0]
            for i in cutlass.range_constexpr(3):
                cnt = Int32(cute.arch.popc(word))
                step = (part == i) & (r >= cnt)
                r = select_i32(step, r - cnt, r)
                part = select_i32(step, part + 1, part)
                word = select_i32(step, w[i + 1], word)
            keys.append(kv_start + blks[j] * BLOCK_KV + part * 32 + find_nth_set(word, r + 1))
        return keys[0], keys[1], keys[2], keys[3]

    @cute.jit
    def _run_math(
        self, has_work, tidx, lane, warp_idx, tmem, tCtAcc_fake, sW, sWarpWeights, sMetaIdx, sMetaScale, sGate, sWin,
        sCounts, q_bar, count_bar, full_tmem, empty_tmem, empty_tile, query_start, mCandVal, mCandIdx, mCandCnt,
    ):
        cute.arch.setmaxregister_increase(self.math_wg_registers)
        tmem.allocate(2 * self.num_tmem_alloc_cols)
        named_barrier_sync_unaligned(288)
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        wg = warp_idx // 4
        key = tidx % 128
        epi_sub_mn = (BLOCK_KV, self.N // NUM_EPI_SUBTILES)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.mma_tiler, utils.LayoutEnum.ROW_MAJOR, self.acc_dtype, self.acc_dtype, (BLOCK_KV, 32), False
        )
        tCtAcc = cute.make_tensor(tmem_ptr + wg * self.num_tmem_alloc_cols, tCtAcc_fake.layout)
        tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0, None)], epi_sub_mn)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(key)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)[(None, None, None, None, None, 0)]
        tTR_cC = thr_copy_t2r.partition_D(cute.make_identity_tensor(epi_sub_mn))
        tTR_rAcc = cute.make_rmem_tensor_like(tTR_cC, self.acc_dtype)
        slot = tTR_cC[0][0]  # the compact tile's row (TMEM lane) this thread reads
        lane_mask = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
        cap = mCandVal.layout.shape[1]

        w_cache = cute.make_rmem_tensor(cute.make_layout((BLOCK_Q, NUM_W_IN_REG), stride=(NUM_W_IN_REG, 1)), Float32)
        if has_work:
            cute.arch.mbarrier_wait(count_bar, 0)
            n = sCounts[0]
            tiles = (n + BLOCK_KV - 1) // BLOCK_KV
            cute.arch.mbarrier_wait(q_bar, 0)
            # The warp's replica of the weights (lane = head): x + |x| heads w / 2, integer-ReLU heads w.
            factor = Float32(0.5)
            if (lane // 4) % 4 == 2:
                factor = Float32(1.0)
            for r in cutlass.range_constexpr(BLOCK_Q):
                sWarpWeights[(warp_idx, r, lane)] = sW[(r, lane, 0)] * factor
            cute.arch.sync_warp()
            for r in cutlass.range_constexpr(BLOCK_Q):
                for h in cutlass.range_constexpr(NUM_W_IN_REG):
                    w_cache[(r, h)] = sWarpWeights[(warp_idx, r, h)]

            for t in cutlass.range(wg, tiles, 2, unroll=1):
                s = t % NUM_TILES
                cute.arch.mbarrier_wait(full_tmem + wg, (t // 2) % 2)
                # TMEM loads are warp-aligned; a barrier wait may return with the warp diverged.
                cute.arch.sync_warp()
                valid = t * BLOCK_KV + slot < n
                kv_idx = select_i32(valid, sMetaIdx[(s, slot)], 0)
                scale = sMetaScale[(s, slot)]
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 0)], tTR_rAcc)
                cute.arch.fence_view_async_tmem_load()
                red = []
                for e in cutlass.range_constexpr(NUM_EPI_SUBTILES):
                    if e > 0:
                        cute.arch.fence_view_async_tmem_load()
                    acc = tTR_rAcc.load()
                    if e + 1 < NUM_EPI_SUBTILES:
                        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, e + 1)], tTR_rAcc)
                    else:
                        # Every subtile is in registers: hand the stage back to the tensor core.
                        cute.arch.sync_warp()
                        if lane == 0:
                            cute.arch.mbarrier_arrive(empty_tmem + wg)
                    red.extend(self._fast_sums(acc, w_cache, sWarpWeights, warp_idx, 2 * e))
                masks = []
                for r in cutlass.range_constexpr(0, BLOCK_Q, 2):
                    raw = cute.arch.mul_packed_f32x2((scale, scale), (red[r], red[r + 1]), rnd=_RND_RN)
                    for i in cutlass.range_constexpr(2):
                        row = r + i
                        take = valid & (raw[i] > sGate[row]) & (
                            cutlass.Uint32(kv_idx - sWin[(row, 0)]) < cutlass.Uint32(sWin[(row, 1)])
                        )
                        masks.append((cutlass.Uint32(cute.arch.vote_ballot_sync(take)), raw[i]))
                # Every row's slots in one round trip: lane r reserves row r's.
                lane_rows = cutlass.Uint32(0)
                for r in cutlass.range_constexpr(BLOCK_Q):
                    lane_rows = cutlass.Uint32(select_i32(lane == r, Int32(masks[r][0]), Int32(lane_rows)))
                lane_base = Int32(0)
                if lane_rows != cutlass.Uint32(0):
                    lane_base = atomic_add_global_i32(
                        mCandCnt.iterator.toint() + cutlass.Int64(query_start + lane) * 4, Int32(cute.arch.popc(lane_rows))
                    )
                for r in cutlass.range_constexpr(BLOCK_Q):
                    m, raw_r = masks[r]
                    if m != cutlass.Uint32(0):
                        base = cute.arch.shuffle_sync(lane_base, r)
                        if (m >> cutlass.Uint32(lane)) & cutlass.Uint32(1) != cutlass.Uint32(0):
                            pos = base + Int32(cute.arch.popc(m & lane_mask))
                            if pos < cap:
                                out_row = cutlass.Int64(query_start + r)
                                store_global_b32(
                                    mCandVal.iterator.toint() + (out_row * mCandVal.layout.stride[0] + pos) * 4,
                                    fp32_to_i32_bits(-raw_r),
                                )
                                store_global_b32(
                                    mCandIdx.iterator.toint() + (out_row * mCandIdx.layout.stride[0] + pos) * 4, kv_idx
                                )
                # The tile's metadata is used: it may be refilled.
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(empty_tile + s)
        cute.arch.barrier(barrier_id=2, number_of_threads=self.num_math_threads)
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    @cute.jit
    def _fast_sums(self, acc, w_cache, sWarpWeights, warp, t0):
        """The subtile's two rows' head sums, in scan_sm100.py's order."""
        out = []
        for r in cutlass.range_constexpr(2):
            t = t0 + r
            s0 = (Float32(0.0), Float32(0.0))
            s1 = (Float32(0.0), Float32(0.0))
            for h in cutlass.range_constexpr(0, NUM_HEADS, 4):
                n0 = r * NUM_HEADS + h
                if cutlass.const_expr((h // 4) % 4 != 2):
                    a01 = cute.arch.add_packed_f32x2((acc[n0], acc[n0 + 1]), (abs_f32(acc[n0]), abs_f32(acc[n0 + 1])))
                    a23 = cute.arch.add_packed_f32x2(
                        (acc[n0 + 2], acc[n0 + 3]), (abs_f32(acc[n0 + 2]), abs_f32(acc[n0 + 3]))
                    )
                else:
                    a01 = tuple(i32_bits_to_fp32(cutlass.max(fp32_to_i32_bits(acc[n0 + i]), Int32(0))) for i in (0, 1))
                    a23 = tuple(i32_bits_to_fp32(cutlass.max(fp32_to_i32_bits(acc[n0 + i]), Int32(0))) for i in (2, 3))
                w = [self._weight(w_cache, sWarpWeights, warp, t, h + i) for i in range(4)]
                s0 = cute.arch.fma_packed_f32x2(a01, (w[0], w[1]), s0, rnd=_RND_RN)
                s1 = cute.arch.fma_packed_f32x2(a23, (w[2], w[3]), s1, rnd=_RND_RN)
            sx, sy = cute.arch.add_packed_f32x2(s0, s1)
            out.append(sx + sy)
        return out

    @cute.jit
    def _weight(self, w_cache, sWarpWeights, warp, t, h: cutlass.Constexpr):
        if cutlass.const_expr(h < NUM_W_IN_REG):
            return w_cache[(t, h)]
        return sWarpWeights[(warp, t, h)]
