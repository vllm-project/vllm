# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-batch routed experts: ``ClusterExpertsTP16`` (TP16 B1/B2, six-CTA split-K
clusters exchanging quantized hidden rows) and ``ClusterExpertsTP8`` (TP8 B1, one
six-CTA cluster per routing slot)."""

from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cute.arch import store_async_dsmem

from ...common.primitives import (
    acquire_tensormap_asm,
    alloc,
    chunk_mma,
    commit_asm,
    copy_evict,
    copy_scale,
    fence_tmem,
    float_bits,
    fp8_bits,
    launch,
    load_tma3,
    load_tmem,
    load_volatile,
    load_words,
    mma_steps,
    mx_exponent,
    mx_rescale,
    pack_fp8,
    prefetch_bytes,
    prefetch_map,
    published,
    release_tmem,
    require_zero_tmem,
    shared_address,
    situ,
    smem_desc,
    split,
    store_bf16_pair,
    store_pair,
    store_shared_words,
    thread,
    unpack_bf16,
    view,
    wait,
    wait_tmem,
    warp_max_abs,
)


@cute.jit
def quantize_mx8(packed, row):
    """(E8M0 scale, FP8 words) of eight BF16 values; four lanes share a 32-block."""
    values = cute.make_rmem_tensor(8, Float32)
    for k in cutlass.range_constexpr(4):
        values[2 * k], values[2 * k + 1] = unpack_bf16(packed[row, k])
    amax = Float32(0)
    for k in cutlass.range_constexpr(8):
        amax = max(amax, abs(values[k]))
    for k in (1, 2):
        amax = max(amax, cute.arch.shuffle_sync_bfly(amax, k))
    sb = mx_exponent(amax)
    rs = mx_rescale(sb)
    lo, hi = (pack_fp8([values[4 * h + k] * rs for k in range(4)]) for h in (0, 1))
    return sb, (lo, hi)


class _ClusterExperts:
    """Shared launch of the six-CTA-cluster expert kernels."""

    kernel: Any
    threads: int

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, 96, self.threads, cluster=6, min_blocks_per_mp=1)


class ClusterExpertsTP16(_ClusterExperts):
    """Paired CTAs split FC1 and exchange quantized hidden rows; the TP16 intermediate
    has 192 live values padded to 256. IDs [B, 16], x BF16 [B, 3584], output BF16
    [B * 16, 3584]. Polled IDs are re-armed per cluster; the caller re-arms latent rows
    after all consumers complete."""

    threads = 288

    def __init__(self, batch: int):
        """Polled IDs and latent rows; successors are released before the final wait."""
        self.batch = batch
        self.off_sfa = 6 * 32768
        self.off_x = self.off_sfa + 6 * 1024
        self.off_h = self.off_x + 7 * 2048
        self.off_xsf = self.off_h + batch * 2048
        self.off_hsf = self.off_xsf + 14 * 128 + 384
        self.off_partial = self.off_hsf + batch * 1152
        self.off_misc = self.off_partial + batch * 256
        self.off_acc1 = self.off_misc + 104
        self.off_acc2 = self.off_acc1 + batch * 8
        self.off_precv = self.off_acc2 + batch * 40
        self.off_hrecv = self.off_precv + batch * 8
        self.off_tmem = self.off_hrecv + batch * 8
        self.off_amax = self.off_tmem + 4
        self.off_stage = (self.off_amax + batch * 16 + 15) // 16 * 16
        # 512 B unused reserve keeps the tuned SMEM footprint.
        self.smem_bytes = (self.off_stage + batch * 64 + 512 + 15) // 16 * 16
        cols = (8, 64, 128, 184) if batch == 1 else (16, 96, 144, 200)
        self.col_acc2, self.col_sfa, self.col_xsf, self.col_hsf = cols

    @cute.jit
    def get_id(self, ids, j, token):
        expert = Int32(load_volatile(ids.iterator + token * 16 + j)[0])
        while expert < 0:
            expert = Int32(load_volatile(ids.iterator + token * 16 + j)[0])
        return expert

    @cute.jit
    def issue(self, sm, maps, s13, s2, full, n, e0, e1, rank, tile0, tiles):
        slot = n % 6
        cute.arch.mbarrier_arrive_and_expect_tx(full.iterator + slot, 16384 + 1024)
        if n < 7 * self.batch:
            expert = e0 if n < 7 else e1
            ks = 7 * (rank % 2) + n % 7
            row = expert * 512 + rank // 2 * 128
            dst = sm.iterator + slot * 32768
            load_tma3(dst, maps.iterator, full.iterator + slot, row, ks * 2)
            dst = sm.iterator + self.off_sfa + slot * 1024
            src = s13.iterator + Int64(expert) * 57344 + rank // 2 * 14336 + ks * 1024
            copy_evict(dst, src, size=1024, barrier=full.iterator + slot)
        else:
            u = n - 7 * self.batch
            expert = e0 if u < tiles else e1
            tile = tile0 + u % tiles
            dst, desc = sm.iterator + slot * 32768, maps.iterator + 128
            load_tma3(dst, desc, full.iterator + slot, expert * 3584 + tile * 128, 0)
            dst = sm.iterator + self.off_sfa + slot * 1024
            src = s2.iterator + Int64(expert) * 28672 + tile * 1024
            copy_evict(dst, src, size=1024, barrier=full.iterator + slot)

    @cute.kernel
    def kernel(self, args):
        x, ids, maps, s13, s2, output, *situ_args = args
        tid, warp, lane, cta = thread()
        rank, j = cta % 6, cta // 6
        half, row_tile = rank % 2, rank // 2
        tile0, tiles = split(28, rank, 6)
        sm = alloc(cutlass.utils.SmemAllocator(), Uint8, self.smem_bytes, 1024)
        full = view(sm.iterator, self.off_misc, Uint64, 6)
        empty = view(sm.iterator, self.off_misc + 48, Uint64, 6)
        xall = view(sm.iterator, self.off_misc + 96, Uint64, 1)
        acc1 = view(sm.iterator, self.off_acc1, Uint64, self.batch)
        acc2 = view(sm.iterator, self.off_acc2, Uint64, self.batch * 5)
        precv = view(sm.iterator, self.off_precv, Uint64, self.batch)
        hrecv = view(sm.iterator, self.off_hrecv, Uint64, self.batch)
        tmem_base = view(sm.iterator, self.off_tmem, Uint32, 1)
        if tid == 0:
            for s in cutlass.range_constexpr(6):
                cute.arch.mbarrier_init(full.iterator + s, 1)
                cute.arch.mbarrier_init(empty.iterator + s, 1)
            cute.arch.mbarrier_init(xall.iterator, 7)
            for t in cutlass.range_constexpr(self.batch):
                cute.arch.mbarrier_init(acc1.iterator + t, 1)
                for i in cutlass.range_constexpr(5):
                    cute.arch.mbarrier_init(acc2.iterator + 5 * t + i, 1)
                cute.arch.mbarrier_init(precv.iterator + t, 1)
                cute.arch.mbarrier_init(hrecv.iterator + t, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(precv.iterator + t, 256)
                cute.arch.mbarrier_arrive_and_expect_tx(hrecv.iterator + t, 288)
            cute.arch.mbarrier_init_fence()
            acquire_tensormap_asm(maps.iterator)
            acquire_tensormap_asm(maps.iterator + 128)
        if warp == 2:
            cute.arch.alloc_tmem(256, tmem_base.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.cluster_arrive_relaxed()
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        tmem = tmem_base[0]
        cute.arch.griddepcontrol_launch_dependents()

        if warp == 0:
            if lane == 0:
                e0 = e1 = self.get_id(ids, j, 0)
                for n in cutlass.range_constexpr(6):
                    self.issue(sm, maps, s13, s2, full, n, e0, e1, rank, tile0, tiles)
                ks = 7 * half + 6
                prefetch_map(maps.iterator, e0 * 512 + row_tile * 128, 2 * ks)
                src = s13.iterator + Int64(e0) * 57344 + row_tile * 14336 + ks * 1024
                prefetch_bytes(src, 1024)
                if cutlass.const_expr(self.batch == 2):
                    e1 = self.get_id(ids, j, 1)
                for n in cutlass.range(6, self.batch * (7 + tiles), unroll=1):
                    slot = n % 6
                    wait(empty.iterator + slot, ((n // 6) - 1) % 2)
                    self.issue(sm, maps, s13, s2, full, n, e0, e1, rank, tile0, tiles)
        elif warp == 1:
            require_zero_tmem(tmem)
            adesc = smem_desc(sm.iterator, True)
            sfa = smem_desc(sm.iterator + self.off_sfa, False)
            xdesc = smem_desc(sm.iterator + self.off_x, True)
            xsf = smem_desc(sm.iterator + self.off_xsf, False)
            wait_tmem(xall.iterator)
            with cute.arch.elect_one():
                for n in cutlass.range_constexpr(14):
                    copy_scale(self.col_xsf + n * 4, xsf + n * 8)
            cute.arch.sync_warp()
            for t in cutlass.range(self.batch, unroll=1):
                for n in cutlass.range_constexpr(7):
                    g = 7 * t + n
                    slot = (t + n) % 6
                    wait_tmem(full.iterator + slot, (g // 6) % 2)
                    with cute.arch.elect_one():
                        col = self.col_sfa + 8 * slot
                        copy_scale(col, sfa + 64 * slot)
                        copy_scale(col + 4, sfa + 64 * slot + 32)
                        d, a, b = 8 * t, adesc + 2048 * slot, xdesc + 128 * n
                        mma_steps(d, a, b, col, self.col_xsf + 8 * n, n != 0, False)
                        commit_asm(empty.iterator + slot)
                        if cutlass.const_expr(n == 6):
                            commit_asm(acc1.iterator + t)
                    cute.arch.sync_warp()
            for t in cutlass.range(self.batch, unroll=1):
                wait(hrecv.iterator + t, test=True)
                cute.arch.fence_proxy("async.shared", space="cta")
                fence_tmem()
                hdesc = smem_desc(sm.iterator + self.off_h + t * 2048, True)
                hsf = smem_desc(sm.iterator + self.off_hsf + t * 1152, False)
                with cute.arch.elect_one():
                    for k in cutlass.range_constexpr(6):
                        copy_scale(self.col_hsf + 24 * t + 4 * k, hsf + k * 8)
                cute.arch.sync_warp()
                for i in cutlass.range_constexpr(5):
                    if i < tiles:
                        g = self.batch * 7 + t * tiles + i
                        slot = g % 6
                        wait_tmem(full.iterator + slot, (g // 6) % 2)
                        with cute.arch.elect_one():
                            col = self.col_sfa + 8 * slot
                            copy_scale(col, sfa + 64 * slot)
                            copy_scale(col + 4, sfa + 64 * slot + 32)
                            d, a = self.col_acc2 + 8 * (5 * t + i), adesc + 2048 * slot
                            mma_steps(d, a, hdesc, col, self.col_hsf + 24 * t, 0, True)
                            commit_asm(empty.iterator + slot)
                            commit_asm(acc2.iterator + 5 * t + i)
                        cute.arch.sync_warp()
        else:
            n = warp - 2
            packed = cute.make_rmem_tensor((self.batch, 4), Uint32)
            for t in cutlass.range_constexpr(self.batch):
                pointer = x.iterator + t * 3584 + 1792 * half + 256 * n + lane * 8
                ready = False
                while not ready:
                    words = load_volatile(pointer, count=4)
                    ready = published(words)
                    for k in cutlass.range_constexpr(4):
                        packed[t, k] = words[k]
            block, k = lane // 4, lane % 4
            kc, kb = 16 * n + 2 * block + k // 2, 8 * n + block
            for t in cutlass.range_constexpr(self.batch):
                sb, words = quantize_mx8(packed, t)
                offset = self.off_x + kc // 8 * 1024 + t * 128 + ((kc % 8) ^ t) * 16
                offset += (k % 2) * 8
                store_shared_words(sm.iterator + offset, words)
                if k == 0:
                    sm[self.off_xsf + kb // 4 * 128 + 16 * t + kb % 4] = Uint8(sb)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(xall.iterator)
            if (warp >= 4) & (warp < 8):
                ew = warp % 4
                own = ew // 2 == half
                tl = tmem + (Uint32(32 * ew) << 16)
                cute.arch.cluster_wait()
                partial = view(sm.iterator, self.off_partial, Float32, self.batch * 64)
                maxima = view(sm.iterator, self.off_amax, Float32, self.batch * 4)
                for t in cutlass.range(self.batch, unroll=1):
                    wait_tmem(acc1.iterator + t)
                    value = load_tmem(tl + 8 * t + t)
                    index = 64 * t + (ew % 2) * 32 + lane
                    if not own:
                        dst = cute.recast_ptr(partial.iterator + index, dtype=Uint32)
                        bits = Uint32(float_bits(value))
                        store_async_dsmem(dst, bits, precv.iterator + t, rank ^ 1)
                    else:
                        wait(precv.iterator + t, test=True)
                        value += partial[index]
                        other = cute.arch.shuffle_sync_bfly(value, 8)
                        up = (lane & 8) == 0
                        hidden = situ(other, value, *situ_args) if up else Float32(0)
                        amax = warp_max_abs(hidden)
                        if lane == 0:
                            maxima[4 * t + ew] = amax
                        cute.arch.barrier(barrier_id=1, number_of_threads=64)
                        sb = mx_exponent(max(amax, maxima[4 * t + (ew ^ 1)]))
                        if up:
                            byte = fp8_bits(hidden * mx_rescale(sb))
                            stage = self.off_stage + 64 * t + 16 * ew
                            sm[stage + 2 * (lane % 8) + lane // 16] = byte
                        cute.arch.sync_warp()
                        if lane < 6:
                            src = sm.iterator + self.off_stage + 64 * t + 16 * ew
                            h_words = load_words(src, count=4, shared=True)
                            kc = 4 * row_tile + ew
                            h_offset = self.off_h + 2048 * t + kc // 8 * 1024
                            h_offset += kc % 8 * 16
                            dst = sm.iterator + cute.assume(h_offset, divby=16)
                            store_async_dsmem(dst, h_words, hrecv.iterator + t, lane)
                            if ew % 2 == 1:
                                sw = (Uint32(sb) * Uint32(0x01010101),) * 4
                                scale_offset = self.off_hsf + 1152 * t + 128 * rank
                                dst = sm.iterator + cute.assume(scale_offset, divby=16)
                                store_async_dsmem(dst, sw, hrecv.iterator + t, lane)
                for t in cutlass.range(self.batch, unroll=1):
                    for i in range(tiles):
                        wait_tmem(acc2.iterator + 5 * t + i)
                        value = load_tmem(tl + self.col_acc2 + 8 * (5 * t + i))
                        other = cute.arch.shuffle_sync_down(value, 8)
                        tile = tile0 + i
                        column = tile * 128 + 32 * ew + 4 * (lane % 8) + lane // 8
                        if (lane & 8) == 0:
                            dst = output.iterator + (t * 16 + j) * 3584 + column
                            store_pair(dst, value, other)
        release_tmem(tmem, 256, warp, 2)
        if (tid == 0) & (rank == 0):
            for t in cutlass.range(self.batch, unroll=1):
                ids[t * 16 + j] = Int32(-1)
        if (cta == 0) & (tid == 0):  # one-thread exit wait, as in the front
            cute.arch.griddepcontrol_wait()


def _align_1024(value):
    return (value + 1023) // 1024 * 1024


class ClusterExpertsTP8(_ClusterExperts):
    """FP4 x MXFP8 FC1 -> SiTU -> MXFP8 -> FC2 of x BF16 [3584] and IDs int32 [16] into
    BF16 [16, 3584]. Weights stay in the loaded MoE weights' shuffled layout (three TMA
    maps, two scale arrays); mailbox mode consumes/re-arms IDs and the latent row via a
    grid ticket."""

    slots = 6
    slot_bytes = 32768
    off_sfa = slots * slot_bytes
    off_x = _align_1024(off_sfa + slots * 1024)
    off_xsf = _align_1024(off_x + 27 * 128 + 1024)
    off_h = _align_1024(off_xsf + 28 * 128 + 384)
    off_hsf = off_h + 3 * 1024
    off_hsc = off_hsf + 3 * 128 + 384
    off_partial = off_hsc + 32
    off_stage = off_partial + 1024
    off_amax = off_stage + 64
    off_bar = off_amax + 16
    off_tmem = off_bar + (2 * slots + 8 + 3) * 8
    smem_bytes = off_tmem + 16
    threads = 256

    @cute.jit
    def issue(self, sm, maps, scales, index, row, chunk, offset, n, full, empty, count):
        slot = n % 6
        phase = (n // 6) % 2
        if n >= 6:
            wait(empty.iterator + slot, phase ^ 1)
        cute.arch.mbarrier_arrive_and_expect_tx(full.iterator + slot, count * 8704)
        dst, desc = sm.iterator + slot * 32768, maps.iterator + index * 128
        load_tma3(dst, desc, full.iterator + slot, row, chunk)
        dst, src = sm.iterator + self.off_sfa + slot * 1024, scales.iterator + offset
        copy_evict(dst, src, size=count * 512, barrier=full.iterator + slot)

    @cute.kernel
    def kernel(self, args):
        x, ids, maps, s13, s2, output, counter, *situ_args = args
        tid, warp, lane, cta = thread()
        rank, j = cta % 6, cta // 6
        sm = alloc(cutlass.utils.SmemAllocator(), Uint8, self.smem_bytes, 1024)
        full = view(sm.iterator, self.off_bar, Uint64, 6)
        empty = view(sm.iterator, self.off_bar + 48, Uint64, 6)
        accb = view(sm.iterator, self.off_bar + 96, Uint64, 8)
        xbar, hbar = accb.iterator + 8, accb.iterator + 9
        hsc = view(sm.iterator, self.off_hsc, Uint32, 6)
        tmem_base = view(sm.iterator, self.off_tmem, Uint32, 1)
        t2lo, t2hi = 28 * rank // 6, 28 * (rank + 1) // 6
        if tid == 0:
            for s in cutlass.range_constexpr(6):
                cute.arch.mbarrier_init(full.iterator + s, 1)
                cute.arch.mbarrier_init(empty.iterator + s, 1)
            for a in cutlass.range_constexpr(8):
                cute.arch.mbarrier_init(accb.iterator + a, 1)
            cute.arch.mbarrier_init(xbar, 6)
            cute.arch.mbarrier_init(hbar, 1)
            cute.arch.mbarrier_arrive_and_expect_tx(hbar, 6 * 68)
            cute.arch.mbarrier_init_fence()
            for i in cutlass.range_constexpr(3):
                acquire_tensormap_asm(maps.iterator + i * 128)
        if warp == 1:
            cute.arch.alloc_tmem(256, tmem_base.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
            fence_tmem(before=True)
        cute.arch.sync_threads()
        cute.arch.cluster_arrive_relaxed()
        fence_tmem()
        tmem = tmem_base[0]
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()
        if warp == 0:
            if lane == 0:
                expert = ids[j]
                for n in range(14):
                    row, chunk = expert * 768 + rank * 128, 2 * n
                    offset = Int64(expert) * 86016 + rank * 14336 + n * 1024
                    self.issue(sm, maps, s13, 0, row, chunk, offset, n, full, empty, 2)
                n = Int32(14)
                for tile in range(t2lo, t2hi):
                    offset = Int64(expert) * 43008 + tile * 1536
                    row = expert * 3584 + tile * 128
                    self.issue(sm, maps, s2, 1, row, 0, offset, n, full, empty, 2)
                    n += 1
                    offset += 1024
                    self.issue(sm, maps, s2, 2, row, 2, offset, n, full, empty, 1)
                    n += 1
        elif warp == 1:
            adesc = smem_desc(sm.iterator, True)
            sfa = smem_desc(sm.iterator + self.off_sfa, False)
            xdesc = smem_desc(sm.iterator + self.off_x, True)
            xsf = smem_desc(sm.iterator + self.off_xsf, False)
            hdesc = smem_desc(sm.iterator + self.off_h, True)
            hsf = smem_desc(sm.iterator + self.off_hsf, False)
            wait_tmem(xbar)
            for n in range(14):
                slot = n % 6
                wait_tmem(full.iterator + slot, (n // 6) % 2)
                for half in cutlass.range_constexpr(2):
                    kc = 2 * n + half
                    empty_address = acc_address = Uint32(0)
                    if cutlass.const_expr(half == 1):
                        empty_address = shared_address(empty.iterator + slot)
                        if n == 13:
                            acc_address = shared_address(accb.iterator)
                    a = (adesc + slot * 2048 + half * 1024, xdesc + kc * 8)
                    sf = (tmem + 64 + 8 * slot + 4 * half, sfa + slot * 64 + half * 32)
                    sfx = (tmem + 128 + 4 * kc, xsf + kc * 8, 1, Int32(kc != 0))
                    chunk_mma(tmem, *a, *sf, *sfx, empty_address, acc_address)
                cute.arch.sync_warp()
            wait(hbar)
            if lane < 3:
                word = (hsc[2 * lane] & 0xFFFF) | (hsc[2 * lane + 1] << 16)
                dst = sm.iterator + self.off_hsf + lane * 128
                store_shared_words(dst, (word,) * 4)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_warp()
            n = Int32(14)
            for tile in range(t2lo, t2hi):
                a = 2 + tile - t2lo
                for stage in cutlass.range_constexpr(2):
                    slot = n % 6
                    wait_tmem(full.iterator + slot, (n // 6) % 2)
                    for half in cutlass.range_constexpr(2 if stage == 0 else 1):
                        kh: cutlass.Constexpr = 2 * stage + half
                        empty_address = acc_address = Uint32(0)
                        if cutlass.const_expr(half == (1 if stage == 0 else 0)):
                            empty_address = shared_address(empty.iterator + slot)
                        if cutlass.const_expr(kh == 2):
                            acc_address = shared_address(accb.iterator + a)
                        ops: tuple = (tmem + 8 * a, adesc + slot * 2048 + half * 1024)
                        ops += (hdesc + kh * 64, tmem + 64 + 8 * slot + 4 * half)
                        ops += (sfa + slot * 64 + half * 32, tmem + 240 + 4 * kh)
                        ops += (hsf + kh * 8, Int32(tile == t2lo), int(kh != 0))
                        chunk_mma(*ops, empty_address, acc_address)
                    cute.arch.sync_warp()
                    n += 1
        else:
            xtid = tid - 64
            packed = cute.make_rmem_tensor((3, 4), Uint32)
            for i in cutlass.range_constexpr(3):
                if 32 * (warp - 2) + 192 * i < 448:
                    words = load_words(
                        x.iterator + 8 * (xtid + 192 * i), count=4, mutable=True
                    )
                    for k in cutlass.range_constexpr(4):
                        packed[i, k] = words[k]
            for i in cutlass.range_constexpr(3):
                f = xtid + 192 * i
                if 32 * (warp - 2) + 192 * i < 448:
                    sb, words = quantize_mx8(packed, i)
                    kc, chunk = f // 16, (f % 16) // 2
                    offset = self.off_x + kc * 128 + (chunk ^ kc % 8) * 16 + f % 2 * 8
                    store_shared_words(sm.iterator + offset, words)
                    kb = f // 4
                    if f % 4 == 0:
                        sm[self.off_xsf + (kb // 4) * 128 + kb % 4] = Uint8(sb)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(xbar)
            if warp >= 4:
                ew = warp % 4
                address = tmem + (Uint32(32 * ew) << 16)
                up = (lane & 8) == 0
                local = 2 * (lane % 8) + lane // 16
                cute.arch.cluster_wait()
                wait_tmem(accb.iterator)
                value = load_tmem(address)
                other = cute.arch.shuffle_sync_bfly(value, 8)
                hidden = situ(other, value, *situ_args) if up else Float32(0)
                amax = abs(hidden)
                for off in cutlass.range_constexpr(5):
                    amax = max(amax, cute.arch.shuffle_sync_bfly(amax, 1 << off))
                maxima = view(sm.iterator, self.off_amax, Float32, 4)
                if lane == 0:
                    maxima[ew] = amax
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                sb = mx_exponent(max(amax, maxima[ew ^ 1]), True)
                if up:
                    byte = fp8_bits(hidden * mx_rescale(sb))
                    sm[self.off_stage + ew * 16 + local] = byte
                cute.arch.sync_warp()
                if lane == 0:
                    src = sm.iterator + self.off_stage + ew * 16
                    words = load_words(src, count=4, shared=True)
                    kc = 4 * rank + ew
                    offset = self.off_h + (kc // 8) * 1024 + (kc % 8) * 16
                    sw = Uint32(mx_exponent(max(maxima[0], maxima[1]), True))
                    sw |= Uint32(mx_exponent(max(maxima[2], maxima[3]), True)) << 8
                    for peer in range(6):
                        dst = sm.iterator + cute.assume(offset, divby=16)
                        store_async_dsmem(dst, words, hbar, peer)
                        if ew == 0:
                            store_async_dsmem(hsc.iterator + rank, sw, hbar, peer)
                for tile in range(t2lo, t2hi):
                    a = 2 + tile - t2lo
                    wait_tmem(accb.iterator + a)
                    value = load_tmem(address + 8 * a)
                    other = cute.arch.shuffle_sync_down(value, 8)
                    if up:
                        column = tile * 128 + 32 * ew + 4 * (lane % 8) + lane // 8
                        dst = output.iterator + j * 3584 + column
                        store_bf16_pair(dst, value, other)
        release_tmem(tmem, 256, warp, 1)
