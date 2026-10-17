# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP8 B1 small-batch front (full router staged in SMEM, x_moe polling, threshold top16,
latent multicast and gather, shared gate/up) and the shared-expert finisher."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Uint8, Uint32, Uint64

from ...common.primitives import (
    alloc,
    bf16,
    copy_bulk,
    copy_g2s,
    div_rn,
    early_weights,
    fma,
    gate_bf16_bits,
    init_barriers,
    launch,
    load_f32x4,
    mailbox_load,
    mailbox_store,
    mma_into,
    multicast_words,
    named_barrier,
    published,
    routing_add,
    routing_sigmoid,
    routing_weight,
    sigmoid,
    smem_words,
    split,
    store_gate_mailbox,
    store_relaxed,
    table_load,
    thread,
    unpack_bf16,
    view,
    wait,
    warp_sum,
    zeros,
)
from ..routing import Top16Fast, bf16_word, gather_x, score_record, stage_rows


class MoeSharedTP8:
    ring_bytes = 57344
    row_bytes = 14400
    lsu_rows = 92
    shared_bytes = ring_bytes + row_bytes + lsu_rows * 1536 + 4096 + 24

    def __init__(self, grid=52):
        self.grid = grid

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, 512, min_blocks_per_mp=1)

    @cute.jit
    def stage(self, dst, down, low, high, barrier):
        cute.arch.mbarrier_arrive_and_expect_tx(barrier, (high - low) * 1536)
        if high > low:
            copy_bulk(dst, down.iterator + low * 768, barrier, (high - low) * 1536)

    @cute.jit
    def rows_out(self, base, low, start, high, hsm, output, par3, lane):
        packed = cute.make_rmem_tensor((6, 2), Uint32)
        for row in range(start, high, 16):
            for chunk in cutlass.range_constexpr(6):
                pointer = base + (row - low) * 1536 + (lane + 32 * chunk) * 8
                values = smem_words(pointer, count=2)
                packed[chunk, 0], packed[chunk, 1] = values[0], values[1]
            self.row_out(row, packed, hsm, output, par3, lane)

    @cute.jit
    def row_out(self, row, packed, hsm, output, par3, lane):
        value = Float32(0)
        for chunk in cutlass.range_constexpr(6):
            h = load_f32x4(hsm.iterator + 4 * (lane + 32 * chunk), shared=True)
            w0, w1 = unpack_bf16(packed[chunk, 0])
            w2, w3 = unpack_bf16(packed[chunk, 1])
            value = fma(w0, h[0], fma(w1, h[1], fma(w2, h[2], fma(w3, h[3], value))))
        value = warp_sum(value, 32)
        if lane == 0:
            pointer = output.iterator + par3 * output.shape[1] * 7168 + row
            store_gate_mailbox(pointer, gate_bf16_bits(value))

    @cute.kernel
    def kernel(self, args):
        xmoe, down, tables, hg, output, par, par3 = args[0:7]
        tid, warp, lane, cta = thread()
        first = 7168 * cta // self.grid
        end = 7168 * (cta + 1) // self.grid
        first_end = min(end, first + 37)
        second_end = min(end, first_end + 9)
        lsu_count = min(92, end - second_end)
        storage = alloc(cutlass.utils.SmemAllocator(), Uint8, self.shared_bytes, 1024)
        ring = storage.iterator
        xslot = ring + self.ring_bytes
        lsu = xslot + self.row_bytes
        partial = lsu + self.lsu_rows * 1536
        hsm = view(partial, 1024, Float32, 768)
        bars = cute.recast_ptr(partial + 4096, dtype=Uint64)
        cute.arch.griddepcontrol_launch_dependents()
        if tid == 0:
            init_barriers(bars, 3, async_proxy=True)
            self.stage(ring, down, first, first_end, bars + 1)
        active = lane < 28
        gather_x(xmoe, par, cta, xslot, 56 * warp + (lane if active else 0), active)
        cute.arch.sync_warp()
        if lsu_count > 0:
            for vector in range(tid, lsu_count * 96, 512):
                dst = lsu + vector * 16
                copy_g2s(dst, down.iterator + second_end * 768 + vector * 8)
            cute.arch.cp_async_commit_group()
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        if tid == 0:
            cute.arch.fence_proxy("async.shared", space="cta")
            self.stage(xslot, down, first_end, second_end, bars + 2)
        if tid < 384:
            column = 2 * tid
            gate = up = (Uint32(0), Uint32(0))
            valid = False
            while not valid:
                gate = mailbox_load(hg.iterator + par * 2 * 1536 + column, count=2)
                up = mailbox_load(hg.iterator + par * 2 * 1536 + 768 + column, count=2)
                valid = ((gate[0] & gate[1] & up[0] & up[1]) & Uint32(1)) != 0
            ga0 = table_load(tables.iterator + (gate[0] >> 16))
            ua0 = table_load(tables.iterator + 65536 + (up[0] >> 16))
            ga1 = table_load(tables.iterator + (gate[1] >> 16))
            ua1 = table_load(tables.iterator + 65536 + (up[1] >> 16))
            hsm[column] = Float32(BFloat16(ga0 * ua0))
            hsm[column + 1] = Float32(BFloat16(ga1 * ua1))
        cute.arch.sync_threads()
        out = (hsm, output, par3, lane)
        if lsu_count > 0:
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            self.rows_out(
                lsu, second_end, second_end + warp, second_end + lsu_count, *out
            )
        parts = ((ring, first, first_end), (xslot, first_end, second_end))
        for part, (base, low, high) in enumerate(parts):
            if high > low:
                if warp == 0:
                    wait(bars + 1 + part)
                cute.arch.sync_threads()
                self.rows_out(base, low, low + warp, high, *out)
        if cta == 0:
            cute.arch.griddepcontrol_wait()


class SharedSituTables:
    """Separate FP32 gate/up lookup tables over all BF16 patterns."""

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, 256, 256, pdl=False)

    @cute.kernel
    def kernel(self, args):
        tables, beta, linear_beta = args[0:3]
        tid, _, _, cta = thread()
        index = 256 * cta + tid
        value = bf16(Uint32(index))
        gate = div_rn(value, beta)
        probability = sigmoid(value)
        tables[index] = (beta * cute.math.tanh(gate, fastmath=False)) * probability
        up = value
        if linear_beta > 0:
            scaled = div_rn(value, linear_beta)
            up = linear_beta * cute.math.tanh(scaled, fastmath=False)
        tables[65536 + index] = up


class MoeFrontTP8:
    row_bytes = 14400
    shared_bytes = 15 * row_bytes + (16 * 16 + 16 * 4) * 4 + 24

    def __init__(self, grid=140, *, renorm=True):
        self.grid = grid
        self.renorm = renorm

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, 512, min_blocks_per_mp=1)

    @cute.jit
    def gemv(self, rows, nrows, activation, partials, warp, lane):
        group, quarter = lane // 4, lane % 4
        acc = zeros((8, 4), Float32)
        for iteration in cutlass.range_constexpr(14):
            offset = (warp * 448 + iteration * 32 + 8 * quarter) * 2
            lo = hi = x = (Uint32(0),) * 4
            if group < nrows:
                lo = smem_words(rows + group * self.row_bytes + offset)
            if group + 8 < nrows:
                hi = smem_words(rows + (group + 8) * self.row_bytes + offset)
            if group < 1:
                x = smem_words(activation + offset)
            for half in cutlass.range_constexpr(2):
                chain: cutlass.Constexpr = (2 * iteration + half) % 8
                a = (lo[2 * half], hi[2 * half], lo[2 * half + 1], hi[2 * half + 1])
                mma_into(acc[chain, None], a, (x[2 * half], x[2 * half + 1]), bf16=True)
        if quarter == 0:
            for half in cutlass.range_constexpr(2):
                comp: cutlass.Constexpr = 2 * half
                partials[warp * 16 + group + 8 * half] = (
                    (acc[0, comp] + acc[1, comp]) + (acc[2, comp] + acc[3, comp])
                ) + ((acc[4, comp] + acc[5, comp]) + (acc[6, comp] + acc[7, comp]))

    @cute.jit
    def gate_pass(self, rows, count, start, barrier, ctx):
        activation, partials, hg, par, tid, warp, lane = ctx
        wait(barrier)
        self.gemv(rows, count, activation, partials, warp, lane)
        cute.arch.sync_threads()
        if tid < count:
            value = Float32(0)
            for w in cutlass.range_constexpr(16):
                value += partials[w * 16 + tid]
            bits = Uint32(gate_bf16_bits(value))
            pointer = hg.iterator + (par * 2) * 1536 + start + tid
            mailbox_store(pointer, ((bits << 16) | Uint32(1),))
        cute.arch.sync_threads()

    @cute.kernel
    def kernel(self, args):
        router, bias, xmoe, epoch, scores, ids, weights, down = args[:8]
        latent, latent_out, gate_up, hg, par, par3, rank, multicast_base = args[8:16]
        scale = args[16]
        tid, warp, lane, cta = thread()
        first_router, router_rows = split(896, cta, self.grid)
        first_pair, pairs = split(224, cta, self.grid)
        gu_start, gu_rows = Int32(0), Int32(0)
        if cta > 0:
            gu_start, gu_rows = split(1536, cta - 1, self.grid - 1)
        gu_first = min(gu_rows, 7)
        gu_second = gu_rows - gu_first
        storage = alloc(cutlass.utils.SmemAllocator(), Uint8, self.shared_bytes, 128)
        slots = storage.iterator
        activation = slots + 7 * self.row_bytes
        dst = (slots, self.row_bytes)
        top = 15 * self.row_bytes
        partials = view(slots, top, Float32, 256)
        latent_partial = view(slots, top + 1024, Float32, (16, 4), (4, 1))
        bars = cute.recast_ptr(slots + top + 1280, dtype=Uint64)
        cute.arch.griddepcontrol_launch_dependents()
        active_lane = lane < 28
        first_chunk = 56 * warp + (lane if active_lane else 0)
        second_chunk = first_chunk + 28
        latent_weights = zeros((4, 2, 4), Uint32)
        for row in cutlass.range_constexpr(4):
            if active_lane & (row < 2 * pairs):
                for chunk in cutlass.range_constexpr(2):
                    col = first_chunk if chunk == 0 else second_chunk
                    pointer = down.iterator + (2 * first_pair + row) * 7168 + 8 * col
                    words = early_weights(pointer, count=4)
                    for j in cutlass.range_constexpr(4):
                        latent_weights[row, chunk, j] = words[j]
        cute.arch.cp_async_commit_group()
        if tid == 0:
            init_barriers(bars, 3, async_proxy=True)
            stage_rows(*dst, 0, router, first_router, router_rows, bars)
            if cta == 0:
                mailbox_store(epoch.iterator + (par ^ 1) * 32, (Uint32(0),))
            observed = mailbox_load(epoch.iterator + par * 32)[0]
            while observed < 28:
                observed = mailbox_load(epoch.iterator + par * 32)[0]
        else:
            for i in range(tid - 1, 2 * router_rows, 511):
                token, row = i // router_rows, i % router_rows
                index = ((par ^ 1) * 2 + token) * 896 + first_router + row
                mailbox_store(scores.iterator + index * 2, (Uint32(0x80000000),) * 2)
            if cta > 0:
                for i in range(tid - 1, 2 * gu_rows, 511):
                    token, row = i // gu_rows, i % gu_rows
                    hg[(par ^ 1) * 2 * 1536 + token * 1536 + gu_start + row] = Uint32(0)
            slot_vectors: cutlass.Constexpr = xmoe.shape[1] * xmoe.shape[2] * 896
            for i in range(cta * 511 + tid - 1, slot_vectors, self.grid * 511):
                pointer = xmoe.iterator + (par ^ 1) * slot_vectors * 8 + i * 8
                mailbox_store(pointer, (Uint32(0x80000000),) * 4)
        my_bias = Float32(0)
        if tid < router_rows:
            my_bias = bias[first_router + tid]
        cute.arch.sync_threads()
        input_words, _ = gather_x(xmoe, par, cta, activation, first_chunk, active_lane)
        for row in cutlass.range_constexpr(4):
            if row < 2 * pairs:
                total = Float32(0)
                for chunk in cutlass.range_constexpr(2):
                    for j in cutlass.range_constexpr(4):
                        w0, w1 = unpack_bf16(latent_weights[row, chunk, j])
                        x0, x1 = unpack_bf16(input_words[0, chunk, j])
                        total = fma(w0, x0, fma(w1, x1, total))
                total = warp_sum(total, 32)
                if lane == 0:
                    latent_partial[warp, row] = total
        cute.arch.sync_warp()
        if warp == 1:
            cute.arch.barrier(barrier_id=3, number_of_threads=512)
            if lane < pairs:
                lo, hi = Float32(0), Float32(0)
                for w in cutlass.range_constexpr(16):
                    lo += latent_partial[w, 2 * lane]
                    hi += latent_partial[w, 2 * lane + 1]
                packed = bf16_word(lo, hi)
                offset = par * 8 * 3584 * 2 + rank * 896 + (first_pair + lane) * 4
                multicast_words(multicast_base + Uint64(offset), (packed,), kind="b32")
        else:
            named_barrier(3, 512, arrive=True)
        if tid == 0:  # noqa: SIM102
            if gu_first > 0:
                stage_rows(*dst, 8, gate_up, gu_start, gu_first, bars + 1)
        wait(bars)
        self.gemv(slots, router_rows, activation, partials, warp, lane)
        cute.arch.sync_threads()
        if tid == 0:  # noqa: SIM102
            if gu_second > 0:
                cute.arch.fence_proxy("async.shared", space="cta")
                stage_rows(*dst, 0, gate_up, gu_start, gu_second, bars + 2, gu_first)
        if tid < router_rows:
            logit = Float32(0)
            for w in cutlass.range_constexpr(16):
                logit += partials[w * 16 + tid]
            score = routing_sigmoid(logit)
            record = score_record(score, routing_add(score, my_bias))
            index = par * 2 * 896 + first_router + tid
            mailbox_store(scores.iterator + index * 2, record)
        if cta == 0:
            if warp < 8:
                selected = Top16Fast(2).run(scores, ids, slots, par, 0)
                if warp == 0:
                    score = selected[lane] if lane < 16 else Float32(0)
                    total = warp_sum(score, 16)
                    if lane < 16:
                        value = routing_weight(score, total, scale, renorm=self.renorm)
                        pointer = weights.iterator + par3 * weights.shape[1] * 16 + lane
                        store_gate_mailbox(pointer, gate_bf16_bits(value))
                    cute.arch.sync_warp()
                for vector in range(tid, 448, 256):
                    pointer = latent.iterator + par * 8 * 3584 + vector * 8
                    words = mailbox_load(pointer, count=4, scope="sys")
                    ready = published(words)
                    while not ready:
                        words = mailbox_load(pointer, count=4, scope="sys")
                        ready = published(words)
                    mailbox_store(latent_out.iterator + vector * 8, words)
                    store_relaxed(pointer, (Uint32(0x80000000),) * 4)
            cute.arch.griddepcontrol_wait()
        else:
            cute.arch.sync_threads()
            ctx = (activation, partials, hg, par, tid, warp, lane)
            if gu_first > 0:
                self.gate_pass(
                    slots + 8 * self.row_bytes, gu_first, gu_start, bars + 1, ctx
                )
            if gu_second > 0:
                self.gate_pass(slots, gu_second, gu_start + gu_first, bars + 2, ctx)
