# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP16 B1/B2 MoE front: SM placement, early router/latent-down staging, MMA router
and shared gate/up, FP32 routing, latent multicast/gather and shared SiTU/down rows."""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Uint8, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from vllm.models.common.ops.moe_tail_fusion.primitives import to_cute
from ...common.primitives import (
    alloc,
    asm,
    bf16,
    copy_bulk,
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
    prefetch_bytes,
    shared_situ,
    sigmoid,
    sleep_ns,
    smem_words,
    split,
    thread,
    unpack_bf16,
    view,
    wait,
    warp_sum,
    zeros,
)
from ..routing import Top16Fast, bf16_word, gather_x, score_record, stage_rows


@asm()
def sreg(name, dtype=Uint32, *, pure=False):
    bits, kind = (64, "l") if dtype is Uint64 else (32, "r")
    return f"mov.u{bits} $0,%{name};", f"={kind}", (), dtype, pure


@asm(Uint32)
def claim_add(pointer):
    return "atom.relaxed.gpu.global.add.u32 $0,[$1],1;", "=r,l,~{memory}", (pointer,)


@asm()
def counter_reduce(pointer, value, *, op="add", release=False):
    ordering = "release" if release else "relaxed"
    code = f"red.{ordering}.gpu.global.{op}.{'b32' if op == 'or' else 'u32'} [$0],$1;"
    return code, "l,r,~{memory}", (pointer, Uint32(value))


@asm(Uint32)
def load_arrivals(pointer):
    return "ld.acquire.gpu.global.u32 $0,[$1];", "=r,l,~{memory}", (pointer,)


class FrontPlacement:
    """One claiming thread per CTA; every CTA publishes its arrival."""

    def __init__(self, batch):
        self.batch = batch
        self.offset = 8 if batch == 2 else 0

    @cute.jit
    def claim(self, epoch, designated, par, cta):
        pointer = epoch.iterator + (2 + par) * 32 + self.offset
        sm = sreg("smid", pure=True)
        claimed = Int32(-1)
        if sm < designated.shape[0] - 16:  # noqa: SIM102 - preserve guarded load
            if designated[sm] != 0:
                claimed = Int32(claim_add(pointer))
                counter_reduce(pointer + 2, Uint32(0xFFFFFFFF) - Uint32(cta), op="max")
                if cutlass.const_expr(self.batch == 2):
                    mask = pointer + 3 + (cta // 32)
                    counter_reduce(mask, Uint32(1) << (cta % 32), op="or")
        counter_reduce(pointer + 1, Uint32(1), release=True)
        return claimed

    @cute.jit
    def rearm(self, epoch, par):
        pointer = epoch.iterator + (2 + (par ^ 1)) * 32 + self.offset
        for i in cutlass.range_constexpr(8 if self.batch == 2 else 3):
            mailbox_store(pointer + i, (Uint32(0),))

    @cute.jit
    def role(self, epoch, par, cta, claimed, grid):
        pointer = epoch.iterator + (2 + par) * 32 + self.offset
        count = Int32(0)
        role = Int32(-1)
        if claimed >= 0:
            arrived = load_arrivals(pointer + 1)
            while arrived < grid:
                sleep_ns(64)
                arrived = load_arrivals(pointer + 1)
            count = Int32(mailbox_load(pointer)[0])
            first = Uint32(0xFFFFFFFF) - mailbox_load(pointer + 2)[0]
            if cutlass.const_expr(self.batch == 1):
                role = claimed | (Int32(first == cta) << 16)
            else:
                order = Int32(0)
                for word in cutlass.range_constexpr(5):
                    bits = mailbox_load(pointer + 3 + word)[0]
                    if word < cta // 32:
                        order += Int32(cute.arch.popc(bits))
                    elif word == cta // 32:
                        below = (Uint32(1) << (cta % 32)) - Uint32(1)
                        order += Int32(cute.arch.popc(bits & below))
                role = order | (Int32(order < 2) << 16)
        return role, count


class FrontPlacementProbe:
    def __init__(self, cluster):
        self.cluster = cluster

    @cute.jit
    def __call__(self, *args):
        grid, smem = args[0].shape[0] // 2, 220 * 1024
        launch(self.kernel, args, grid, 128, self.cluster, pdl=False, smem=smem)

    @cute.kernel
    def kernel(self, args):
        output, spin = args
        tid, _, _, cta = thread()
        sm, cluster = sreg("smid", pure=True), sreg("clusterid.x")
        start = sreg("globaltimer", Uint64)
        if tid == 0:
            output[2 * cta] = Int32(sm)
            output[2 * cta + 1] = Int32(cluster)
        while sreg("globaltimer", Uint64) - start < spin:
            pass


@functools.cache
def _compile_probe(device, cluster, length):
    output = make_fake_tensor(Int32, (length,), (1,), assumed_align=16)
    with torch.accelerator.device_index(device):
        probe = FrontPlacementProbe(cluster)
        return cute.compile(probe, output, Uint64(20000), make_fake_stream())


def plan_designated(groups, *, engine_clusters=16, cluster_size=6):
    """Largest-capacity-first plan of the routed-expert clusters over the SM groups."""
    capacities = [len(group) // cluster_size for group in groups]
    if sum(capacities) < engine_clusters:
        raise ValueError("SM probe found insufficient routed-engine cluster capacity")
    order = sorted(range(len(groups)), key=lambda i: (-capacities[i], -len(groups[i])))
    counts = [0] * len(groups)
    remaining = engine_clusters
    for i in order:
        counts[i] = min(capacities[i], remaining)
        remaining -= counts[i]
    keep = set()
    for group, count in zip(groups, counts):
        quota = len(group) - cluster_size * count
        chosen: list[int] = []
        for sm in sorted(group):
            if sm % 2 == 0 and sm + 1 in group and len(chosen) + 2 <= quota:
                chosen.extend((sm, sm + 1))
        for sm in sorted(group):
            if len(chosen) < quota and sm not in chosen:
                chosen.append(sm)
        keep.update(chosen)
    return keep, counts


def initialize_designated(table: torch.Tensor):
    """Probe SM groups and fill the model-owned int32[nsm+16] table; returns CPU audit
    data. Probe storage is freed before capture. Probes: 1- and 3-wave grids of
    2..16-CTA clusters, 220 KiB SMEM (one CTA per SM), a 20 us global-timer spin."""
    device = table.device
    nsm = torch.cuda.get_device_properties(device).multi_processor_count
    parent = list(range(4096))
    seen = set()
    launches = []

    def find(sm):
        while parent[sm] != sm:
            parent[sm] = parent[parent[sm]]
            sm = parent[sm]
        return sm

    with torch.accelerator.device_index(device.index):
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        for cluster in range(2, 17):
            for waves in (1, 3):
                clusters = max(1, (nsm // cluster) * waves)
                output = table.new_full((2 * cluster * clusters,), -1)
                try:
                    compiled = _compile_probe(device.index, cluster, output.numel())
                    compiled(to_cute(output), Uint64(20000), stream)
                except RuntimeError as e:
                    launches.append(dict(cluster=cluster, waves=waves, error=str(e)))
                    break
                torch.accelerator.synchronize(device)
                members: dict[int, list[int]] = {}
                for sm, group in output.view(-1, 2).cpu().tolist():
                    if sm >= 0:
                        members.setdefault(group, []).append(sm)
                        seen.add(sm)
                for group in members.values():
                    for sm in group[1:]:
                        parent[find(sm)] = find(group[0])
                launches.append(dict(cluster=cluster, waves=waves, clusters=clusters))
    if len(seen) != nsm:
        raise RuntimeError(f"Placement probe saw {len(seen)} of {nsm} SMs")
    group_map: dict[int, list[int]] = {}
    for sm in sorted(seen):
        group_map.setdefault(find(sm), []).append(sm)
    groups = sorted(group_map.values(), key=lambda group: group[0])
    keep, counts = plan_designated(groups)
    values = [int(sm in keep) for sm in range(nsm + 16)]
    table.copy_(torch.tensor(values, dtype=torch.int32, device=device))
    return dict(
        groups=groups,
        engine_clusters=counts,
        designated=sorted(keep),
        launches=launches,
    )


class MoeFrontTP16:
    def __init__(self, batch: int, *, grid=None, renorm=True):
        self.batch = batch
        self.grid = grid if grid is not None else 141 if batch == 1 else 134
        self.renorm = renorm
        self.shared_bytes = 15 * 14336 + (512 * batch + 128) * 4 + 16 + 8
        self.placement = FrontPlacement(batch)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, 512, min_blocks_per_mp=1)

    @cute.jit
    def gemv(self, rows, nrows, activation, partials, warp, lane):
        group, quarter = lane // 4, lane % 4
        acc = zeros((4, 4), Float32)
        for iteration in cutlass.range_constexpr(14):
            offset = (warp * 448 + iteration * 32 + 8 * quarter) * 2
            weight = x = (Uint32(0),) * 4
            if group < nrows:
                weight = smem_words(rows + group * 14336 + offset)
            if group < 2 * self.batch:
                x = smem_words(activation + (group // 2) * 14336 + offset)
            b0 = x[2] if (group & 1) != 0 else x[0]
            b1 = x[3] if (group & 1) != 0 else x[1]
            chain: cutlass.Constexpr = iteration % 4
            a = (weight[0], weight[2], weight[1], weight[3])
            mma_into(acc[chain, None], a, (b0, b1), bf16=True)
        if (quarter < self.batch) & (group < nrows):
            chain_sum = lambda c: acc[c, 0] + acc[c, 3]
            total = (chain_sum(0) + chain_sum(1)) + (chain_sum(2) + chain_sum(3))
            partials[(warp * 16 + group) * self.batch + quarter] = total

    @cute.jit
    def gate_pass(self, slots, rows, start, bars, partials, hg, par, tid):
        wait(bars + 1)
        warp, lane = tid // 32, tid % 32
        self.gemv(slots + 9 * 14336, rows, slots + 7 * 14336, partials, warp, lane)
        cute.arch.sync_threads()
        if tid < self.batch * rows:
            token, row = tid // rows, tid % rows
            total = Float32(0)
            for w in cutlass.range_constexpr(16):
                total += partials[(w * 16 + row) * self.batch + token]
            bits = Uint32(gate_bf16_bits(total))
            pointer = hg.iterator + (par * self.batch + token) * 768 + start + row
            mailbox_store(pointer, ((bits << 16) | Uint32(1),))

    @cute.jit
    def poll_flag(self, pointer):
        word = mailbox_load(pointer)[0]
        while (word & 1) == 0:
            if cutlass.const_expr(self.batch == 2):
                sleep_ns(128)
            word = mailbox_load(pointer)[0]
        return word

    @cute.jit
    def shared_dot(self, packed, hq, token):
        value = Float32(0)
        for chunk in cutlass.range_constexpr(3):
            w = (*unpack_bf16(packed[chunk, 0]), *unpack_bf16(packed[chunk, 1]))
            for j in cutlass.range_constexpr(3, -1, -1):
                value = fma(w[j], hq[token][chunk][j], value)
        return warp_sum(value, 32)

    @cute.jit
    def load_down(self, down, packed, pair, end_pair, lane, fits=True):
        for row in cutlass.range_constexpr(4):
            for chunk in cutlass.range_constexpr(3):
                words = (Uint32(0), Uint32(0))
                if fits & (pair + row // 2 < end_pair):
                    words = early_weights(
                        down.iterator + (2 * pair + row) * 384 + 4 * (lane + 32 * chunk)
                    )
                packed[row, chunk, 0], packed[row, chunk, 1] = words[0], words[1]

    @cute.jit
    def down_dots(self, packed, values, hq, token):
        for row in cutlass.range_constexpr(4):
            values[token, row] = self.shared_dot(packed[row, None, None], hq, token)

    @cute.jit
    def lane_word(self, values, token, lane):
        lo = values[token, 0] if lane == 0 else values[token, 2]
        hi = values[token, 1] if lane == 0 else values[token, 3]
        return bf16_word(lo, hi)

    @cute.jit
    def store_word(self, output, par3, token, column, word):
        row = par3 * output.shape[1] + token
        mailbox_store(output.iterator + row * 7168 + 2 * column, (word,))

    @cute.jit
    def stream_pairs(self, down, hq, output, par3, begin_pair, end_pair, warp, lane):
        packed = cute.make_rmem_tensor((4, 3, 2), Uint32)
        values = cute.make_rmem_tensor((self.batch, 4), Float32)
        for pair in range(begin_pair + 2 * warp, end_pair, 32):
            self.load_down(down, packed, pair, end_pair, lane)
            for token in cutlass.range_constexpr(self.batch):
                self.down_dots(packed, values, hq, token)
                if (lane < 2) & (pair + lane < end_pair):
                    word = self.lane_word(values, token, lane)
                    self.store_word(output, par3, token, pair + lane, word)

    @cute.jit
    def shared_outputs(self, down, hsm, output, par3, first, end, warp, lane):
        begin_pair, end_pair = (first + 1) // 2, end // 2
        hq = [
            [
                load_f32x4(hsm.iterator + t * 384 + 4 * (lane + 32 * c), shared=True)
                for c in range(3)
            ]
            for t in range(self.batch)
        ]
        packed = cute.make_rmem_tensor((4, 3, 2), Uint32)
        values = cute.make_rmem_tensor((self.batch, 4), Float32)
        saved = cute.make_rmem_tensor((3, self.batch), Uint32)
        fits = (end_pair - begin_pair) <= 96
        if cutlass.const_expr(self.batch == 2):
            for iteration in cutlass.range_constexpr(3):
                pair = begin_pair + 2 * warp + 32 * iteration
                self.load_down(down, packed, pair, end_pair, lane, fits)
                for token in cutlass.range_constexpr(2):
                    self.down_dots(packed, values, hq, token)
                    saved[iteration, token] = self.lane_word(values, token, lane)
        if cutlass.const_expr(self.batch == 1) or not fits:
            self.stream_pairs(down, hq, output, par3, begin_pair, end_pair, warp, lane)
        boundary = cute.make_rmem_tensor(self.batch, Uint32)
        if ((end & 1) != 0) & (warp == 0):
            edge = end // 2
            for row in cutlass.range_constexpr(2):
                for chunk in cutlass.range_constexpr(3):
                    words = early_weights(
                        down.iterator + (2 * edge + row) * 384 + 4 * (lane + 32 * chunk)
                    )
                    packed[row, chunk, 0], packed[row, chunk, 1] = words[0], words[1]
            for token in cutlass.range_constexpr(self.batch):
                lo = self.shared_dot(packed[0, None, None], hq, token)
                hi = self.shared_dot(packed[1, None, None], hq, token)
                boundary[token] = bf16_word(lo, hi)
        if cutlass.const_expr(self.batch == 2):
            for iteration in cutlass.range_constexpr(3):
                pair = begin_pair + 2 * warp + 32 * iteration
                if fits & (lane < 2) & (pair + lane < end_pair):
                    for token in cutlass.range_constexpr(2):
                        word = saved[iteration, token]
                        self.store_word(output, par3, token, pair + lane, word)
        if ((end & 1) != 0) & (warp == 0) & (lane == 0):
            for token in cutlass.range_constexpr(self.batch):
                self.store_word(output, par3, token, end // 2, boundary[token])

    @cute.kernel
    def kernel(self, args):
        router, bias, xmoe, epoch, scores, ids, weights, down, latent = args[0:9]
        gate_up, hg, shared_down, output, designated, par, par3, rank = args[9:17]
        multicast_base, scale, situ = args[17], args[18], args[19:23]
        tid, warp, lane, cta = thread()
        router_first, router_rows = split(896, cta, self.grid)
        pair_first, pairs = split(112, cta, self.grid)
        gate_first, gate_rows = Int32(0), Int32(0)
        if cta >= self.batch:
            part, parts = cta - self.batch, self.grid - self.batch
            gate_first, gate_rows = split(768, part, parts)
        slots = alloc(cutlass.utils.SmemAllocator(), Uint8, self.shared_bytes).iterator
        top, size = 15 * 14336, 256 * self.batch
        partials = view(slots, top, Float32, size)
        latent_partial = view(slots, top + 4 * size, Float32, (16, 8), (8, 1))
        gate_partial = view(slots, top + 4 * size + 512, Float32, size)
        bars = cute.recast_ptr(slots + top + 8 * size + 512, dtype=Uint64)
        roles = view(slots, top + 8 * size + 528, Int32, 2)
        hsm = view(slots, 0, Float32, self.batch * 384)
        gate_args = (slots, gate_rows, gate_first, bars, gate_partial, hg, par, tid)
        cute.arch.griddepcontrol_launch_dependents()
        active_lane = lane < 28
        first_chunk = 56 * warp + (lane if active_lane else 0)
        h2_weights = zeros((2, 4 if self.batch == 2 else 2, 4), Uint32)
        for row in cutlass.range_constexpr(2):
            for chunk in cutlass.range_constexpr(4 if self.batch == 2 else 2):
                valid = active_lane & (pairs > 0)
                index = first_chunk + 28 * chunk
                if cutlass.const_expr(self.batch == 2):
                    valid = (warp < 7) & (pairs > 0)
                    index = 32 * warp + lane + 224 * chunk
                if valid:
                    pointer = down.iterator + (2 * pair_first + row) * 7168 + index * 8
                    words = early_weights(pointer, count=4)
                    for j in cutlass.range_constexpr(4):
                        h2_weights[row, chunk, j] = words[j]
        if tid == 0:
            init_barriers(bars, 2, async_proxy=True)
            claimed = self.placement.claim(epoch, designated, par, cta)
            stage_rows(slots, 14336, 0, router, router_first, router_rows, bars)
            if cta == 0:
                mailbox_store(epoch.iterator + (par ^ 1) * 32, (Uint32(0),))
                self.placement.rearm(epoch, par)
            observed = mailbox_load(epoch.iterator + par * 32)[0]
            while observed < 28 * self.batch:
                sleep_ns(32)
                observed = mailbox_load(epoch.iterator + par * 32)[0]
            role, count = self.placement.role(epoch, par, cta, claimed, self.grid)
            roles[0], roles[1] = role, count
        else:
            if (tid >= 32) & (tid < 32 + self.batch * router_rows):
                offset = tid - 32
                token, row = offset // router_rows, offset % router_rows
                index = ((par ^ 1) * self.batch + token) * 896 + router_first + row
                mailbox_store(scores.iterator + index * 2, (Uint32(0x80000000),) * 2)
            if (tid >= 64) & (tid < 64 + self.batch * gate_rows):
                offset = tid - 64
                token, row = offset // max(gate_rows, 1), offset % max(gate_rows, 1)
                index = ((par ^ 1) * self.batch + token) * 768 + gate_first + row
                hg[index] = Uint32(0)
            vectors: cutlass.Constexpr = xmoe.shape[1] * xmoe.shape[2] * 896
            rearm = ((xmoe, vectors, vectors), (latent, 3584, self.batch * 448))
            for tensor, stride, count in rearm:
                for vector in range(cta * 511 + tid - 1, count, self.grid * 511):
                    pointer = tensor.iterator + (par ^ 1) * stride * 8 + vector * 8
                    mailbox_store(pointer, (Uint32(0x80000000),) * 4)
        my_bias = Float32(0)
        if tid < self.batch * router_rows:
            my_bias = bias[router_first + tid % router_rows]
        cute.arch.sync_threads()
        role, count = roles[0], roles[1]
        shared_resident = role >= 0
        consumer = shared_resident & ((role & 0x10000) != 0)
        dense = role & 0xFFFF
        consumer_token = dense if self.batch == 2 else Int32(0)
        shared_first, shared_end = Int32(0), Int32(0)
        if shared_resident:
            shared_first = 7168 * dense // count
            shared_end = 7168 * (dense + 1) // count
        h2 = (warp, lane, pairs) if self.batch == 2 else None
        words_x, words_h2 = gather_x(
            xmoe, par, cta, slots + 7 * 14336, first_chunk, active_lane, self.batch, h2
        )
        if cutlass.const_expr(self.batch == 1):
            for row in cutlass.range_constexpr(2):
                value = Float32(0)
                if pairs > 0:
                    for chunk in cutlass.range_constexpr(2):
                        for j in cutlass.range_constexpr(4):
                            w0, w1 = unpack_bf16(h2_weights[row, chunk, j])
                            x0, x1 = unpack_bf16(words_x[0, chunk, j])
                            value = fma(w0, x0, fma(w1, x1, value))
                value = warp_sum(value, 32)
                if lane == 0:
                    latent_partial[warp, row] = value
        elif (warp < 7) & (pairs > 0):
            for token in cutlass.range_constexpr(2):
                for row in cutlass.range_constexpr(2):
                    value = Float32(0)
                    for chunk in cutlass.range_constexpr(4):
                        for j in cutlass.range_constexpr(4):
                            w0, w1 = unpack_bf16(h2_weights[row, chunk, j])
                            x0, x1 = unpack_bf16(words_h2[token, chunk, j])
                            value = fma(w1, x1, fma(w0, x0, value))
                    value = warp_sum(value, 32)
                    if lane == 0:
                        latent_partial[warp, 2 * token + row] = value
        cute.arch.sync_warp()
        if cutlass.const_expr(self.batch == 1):
            if warp == 1:
                cute.arch.barrier(barrier_id=3, number_of_threads=512)
            else:
                named_barrier(3, 512, arrive=True)
        elif (warp == 1) & (pairs > 0):
            cute.arch.barrier(barrier_id=3, number_of_threads=224)
        elif (warp < 7) & (pairs > 0):
            named_barrier(3, 224, arrive=True)
        if (warp == 1) & (pairs > 0) & (lane < self.batch):
            lo, hi = Float32(0), Float32(0)
            for w in cutlass.range_constexpr(16 if self.batch == 1 else 7):
                lo += latent_partial[w, 2 * lane]
                hi += latent_partial[w, 2 * lane + 1]
            word = bf16_word(lo, hi)
            offset = par * 8 * 3584 + lane * 3584 + rank * 224 + 2 * pair_first
            multicast_words(multicast_base + Uint64(offset * 2), (word,), kind="b32")
        if warp == 1:
            cute.arch.sync_warp()
        if tid == 0:
            if gate_rows > 0:
                cute.arch.mbarrier_arrive_and_expect_tx(bars + 1, gate_rows * 14336)
                dst, src = slots + 9 * 14336, gate_up.iterator + gate_first * 7168
                copy_bulk(dst, src, bars + 1, gate_rows * 14336)
            if shared_resident & (shared_end > shared_first):
                src = shared_down.iterator + shared_first * 384
                prefetch_bytes(src, (shared_end - shared_first) * 768)
        wait(bars)
        self.gemv(slots, router_rows, slots + 7 * 14336, partials, warp, lane)
        cute.arch.sync_threads()
        if tid < self.batch * router_rows:
            token, row = tid // router_rows, tid % router_rows
            logit = Float32(0)
            for w in cutlass.range_constexpr(16):
                logit += partials[(w * 16 + row) * self.batch + token]
            score = sigmoid(logit)
            record = score_record(score, score + my_bias)
            index = (par * self.batch + token) * 896 + router_first + row
            mailbox_store(scores.iterator + index * 2, record)
        if (gate_rows > 0) & (~consumer):
            self.gate_pass(*gate_args)
        cute.arch.sync_threads()
        if consumer & (warp < 8):
            top16 = Top16Fast(self.batch)
            selected = top16.run(scores, ids, slots, par, consumer_token)
            if warp == 0:
                score = selected[lane] if lane < 16 else Float32(0)
                total = warp_sum(score, 16)
                if lane < 16:
                    normalized = score
                    if cutlass.const_expr(self.renorm):
                        # Like the TP8 front and vLLM's grouped_topk kernels.
                        normalized = div_rn(score, total + Float32(1e-20))
                    bits = Uint32(gate_bf16_bits(normalized * scale))
                    if bits == 0x8000:
                        bits = Uint32(0)
                    high = cute.arch.shuffle_sync_down(bits, 1, mask=0xFFFF)
                    if (lane & 1) == 0:
                        slot = par3 * weights.shape[1] + consumer_token
                        pointer = weights.iterator + slot * 16 + lane
                        mailbox_store(pointer, (bits | (high << 16),))
                cute.arch.sync_warp()
        if consumer & (gate_rows > 0):
            cute.arch.sync_threads()
            self.gate_pass(*gate_args)
        if shared_resident:
            if consumer:
                cute.arch.sync_threads()
            for element in range(tid, self.batch * 384, 512):
                token, column = element // 384, element % 384
                gp = hg.iterator + (par * self.batch + token) * 768 + column
                gate, up = self.poll_flag(gp), self.poll_flag(gp + 384)
                result = shared_situ(bf16(gate, True), bf16(up, True), *situ)
                hsm[element] = Float32(BFloat16(result))
            cute.arch.sync_threads()
            self.shared_outputs(
                shared_down, hsm, output, par3, shared_first, shared_end, warp, lane
            )
            cute.arch.sync_threads()
        # One-thread exit wait: completion then covers the o-proj consumer (and its
        # mailbox re-arm) for later waits.
        if (cta == 0) & (tid == 0):
            cute.arch.griddepcontrol_wait()
