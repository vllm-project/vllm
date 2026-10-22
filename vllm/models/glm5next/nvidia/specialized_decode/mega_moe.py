# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed experts of a few decode tokens as one kernel, one 16-CTA cluster per
(token, expert) route, on the TRTLLM-Gen FP8 weights (DeepSeek block-128 FP32
scales, BlockMajorK [E, K/128, N, 128] with rows shuffled within 64-row tiles).

The block scales are not powers of two, so the tensor cores cannot apply them.
Instead each of a CTA's 16 K blocks puts the token in its own B column (the
rest zero): one TMEM accumulator then holds every K block's unscaled partial in
its own column, the MMAs run back to back as weight stages land, and the
epilogue scales and sums the 16 columns once.

GEMM1 (W31: up rows, then gate rows): CTA r owns row tile r // 2 (128 rows) and
K half r % 2; the K halves meet over DSMEM.

Every warp redoes its token's top-k from the router logits (cheaper than a
kernel boundary), and the epilogue warps quantize the BF16 token into the B
tiles, so routing and the activation quant add no launches.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cute.arch import store_async_dsmem

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    stream,
    tensor_maps,
    to_cute,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    acquire_tensormap_asm,
    alloc,
    asm,
    bits_float,
    div_rn,
    fence_tmem,
    float_bits,
    load_bf16x8,
    load_tma3,
    load_tmem,
    pack_fp8,
    redux,
    release_tmem,
    sigmoid,
    smem_desc,
    view,
    wait,
    wait_tmem,
)

CLUSTER = 16
TILE = 128  # weight rows per MMA (UMMA M) and per K block
COLUMNS = 16  # UMMA N: the route's token in column 0, the rest zero
# Few enough stages (~106 KB of shared memory) that two CTAs share an SM:
# 16-CTA clusters are GPC-bound, and packing two per SM fits more at once.
STAGES = 4
FP8_MAX = 448.0
STAGE_BYTES = TILE * TILE  # one K block of one row tile, FP8
B_BYTES = COLUMNS * TILE  # one K block of the activation operand
# UMMA instruction: D FP32, A and B E4M3, both K-major, N = 16, M = 128.
IDESC = (1 << 4) | ((COLUMNS >> 3) << 17) | ((TILE >> 4) << 24)


@asm()
def _umma_block(d, a, b, instruction, accumulate, empty):
    """Four K32 FP8 UMMAs of one 128-K block into ``d`` (the first accumulates
    iff ``accumulate``) from the elected lane, then commit the weight stage."""
    mma = "tcgen05.mma.cta_group::1.kind::f8f6f4"
    commit = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    code = "{ .reg .pred e,p0,pt; .reg .b64 a1,a2,a3,b1,b2,b3; "
    code += "elect.sync _|e,0xffffffff; setp.ne.b32 p0,$4,0; setp.eq.b32 pt,0,0; "
    code += "add.s64 a1,$1,2; add.s64 a2,$1,4; add.s64 a3,$1,6; "
    code += "add.s64 b1,$2,2; add.s64 b2,$2,4; add.s64 b3,$2,6; "
    code += f"@e {mma} [$0],$1,$2,$3,p0; @e {mma} [$0],a1,b1,$3,pt; "
    code += f"@e {mma} [$0],a2,b2,$3,pt; @e {mma} [$0],a3,b3,$3,pt; "
    code += f"@e {commit} [$5]; }}"
    operands = (Uint32(d), Uint64(a), Uint64(b), Uint32(instruction))
    return code, "r,l,l,r,r,r", (*operands, Uint32(accumulate), empty)


@asm()
def _store_bf16(pointer, value):
    code = "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; st.global.b16 [$0], h; }"
    return code, "l,f", (pointer, value)


@asm(Uint32, pure=True)
def _ordered(value):
    """Monotone uint32 key of a float: flip all bits of negatives, the sign of
    the rest."""
    code = "{ .reg .b32 m; shr.s32 m,$1,31; or.b32 m,m,0x80000000; xor.b32 $0,$1,m; }"
    return code, "=r,r", (float_bits(value),)


@asm()
def _commit(barrier):
    commit = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    code = "{ .reg .pred e; elect.sync _|e,0xffffffff; " + f"@e {commit} [$0]; }}"
    return code, "r", (barrier,)


class _MegaKernel:
    """BF16 [routes, hidden] expert outputs of each route (finalize-ready)."""

    threads = 192  # producer, MMA, 4 epilogue warps

    def __init__(
        self, *, hidden: int, inter: int, experts: int, top_k: int, limit: float,
        scale: float, ue8m0: bool,
    ):  # fmt: skip
        self.hidden, self.inter, self.top_k, self.limit = hidden, inter, top_k, limit
        self.experts, self.scale, self.ue8m0 = experts, scale, ue8m0
        assert experts % 32 == 0
        self.nk1 = hidden // TILE  # GEMM1 K blocks
        self.nk2 = inter // TILE  # GEMM2 K blocks (= up tiles)
        self.tiles1 = 2 * inter // TILE  # GEMM1 row tiles (W31: up, then gate)
        assert self.tiles1 * 2 == CLUSTER and self.nk1 // 2 == COLUMNS
        self.blocks1 = self.nk1 // 2
        self.rows2 = hidden // CLUSTER  # GEMM2 rows per CTA
        self.tiles2 = self.rows2 // TILE
        self.loads = self.blocks1 + self.tiles2 * self.nk2
        self.off_b1 = STAGES * STAGE_BYTES
        self.off_b2 = self.off_b1 + self.blocks1 * B_BYTES
        self.off_partial = self.off_b2 + self.nk2 * B_BYTES
        self.off_gate = self.off_partial + TILE * 4
        self.off_stage = self.off_gate + TILE * 4
        self.off_amax = self.off_stage + TILE
        self.off_hscale = self.off_amax + 16
        self.off_xscale = self.off_hscale + 16
        self.off_bar = self.off_xscale + COLUMNS * 4
        self.bars = 2 * STAGES + 2 + self.tiles2 + 4
        self.off_tmem = self.off_bar + self.bars * 8
        self.smem = self.off_tmem + 16

    @cute.jit
    def __call__(
        self,
        maps: cute.Tensor,
        w13_scale: cute.Tensor,
        w2_scale: cute.Tensor,
        x: cute.Tensor,
        logits: cute.Tensor,
        bias: cute.Tensor,
        rows1: cute.Tensor,
        rows2: cute.Tensor,
        out: cute.Tensor,
        ids: cute.Tensor,
        weights: cute.Tensor,
        routes: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            maps, w13_scale, w2_scale, x, logits, bias, rows1, rows2, out, ids, weights
        ).launch(
            grid=(CLUSTER * routes, 1, 1),
            block=(self.threads, 1, 1),
            cluster=(CLUSTER, 1, 1),
            smem=self.smem + 1024,
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=2,
        )

    @cute.kernel
    def kernel(
        self,
        maps: cute.Tensor,
        w13_scale: cute.Tensor,
        w2_scale: cute.Tensor,
        x: cute.Tensor,
        logits: cute.Tensor,
        bias: cute.Tensor,
        rows1: cute.Tensor,
        rows2: cute.Tensor,
        out: cute.Tensor,
        ids: cute.Tensor,
        weights: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        cta, _, _ = cute.arch.block_idx()
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        route = cta // CLUSTER
        tile = rank // 2
        half = rank % 2
        blocks1, nk2, tiles2 = self.blocks1, self.nk2, self.tiles2
        first = half * blocks1

        sm = alloc(cutlass.utils.SmemAllocator(), Uint8, self.smem, 1024)
        bars = view(sm.iterator, self.off_bar, Uint64, self.bars)
        full = bars.iterator
        empty = full + STAGES
        acc1 = full + 2 * STAGES
        b_ready = acc1 + 1
        acc2 = acc1 + 2
        pair = acc2 + tiles2
        gate_recv = pair + 1
        hrecv = pair + 2
        tmem_base = view(sm.iterator, self.off_tmem, Uint32, 1)
        partial = view(sm.iterator, self.off_partial, Float32, TILE)
        gate_buf = view(sm.iterator, self.off_gate, Float32, TILE)
        amax_buf = view(sm.iterator, self.off_amax, Float32, 4)
        hscale = view(sm.iterator, self.off_hscale, Float32, 4)
        xscale = view(sm.iterator, self.off_xscale, Float32, COLUMNS)
        offsets = cute.make_rmem_tensor(
            cute.make_layout((self.experts // 32,)), Float32
        )
        for j in cutlass.range_constexpr(self.experts // 32):
            offsets[j] = bias[j * 32 + lane]
        if tid == 0:
            for s in cutlass.range_constexpr(STAGES):
                cute.arch.mbarrier_init(full + s, 1)
                cute.arch.mbarrier_init(empty + s, 1)
            cute.arch.mbarrier_init(acc1, 1)
            cute.arch.mbarrier_init(b_ready, 4)
            for t in cutlass.range_constexpr(tiles2):
                cute.arch.mbarrier_init(acc2 + t, 1)
            for b in cutlass.range_constexpr(3):
                cute.arch.mbarrier_init(pair + b, 1)
            if half == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(pair, TILE * 4)
                if tile < nk2:
                    cute.arch.mbarrier_arrive_and_expect_tx(gate_recv, TILE * 4)
            # Every up tile's FP8 hidden block and its scale.
            cute.arch.mbarrier_arrive_and_expect_tx(hrecv, nk2 * (TILE + 4))
            cute.arch.mbarrier_init_fence()
            acquire_tensormap_asm(maps.iterator)
            acquire_tensormap_asm(maps.iterator + 128)
        if warp == 1:
            cute.arch.alloc_tmem(64, tmem_base.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        # GEMM2's B tiles get remote hidden rows: zero them before the cluster
        # barrier releases any sender.
        b2_words = cute.recast_ptr(sm.iterator + self.off_b2, dtype=Uint32)
        chunks = nk2 * B_BYTES // 16
        for i in cutlass.range_constexpr(-(-chunks // self.threads)):
            if i * self.threads + tid < chunks:
                zero = b2_words + (i * self.threads + tid) * 4
                for w in cutlass.range_constexpr(4):
                    cute.arch.store((zero + w).llvm_ptr, Uint32(0))
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.cluster_arrive()
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        tmem = tmem_base[0]

        # No early launch_dependents: the finalize publish reads the routing
        # weights this kernel writes before its own PDL wait, so it may only
        # launch once this grid has exited.
        cute.arch.griddepcontrol_wait()
        pick, weight = self.top_k_pick(logits, offsets, route, lane)
        expert = Int64(pick)
        if rank == 0 and tid == 0:
            ids[route] = pick
            weights[route] = weight

        if warp == 0:
            if lane == 0:
                for n in cutlass.range(self.loads, unroll=1):
                    slot = n % STAGES
                    if n >= STAGES:
                        wait(empty + slot, ((n // STAGES) - 1) % 2, scope="cta")
                    cute.arch.mbarrier_arrive_and_expect_tx(full + slot, STAGE_BYTES)
                    dst = sm.iterator + slot * STAGE_BYTES
                    if n < blocks1:
                        load_tma3(
                            dst, maps.iterator, full + slot, Int32(tile * TILE),
                            Int32(expert * self.nk1 + first + n),
                        )  # fmt: skip
                    else:
                        m = n - blocks1
                        load_tma3(
                            dst, maps.iterator + 128, full + slot,
                            Int32(rank * self.rows2 + (m // nk2) * TILE),
                            Int32(expert * nk2 + m % nk2),
                        )  # fmt: skip
        elif warp == 1:
            adesc = smem_desc(sm.iterator, True)
            b1desc = smem_desc(sm.iterator + self.off_b1, True)
            b2desc = smem_desc(sm.iterator + self.off_b2, True)
            wait(b_ready, 0, scope="cta")
            for n in cutlass.range(blocks1, unroll=1):
                slot = n % STAGES
                wait_tmem(full + slot, (n // STAGES) % 2, scope="cta")
                _umma_block(
                    tmem, adesc + slot * (STAGE_BYTES >> 4),
                    b1desc + n * (B_BYTES >> 4), IDESC, n > 0, empty + slot,
                )  # fmt: skip
                cute.arch.sync_warp()
            _commit(acc1)
            wait(hrecv, 0, scope="cta")
            cute.arch.fence_proxy("async.shared", space="cta")
            for m in cutlass.range(tiles2 * nk2, unroll=1):
                n = blocks1 + m
                slot = n % STAGES
                kb = m % nk2
                wait_tmem(full + slot, (n // STAGES) % 2, scope="cta")
                _umma_block(
                    tmem + COLUMNS * (1 + m // nk2), adesc + slot * (STAGE_BYTES >> 4),
                    b2desc + kb * (B_BYTES >> 4), IDESC, kb > 0, empty + slot,
                )  # fmt: skip
                cute.arch.sync_warp()
                if kb == nk2 - 1:
                    _commit(acc2 + m // nk2)
        else:
            self.epilogue(
                sm, tmem, warp, lane, rank, route, tile, half, first, expert,
                w13_scale, w2_scale, x, rows1, rows2, out,
                b_ready, acc1, acc2, pair, gate_recv, hrecv,
                partial, gate_buf, amax_buf, hscale, xscale,
            )  # fmt: skip
        if warp < 2:
            cute.arch.cluster_wait()
        release_tmem(tmem, 64, warp, 1)

    @cute.jit
    def top_k_pick(self, logits, offsets, route, lane):
        """Expert ``route % top_k`` of the route's token and its weight: sigmoid
        scores (approximate exp and divide, a few ulp), top-k by score plus bias
        (ties: lower expert), renormalized and scaled. Lane l holds experts
        l + 32 j."""
        token = route // self.top_k
        k = route % self.top_k
        per = self.experts // 32
        scores = cute.make_rmem_tensor(cute.make_layout((per,)), Float32)
        keys = cute.make_rmem_tensor(cute.make_layout((per,)), Uint32)
        # Every load first: interleaved with the sigmoids they serialize.
        for j in cutlass.range_constexpr(per):
            scores[j] = Float32(logits[token, j * 32 + lane])
        for j in cutlass.range_constexpr(per):
            scores[j] = sigmoid(scores[j], fast_math=True)
            keys[j] = _ordered(scores[j] + offsets[j])
        total = Float32(0.0)
        pick = Int32(0)
        picked = Float32(0.0)
        for r in cutlass.range_constexpr(self.top_k):
            best = Uint32(0)
            for j in cutlass.range_constexpr(per):
                if keys[j] > best:
                    best = keys[j]
            top = redux(best, "max")
            candidate = Uint32(0xFFFFFFFF)
            for j in cutlass.range_constexpr(per - 1, -1, -1):
                if keys[j] == top:
                    candidate = Uint32(j * 32) + Uint32(lane)
            winner = Int32(redux(candidate, "min"))
            score = Float32(0.0)
            for j in cutlass.range_constexpr(per):
                if winner == j * 32 + lane:
                    score = scores[j]
                    keys[j] = Uint32(0)
            score = cute.arch.shuffle_sync(score, winner % 32)
            total = total + score
            if k == r:
                pick = winner
                picked = score
        return pick, div_rn(picked, total) * Float32(self.scale)

    @cute.jit
    def epilogue(
        self, sm, tmem, warp, lane, rank, route, tile, half, first, expert,
        w13_scale, w2_scale, x, rows1, rows2, out,
        b_ready, acc1, acc2, pair, gate_recv, hrecv,
        partial, gate_buf, amax_buf, hscale, xscale,
    ):  # fmt: skip
        nk2, blocks1 = self.nk2, self.blocks1
        e = (warp - 2) * 32 + lane
        quadrant = warp % 4
        row = quadrant * 32 + lane
        tl = tmem + (Uint32(quadrant * 32) << 16)
        token = Int64(route // self.top_k)

        # K block n of the token, FP8 per 128 like per_token_group_quant_fp8 (with
        # its UE8M0 scale rounding when the stock path uses it), as
        # row (B column) n of B tile n, the rest zero; rows are 128B-swizzled
        # (16B chunk c at c ^ (row % 8)). Thread e quantizes 16 values.
        kb = e // 8
        chunk = e % 8
        source = x.iterator + token * self.hidden + (first + kb) * TILE + chunk * 16
        values = cute.make_rmem_tensor(cute.make_layout((16,)), Float32)
        for h in cutlass.range_constexpr(2):
            got = load_bf16x8(source + h * 8)
            for v in cutlass.range_constexpr(8):
                values[h * 8 + v] = got[v]
        amax = Float32(1e-10)
        for v in cutlass.range_constexpr(16):
            amax = cute.arch.fmax(amax, cute.arch.fmax(values[v], -values[v]))
        for offset in cutlass.range_constexpr(3):
            amax = cute.arch.fmax(
                amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << offset, mask=-1)
            )
        x_scale = div_rn(amax, Float32(FP8_MAX))
        if cutlass.const_expr(self.ue8m0):
            bits = Uint32(float_bits(cute.arch.fmax(x_scale, Float32(1e-10))))
            exponent = (bits >> 23) + Uint32((bits & Uint32(0x7FFFFF)) != 0)
            x_scale = bits_float(exponent << 23)
        inverse = div_rn(Float32(1.0), x_scale)
        words = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        for w in cutlass.range_constexpr(4):
            words[w] = pack_fp8(tuple(values[4 * w + v] * inverse for v in range(4)))
        if chunk == 0:
            xscale[kb] = x_scale
        b1_words = cute.recast_ptr(sm.iterator + self.off_b1, dtype=Uint32)
        for i in cutlass.range_constexpr(blocks1 * B_BYTES // (128 * 16)):
            zero = b1_words + (i * 128 + e) * 4
            for w in cutlass.range_constexpr(4):
                cute.arch.store((zero + w).llvm_ptr, Uint32(0))
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        target = b1_words + (kb * B_BYTES + kb * TILE + (chunk ^ (kb % 8)) * 16) // 4
        for w in cutlass.range_constexpr(4):
            cute.arch.store((target + w).llvm_ptr, words[w])
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()
        if lane == 0:
            cute.arch.mbarrier_arrive(b_ready)

        # GEMM1: scale and sum the 16 K-block columns.
        w_scale = w13_scale.iterator + (expert * self.tiles1 + tile) * self.nk1
        scales = cute.make_rmem_tensor(cute.make_layout((COLUMNS,)), Float32)
        for n in cutlass.range_constexpr(COLUMNS):
            scales[n] = (
                cute.arch.load((w_scale + first + n).llvm_ptr, Float32) * xscale[n]
            )
        wait_tmem(acc1, 0, scope="cta")
        sums = load_tmem(tl, count=COLUMNS)
        acc = Float32(0.0)
        for n in cutlass.range_constexpr(COLUMNS):
            acc = acc + sums[n] * scales[n]
        local = (
            cute.arch.load((rows1.iterator + tile * TILE + row).llvm_ptr, Int32)
            - tile * TILE
        )

        cute.arch.cluster_wait()
        if half == 1:
            dst = cute.recast_ptr(partial.iterator + row, dtype=Uint32)
            store_async_dsmem(dst, Uint32(float_bits(acc)), pair, rank - 1)
        else:
            wait(pair, 0, scope="cta")
            acc = acc + partial[row]
            if tile >= nk2:
                # Gate tile: hand the gate values to the matching up tile.
                dst = cute.recast_ptr(gate_buf.iterator + local, dtype=Uint32)
                peer = 2 * (tile - nk2)
                store_async_dsmem(dst, Uint32(float_bits(acc)), gate_recv, peer)
            else:
                self.hidden_block(
                    sm, acc, local, lane, quadrant, e, tile, gate_buf, gate_recv,
                    amax_buf, hrecv,
                )  # fmt: skip

        # GEMM2: this CTA's rows, K blocks in B columns 0..3.
        wait(hrecv, 0, scope="cta")
        h_scales = cute.make_rmem_tensor(cute.make_layout((nk2,)), Float32)
        for kb2 in cutlass.range_constexpr(nk2):
            h_scales[kb2] = hscale[kb2]
        for t in cutlass.range_constexpr(self.tiles2):
            first_row = rank * self.rows2 + t * TILE
            w2 = (
                w2_scale.iterator
                + (expert * (self.hidden // TILE) + first_row // TILE) * nk2
            )
            wait_tmem(acc2 + t, 0, scope="cta")
            parts = load_tmem(tl + COLUMNS * (1 + t), count=nk2)
            value = Float32(0.0)
            for kb2 in cutlass.range_constexpr(nk2):
                value = value + parts[kb2] * (
                    cute.arch.load((w2 + kb2).llvm_ptr, Float32) * h_scales[kb2]
                )
            logical = cute.arch.load((rows2.iterator + first_row + row).llvm_ptr, Int32)
            _store_bf16(out.iterator + Int64(route) * self.hidden + logical, value)

    @cute.jit
    def hidden_block(
        self, sm, up, local, lane, quadrant, e, tile, gate_buf, gate_recv, amax_buf,
        hrecv,
    ):  # fmt: skip
        """Clamped SwiGLU of this up tile, FP8 per 128 (scale amax / 448), sent as
        hidden K block ``tile`` (logical order, row ``tile`` of B tile ``tile``)
        with its scale to every CTA of the cluster."""
        wait(gate_recv, 0, scope="cta")
        limit = Float32(self.limit)
        gate = cute.arch.fmin(gate_buf[local], limit)
        up = cute.arch.fmax(cute.arch.fmin(up, limit), -limit)
        act = gate / (Float32(1.0) + cute.math.exp(-gate)) * up
        amax = cute.arch.fmax(act, -act)
        for offset in cutlass.range_constexpr(5):
            amax = cute.arch.fmax(
                amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << offset, mask=-1)
            )
        if lane == 0:
            amax_buf[quadrant] = amax
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        amax = cute.arch.fmax(
            cute.arch.fmax(amax_buf[0], amax_buf[1]),
            cute.arch.fmax(amax_buf[2], amax_buf[3]),
        )
        scale = cute.arch.fmax(amax, Float32(1e-10)) / Float32(FP8_MAX)
        stage = sm.iterator + self.off_stage
        byte = pack_fp8((act / scale, Float32(0.0), Float32(0.0), Float32(0.0)))
        cute.arch.store((stage + local).llvm_ptr, Uint8(byte & Uint32(0xFF)))
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        # Thread e sends 16-byte chunk e % 8 to CTA e // 8 (16 CTAs x 8 chunks).
        peer = e // 8
        chunk = e % 8
        words = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        source = cute.recast_ptr(stage + chunk * 16, dtype=Uint32)
        for w in cutlass.range_constexpr(4):
            words[w] = cute.arch.load((source + w).llvm_ptr, Uint32)
        offset = self.off_b2 + tile * B_BYTES + tile * TILE + (chunk ^ (tile % 8)) * 16
        dst = cute.recast_ptr(sm.iterator + cute.assume(offset, divby=16), dtype=Uint32)
        store_async_dsmem(dst, (words[0], words[1], words[2], words[3]), hrecv, peer)
        if chunk == 0:
            scale_offset = self.off_hscale + tile * 4
            scale_dst = cute.recast_ptr(
                sm.iterator + cute.assume(scale_offset, divby=4), dtype=Uint32
            )
            store_async_dsmem(scale_dst, Uint32(float_bits(scale)), hrecv, peer)


def _weight_map(weight: torch.Tensor, box_rows: int) -> torch.Tensor:
    n = weight.shape[2]
    spec = (
        weight.data_ptr(),
        cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
        (TILE, n, weight.numel() // (TILE * n)),
        (TILE, TILE * n),
        (TILE, box_rows, 1),
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
    )
    return tensor_maps([spec], weight.device)


def _logical_rows(n: int, device) -> torch.Tensor:
    from flashinfer.utils import get_shuffle_matrix_a_row_indices

    order = get_shuffle_matrix_a_row_indices(
        torch.empty(n, TILE, dtype=torch.uint8), 64
    )
    return order.to(device=device, dtype=torch.int32)


# Compiled kernels by shape, shared by every layer.
_compiled: dict = {}


class MegaExperts:
    """Routed experts of up to a few tokens: routing, GEMM1, clamped SwiGLU, FP8
    hidden and GEMM2 of every (token, expert) route in one kernel."""

    def __init__(
        self,
        w13,
        w13_scale,
        w2,
        w2_scale,
        top_k: int,
        limit: float,
        scale: float,
        ue8m0: bool,
    ):
        experts, nk1, n1, _ = w13.shape
        hidden = nk1 * TILE
        self.top_k, self.hidden = top_k, hidden
        self.w13, self.w2 = w13, w2
        self.w13_scale = w13_scale.contiguous()
        self.w2_scale = w2_scale.contiguous()
        self.maps = torch.cat([_weight_map(w13, TILE), _weight_map(w2, TILE)])
        self.rows1 = _logical_rows(n1, w13.device)
        self.rows2 = _logical_rows(w2.shape[2], w13.device)
        self.kernel = _MegaKernel(
            hidden=hidden, inter=n1 // 2, experts=experts, top_k=top_k, limit=limit,
            scale=scale, ue8m0=ue8m0,
        )  # fmt: skip
        self.compiled = _compiled.setdefault(
            (hidden, n1 // 2, experts, top_k, limit, scale, ue8m0), {}
        )

    def prepare(self, tokens: list[int], logits_dtype: torch.dtype) -> None:
        """Compiles the kernel for these token counts before graph capture."""
        device = self.w13.device
        experts = self.w13.shape[0]
        bias = torch.zeros(experts, device=device)
        for m in tokens:
            x = torch.zeros(m, self.hidden, dtype=torch.bfloat16, device=device)
            logits = torch.zeros(m, experts, dtype=logits_dtype, device=device)
            self(x, logits, bias)

    def __call__(self, x, logits, bias):
        """BF16 [M * top_k, hidden] expert outputs (one row per route), int32
        [M, top_k] experts and FP32 [M, top_k] routing weights.

        Args:
            x: [M, hidden] BF16 tokens.
            logits: [M, experts] router logits.
            bias: [experts] FP32 score correction bias.

        """
        m = x.shape[0]
        routes = m * self.top_k
        device = x.device
        out = torch.empty(routes, self.hidden, dtype=torch.bfloat16, device=device)
        ids = torch.empty(m, self.top_k, dtype=torch.int32, device=device)
        weights = torch.empty(m, self.top_k, dtype=torch.float32, device=device)
        args = (
            self.maps, self.w13_scale, self.w2_scale, x, logits, bias, self.rows1,
            self.rows2, out, ids.view(-1), weights.view(-1),
        )  # fmt: skip
        st = stream(device)
        key = (routes, logits.dtype)
        if key not in self.compiled:
            self.compiled[key] = cute.compile(
                self.kernel, *(to_cute(t, 16) for t in args), Int32(routes), st
            )
        self.compiled[key](*(to_cute(t, 16) for t in args), Int32(routes), st)
        return out, ids, weights
