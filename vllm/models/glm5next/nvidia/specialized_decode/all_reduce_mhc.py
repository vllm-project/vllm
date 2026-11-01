# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lamport TP all-reduce fused with GLM-5.3-Flash's mHC sublayer boundary.

A publish kernel multicasts each rank's partial sublayer output into a Lamport
mailbox (DSV4.1's ``AllReduceMHC`` protocol). One cluster of CTAs per token
then

1. polls its fragments of every rank's contribution (lane ``r`` of each
   four-lane group reads rank ``r``) and sums them,
2. post-mixes the four residual streams (lane ``j`` owns stream ``j``),
3. reduces the 24-way prenorm GEMV and the sum of squares over the cluster
   through distributed shared memory,
4. collapses the streams with the pre-mix and RMS-normalizes them after a
   second, one-value cluster reduction.

Unlike DSV4.1's delayed mHC, GLM computes the pre-mix from the new residual,
hence the GEMV inside the kernel. The post-mix and the Sinkhorn residual mix
only feed the next boundary: the boundary leaves those mixes raw and a
one-warp kernel on a side stream finishes them while the next sublayer runs.
hc_fn rows are preloaded into registers before the PDL wait.

GB300 runs at most 15 of these 8-CTA clusters at once, so larger batches give
each cluster several tokens: they share its hc_fn registers and cluster
reductions, and the GEMV is one mixed-precision BF16 FMA per element.
"""

from __future__ import annotations

from collections.abc import Mapping

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.distributed import get_tp_group
from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput
from vllm.models.deepseek_v41.nvidia.ops.cute_dsl.all_reduce_mhc import (
    ACTIVE_STAGE,
    LAMPORT_GENERATIONS,
    NEXT_STAGE,
    _QuadFinalizePublishDeviceKernel,
)
from vllm.models.deepseek_v41.nvidia.ops.cute_dsl.primitives import (
    VEC_BF16,
    WARP_SIZE,
    _asm,
    bf16x8_to_packed_u32x4,
    current_cu_stream,
    fragment_has_negative_zero,
    load_global_u32x4,
    load_volatile_u32,
    make_fake_dynamic_compact_tensor,
    map_shared_to_peer,
    packed_u32x4_to_bf16x8,
    sanitize_negative_zero_u32x4,
    stmc_bf16x8,
    store_global_u32,
    store_global_u32x4,
    store_lamport_sentinel_u32x4,
    store_shared_cluster_f32,
    to_cute,
    to_cute_dynamic,
)

HC = 4
MIX = (2 + HC) * HC  # pre 4, post 4, comb 16


@cute.jit
def _gmem_ptr(tensor: cute.Tensor, element: Int64, align: cutlass.Constexpr[int]):
    return cute.make_ptr(
        tensor.element_type,
        (tensor.iterator + element).llvm_ptr,
        cute.AddressSpace.gmem,
        assumed_align=align,
    )


@cute.jit
def _sigmoid(value: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + cute.math.exp(-value))


@cute.jit
def _rcp(value: Float32) -> Float32:
    # The Sinkhorn is 40 dependent normalizations; IEEE reciprocals take ~3x as
    # long as the approximate ones (<= 1 ulp).
    return cute.math.rcp(value, approx=True, ftz=True)


class _PublishDeviceKernel:
    """Multicasts this rank's [M, hidden] partial into the Lamport mailbox.

    Unlike DSV4.1's publish, it lets the boundary launch before its stores:
    the boundary reads the mailbox stage only after its own PDL wait."""

    def __init__(self, *, hidden: int, tp: int, rank: int, capacity_m: int):
        self.hidden = hidden
        self.tp = tp
        self.rank = rank
        self.capacity_m = capacity_m
        self.threads = 128
        self.ctas_per_token = hidden // VEC_BF16 // self.threads

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        stage_state: cute.Tensor,
        multicast: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(x, stage_state, multicast).launch(
            grid=(m * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(self, x: cute.Tensor, stage_state: cute.Tensor, multicast: Int64):
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        token = block // self.ctas_per_token
        column = Int64((block % self.ctas_per_token) * self.threads + tidx) * VEC_BF16
        # Triggering after the wait keeps everything before the producer
        # complete when the boundary's prologue runs.
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        element = Int64(token) * self.hidden + column
        words = sanitize_negative_zero_u32x4(
            load_global_u32x4(_gmem_ptr(x, element, 16))
        )
        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        slot = (
            (Int64(stage) * self.tp + self.rank) * self.capacity_m + Int64(token)
        ) * self.hidden + column
        stmc_bf16x8(multicast + slot * 2, words)
        if block == 0 and tidx == 0:
            store_global_u32(stage_state.iterator + ACTIVE_STAGE, stage)


class _BoundaryDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        tp: int,
        capacity_m: int,
        cluster_size: int,
        rms_eps: float,
        pre_eps: float,
        norm_eps: float,
    ) -> None:
        if tp != HC:
            raise ValueError("one rank lane per hc stream: tp must be 4")
        fragments = hidden // VEC_BF16
        if fragments % cluster_size:
            raise ValueError("hidden fragments must split evenly over the cluster")
        self.hidden = hidden
        self.tp = tp
        self.capacity_m = capacity_m
        self.cluster_size = cluster_size
        self.threads = fragments // cluster_size * HC
        if self.threads % WARP_SIZE or self.threads > 1024:
            raise ValueError("unsupported cluster size")
        self.warps = self.threads // WARP_SIZE
        self.groups_per_cta = self.threads // HC
        self.values = MIX + 1
        self.rms_eps = rms_eps
        self.pre_eps = pre_eps
        self.norm_eps = norm_eps

    def smem_size_in_bytes(self) -> int:
        first = (self.warps + self.cluster_size + 1) * self.values
        second = self.warps + self.cluster_size
        return (first + second) * 4

    @cute.jit
    def __call__(
        self,
        mailbox: cute.Tensor,
        residual: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        fn: cute.Tensor,
        scale: cute.Tensor,
        base: cute.Tensor,
        gamma: cute.Tensor,
        residual_output: cute.Tensor,
        mixes_output: cute.Tensor,
        norm_output: cute.Tensor,
        stage_state: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            mailbox,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            gamma,
            residual_output,
            mixes_output,
            norm_output,
            stage_state,
        ).launch(
            grid=(m, self.cluster_size, 1),
            block=(self.threads, 1, 1),
            cluster=(1, self.cluster_size, 1),
            smem=self.smem_size_in_bytes(),
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def _block_reduce_to_cluster(
        self,
        values: cute.Tensor,
        count: cutlass.Constexpr[int],
        warp_sums: cute.Pointer,
        cluster_sums: cute.Pointer,
        cluster_rank: Int32,
    ) -> None:
        """Sums ``values`` over the CTA and stores the sums into every peer's
        ``cluster_sums[cluster_rank]`` slot."""
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        for k in cutlass.range_constexpr(count):
            value = values[k]
            for offset in cutlass.range_constexpr(5):
                value = value + cute.arch.shuffle_sync_bfly(
                    value, offset=1 << offset, mask=-1, mask_and_clamp=31
                )
            values[k] = value
        if lane == 0:
            for k in cutlass.range_constexpr(count):
                cute.arch.store((warp_sums + warp * count + k).llvm_ptr, values[k])
        cute.arch.barrier()
        if tidx < Int32(count * self.cluster_size):
            k = tidx % count
            peer = tidx // count
            total = Float32(0.0)
            for w in cutlass.range_constexpr(self.warps):
                total = total + cute.arch.load(
                    (warp_sums + w * count + k).llvm_ptr, Float32
                )
            remote = map_shared_to_peer(cluster_sums + cluster_rank * count + k, peer)
            store_shared_cluster_f32(remote, total)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

    @cute.kernel
    def kernel(
        self,
        mailbox: cute.Tensor,
        residual: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        fn: cute.Tensor,
        scale: cute.Tensor,
        base: cute.Tensor,
        gamma: cute.Tensor,
        residual_output: cute.Tensor,
        mixes_output: cute.Tensor,
        norm_output: cute.Tensor,
        stage_state: cute.Tensor,
    ) -> None:
        hidden = self.hidden
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        lane = tidx % HC  # source rank while polling, hc stream afterwards
        fragment = cluster_rank * self.groups_per_cta + tidx // HC
        column = Int64(fragment) * VEC_BF16

        # Everything below until the PDL wait is a weight or was written before
        # the publish kernel's producer started.
        fn_rows = cute.make_rmem_tensor(
            cute.make_layout((MIX, VEC_BF16), stride=(VEC_BF16, 1)), BFloat16
        )
        for row in cutlass.range_constexpr(MIX):
            element = Int64(row) * (HC * hidden) + Int64(lane) * hidden + column
            fn_rows[row, None].store(
                packed_u32x4_to_bf16x8(load_global_u32x4(_gmem_ptr(fn, element, 16)))
            )
        streams = cute.make_rmem_tensor(
            cute.make_layout((HC, VEC_BF16), stride=(VEC_BF16, 1)), BFloat16
        )
        for source in cutlass.range_constexpr(HC):
            element = (Int64(token) * HC + source) * hidden + column
            streams[source, None].store(
                packed_u32x4_to_bf16x8(
                    load_global_u32x4(_gmem_ptr(residual, element, 16))
                )
            )
        gamma_values = packed_u32x4_to_bf16x8(
            load_global_u32x4(_gmem_ptr(gamma, column, 16))
        )
        post_mix = cute.arch.load(
            (post.iterator + Int64(token) * HC + Int64(lane)).llvm_ptr, Float32
        )
        comb_mix = cute.make_rmem_tensor(cute.make_layout((HC,)), Float32)
        for source in cutlass.range_constexpr(HC):
            index = (Int64(token) * HC + source) * HC + Int64(lane)
            comb_mix[source] = cute.arch.load((comb.iterator + index).llvm_ptr, Float32)

        cute.arch.griddepcontrol_wait()

        active_stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        slot = (
            (Int64(active_stage) * self.tp + Int64(lane)) * self.capacity_m
            + Int64(token)
        ) * hidden + column
        packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        dirty = True
        while dirty:
            loaded = load_global_u32x4(_gmem_ptr(mailbox, slot, 16), volatile=True)
            dirty = fragment_has_negative_zero(loaded)
            packed.store(loaded)
        reduced = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
        reduced.store(packed_u32x4_to_bf16x8(packed.load()).to(Float32))
        for offset in cutlass.range_constexpr(2):
            for k in cutlass.range_constexpr(VEC_BF16):
                reduced[k] = reduced[k] + cute.arch.shuffle_sync_bfly(
                    reduced[k], offset=1 << offset, mask=-1, mask_and_clamp=31
                )

        if token == 0 and cluster_rank == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + NEXT_STAGE,
                (active_stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )
        cute.arch.griddepcontrol_launch_dependents()
        store_lamport_sentinel_u32x4(Int64((mailbox.iterator + slot).toint()))

        # The all-reduced sublayer output is BF16, as the collective returns it.
        x = reduced.load().to(BFloat16).to(Float32)
        mixed = x * post_mix
        for source in cutlass.range_constexpr(HC):
            mixed = mixed + streams[source, None].load().to(Float32) * comb_mix[source]
        stream_bf16 = mixed.to(BFloat16)
        output = (Int64(token) * HC + Int64(lane)) * hidden + column
        store_global_u32x4(
            Int64((residual_output.iterator + output).toint()),
            bf16x8_to_packed_u32x4(stream_bf16),
        )
        stream_values = stream_bf16.to(Float32)

        partial = cute.make_rmem_tensor(cute.make_layout((self.values,)), Float32)
        for row in cutlass.range_constexpr(MIX):
            products = fn_rows[row, None].load().to(Float32) * stream_values
            partial[row] = products.reduce(
                cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
            )
        partial[MIX] = (stream_values * stream_values).reduce(
            cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
        )

        smem = cutlass.utils.SmemAllocator()
        warp_sums = smem.allocate_array(Float32, self.warps * self.values)
        cluster_sums = smem.allocate_array(Float32, self.cluster_size * self.values)
        totals = smem.allocate_array(Float32, self.values)
        warp_sums2 = smem.allocate_array(Float32, self.warps)
        cluster_sums2 = smem.allocate_array(Float32, self.cluster_size)

        self._block_reduce_to_cluster(
            partial, self.values, warp_sums, cluster_sums, cluster_rank
        )
        if tidx < Int32(self.values):
            total = Float32(0.0)
            for peer in cutlass.range_constexpr(self.cluster_size):
                total = total + cute.arch.load(
                    (cluster_sums + peer * self.values + tidx).llvm_ptr, Float32
                )
            cute.arch.store((totals + tidx).llvm_ptr, total)
        cute.arch.barrier()

        rstd = cute.math.rsqrt(
            cute.arch.load((totals + MIX).llvm_ptr, Float32) / Float32(HC * hidden)
            + Float32(self.rms_eps)
        )
        if cluster_rank == 0 and tidx < Int32(MIX):
            cute.arch.store(
                (mixes_output.iterator + Int64(token) * MIX + Int64(tidx)).llvm_ptr,
                cute.arch.load((totals + tidx).llvm_ptr, Float32) * rstd,
            )
        mix = cute.arch.load((totals + lane).llvm_ptr, Float32) * rstd
        pre_mix = _sigmoid(
            mix * cute.arch.load(scale.iterator.llvm_ptr, Float32)
            + cute.arch.load((base.iterator + lane).llvm_ptr, Float32)
        ) + Float32(self.pre_eps)

        collapse = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
        collapse.store(stream_values * pre_mix)
        for offset in cutlass.range_constexpr(2):
            for k in cutlass.range_constexpr(VEC_BF16):
                collapse[k] = collapse[k] + cute.arch.shuffle_sync_bfly(
                    collapse[k], offset=1 << offset, mask=-1, mask_and_clamp=31
                )
        prenorm = collapse.load().to(BFloat16).to(Float32)

        square = cute.make_rmem_tensor(cute.make_layout((1,)), Float32)
        square[0] = Float32(0.0)
        if lane == 0:
            square[0] = (prenorm * prenorm).reduce(
                cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
            )
        self._block_reduce_to_cluster(
            square, 1, warp_sums2, cluster_sums2, cluster_rank
        )
        sum_square = Float32(0.0)
        for peer in cutlass.range_constexpr(self.cluster_size):
            sum_square = sum_square + cute.arch.load(
                (cluster_sums2 + peer).llvm_ptr, Float32
            )
        inv_rms = cute.math.rsqrt(sum_square / Float32(hidden) + Float32(self.norm_eps))
        if lane == 0:
            result = (prenorm * inv_rms * gamma_values.to(Float32)).to(BFloat16)
            store_global_u32x4(
                Int64((norm_output.iterator + Int64(token) * hidden + column).toint()),
                bf16x8_to_packed_u32x4(result),
            )


@dsl_user_op
def _store_shared_u32x4(address: Int32, packed, *, loc=None, ip=None) -> None:
    words = [packed[index].ir_value(loc=loc, ip=ip) for index in range(4)]
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _load_shared_u32x4(address: Int32, *, loc=None, ip=None):
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), loaded, [k], loc=loc, ip=ip) for k in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def _dot_bf16x8(a, b, *, loc=None, ip=None) -> Float32:
    """FP32 dot product of two packed BF16x8 fragments, one exact-product FMA
    per element and no conversions."""
    a_words = [a[k].ir_value(loc=loc, ip=ip) for k in range(4)]
    b_words = [b[k].ir_value(loc=loc, ip=ip) for k in range(4)]
    text = ["{", ".reg .b16 a<8>, b<8>;"]
    for k in range(4):
        text.append(f"mov.b32 {{a{2 * k}, a{2 * k + 1}}}, ${1 + k};")
        text.append(f"mov.b32 {{b{2 * k}, b{2 * k + 1}}}, ${5 + k};")
    text.append("mov.b32 $0, 0f00000000;")
    for k in range(VEC_BF16):
        text.append(f"fma.rn.f32.bf16 $0, a{k}, b{k}, $0;")
    text.append("}")
    return Float32(
        _asm(
            T.f32(),
            [*a_words, *b_words],
            "\n\t".join(text),
            "=f," + ",".join(["r"] * 8),
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _warp_reduce_scatter(values: list) -> Float32:
    """Sums 32 per-lane values over the warp; lane ``i`` returns sum ``i``."""
    lane = cute.arch.lane_idx()
    for step in cutlass.range_constexpr(5):
        width = WARP_SIZE >> (step + 1)
        upper = (lane & Int32(width)) != Int32(0)
        kept = []
        for i in cutlass.range_constexpr(width):
            low, high = values[i], values[i + width]
            sent = Float32(cutlass.select_(upper, low, high))
            kept.append(
                Float32(cutlass.select_(upper, high, low))
                + cute.arch.shuffle_sync_bfly(
                    sent, offset=width, mask=-1, mask_and_clamp=31
                )
            )
        values = kept
    return values[0]


class _GroupBoundaryDeviceKernel(_BoundaryDeviceKernel):
    """The boundary for ``tokens`` tokens per cluster, for large batches: one
    load of the hc_fn rows into registers serves every token of the cluster,
    the residual streams are staged in shared memory before the PDL wait, and
    the tokens share the two cluster reductions."""

    def __init__(self, *, tokens: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.values > WARP_SIZE:
            raise ValueError("the mixes must fit one warp reduce-scatter")
        self.tokens = tokens
        # Per token: post-mix and residual mix, then the reductions.
        self.inputs = HC + HC * HC

    def smem_size_in_bytes(self) -> int:
        staged = self.tokens * HC * self.groups_per_cta * 16
        floats = self.tokens * (
            self.inputs
            + (self.warps + self.cluster_size + 1) * self.values
            + self.warps
            + self.cluster_size
        )
        return staged + floats * 4

    @cute.jit
    def __call__(
        self,
        mailbox: cute.Tensor,
        residual: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        fn: cute.Tensor,
        scale: cute.Tensor,
        base: cute.Tensor,
        gamma: cute.Tensor,
        residual_output: cute.Tensor,
        mixes_output: cute.Tensor,
        norm_output: cute.Tensor,
        stage_state: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            mailbox,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            gamma,
            residual_output,
            mixes_output,
            norm_output,
            stage_state,
            m,
        ).launch(
            grid=((m + self.tokens - 1) // self.tokens, self.cluster_size, 1),
            block=(self.threads, 1, 1),
            cluster=(1, self.cluster_size, 1),
            smem=self.smem_size_in_bytes(),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mailbox: cute.Tensor,
        residual: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        fn: cute.Tensor,
        scale: cute.Tensor,
        base: cute.Tensor,
        gamma: cute.Tensor,
        residual_output: cute.Tensor,
        mixes_output: cute.Tensor,
        norm_output: cute.Tensor,
        stage_state: cute.Tensor,
        m: Int32,
    ) -> None:
        hidden = self.hidden
        tokens = self.tokens
        values = self.values
        cluster = self.cluster_size
        groups = self.groups_per_cta
        tidx, _, _ = cute.arch.thread_idx()
        group, _, _ = cute.arch.block_idx()
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        warp = cute.arch.warp_idx()
        lane = tidx % HC  # source rank while polling, hc stream afterwards
        local = tidx // HC
        fragment = cluster_rank * groups + local
        column = Int64(fragment) * VEC_BF16
        first = group * tokens
        # A short last group repeats its last token; the repeats rewrite the
        # same outputs.
        last = m - 1

        smem = cutlass.utils.SmemAllocator()
        staged = smem.allocate_array(
            Uint32, tokens * HC * groups * 4, byte_alignment=16
        )
        mix_inputs = smem.allocate_array(Float32, tokens * self.inputs)
        warp_sums = smem.allocate_array(Float32, tokens * self.warps * values)
        cluster_sums = smem.allocate_array(Float32, tokens * cluster * values)
        totals = smem.allocate_array(Float32, tokens * values)
        warp_sums2 = smem.allocate_array(Float32, tokens * self.warps)
        cluster_sums2 = smem.allocate_array(Float32, tokens * cluster)

        # Everything below until the PDL wait is a weight or was written before
        # the publish kernel's producer started.
        fn_rows = cute.make_rmem_tensor(
            cute.make_layout((MIX, 4), stride=(4, 1)), Uint32
        )
        for row in cutlass.range_constexpr(MIX):
            element = Int64(row) * (HC * hidden) + Int64(lane) * hidden + column
            fn_rows[row, None].store(load_global_u32x4(_gmem_ptr(fn, element, 16)))
        gamma_values = packed_u32x4_to_bf16x8(
            load_global_u32x4(_gmem_ptr(gamma, column, 16))
        )
        for t in cutlass.range_constexpr(tokens):
            token = cutlass.min(first + t, last)
            element = (Int64(token) * HC + Int64(lane)) * hidden + column
            _store_shared_u32x4(
                (staged + ((t * HC + lane) * groups + local) * 4).toint(),
                load_global_u32x4(_gmem_ptr(residual, element, 16)),
            )
        for k in cutlass.range_constexpr(
            (tokens * self.inputs + self.threads - 1) // self.threads
        ):
            index = tidx + k * self.threads
            if index < Int32(tokens * self.inputs):
                t = index // self.inputs
                j = index % self.inputs
                row = Int64(cutlass.min(first + t, last))
                if j < Int32(HC):
                    value = cute.arch.load(
                        (post.iterator + row * HC + Int64(j)).llvm_ptr, Float32
                    )
                    cute.arch.store((mix_inputs + index).llvm_ptr, value)
                else:
                    value = cute.arch.load(
                        (comb.iterator + row * (HC * HC) + Int64(j - HC)).llvm_ptr,
                        Float32,
                    )
                    cute.arch.store((mix_inputs + index).llvm_ptr, value)
        cute.arch.barrier()

        cute.arch.griddepcontrol_wait()

        # Poll every token at once: a repeated token reads its slot again
        # before the reset below.
        active_stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        stage_base = Int64(active_stage) * self.tp + Int64(lane)
        kept = cute.make_rmem_tensor(
            cute.make_layout((tokens, 4), stride=(4, 1)), Uint32
        )
        dirty = True
        while dirty:
            dirty = False
            for t in cutlass.range_constexpr(tokens):
                token = cutlass.min(first + t, last)
                slot = (stage_base * self.capacity_m + Int64(token)) * hidden + column
                loaded = load_global_u32x4(_gmem_ptr(mailbox, slot, 16), volatile=True)
                dirty = dirty | fragment_has_negative_zero(loaded)
                kept[t, None].store(loaded)
        # Lanes leave the poll at different times; without reconverging, every
        # shuffle below takes the divergent-warp path.
        cute.arch.sync_warp()
        reduced = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
        for t in cutlass.range_constexpr(tokens):
            reduced.store(packed_u32x4_to_bf16x8(kept[t, None].load()).to(Float32))
            for offset in cutlass.range_constexpr(2):
                for k in cutlass.range_constexpr(VEC_BF16):
                    reduced[k] = reduced[k] + cute.arch.shuffle_sync_bfly(
                        reduced[k], offset=1 << offset, mask=-1, mask_and_clamp=31
                    )
            # The all-reduced sublayer output is BF16, as the collective
            # returns it.
            kept[t, None].store(bf16x8_to_packed_u32x4(reduced.load().to(BFloat16)))

        if group == 0 and cluster_rank == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + NEXT_STAGE,
                (active_stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )
        cute.arch.griddepcontrol_launch_dependents()
        for t in cutlass.range_constexpr(tokens):
            token = cutlass.min(first + t, last)
            slot = (stage_base * self.capacity_m + Int64(token)) * hidden + column
            store_lamport_sentinel_u32x4(Int64((mailbox.iterator + slot).toint()))

        for t in cutlass.range_constexpr(tokens):
            token = cutlass.min(first + t, last)
            inputs = mix_inputs + t * self.inputs
            mixed = packed_u32x4_to_bf16x8(kept[t, None].load()).to(
                Float32
            ) * cute.arch.load((inputs + lane).llvm_ptr, Float32)
            for source in cutlass.range_constexpr(HC):
                stream = packed_u32x4_to_bf16x8(
                    _load_shared_u32x4(
                        (staged + ((t * HC + source) * groups + local) * 4).toint()
                    )
                )
                mixed = mixed + stream.to(Float32) * cute.arch.load(
                    (inputs + HC + source * HC + lane).llvm_ptr, Float32
                )
            stream_bf16 = mixed.to(BFloat16)
            words = bf16x8_to_packed_u32x4(stream_bf16)
            kept[t, None].store(words)
            output = (Int64(token) * HC + Int64(lane)) * hidden + column
            store_global_u32x4(
                Int64((residual_output.iterator + output).toint()), words
            )
            partial = [
                _dot_bf16x8(fn_rows[row, None].load(), words) for row in range(MIX)
            ]
            partial.append(_dot_bf16x8(words, words))
            for _ in cutlass.range_constexpr(WARP_SIZE - values):
                partial.append(Float32(0.0))
            total = _warp_reduce_scatter(partial)
            if cute.arch.lane_idx() < Int32(values):
                cute.arch.store(
                    (
                        warp_sums
                        + (t * self.warps + warp) * values
                        + cute.arch.lane_idx()
                    ).llvm_ptr,
                    total,
                )
        cute.arch.barrier()

        # Every CTA's per-token sums into every peer's slot, then the totals.
        for k in cutlass.range_constexpr(
            (tokens * values + self.threads - 1) // self.threads
        ):
            index = tidx + k * self.threads
            if index < Int32(tokens * values):
                t = index // values
                v = index % values
                total = Float32(0.0)
                for w in cutlass.range_constexpr(self.warps):
                    total = total + cute.arch.load(
                        (warp_sums + (t * self.warps + w) * values + v).llvm_ptr,
                        Float32,
                    )
                for peer in cutlass.range_constexpr(cluster):
                    remote = map_shared_to_peer(
                        cluster_sums + (t * cluster + cluster_rank) * values + v,
                        Int32(peer),
                    )
                    store_shared_cluster_f32(remote, total)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        for k in cutlass.range_constexpr(
            (tokens * values + self.threads - 1) // self.threads
        ):
            index = tidx + k * self.threads
            if index < Int32(tokens * values):
                t = index // values
                v = index % values
                total = Float32(0.0)
                for peer in cutlass.range_constexpr(cluster):
                    total = total + cute.arch.load(
                        (cluster_sums + (t * cluster + peer) * values + v).llvm_ptr,
                        Float32,
                    )
                cute.arch.store((totals + index).llvm_ptr, total)
        cute.arch.barrier()

        scale_pre = cute.arch.load(scale.iterator.llvm_ptr, Float32)
        base_pre = cute.arch.load((base.iterator + lane).llvm_ptr, Float32)
        squares = []
        for t in cutlass.range_constexpr(tokens):
            rstd = cute.math.rsqrt(
                cute.arch.load((totals + t * values + MIX).llvm_ptr, Float32)
                / Float32(HC * hidden)
                + Float32(self.rms_eps)
            )
            if cluster_rank == 0 and tidx < Int32(MIX) and first + t <= last:
                cute.arch.store(
                    (
                        mixes_output.iterator + Int64(first + t) * MIX + Int64(tidx)
                    ).llvm_ptr,
                    cute.arch.load((totals + t * values + tidx).llvm_ptr, Float32)
                    * rstd,
                )
            mix = cute.arch.load((totals + t * values + lane).llvm_ptr, Float32) * rstd
            pre_mix = _sigmoid(mix * scale_pre + base_pre) + Float32(self.pre_eps)
            collapse = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
            collapse.store(
                packed_u32x4_to_bf16x8(kept[t, None].load()).to(Float32) * pre_mix
            )
            for offset in cutlass.range_constexpr(2):
                for k in cutlass.range_constexpr(VEC_BF16):
                    collapse[k] = collapse[k] + cute.arch.shuffle_sync_bfly(
                        collapse[k], offset=1 << offset, mask=-1, mask_and_clamp=31
                    )
            prenorm = collapse.load().to(BFloat16)
            kept[t, None].store(bf16x8_to_packed_u32x4(prenorm))
            square = (prenorm.to(Float32) * prenorm.to(Float32)).reduce(
                cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
            )
            squares.append(
                Float32(cutlass.select_(lane == Int32(0), square, Float32(0.0)))
            )
        for t in cutlass.range_constexpr(tokens):
            for offset in cutlass.range_constexpr(5):
                squares[t] = squares[t] + cute.arch.shuffle_sync_bfly(
                    squares[t], offset=1 << offset, mask=-1, mask_and_clamp=31
                )
        if cute.arch.lane_idx() == 0:
            for t in cutlass.range_constexpr(tokens):
                cute.arch.store(
                    (warp_sums2 + t * self.warps + warp).llvm_ptr, squares[t]
                )
        cute.arch.barrier()
        if tidx < Int32(tokens * cluster):
            t = tidx // cluster
            peer = tidx % cluster
            total = Float32(0.0)
            for w in cutlass.range_constexpr(self.warps):
                total = total + cute.arch.load(
                    (warp_sums2 + t * self.warps + w).llvm_ptr, Float32
                )
            remote = map_shared_to_peer(
                cluster_sums2 + t * cluster + cluster_rank, peer
            )
            store_shared_cluster_f32(remote, total)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        if lane == 0:
            for t in cutlass.range_constexpr(tokens):
                token = cutlass.min(first + t, last)
                sum_square = Float32(0.0)
                for peer in cutlass.range_constexpr(cluster):
                    sum_square = sum_square + cute.arch.load(
                        (cluster_sums2 + t * cluster + peer).llvm_ptr, Float32
                    )
                inv_rms = cute.math.rsqrt(
                    sum_square / Float32(hidden) + Float32(self.norm_eps)
                )
                normed = packed_u32x4_to_bf16x8(kept[t, None].load()).to(Float32)
                result = (normed * inv_rms * gamma_values.to(Float32)).to(BFloat16)
                store_global_u32x4(
                    Int64(
                        (norm_output.iterator + Int64(token) * hidden + column).toint()
                    ),
                    bf16x8_to_packed_u32x4(result),
                )


class _MixesDeviceKernel:
    """One thread per token: the post-mix and the Sinkhorn residual mix from a
    boundary's raw mixes."""

    def __init__(
        self, *, sinkhorn_eps: float, post_mult: float, repeat: int, capacity_m: int
    ):
        self.sinkhorn_eps = sinkhorn_eps
        self.post_mult = post_mult
        self.repeat = repeat
        self.threads = -(-capacity_m // WARP_SIZE) * WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mixes: cute.Tensor,
        scale: cute.Tensor,
        base: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(mixes, scale, base, post, comb, m).launch(
            grid=(1, 1, 1), block=(self.threads, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mixes: cute.Tensor,
        scale: cute.Tensor,
        base: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        m: Int32,
    ) -> None:
        token, _, _ = cute.arch.thread_idx()
        if token < m:
            row = mixes.iterator + Int64(token) * MIX
            post_scale = cute.arch.load((scale.iterator + 1).llvm_ptr, Float32)
            for j in cutlass.range_constexpr(HC):
                value = _sigmoid(
                    cute.arch.load((row + HC + j).llvm_ptr, Float32) * post_scale
                    + cute.arch.load((base.iterator + HC + j).llvm_ptr, Float32)
                ) * Float32(self.post_mult)
                cute.arch.store((post.iterator + Int64(token) * HC + j).llvm_ptr, value)
            c = self._sinkhorn(row, scale, base)
            for k in cutlass.range_constexpr(HC * HC):
                cute.arch.store(
                    (comb.iterator + Int64(token) * HC * HC + k).llvm_ptr, c[k]
                )

    @cute.jit
    def _sinkhorn(self, row: cute.Pointer, scale: cute.Tensor, base: cute.Tensor):
        """The residual mix, row-major in 16 registers."""
        eps = Float32(self.sinkhorn_eps)
        comb_scale = cute.arch.load((scale.iterator + 2).llvm_ptr, Float32)
        c = []
        for i in cutlass.range_constexpr(HC * HC):
            mix = cute.arch.load((row + 2 * HC + i).llvm_ptr, Float32)
            bias = cute.arch.load((base.iterator + 2 * HC + i).llvm_ptr, Float32)
            c.append(mix * comb_scale + bias)
        for r in cutlass.range_constexpr(HC):
            i = r * HC
            peak = cute.arch.fmax(
                cute.arch.fmax(c[i], c[i + 1]), cute.arch.fmax(c[i + 2], c[i + 3])
            )
            for k in cutlass.range_constexpr(HC):
                c[i + k] = cute.math.exp(c[i + k] - peak)
            inv = _rcp((c[i] + c[i + 1]) + (c[i + 2] + c[i + 3]))
            for k in cutlass.range_constexpr(HC):
                c[i + k] = c[i + k] * inv + eps
        for iteration in cutlass.range_constexpr(self.repeat):
            if cutlass.const_expr(iteration > 0):
                for r in cutlass.range_constexpr(HC):
                    i = r * HC
                    inv = _rcp((c[i] + c[i + 1]) + (c[i + 2] + c[i + 3]) + eps)
                    for k in cutlass.range_constexpr(HC):
                        c[i + k] = c[i + k] * inv
            for k in cutlass.range_constexpr(HC):
                inv = _rcp((c[k] + c[HC + k]) + (c[2 * HC + k] + c[3 * HC + k]) + eps)
                for r in cutlass.range_constexpr(HC):
                    c[r * HC + k] = c[r * HC + k] * inv
        return c


class AllReduceMhcBoundary:
    """``tensor_model_parallel_all_reduce`` + ``hc_fused_post_pre`` for up to
    ``max_num_tokens`` tokens: two kernels on the current stream and one on a
    side stream.

    The returned post-mix and residual mix are written on the side stream; the
    next call joins it, and a stock consumer must call ``join`` first.
    Construction is collective over the TP group; one instance serves every
    boundary of the model, which run one after another. ``cluster_tokens``
    maps a token count to the tokens each cluster takes (1 if absent).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        max_num_tokens: int,
        rms_eps: float,
        hc_eps: float,
        post_mult: float,
        sinkhorn_repeat: int,
        norm_eps: float,
        device: torch.device,
        top_k: int,
        cluster_size: int = 8,
        cluster_tokens: Mapping[int, int] | None = None,
    ) -> None:
        group = get_tp_group().device_group
        self.top_k = top_k
        self.hidden_size = hidden = hidden_size
        self.capacity = capacity = max_num_tokens
        tp = dist.get_world_size(group)
        rank = dist.get_rank(group)
        device = torch.device(device)

        def dynamic(dtype, alignment, divisibility):
            return make_fake_dynamic_compact_tensor(
                dtype, alignment=alignment, divisibility=divisibility
            )

        with torch.accelerator.device_index(device.index):
            self._publish = cute.compile(
                _PublishDeviceKernel(
                    hidden=hidden, tp=tp, rank=rank, capacity_m=capacity
                ),
                dynamic(BFloat16, 16, hidden),
                make_fake_compact_tensor(Int32, (2,), assumed_align=4),
                Int64(0),
                Int32(capacity),
                current_cu_stream(),
            )
            # By routing-weight dtype; the routing folds in the routed scale.
            self._finalize_publish = {
                dtype: cute.compile(
                    _QuadFinalizePublishDeviceKernel(
                        hidden=hidden,
                        top_k=top_k,
                        tp=tp,
                        rank=rank,
                        capacity_m=capacity,
                        threads=128,
                        routed_scaling_factor=1.0,
                        include_shared_expert=True,
                        load_shared_expert_before_pdl=False,
                        enable_pdl=True,
                        prefetch_group=top_k,
                        fp32_weights=dtype == torch.float32,
                    ),
                    dynamic(BFloat16, 16, hidden),
                    dynamic(
                        Float32 if dtype == torch.float32 else BFloat16,
                        4 if dtype == torch.float32 else 2,
                        top_k,
                    ),
                    dynamic(Int32, 4, top_k),
                    dynamic(BFloat16, 16, hidden),
                    make_fake_compact_tensor(Int32, (2,), assumed_align=4),
                    Int64(0),
                    Int32(capacity),
                    current_cu_stream(),
                )
                for dtype in (torch.float32, torch.bfloat16)
            }
            boundary = dict(
                hidden=hidden,
                tp=tp,
                capacity_m=capacity,
                cluster_size=cluster_size,
                rms_eps=rms_eps,
                pre_eps=hc_eps,
                norm_eps=norm_eps,
            )
            self.cluster_tokens = dict(cluster_tokens or {})
            self._boundary = {}
            for tokens in {1, *self.cluster_tokens.values()}:
                kernel = (
                    _GroupBoundaryDeviceKernel(tokens=tokens, **boundary)
                    if tokens > 1
                    else _BoundaryDeviceKernel(**boundary)
                )
                self._boundary[tokens] = cute.compile(
                    kernel,
                    make_fake_compact_tensor(
                        BFloat16,
                        (LAMPORT_GENERATIONS * tp * capacity * hidden,),
                        assumed_align=16,
                    ),
                    dynamic(BFloat16, 16, HC * hidden),
                    dynamic(Float32, 4, HC),
                    dynamic(Float32, 4, HC * HC),
                    make_fake_compact_tensor(
                        BFloat16, (MIX * HC * hidden,), assumed_align=16
                    ),
                    make_fake_compact_tensor(Float32, (3,), assumed_align=4),
                    make_fake_compact_tensor(Float32, (MIX,), assumed_align=4),
                    make_fake_compact_tensor(BFloat16, (hidden,), assumed_align=16),
                    dynamic(BFloat16, 16, HC * hidden),
                    dynamic(Float32, 4, MIX),
                    dynamic(BFloat16, 16, hidden),
                    make_fake_compact_tensor(Int32, (2,), assumed_align=4),
                    Int32(capacity),
                    current_cu_stream(),
                )
            self._mixes = cute.compile(
                _MixesDeviceKernel(
                    sinkhorn_eps=hc_eps,
                    post_mult=post_mult,
                    repeat=sinkhorn_repeat,
                    capacity_m=capacity,
                ),
                dynamic(Float32, 4, MIX),
                make_fake_compact_tensor(Float32, (3,), assumed_align=4),
                make_fake_compact_tensor(Float32, (MIX,), assumed_align=4),
                dynamic(Float32, 4, HC),
                dynamic(Float32, 4, HC * HC),
                Int32(capacity),
                current_cu_stream(),
            )

            self._mailbox = symm_mem.empty(
                (LAMPORT_GENERATIONS, tp, capacity, hidden),
                dtype=torch.bfloat16,
                device=device,
            )
            self._mailbox_handle = symm_mem.rendezvous(self._mailbox, group)
            multicast = int(self._mailbox_handle.multicast_ptr or 0)
            if not multicast or multicast % 16:
                raise RuntimeError("NVLink multicast mapping is unavailable")
            self._multicast = multicast
            # Every slot starts as the Lamport sentinel, BF16 negative zero.
            self._mailbox.view(torch.int16).fill_(-32768)
            self._stage_state = torch.zeros(2, dtype=torch.int32, device=device)
            self._side = torch.cuda.Stream(device)
            self._pending: torch.cuda.Event | None = None
            # The raw mixes the side stream reads: allocated on the current
            # stream, so they must outlive the side kernel.
            self._in_flight: torch.Tensor | None = None
            torch.accelerator.synchronize()
        dist.barrier(group=group)

    def supports(self, tokens: int) -> bool:
        return 1 <= tokens <= self.capacity

    def join(self) -> None:
        """Makes the current stream wait for the last mixes kernel."""
        if self._pending is not None:
            torch.cuda.current_stream().wait_event(self._pending)
            self._pending = self._in_flight = None

    def __call__(
        self,
        x: torch.Tensor | MoEOutput,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the residual streams, post-mix, residual mix and layer input
        of ``hc_fused_post_pre`` applied to the TP sum of ``x``.

        Args:
            x: [M, hidden] BF16 TP-partial sublayer output, or a MoE output
                whose routed top-k reduction is still open (its shared and
                routed parts are TP-partial).
            residual: [M, hc, hidden] BF16 residual streams.
            post: [M, hc, 1] FP32 post-mix.
            comb: [M, hc, hc] FP32 residual mix, indexed [source, target].
            fn: [24, hc * hidden] BF16 prenorm GEMV weight.
            scale: [3] FP32 mix scales.
            base: [24] FP32 mix biases.
            norm_weight: [hidden] BF16 RMSNorm weight.

        """
        hidden = self.hidden_size
        self.join()
        if isinstance(x, MoEOutput):
            routed, shared = x.routed, x.shared_output
            assert shared is not None and not isinstance(routed, torch.Tensor)
            m = shared.shape[0]
            top_k = self.top_k
            weights = routed.expert_weights
            self._finalize_publish[weights.dtype](
                to_cute_dynamic(
                    routed.gemm2_permuted.flatten(), 16, divisibility=hidden
                ),
                to_cute_dynamic(
                    weights.flatten(), weights.element_size(), divisibility=top_k
                ),
                to_cute_dynamic(
                    routed.expanded_idx_to_permuted_idx.flatten(),
                    4,
                    divisibility=top_k,
                ),
                to_cute_dynamic(shared.flatten(), 16, divisibility=hidden),
                to_cute(self._stage_state, 4),
                Int64(self._multicast),
                Int32(m),
                current_cu_stream(),
            )
            layer_input = torch.empty_like(shared)
        else:
            m = x.shape[0]
            self._publish(
                to_cute_dynamic(x.flatten(), 16, divisibility=hidden),
                to_cute(self._stage_state, 4),
                Int64(self._multicast),
                Int32(m),
                current_cu_stream(),
            )
            layer_input = torch.empty_like(x)
        device = residual.device
        residual_output = torch.empty_like(residual)
        mixes = torch.empty((m, MIX), dtype=torch.float32, device=device)
        post_output = torch.empty((m, HC, 1), dtype=torch.float32, device=device)
        comb_output = torch.empty((m, HC, HC), dtype=torch.float32, device=device)
        self._boundary[self.cluster_tokens.get(m, 1)](
            to_cute(self._mailbox.flatten(), 16),
            to_cute_dynamic(residual.flatten(), 16, divisibility=HC * hidden),
            to_cute_dynamic(post.flatten(), 4, divisibility=HC),
            to_cute_dynamic(comb.flatten(), 4, divisibility=HC * HC),
            to_cute(fn.flatten(), 16),
            to_cute(scale, 4),
            to_cute(base, 4),
            to_cute(norm_weight, 16),
            to_cute_dynamic(residual_output.flatten(), 16, divisibility=HC * hidden),
            to_cute_dynamic(mixes.flatten(), 4, divisibility=MIX),
            to_cute_dynamic(layer_input.flatten(), 16, divisibility=hidden),
            to_cute(self._stage_state, 4),
            Int32(m),
            current_cu_stream(),
        )
        self._side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._side):
            self._mixes(
                to_cute_dynamic(mixes.flatten(), 4, divisibility=MIX),
                to_cute(scale, 4),
                to_cute(base, 4),
                to_cute_dynamic(post_output.flatten(), 4, divisibility=HC),
                to_cute_dynamic(comb_output.flatten(), 4, divisibility=HC * HC),
                Int32(m),
                current_cu_stream(),
            )
            self._pending = torch.cuda.Event()
            self._pending.record()
        self._in_flight = mixes
        return residual_output, post_output, comb_output, layer_input
