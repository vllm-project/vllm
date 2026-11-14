# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM90 boundary kernel of the fused TP all-reduce + mHC boundary.

Adapted from `all_reduce_mhc._GroupBoundaryDeviceKernel`, which serves any
number of tokens per 8-CTA cluster (one token when `tokens` is 1). Changes:

- TP 2 or 4: lane `r` of each four-lane stream group polls rank `r` of the
  Lamport mailbox and the lanes past the TP size add zero (SM100 requires
  TP 4, one lane per rank).
- `reduced`: the input is the stock all-reduce's output instead of the
  mailbox; lane 0 of each group loads it.
- The prenorm GEMV unpacks BF16 pairs and runs FP32 FMAs; SM100's mixed
  `fma.rn.f32.bf16` does not exist on SM90. The products are exact either way,
  so the sums are the same.

The publish kernel and the post-mix / Sinkhorn kernel are main's
(`_PublishDeviceKernel`, `_MixesDeviceKernel`); they compile for SM90 as is.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.models.common.ops.moe_tail_fusion.primitives import (
    ACTIVE_STAGE,
    LAMPORT_GENERATIONS,
    NEXT_STAGE,
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
    store_global_u32,
    store_global_u32x4,
    store_lamport_sentinel_u32x4,
    store_shared_cluster_f32,
)

from ..all_reduce_mhc import (
    HC,
    MIX,
    _gmem_ptr,
    _load_shared_u32x4,
    _MixesDeviceKernel,
    _sigmoid,
    _store_shared_u32x4,
    _warp_reduce_scatter,
)

CLUSTER_SIZE = 8


@dsl_user_op
def _dot_bf16x8(a, b, *, loc=None, ip=None) -> Float32:
    """FP32 dot product of two packed BF16x8 fragments in element order, one
    FP32 FMA per (exact) product."""
    a_words = [a[k].ir_value(loc=loc, ip=ip) for k in range(4)]
    b_words = [b[k].ir_value(loc=loc, ip=ip) for k in range(4)]
    text = ["{", ".reg .b32 al, ah, bl, bh;", "mov.b32 $0, 0f00000000;"]
    for k in range(4):
        text.append(f"shl.b32 al, ${1 + k}, 16;")
        text.append(f"and.b32 ah, ${1 + k}, 0xffff0000;")
        text.append(f"shl.b32 bl, ${5 + k}, 16;")
        text.append(f"and.b32 bh, ${5 + k}, 0xffff0000;")
        text.append("fma.rn.f32 $0, al, bl, $0;")
        text.append("fma.rn.f32 $0, ah, bh, $0;")
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


class HopperBoundaryKernel:
    """mHC post-mix, prenorm GEMV, pre-mix collapse and RMSNorm of `tokens`
    tokens per cluster, from the Lamport mailbox or a reduced input."""

    def __init__(
        self,
        *,
        tokens: int,
        reduced: bool,
        hidden: int,
        tp: int,
        capacity_m: int,
        rms_eps: float,
        pre_eps: float,
        norm_eps: float,
        cluster_size: int = CLUSTER_SIZE,
    ) -> None:
        if tp > HC or HC % tp:
            raise ValueError("the TP size must divide the four stream lanes")
        fragments = hidden // VEC_BF16
        if fragments % cluster_size:
            raise ValueError("hidden fragments must split evenly over the cluster")
        self.tokens = tokens
        self.reduced = reduced
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
        # Per token: post-mix and residual mix.
        self.inputs = HC + HC * HC
        self.rms_eps = rms_eps
        self.pre_eps = pre_eps
        self.norm_eps = norm_eps

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
        source: cute.Tensor,
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
            source,
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
        source: cute.Tensor,
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
        # the producer of the input started. residual, post and comb come from
        # kernels several launches back, so these reads hold only if (1) the
        # first kernel between two boundary kernels waits before it triggers
        # its dependents, and (2) every kernel in that chain has at least one
        # thread that executes the wait. A kernel that triggers at entry, or
        # whose threads all exit before waiting, lets this kernel read them
        # before they are written.
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

        kept = cute.make_rmem_tensor(
            cute.make_layout((tokens, 4), stride=(4, 1)), Uint32
        )
        kept.fill(Uint32(0))
        if cutlass.const_expr(self.reduced):
            if lane == Int32(0):
                for t in cutlass.range_constexpr(tokens):
                    token = cutlass.min(first + t, last)
                    element = Int64(token) * hidden + column
                    kept[t, None].store(
                        load_global_u32x4(_gmem_ptr(source, element, 16))
                    )
            cute.arch.griddepcontrol_launch_dependents()
        else:
            # Poll every token at once: a repeated token reads its slot again
            # before the reset below.
            active_stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
            stage_base = Int64(active_stage) * self.tp + Int64(lane)
            dirty = lane < Int32(self.tp)
            while dirty:
                dirty = False
                for t in cutlass.range_constexpr(tokens):
                    token = cutlass.min(first + t, last)
                    slot = (
                        stage_base * self.capacity_m + Int64(token)
                    ) * hidden + column
                    loaded = load_global_u32x4(
                        _gmem_ptr(source, slot, 16), volatile=True
                    )
                    dirty = dirty | fragment_has_negative_zero(loaded)
                    kept[t, None].store(loaded)
            # Lanes leave the poll at different times; without reconverging,
            # every shuffle below takes the divergent-warp path.
            cute.arch.sync_warp()
            if group == 0 and cluster_rank == 0 and tidx == 0:
                store_global_u32(
                    stage_state.iterator + NEXT_STAGE,
                    (active_stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
                )
            cute.arch.griddepcontrol_launch_dependents()
            if lane < Int32(self.tp):
                for t in cutlass.range_constexpr(tokens):
                    token = cutlass.min(first + t, last)
                    slot = (
                        stage_base * self.capacity_m + Int64(token)
                    ) * hidden + column
                    store_lamport_sentinel_u32x4(
                        Int64((source.iterator + slot).toint())
                    )

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

        for t in cutlass.range_constexpr(tokens):
            token = cutlass.min(first + t, last)
            inputs = mix_inputs + t * self.inputs
            mixed = packed_u32x4_to_bf16x8(kept[t, None].load()).to(
                Float32
            ) * cute.arch.load((inputs + lane).llvm_ptr, Float32)
            for src in cutlass.range_constexpr(HC):
                stream = packed_u32x4_to_bf16x8(
                    _load_shared_u32x4(
                        (staged + ((t * HC + src) * groups + local) * 4).toint()
                    )
                )
                mixed = mixed + stream.to(Float32) * cute.arch.load(
                    (inputs + HC + src * HC + lane).llvm_ptr, Float32
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


def _dynamic(dtype, alignment: int, divisibility: int):
    return make_fake_dynamic_compact_tensor(
        dtype, alignment=alignment, divisibility=divisibility
    )


def compile_boundary(
    *,
    tokens: int,
    reduced: bool,
    hidden: int,
    tp: int,
    capacity: int,
    rms_eps: float,
    pre_eps: float,
    norm_eps: float,
):
    """The compiled boundary kernel. Its first argument is the flattened
    `[3, tp, capacity, hidden]` mailbox, or with `reduced` the `[m, hidden]`
    all-reduced input."""
    kernel = HopperBoundaryKernel(
        tokens=tokens,
        reduced=reduced,
        hidden=hidden,
        tp=tp,
        capacity_m=capacity,
        rms_eps=rms_eps,
        pre_eps=pre_eps,
        norm_eps=norm_eps,
    )
    source = (
        _dynamic(BFloat16, 16, hidden)
        if reduced
        else make_fake_compact_tensor(
            BFloat16, (LAMPORT_GENERATIONS * tp * capacity * hidden,), assumed_align=16
        )
    )
    return cute.compile(
        kernel,
        source,
        _dynamic(BFloat16, 16, HC * hidden),
        _dynamic(Float32, 4, HC),
        _dynamic(Float32, 4, HC * HC),
        make_fake_compact_tensor(BFloat16, (MIX * HC * hidden,), assumed_align=16),
        make_fake_compact_tensor(Float32, (3,), assumed_align=4),
        make_fake_compact_tensor(Float32, (MIX,), assumed_align=4),
        make_fake_compact_tensor(BFloat16, (hidden,), assumed_align=16),
        _dynamic(BFloat16, 16, HC * hidden),
        _dynamic(Float32, 4, MIX),
        _dynamic(BFloat16, 16, hidden),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        Int32(capacity),
        current_cu_stream(),
    )


def compile_mixes(*, sinkhorn_eps: float, post_mult: float, repeat: int, capacity: int):
    """Main's post-mix and Sinkhorn kernel (one thread per token)."""
    return cute.compile(
        _MixesDeviceKernel(
            sinkhorn_eps=sinkhorn_eps,
            post_mult=post_mult,
            repeat=repeat,
            capacity_m=capacity,
        ),
        _dynamic(Float32, 4, MIX),
        make_fake_compact_tensor(Float32, (3,), assumed_align=4),
        make_fake_compact_tensor(Float32, (MIX,), assumed_align=4),
        _dynamic(Float32, 4, HC),
        _dynamic(Float32, 4, HC * HC),
        Int32(capacity),
        current_cu_stream(),
    )
