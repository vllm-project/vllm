# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lamport TP all-reduce fused with a residual add and RMSNorm.

The boundary of the MTP draft layer, which has no mHC streams: the sublayer's
partial output (or its MoE output, top-k reduction still open) is multicast
into a Lamport mailbox by ``all_reduce_mhc``'s publish kernels, then one CTA
per token polls every rank's contribution, sums it, adds the residual and
RMS-normalizes the sum.

The draft's input takes the same mailbox: each rank publishes its vocabulary
shard's embedding rows (zeros for tokens it does not own), and the norm kernel
applies the MTP layer's enorm / hnorm and writes the eh_proj input.
"""

from __future__ import annotations

from typing import Any

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_compact_tensor

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
    bf16x8_to_packed_u32x4,
    current_cu_stream,
    fragment_has_negative_zero,
    load_global_u32x4,
    load_volatile_u32,
    make_fake_dynamic_compact_tensor,
    packed_u32x4_to_bf16x8,
    sanitize_negative_zero_u32x4,
    stmc_bf16x8,
    store_global_u32,
    store_global_u32x4,
    store_lamport_sentinel_u32x4,
    to_cute,
    to_cute_dynamic,
)

from .all_reduce_mhc import _gmem_ptr, _PublishDeviceKernel


class _EmbedPublishDeviceKernel:
    """Multicasts this rank's embedding rows of ``ids`` (zeros for ids outside
    its ``[start, start + rows)`` shard) into the Lamport mailbox."""

    def __init__(
        self, *, hidden: int, tp: int, rank: int, capacity_m: int, start: int, rows: int
    ):
        self.hidden = hidden
        self.tp = tp
        self.rank = rank
        self.capacity_m = capacity_m
        self.start = start
        self.rows = rows
        self.threads = 128
        self.ctas_per_token = hidden // VEC_BF16 // self.threads

    @cute.jit
    def __call__(
        self,
        ids: cute.Tensor,
        weight: cute.Tensor,
        stage_state: cute.Tensor,
        multicast: Int64,
        m: Int32,
        stream,
    ) -> None:
        self.kernel(ids, weight, stage_state, multicast).launch(
            grid=(m * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        ids: cute.Tensor,
        weight: cute.Tensor,
        stage_state: cute.Tensor,
        multicast: Int64,
    ):
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        token = block // self.ctas_per_token
        column = Int64((block % self.ctas_per_token) * self.threads + tidx) * VEC_BF16
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        row = Int64(ids[token]) - self.start
        words = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        words.fill(Uint32(0))
        if (row >= 0) & (row < self.rows):
            words.store(
                sanitize_negative_zero_u32x4(
                    load_global_u32x4(_gmem_ptr(weight, row * self.hidden + column, 16))
                )
            )
        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        slot = (
            (Int64(stage) * self.tp + self.rank) * self.capacity_m + Int64(token)
        ) * self.hidden + column
        stmc_bf16x8(multicast + slot * 2, words.load())
        if block == 0 and tidx == 0:
            store_global_u32(stage_state.iterator + ACTIVE_STAGE, stage)


class _EmbedNormDeviceKernel:
    """One CTA per token: sums the published embedding, zeroes it at position
    0, and writes ``[enorm(embedding), hnorm(previous)]``."""

    def __init__(self, *, hidden: int, tp: int, capacity_m: int, eps: float):
        self.hidden = hidden
        self.tp = tp
        self.capacity_m = capacity_m
        self.eps = eps
        self.threads = hidden // VEC_BF16
        if self.threads % WARP_SIZE or self.threads > 1024:
            raise ValueError("unsupported hidden size")
        self.warps = self.threads // WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mailbox: cute.Tensor,
        positions: cute.Tensor,
        previous: cute.Tensor,
        enorm: cute.Tensor,
        hnorm: cute.Tensor,
        output: cute.Tensor,
        stage_state: cute.Tensor,
        m: Int32,
        stream,
    ) -> None:
        self.kernel(
            mailbox, positions, previous, enorm, hnorm, output, stage_state
        ).launch(
            grid=(m, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.warps * 8,
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mailbox: cute.Tensor,
        positions: cute.Tensor,
        previous: cute.Tensor,
        enorm: cute.Tensor,
        hnorm: cute.Tensor,
        output: cute.Tensor,
        stage_state: cute.Tensor,
    ) -> None:
        hidden = self.hidden
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        column = Int64(tidx) * VEC_BF16
        element = Int64(token) * hidden + column
        enorm_values = packed_u32x4_to_bf16x8(
            load_global_u32x4(_gmem_ptr(enorm, column, 16))
        ).to(Float32)
        hnorm_values = packed_u32x4_to_bf16x8(
            load_global_u32x4(_gmem_ptr(hnorm, column, 16))
        ).to(Float32)

        cute.arch.griddepcontrol_wait()

        previous_values = packed_u32x4_to_bf16x8(
            load_global_u32x4(_gmem_ptr(previous, element, 16))
        ).to(Float32)
        active_stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        reduced = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
        reduced.fill(0.0)
        packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        for source in cutlass.range_constexpr(self.tp):
            slot = (
                (Int64(active_stage) * self.tp + source) * self.capacity_m
                + Int64(token)
            ) * hidden + column
            dirty = True
            while dirty:
                loaded = load_global_u32x4(_gmem_ptr(mailbox, slot, 16), volatile=True)
                dirty = fragment_has_negative_zero(loaded)
                packed.store(loaded)
            reduced.store(
                reduced.load() + packed_u32x4_to_bf16x8(packed.load()).to(Float32)
            )

        if token == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + NEXT_STAGE,
                (active_stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )
        cute.arch.griddepcontrol_launch_dependents()
        for source in cutlass.range_constexpr(self.tp):
            slot = (
                (Int64(active_stage) * self.tp + source) * self.capacity_m
                + Int64(token)
            ) * hidden + column
            store_lamport_sentinel_u32x4(Int64((mailbox.iterator + slot).toint()))

        # Exactly one rank owns each row, so the sum is the BF16 row itself.
        keep = Float32(1.0)
        if positions[token] == 0:
            keep = Float32(0.0)
        embedding = reduced.load().to(BFloat16).to(Float32) * keep
        squares = cute.make_rmem_tensor(cute.make_layout((2,)), Float32)
        squares[0] = (embedding * embedding).reduce(
            cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
        )
        squares[1] = (previous_values * previous_values).reduce(
            cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
        )
        for i in cutlass.range_constexpr(2):
            for offset in cutlass.range_constexpr(5):
                squares[i] = squares[i] + cute.arch.shuffle_sync_bfly(
                    squares[i], offset=1 << offset, mask=-1, mask_and_clamp=31
                )
        smem = cutlass.utils.SmemAllocator()
        warp_sums = smem.allocate_array(Float32, 2 * self.warps)
        if lane == 0:
            cute.arch.store((warp_sums + 2 * warp).llvm_ptr, squares[0])
            cute.arch.store((warp_sums + 2 * warp + 1).llvm_ptr, squares[1])
        cute.arch.barrier()
        total_e = Float32(0.0)
        total_h = Float32(0.0)
        for w in cutlass.range_constexpr(self.warps):
            total_e = total_e + cute.arch.load((warp_sums + 2 * w).llvm_ptr, Float32)
            total_h = total_h + cute.arch.load(
                (warp_sums + 2 * w + 1).llvm_ptr, Float32
            )
        inv_e = cute.math.rsqrt(total_e / Float32(hidden) + Float32(self.eps))
        inv_h = cute.math.rsqrt(total_h / Float32(hidden) + Float32(self.eps))
        row = Int64(token) * (2 * hidden) + column
        store_global_u32x4(
            Int64((output.iterator + row).toint()),
            bf16x8_to_packed_u32x4((embedding * inv_e * enorm_values).to(BFloat16)),
        )
        store_global_u32x4(
            Int64((output.iterator + row + hidden).toint()),
            bf16x8_to_packed_u32x4(
                (previous_values * inv_h * hnorm_values).to(BFloat16)
            ),
        )


class _NormDeviceKernel:
    """One CTA per token, one 8-element fragment per thread."""

    def __init__(self, *, hidden: int, tp: int, capacity_m: int, eps: float):
        self.hidden = hidden
        self.tp = tp
        self.capacity_m = capacity_m
        self.eps = eps
        self.threads = hidden // VEC_BF16
        if self.threads % WARP_SIZE or self.threads > 1024:
            raise ValueError("unsupported hidden size")
        self.warps = self.threads // WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mailbox: cute.Tensor,
        residual: cute.Tensor,
        gamma: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        stage_state: cute.Tensor,
        m: Int32,
        stream,
    ) -> None:
        self.kernel(
            mailbox, residual, gamma, residual_output, norm_output, stage_state
        ).launch(
            grid=(m, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.warps * 4,
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mailbox: cute.Tensor,
        residual: cute.Tensor,
        gamma: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        stage_state: cute.Tensor,
    ) -> None:
        hidden = self.hidden
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        column = Int64(tidx) * VEC_BF16
        element = Int64(token) * hidden + column

        # The residual was written before the publish kernel's producer started.
        residual_values = packed_u32x4_to_bf16x8(
            load_global_u32x4(_gmem_ptr(residual, element, 16))
        ).to(Float32)
        gamma_values = packed_u32x4_to_bf16x8(
            load_global_u32x4(_gmem_ptr(gamma, column, 16))
        ).to(Float32)

        cute.arch.griddepcontrol_wait()

        active_stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        reduced = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
        reduced.fill(0.0)
        packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        for source in cutlass.range_constexpr(self.tp):
            slot = (
                (Int64(active_stage) * self.tp + source) * self.capacity_m
                + Int64(token)
            ) * hidden + column
            dirty = True
            while dirty:
                loaded = load_global_u32x4(_gmem_ptr(mailbox, slot, 16), volatile=True)
                dirty = fragment_has_negative_zero(loaded)
                packed.store(loaded)
            reduced.store(
                reduced.load() + packed_u32x4_to_bf16x8(packed.load()).to(Float32)
            )

        if token == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + NEXT_STAGE,
                (active_stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )
        cute.arch.griddepcontrol_launch_dependents()
        for source in cutlass.range_constexpr(self.tp):
            slot = (
                (Int64(active_stage) * self.tp + source) * self.capacity_m
                + Int64(token)
            ) * hidden + column
            store_lamport_sentinel_u32x4(Int64((mailbox.iterator + slot).toint()))

        # The all-reduced sublayer output is BF16, as the collective returns it.
        summed = (reduced.load().to(BFloat16).to(Float32) + residual_values).to(
            BFloat16
        )
        store_global_u32x4(
            Int64((residual_output.iterator + element).toint()),
            bf16x8_to_packed_u32x4(summed),
        )
        values = summed.to(Float32)
        square = (values * values).reduce(
            cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
        )
        for offset in cutlass.range_constexpr(5):
            square = square + cute.arch.shuffle_sync_bfly(
                square, offset=1 << offset, mask=-1, mask_and_clamp=31
            )
        smem = cutlass.utils.SmemAllocator()
        warp_sums = smem.allocate_array(Float32, self.warps)
        if lane == 0:
            cute.arch.store((warp_sums + warp).llvm_ptr, square)
        cute.arch.barrier()
        total = Float32(0.0)
        for w in cutlass.range_constexpr(self.warps):
            total = total + cute.arch.load((warp_sums + w).llvm_ptr, Float32)
        inv_rms = cute.math.rsqrt(total / Float32(hidden) + Float32(self.eps))
        store_global_u32x4(
            Int64((norm_output.iterator + element).toint()),
            bf16x8_to_packed_u32x4((values * inv_rms * gamma_values).to(BFloat16)),
        )


class AllReduceNorm:
    """``tensor_model_parallel_all_reduce`` + ``RMSNorm(x, residual)`` for up to
    ``max_num_tokens`` tokens: a publish kernel and a norm kernel.

    Construction is collective over the TP group; one instance serves every
    boundary of its model, which run one after another.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        max_num_tokens: int,
        top_k: int,
        eps: float,
        device: torch.device,
    ) -> None:
        group = get_tp_group().device_group
        self.top_k = top_k
        self.hidden_size = hidden = hidden_size
        self.capacity = capacity = max_num_tokens
        self.eps = eps
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
            self._norm = cute.compile(
                _NormDeviceKernel(hidden=hidden, tp=tp, capacity_m=capacity, eps=eps),
                make_fake_compact_tensor(
                    BFloat16,
                    (LAMPORT_GENERATIONS * tp * capacity * hidden,),
                    assumed_align=16,
                ),
                dynamic(BFloat16, 16, hidden),
                make_fake_compact_tensor(BFloat16, (hidden,), assumed_align=16),
                dynamic(BFloat16, 16, hidden),
                dynamic(BFloat16, 16, hidden),
                make_fake_compact_tensor(Int32, (2,), assumed_align=4),
                Int32(capacity),
                current_cu_stream(),
            )
            self._mailbox = symm_mem.empty(
                (LAMPORT_GENERATIONS, tp, capacity, hidden),
                dtype=torch.bfloat16,
                device=device,
            )
            handle = symm_mem.rendezvous(self._mailbox, group)
            multicast = int(handle.multicast_ptr or 0)
            if not multicast or multicast % 16:
                raise RuntimeError("NVLink multicast mapping is unavailable")
            self._handle = handle
            self._multicast = multicast
            # Every slot starts as the Lamport sentinel, BF16 negative zero.
            self._mailbox.view(torch.int16).fill_(-32768)
            self._stage_state = torch.zeros(2, dtype=torch.int32, device=device)
            torch.accelerator.synchronize()
        dist.barrier(group=group)
        self.tp, self.rank, self.device = tp, rank, device
        self._embed: dict[int, Any] = {}

    def supports(self, tokens: int) -> bool:
        return 1 <= tokens <= self.capacity

    def prepare_embedding(
        self, weight: torch.Tensor, start: int, rows: int, eps: float
    ):
        """Compiles ``embed_norm`` for the vocabulary shard ``weight`` whose
        first ``rows`` rows hold token ids ``[start, start + rows)``."""
        hidden, capacity = self.hidden_size, self.capacity
        assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
        assert weight.shape[1] == hidden
        key = weight.data_ptr()
        if key in self._embed:
            return

        def dynamic(dtype, alignment, divisibility):
            return make_fake_dynamic_compact_tensor(
                dtype, alignment=alignment, divisibility=divisibility
            )

        with torch.accelerator.device_index(self.device.index):
            publish = cute.compile(
                _EmbedPublishDeviceKernel(
                    hidden=hidden,
                    tp=self.tp,
                    rank=self.rank,
                    capacity_m=capacity,
                    start=start,
                    rows=rows,
                ),
                dynamic(Int32, 4, 1),
                make_fake_compact_tensor(BFloat16, (weight.numel(),), assumed_align=16),
                make_fake_compact_tensor(Int32, (2,), assumed_align=4),
                Int64(0),
                Int32(capacity),
                current_cu_stream(),
            )
            norm = cute.compile(
                _EmbedNormDeviceKernel(
                    hidden=hidden, tp=self.tp, capacity_m=capacity, eps=eps
                ),
                make_fake_compact_tensor(
                    BFloat16,
                    (LAMPORT_GENERATIONS * self.tp * capacity * hidden,),
                    assumed_align=16,
                ),
                dynamic(Int64, 8, 1),
                dynamic(BFloat16, 16, hidden),
                make_fake_compact_tensor(BFloat16, (hidden,), assumed_align=16),
                make_fake_compact_tensor(BFloat16, (hidden,), assumed_align=16),
                dynamic(BFloat16, 16, 2 * hidden),
                make_fake_compact_tensor(Int32, (2,), assumed_align=4),
                Int32(capacity),
                current_cu_stream(),
            )
        self._embed[key] = (publish, norm, weight)

    def embed_norm(
        self,
        ids: torch.Tensor,
        positions: torch.Tensor,
        previous: torch.Tensor,
        weight: torch.Tensor,
        enorm: torch.Tensor,
        hnorm: torch.Tensor,
    ) -> torch.Tensor:
        """``fused_eh_norm`` of the vocab-parallel embedding of ``ids``:
        ``[M, 2 * hidden]`` = ``[enorm(embedding), hnorm(previous)]``, the
        embedding zeroed at position 0."""
        publish, norm, _ = self._embed[weight.data_ptr()]
        hidden = self.hidden_size
        m = ids.shape[0]
        state = to_cute(self._stage_state, 4)
        publish(
            to_cute_dynamic(ids, 4, divisibility=1),
            to_cute(weight.flatten(), 16),
            state,
            Int64(self._multicast),
            Int32(m),
            current_cu_stream(),
        )
        output = torch.empty((m, 2 * hidden), dtype=previous.dtype, device=ids.device)
        norm(
            to_cute(self._mailbox.flatten(), 16),
            to_cute_dynamic(positions, 8, divisibility=1),
            to_cute_dynamic(previous.flatten(), 16, divisibility=hidden),
            to_cute(enorm, 16),
            to_cute(hnorm, 16),
            to_cute_dynamic(output.flatten(), 16, divisibility=2 * hidden),
            state,
            Int32(m),
            current_cu_stream(),
        )
        return output

    def __call__(
        self,
        x: torch.Tensor | MoEOutput,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(RMSNorm(sum + residual), sum + residual)`` of the TP sum of ``x``.

        Args:
            x: [M, hidden] BF16 TP-partial sublayer output, or a MoE output
                whose routed top-k reduction is still open.
            residual: [M, hidden] BF16 residual.
            norm_weight: [hidden] BF16 RMSNorm weight.

        """
        hidden = self.hidden_size
        state = to_cute(self._stage_state, 4)
        if isinstance(x, MoEOutput):
            routed, shared = x.routed, x.shared_output
            assert shared is not None and not isinstance(routed, torch.Tensor)
            m = shared.shape[0]
            weights = routed.expert_weights
            self._finalize_publish[weights.dtype](
                to_cute_dynamic(
                    routed.gemm2_permuted.flatten(), 16, divisibility=hidden
                ),
                to_cute_dynamic(
                    weights.flatten(), weights.element_size(), divisibility=self.top_k
                ),
                to_cute_dynamic(
                    routed.expanded_idx_to_permuted_idx.flatten(),
                    4,
                    divisibility=self.top_k,
                ),
                to_cute_dynamic(shared.flatten(), 16, divisibility=hidden),
                state,
                Int64(self._multicast),
                Int32(m),
                current_cu_stream(),
            )
        else:
            m = x.shape[0]
            self._publish(
                to_cute_dynamic(x.flatten(), 16, divisibility=hidden),
                state,
                Int64(self._multicast),
                Int32(m),
                current_cu_stream(),
            )
        residual = residual.contiguous()
        residual_output = torch.empty_like(residual)
        norm_output = torch.empty_like(residual)
        self._norm(
            to_cute(self._mailbox.flatten(), 16),
            to_cute_dynamic(residual.flatten(), 16, divisibility=hidden),
            to_cute(norm_weight, 16),
            to_cute_dynamic(residual_output.flatten(), 16, divisibility=hidden),
            to_cute_dynamic(norm_output.flatten(), 16, divisibility=hidden),
            state,
            Int32(m),
            current_cu_stream(),
        )
        return norm_output, residual_output
