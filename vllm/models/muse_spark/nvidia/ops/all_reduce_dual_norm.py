# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-shot LL all-reduce fused into the gated residual and the next RMSNorm.

Every rank multicasts its full `[M, H]` BF16 partial sum into slot `[stage, rank]` of
each rank's mailbox. The consumer sums the TP slots in rank order (so every rank gets
the same FP32 row) and runs the same dual norm as FlashInfer's all-gather variant,
optionally also writing the unrounded FP32 result for the router.
"""

from dataclasses import dataclass
from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_compact_tensor
from flashinfer.comm.mnnvl_cutedsl.cute_dsl_primitives import (
    VEC_BF16,
    WARP_SIZE,
    bf16x8_to_packed_u32x4,
    fragment_has_negative_zero,
    load_bf16x8_as_f32,
    load_f32x8,
    load_global_u32x4,
    load_volatile_u32,
    packed_u32x4_to_bf16x8,
    store_f32x8,
    store_global_u32,
    store_global_u32x4,
    store_lamport_sentinel_u32x4,
)
from flashinfer.comm.mnnvl_cutedsl.kernel_ll.column_gather import (
    LLColumnGather,
    cluster_row_sum,
)
from flashinfer.comm.mnnvl_cutedsl.kernel_ll.device_kernels import (
    ACTIVE_STAGE,
    LAMPORT_GENERATIONS,
    NEXT_STAGE,
)
from flashinfer.comm.mnnvl_cutedsl.kernel_ll.protocol import (
    compile_shared_only_publish,
    create_ll_state,
)
from flashinfer.comm.mnnvl_cutedsl.runtime import (
    current_cu_stream,
    make_fake_dynamic_compact_tensor,
    to_cute,
    to_cute_dynamic,
)


@dataclass(frozen=True, slots=True)
class DualNormConstants:
    post_eps: float
    next_eps: float
    post_gain_center: float
    next_gain_center: float
    has_post_weight: bool
    has_router: bool


class _AllReduceDualNormKernel:
    def __init__(
        self,
        *,
        hidden: int,
        tp: int,
        capacity_m: int,
        cluster_size: int,
        threads: int,
        constants: DualNormConstants,
        enable_pdl: bool,
    ) -> None:
        assert cluster_size * threads * VEC_BF16 == hidden
        assert threads % WARP_SIZE == 0 and threads >= cluster_size
        self.hidden = hidden
        self.tp = tp
        self.capacity_m = capacity_m
        self.cluster_size = cluster_size
        self.threads = threads
        self.warps = threads // WARP_SIZE
        self.constants = constants
        self.enable_pdl = enable_pdl

    @cute.jit
    def __call__(
        self,
        mailbox: cute.Tensor,
        residual: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        post_weight: cute.Tensor,
        next_weight: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        router_output: cute.Tensor,
        stage_state: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            mailbox,
            residual,
            alpha,
            beta,
            post_weight,
            next_weight,
            residual_output,
            norm_output,
            router_output,
            stage_state,
        ).launch(
            grid=(m, self.cluster_size, 1),
            block=(self.threads, 1, 1),
            cluster=(1, self.cluster_size, 1),
            smem=2 * self.cluster_size * self.warps * 4,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mailbox: cute.Tensor,
        residual: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        post_weight: cute.Tensor,
        next_weight: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        router_output: cute.Tensor,
        stage_state: cute.Tensor,
    ) -> None:
        constants = self.constants
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        fragment = cluster_rank * self.threads + tidx
        column = Int64(fragment) * VEC_BF16
        row_element = Int64(token) * self.hidden + column
        cute.arch.cluster_arrive_relaxed()

        alpha_values = load_f32x8(alpha, column)
        beta_values = load_f32x8(beta, column)
        next_gamma = load_bf16x8_as_f32(next_weight, column) + Float32(
            constants.next_gain_center
        )
        if cutlass.const_expr(constants.has_post_weight):
            post_gamma = load_bf16x8_as_f32(post_weight, column) + Float32(
                constants.post_gain_center
            )
        stream_values = load_f32x8(residual, row_element)

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        active_stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        stage_base = Int64(active_stage) * self.tp
        rank_stride = Int64(self.capacity_m) * self.hidden
        first_element = (stage_base * self.capacity_m + Int64(token)) * self.hidden
        first_pointer = cute.make_ptr(
            BFloat16,
            (mailbox.iterator + first_element + column).llvm_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        packed = cute.make_rmem_tensor(cute.make_layout((4, self.tp)), Uint32)
        # Poll every rank's slot in one pass so the waits overlap.
        dirty = fragment >= Int32(0)
        while dirty:
            dirty = fragment < Int32(0)
            for source_rank in cutlass.range_constexpr(self.tp):
                value = load_global_u32x4(
                    first_pointer + rank_stride * source_rank, volatile=True
                )
                dirty = dirty | fragment_has_negative_zero(value)
                packed[None, source_rank].store(value)
        branch = packed_u32x4_to_bf16x8(packed[None, 0].load()).to(Float32)
        for source_rank in cutlass.range_constexpr(1, self.tp):
            branch = branch + packed_u32x4_to_bf16x8(
                packed[None, source_rank].load()
            ).to(Float32)
        for source_rank in cutlass.range_constexpr(self.tp):
            store_lamport_sentinel_u32x4(
                Int64((first_pointer + rank_stride * source_rank).toint())
            )

        if token == 0 and fragment == 0:
            store_global_u32(
                stage_state.iterator + NEXT_STAGE,
                (active_stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        smem = cutlass.utils.SmemAllocator()
        post_slots = smem.allocate_array(Float32, self.cluster_size * self.warps)
        next_slots = smem.allocate_array(Float32, self.cluster_size * self.warps)

        b = branch
        if cutlass.const_expr(constants.has_post_weight):
            b = b * post_gamma
        cute.arch.cluster_wait()
        post_sum = cluster_row_sum(
            (b * b).reduce(
                cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
            ),
            post_slots,
            self.warps,
            self.cluster_size,
            cluster_rank,
            warp,
            lane,
        )
        post_inv_rms = cute.math.rsqrt(
            post_sum * Float32(1.0 / self.hidden) + Float32(constants.post_eps),
            fastmath=True,
        )
        updated = alpha_values * stream_values + (beta_values * b) * post_inv_rms
        store_f32x8(residual_output, row_element, updated)

        next_sum = cluster_row_sum(
            (updated * updated).reduce(
                cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
            ),
            next_slots,
            self.warps,
            self.cluster_size,
            cluster_rank,
            warp,
            lane,
        )
        next_inv_rms = cute.math.rsqrt(
            next_sum * Float32(1.0 / self.hidden) + Float32(constants.next_eps),
            fastmath=True,
        )
        result = updated * next_inv_rms * next_gamma
        store_global_u32x4(
            Int64((norm_output.iterator + row_element).toint()),
            bf16x8_to_packed_u32x4(result.to(BFloat16)),
        )
        if cutlass.const_expr(constants.has_router):
            store_f32x8(router_output, row_element, result)


class AllReduceDualNorm(LLColumnGather):
    """Collective over `group`; calls must use the same `m` on every rank."""

    def __init__(
        self,
        *,
        hidden_size: int,
        tp_size: int,
        rank: int,
        capacity_m: int,
        group: dist.ProcessGroup,
        cluster_size: int = 8,
        enable_pdl: bool = True,
    ) -> None:
        if hidden_size % (cluster_size * VEC_BF16):
            raise ValueError("hidden_size must split over the cluster")
        self.threads = hidden_size // (cluster_size * VEC_BF16)
        self.cluster_size = cluster_size
        self.enable_pdl = enable_pdl
        # LLColumnGather's mailbox, but every rank publishes the full row.
        self.hidden_size = hidden_size
        self.hidden_local = hidden_size
        self.tp_size = tp_size
        self.capacity_m = capacity_m
        self.state = create_ll_state(group, tp_size, rank, capacity_m, hidden_size)
        self._mailbox = to_cute(self.state.contribution_mailbox.tensor.flatten(), 16)
        self._mailbox_multicast = Int64(
            self.state.contribution_mailbox.multicast_address
        )
        self._stage_state = to_cute(self.state.stage_state, 4)
        self._publisher = compile_shared_only_publish(
            hidden=hidden_size,
            tp=tp_size,
            rank=rank,
            capacity_m=capacity_m,
            elements_per_thread=VEC_BF16,
            threads=128,
            release_before_store=False,
            enable_pdl=enable_pdl,
        )
        self._collectives: dict[Any, Any] = {}
        torch.accelerator.synchronize()
        dist.barrier(group)

    def _compile(self, constants: DualNormConstants) -> Any:
        device_kernel = _AllReduceDualNormKernel(
            hidden=self.hidden_size,
            tp=self.tp_size,
            capacity_m=self.capacity_m,
            cluster_size=self.cluster_size,
            threads=self.threads,
            constants=constants,
            enable_pdl=self.enable_pdl,
        )
        weight = make_fake_compact_tensor(
            BFloat16, (self.hidden_size,), assumed_align=16
        )
        gate = make_fake_compact_tensor(Float32, (self.hidden_size,), assumed_align=32)

        def rows_f32():
            return make_fake_dynamic_compact_tensor(
                Float32, alignment=32, divisibility=self.hidden_size
            )

        return cute.compile(
            device_kernel,
            make_fake_compact_tensor(
                BFloat16,
                (self.state.contribution_mailbox.tensor.numel(),),
                assumed_align=16,
            ),
            rows_f32(),
            gate,
            gate,
            weight,
            weight,
            rows_f32(),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            rows_f32(),
            make_fake_compact_tensor(Int32, (2,), assumed_align=4),
            Int32(self.capacity_m),
            current_cu_stream(),
        )

    def fits(self, partial: torch.Tensor) -> bool:
        return (
            partial.dim() == 2
            and 1 <= partial.shape[0] <= self.capacity_m
            and partial.shape[1] == self.hidden_size
            and partial.dtype == torch.bfloat16
            and partial.is_contiguous()
        )

    def __call__(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        next_weight: torch.Tensor,
        *,
        constants: DualNormConstants,
        post_weight: torch.Tensor | None,
        router_output: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(norm_output, residual_output)`."""
        assert constants.has_router == (router_output is not None)
        residual_output = torch.empty_like(residual)
        norm_output = torch.empty_like(partial)
        collective = self.compiled(constants)
        m = Int32(partial.shape[0])
        stream = current_cu_stream()
        self._publish(partial, m, stream)
        rows_out = to_cute_dynamic(
            residual_output.flatten(), 32, divisibility=self.hidden_size
        )
        collective(
            self._mailbox,
            to_cute_dynamic(residual.flatten(), 32, divisibility=self.hidden_size),
            to_cute(alpha, 32),
            to_cute(beta, 32),
            to_cute(post_weight if post_weight is not None else next_weight, 16),
            to_cute(next_weight, 16),
            rows_out,
            to_cute_dynamic(norm_output.flatten(), 16, divisibility=self.hidden_size),
            (
                to_cute_dynamic(
                    router_output.flatten(), 32, divisibility=self.hidden_size
                )
                if router_output is not None
                else rows_out
            ),
            self._stage_state,
            m,
            stream,
        )
        return norm_output, residual_output
