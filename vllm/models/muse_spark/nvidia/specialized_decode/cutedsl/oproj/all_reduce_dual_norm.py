# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
# The consumer's poll, rank-order sum and re-arm are adapted from FlashInfer
# (flashinfer/comm/mnnvl_cutedsl/kernel_ll/device_kernels.py and
# flashinfer/norm/kernels/gated_residual_next_rmsnorm.py,
# flashinfer-ai/flashinfer@840fb69656b336da91f0a8131c2e61b907bbc2cf,
# Apache-2.0), with unused options removed and the spin bounded; the
# publisher, the primitives and the dual norm are the package's shared copies
# (cutedsl/common).
# FlashInfer NOTICE: "FlashInfer / Copyright 2025-2026 NVIDIA / Copyright
# 2023-2026 FlashInfer community (https://flashinfer.ai/)".

"""The attention boundary at TP4 and TP8: the o_proj all-reduce fused with
the attention boundary's dual norm.

The shared-only publisher multicasts this rank's o_proj partial ([b, 8192]
BF16, -0.0 stored as +0.0) into slot ``rank`` of every rank's Lamport mailbox;
with ``published`` the partial's producer does that instead (TP8: the o_proj
GEMM's zero-copy epilogue). The PDL-chained consumer (the dual norm's grid)
waits for every rank's fragment, sums the ranks in rank order, re-arms the
slots and runs the dual norm with the standalone kernel's reduction. The
sum matches the all-reduce vLLM's TP communicator runs for the group
(the order of the backend it resolved to, which the executor reads once at
build), with FP32 adds that flush subnormals (as that all-reduce, built with
fast math, does):

- ``BF16_ADDS``: from rank 0's value, a BF16 rounding after every add, the
  TRT-LLM node-local one-shot (one node only);
- ``FP32_ADDS``: an FP32 sum from +0.0 rounded to BF16 once, the MNNVL
  one-shot (vLLM's default on one node, and the only backend across trays);
  above ``one_shot_rows`` rows that all-reduce runs its two-shot form, which
  adds the sum to +0.0 once more (-0.0 becomes +0.0).

The result is what the stock all-reduce followed by the stock dual norm
returns, bit for bit.

The mailbox needs NVLink multicast over the group (one NVLink domain);
construction raises on every rank when it has no multicast mapping, and
refuses the BF16 order for a group that spans nodes. Every rank must call with
the same row count, one call at a time; a poll that sees no data for
``SPIN_TIMEOUT_NS`` traps instead of hanging the group.
"""

from collections.abc import Callable
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cu_utils
import torch
import torch.distributed as dist
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, const_expr
from cutlass.cute.runtime import make_fake_compact_tensor

from vllm.distributed.parallel_state import in_the_same_node_as

from ..common.dual_norm import (
    CLUSTER_N,
    NSLOT,
    SMEM_BYTES,
    THREADS,
    DualNormConstants,
    dual_norm_fragment,
    tree_row_sum,
)
from ..common.host import assert_not_capturing
from ..common.mailbox import SPIN_TIMEOUT_NS, LamportMailbox, spin_deadline
from ..common.policy import BF16_ADDS, FP32_ADDS, MAILBOX_CAPACITY, SUM_ORDERS
from ..common.primitives import (
    ACTIVE_STAGE,
    LAMPORT_GENERATIONS,
    NEXT_STAGE,
    VEC_BF16,
    add_ftz_f32,
    bf16x8_to_packed_u32x4,
    current_cu_stream,
    fragment_has_negative_zero,
    load_bf16x8_as_f32,
    load_f32x8,
    load_global_u32x4,
    load_volatile_u32,
    make_fake_dynamic_compact_tensor,
    packed_u32x4_to_bf16x8,
    round_bf16,
    store_f32x8,
    store_global_u32,
    store_global_u32x4,
    store_lamport_sentinel_u32x4,
    to_cute,
    to_cute_dynamic,
)

H = 8192
# The MNNVL all-reduce runs one-shot while rows * H * tp * 2 bytes fit this.
MNNVL_ONE_SHOT_BYTES = 1 << 20


def one_shot_rows(tp: int) -> int:
    """The most rows the MNNVL all-reduce of ``tp`` ranks reduces one-shot."""
    return MNNVL_ONE_SHOT_BYTES // (H * tp * 2)


def stock_one_shot_rows(sum_order: str, tp: int, capacity: int) -> int:
    """Rows above which the stock all-reduce of ``sum_order`` runs its
    two-shot form: the MNNVL one's ``one_shot_rows``; the node-local one never
    does up to the mailbox capacity."""
    return one_shot_rows(tp) if sum_order == FP32_ADDS else capacity


assert CLUSTER_N * THREADS * VEC_BF16 == H


@cute.jit
def rank_order_sum(
    slots: cute.Tensor,
    tp: cutlass.Constexpr[int],
    sum_order: cutlass.Constexpr[str],
    one_shot_rows: cutlass.Constexpr[int],
    ptx_round: cutlass.Constexpr[bool],
):
    """The sum of the ``tp`` BF16x8 fragments in ``slots`` ([tp, 4] u32) in
    rank order, as the stock all-reduce of ``sum_order`` adds them, in an FP32
    [8] register tensor: ``FP32_ADDS`` from +0.0 and, above ``one_shot_rows``
    rows (the grid's), added to +0.0 once more (the MNNVL two-shot);
    ``BF16_ADDS`` from rank 0's value, rounded to BF16 after every add (with
    ``round_bf16`` when ``ptx_round``, else the DSL's conversion). Every add
    flushes subnormals (``add_ftz_f32``)."""
    total = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
    value = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
    if const_expr(sum_order == FP32_ADDS):
        total.fill(0.0)
    for r in cutlass.range_constexpr(tp):
        pr = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        for w in cutlass.range_constexpr(4):
            pr[w] = slots[r, w]
        value.store(packed_u32x4_to_bf16x8(pr.load()).to(Float32))
        if const_expr(sum_order == BF16_ADDS and r == 0):
            total.store(value.load())
        else:
            for i in cutlass.range_constexpr(VEC_BF16):
                total[i] = add_ftz_f32(total[i], value[i])
            if const_expr(sum_order == BF16_ADDS):
                if const_expr(ptx_round):
                    for i in cutlass.range_constexpr(VEC_BF16):
                        total[i] = round_bf16(total[i])
                else:
                    total.store(total.load().to(BFloat16).to(Float32))
    if cute.arch.grid_dim()[0] > Int32(one_shot_rows):
        for i in cutlass.range_constexpr(VEC_BF16):
            total[i] = add_ftz_f32(Float32(0.0), total[i])
    return total


def fused_mailbox(
    group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
    *,
    sum_order: str,
    device: torch.device,
    capacity: int = MAILBOX_CAPACITY,
) -> LamportMailbox:
    """The [3, tp, capacity, H] Lamport mailbox of an all-reduce fused into a
    boundary over ``group`` (the TP device group), summing in ``sum_order``.
    Refuses an unknown order, a group of one rank and the node-local
    ``BF16_ADDS`` order on a group that spans nodes (checked over
    ``cpu_group``, its CPU counterpart). Collective over both groups."""
    if sum_order not in SUM_ORDERS:
        raise ValueError(f"unknown sum order {sum_order!r}")
    tp, rank = dist.get_world_size(group), dist.get_rank(group)
    if tp < 2:
        raise ValueError("a fused all-reduce needs two or more ranks")
    if sum_order == BF16_ADDS and not all(
        in_the_same_node_as(cpu_group, source_rank=0)
    ):
        raise RuntimeError(
            "The BF16-add all-reduce is node-local: its TP group spans nodes."
        )
    return LamportMailbox(
        group, tp=tp, rank=rank, capacity=capacity, width=H, device=device
    )


class _AllReduceDualNormKernel:
    """Grid (m, CLUSTER_N), cluster (1, CLUSTER_N, 1), THREADS threads.

    Polls the ``tp`` slots of its 8 columns until none holds -0.0, advances the
    stage (token 0, fragment 0), sums the slots in rank order (``sum_order``),
    re-arms them, then runs the dual norm. Dependents are
    released after the last store, as the stock dual norm does.
    """

    def __init__(
        self,
        *,
        tp: int,
        capacity: int,
        constants: DualNormConstants,
        sum_order: str,
        router: bool,
        timeout_ns: int,
    ) -> None:
        self.tp = tp
        self.sum_order = sum_order
        self.one_shot_rows = stock_one_shot_rows(sum_order, tp, capacity)
        self.capacity = capacity
        self.constants = constants
        self.router = router
        self.timeout_ns = timeout_ns

    @cute.jit
    def __call__(
        self,
        mailbox: cute.Tensor,
        stream_in: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        post_weight: cute.Tensor,
        next_weight: cute.Tensor,
        stream_out: cute.Tensor,
        out: cute.Tensor,
        router_out: cute.Tensor,
        stage_state: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            mailbox,
            stream_in,
            alpha,
            beta,
            post_weight,
            next_weight,
            stream_out,
            out,
            router_out,
            stage_state,
        ).launch(
            grid=(m, CLUSTER_N, 1),
            block=(THREADS, 1, 1),
            cluster=(1, CLUSTER_N, 1),
            smem=SMEM_BYTES,
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mailbox: cute.Tensor,
        stream_in: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        post_weight: cute.Tensor,
        next_weight: cute.Tensor,
        stream_out: cute.Tensor,
        out: cute.Tensor,
        router_out: cute.Tensor,
        stage_state: cute.Tensor,
    ) -> None:
        constants = self.constants
        tidx, _, _ = cute.arch.thread_idx()
        token, cy, _ = cute.arch.block_idx()
        smem = cu_utils.SmemAllocator()
        mbar = smem.allocate_array(cutlass.Int64, num_elems=2)
        buf = smem.allocate_tensor(
            Float32, cute.make_layout((4, NSLOT * 2), stride=(1, 4)), byte_alignment=16
        )
        fragment = cy * THREADS + tidx
        column = Int64(fragment) * VEC_BF16
        row_element = Int64(token) * H + column

        # Gates and weights do not change between calls: load before the PDL wait.
        alpha_values = None
        if const_expr(constants.alpha_scalar is None):
            alpha_values = load_f32x8(alpha, column)
        elif const_expr(constants.alpha_scalar != 1.0):
            alpha_values = Float32(constants.alpha_scalar)
        beta_values = load_f32x8(beta, column)
        next_gamma = load_bf16x8_as_f32(next_weight, column) + Float32(
            constants.next_gain_center
        )
        post_gamma = None
        if const_expr(constants.post_gain_center is not None):
            post_gamma = load_bf16x8_as_f32(post_weight, column) + Float32(
                constants.post_gain_center
            )
        if tidx == 0:
            cute.arch.mbarrier_init(mbar, 1)
            cute.arch.mbarrier_init(mbar + 1, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        rank = cute.arch.block_idx_in_cluster()

        # The stream was written by the previous boundary: load after the wait.
        cute.arch.griddepcontrol_wait()
        stream_values = load_f32x8(stream_in, row_element)

        active_stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        lane_packed = cute.make_rmem_tensor(cute.make_layout((self.tp, 4)), Uint32)
        lane_packed.fill(Uint32(0))
        deadline = Int64(0)
        # Always true; a runtime value keeps the poll a device loop.
        dirty = fragment >= Int32(0)
        while dirty:
            dirty = False
            for r in cutlass.range_constexpr(self.tp):
                src = (
                    (Int64(active_stage) * self.tp + Int64(r)) * self.capacity
                    + Int64(token)
                ) * H + column
                ptr = cute.make_ptr(
                    BFloat16,
                    (mailbox.iterator + src).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                packed = load_global_u32x4(ptr, volatile=True)
                dirty = dirty | fragment_has_negative_zero(packed)
                for w in cutlass.range_constexpr(4):
                    lane_packed[r, w] = packed[w]
            if dirty:
                deadline = spin_deadline(deadline, self.timeout_ns)
        if token == 0 and fragment == 0:
            store_global_u32(
                stage_state.iterator + NEXT_STAGE,
                (active_stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )

        # Rank-order sum with flushing FP32 adds, as the stock all-reduce.
        total = rank_order_sum(
            lane_packed, self.tp, self.sum_order, self.one_shot_rows, False
        )
        acc = total.load().to(BFloat16)
        for r in cutlass.range_constexpr(self.tp):
            src = (
                (Int64(active_stage) * self.tp + Int64(r)) * self.capacity
                + Int64(token)
            ) * H + column
            store_lamport_sentinel_u32x4(Int64((mailbox.iterator + src).toint()))

        cute.arch.cluster_wait()
        res, result = dual_norm_fragment(
            acc.to(Float32),
            stream_values,
            alpha_values,
            beta_values,
            post_gamma,
            next_gamma,
            constants.post_eps,
            constants.next_eps,
            tree_row_sum(buf, mbar, tidx, lane, warp, rank),
        )
        # No CTA may exit while a peer still reads its shared memory.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        store_f32x8(stream_out, row_element, res)
        store_global_u32x4(
            Int64((out.iterator + row_element).toint()),
            bf16x8_to_packed_u32x4(result.to(BFloat16)),
        )
        if const_expr(self.router):
            store_f32x8(router_out, row_element, result)
        cute.arch.griddepcontrol_launch_dependents()


def compile_consumer(
    *,
    tp: int,
    capacity: int,
    mailbox_numel: int,
    constants: DualNormConstants,
    sum_order: str,
    router: bool,
    timeout_ns: int = SPIN_TIMEOUT_NS,
):
    """cute.compile one consumer; the row count is a runtime argument
    (1 <= m <= capacity). Call before CUDA graph capture."""
    assert_not_capturing("the all-reduce dual-norm consumer")
    weight = make_fake_compact_tensor(BFloat16, (H,), assumed_align=16)
    gate = make_fake_compact_tensor(Float32, (H,), assumed_align=32)
    rows_f32 = make_fake_dynamic_compact_tensor(Float32, alignment=32, divisibility=H)
    rows_bf16 = make_fake_dynamic_compact_tensor(BFloat16, alignment=16, divisibility=H)
    return cute.compile(
        _AllReduceDualNormKernel(
            tp=tp,
            capacity=capacity,
            constants=constants,
            sum_order=sum_order,
            router=router,
            timeout_ns=timeout_ns,
        ),
        make_fake_compact_tensor(BFloat16, (mailbox_numel,), assumed_align=16),
        rows_f32,
        gate,
        gate,
        weight,
        weight,
        rows_f32,
        rows_bf16,
        rows_f32,
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        Int32(capacity),
        current_cu_stream(),
    )


def check_rows(
    capacity: int,
    branch: torch.Tensor,
    stream: torch.Tensor,
    out: torch.Tensor,
    router_input: torch.Tensor | None,
) -> int:
    """The row count of one call, or ValueError. The caller decides on the row
    count alone, the same on every rank; any other mismatch is a bug and must
    raise before anything launches, not leave the other ranks polling."""
    m = branch.shape[0]
    rows = [(branch, torch.bfloat16, 16), (stream, torch.float32, 32)]
    rows.append((out, torch.bfloat16, 16))
    if router_input is not None:
        rows.append((router_input, torch.float32, 32))
    if not 1 <= m <= capacity or not all(
        t.shape == (m, H)
        and t.dtype == dtype
        and t.is_contiguous()
        and t.data_ptr() % align == 0
        for t, dtype, align in rows
    ):
        raise ValueError(
            "branch, out (BF16), stream and router_input (FP32) must be dense "
            f"aligned [m, {H}] tensors with 1 <= m <= {capacity}"
        )
    return m


class AllReduceDualNorm:
    """The mailbox and publisher of one TP group, shared by every attention
    boundary of the model. Construction is collective over ``group`` (the TP
    device group); ``cpu_group`` is its CPU (non-NCCL) counterpart, used to
    check that every rank shares this node when ``sum_order`` is
    ``BF16_ADDS``."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        cpu_group: dist.ProcessGroup,
        *,
        device: torch.device,
        sum_order: str,
        capacity: int = MAILBOX_CAPACITY,
        timeout_ns: int = SPIN_TIMEOUT_NS,
    ) -> None:
        assert_not_capturing("AllReduceDualNorm")
        self.mailbox = fused_mailbox(
            group, cpu_group, sum_order=sum_order, device=device, capacity=capacity
        )
        self.tp, self.rank = dist.get_world_size(group), dist.get_rank(group)
        self.capacity = capacity
        self.sum_order = sum_order
        self.timeout_ns = timeout_ns
        # The shared-only publisher, compiled by the first bind that needs it.
        self.publisher: Callable[..., None] | None = None
        self._consumers: dict[tuple[DualNormConstants, bool], Callable[..., None]] = {}

    def bind(
        self,
        constants: DualNormConstants,
        *,
        router: bool = False,
        published: bool = False,
    ) -> "AllReduceDualNormBoundary":
        """One boundary's op, its consumer compiled once per constant set.
        ``published``: the branch reaches the mailbox from its producer (the
        o_proj GEMM's zero-copy epilogue), so the op runs the consumer only;
        otherwise the publisher is compiled here (before capture) if it is not
        yet."""
        if not published and self.publisher is None:
            self.publisher = self.mailbox.compile_publish()
        key = (constants, router)
        if key not in self._consumers:
            self._consumers[key] = compile_consumer(
                tp=self.tp,
                capacity=self.capacity,
                mailbox_numel=self.mailbox.tensor.numel(),
                constants=constants,
                sum_order=self.sum_order,
                router=router,
                timeout_ns=self.timeout_ns,
            )
        return AllReduceDualNormBoundary(self, self._consumers[key], router, published)


class AllReduceDualNormBoundary(NamedTuple):
    """One attention boundary: the shared mailbox and its compiled consumer;
    with ``published``, the branch's producer has already written this rank's
    slot (zero-copy) and ``run`` launches the consumer alone."""

    owner: AllReduceDualNorm
    consumer: Callable[..., None]
    router: bool
    published: bool = False

    def run(
        self,
        branch: torch.Tensor,
        stream: torch.Tensor,
        post_weight: torch.Tensor | None,
        alpha: torch.Tensor | float,
        beta: torch.Tensor,
        next_weight: torch.Tensor,
        out: torch.Tensor,
        router_input: torch.Tensor | None = None,
    ) -> None:
        """All-reduce ``branch`` (this rank's BF16 [b, H] o_proj partial) over the
        group, then the dual norm as ``DualNorm.run`` takes it: ``stream``
        (FP32 [b, H]) updated in place, ``out`` (BF16 [b, H]) and, when bound
        with ``router``, ``router_input`` (FP32 [b, H], the unrounded pre-norm
        output). Graph-capturable; allocates nothing. Bound with ``published``,
        ``branch`` (checked, not read) is what the producer that ran just before
        on this stream wrote into the mailbox."""
        owner = self.owner
        if (router_input is not None) != self.router:
            raise ValueError("router_input must be given iff bound with router")
        m = check_rows(owner.capacity, branch, stream, out, router_input)
        mailbox = owner.mailbox
        if not self.published:
            mailbox.publish(owner.publisher, branch)
        stream_rows = to_cute_dynamic(stream.flatten(), 32, divisibility=H)
        self.consumer(
            mailbox.view,
            stream_rows,
            to_cute(alpha if isinstance(alpha, torch.Tensor) else beta, 32),
            to_cute(beta, 32),
            to_cute(post_weight if post_weight is not None else next_weight, 16),
            to_cute(next_weight, 16),
            stream_rows,
            to_cute_dynamic(out.flatten(), 16, divisibility=H),
            (
                to_cute_dynamic(router_input.flatten(), 32, divisibility=H)
                if router_input is not None
                else stream_rows
            ),
            mailbox.stage_view,
            Int32(m),
            current_cu_stream(),
        )
