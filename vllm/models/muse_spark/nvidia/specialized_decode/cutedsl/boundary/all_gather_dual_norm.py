# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
# The consumer kernel (owned-fragment gather, parameter loads before the PDL
# wait, the dual norm over the gathered row) is adapted from FlashInfer
# (flashinfer/comm/mnnvl_cutedsl/kernel_ll/all_gather_dual_norm.py and
# column_gather.py, flashinfer-ai/flashinfer@14ba7aaa84718abe7a5648184fa9ad15362e2880,
# Apache-2.0); here it triggers its PDL dependents before the poll, bounds the
# poll, updates the stream in place and writes the aux state.
# FlashInfer NOTICE: "FlashInfer / Copyright 2025-2026 NVIDIA / Copyright
# 2023-2026 FlashInfer community (https://flashinfer.ai/)".

"""The input boundary of every Muse Spark decoder layer, and the final boundary.

The MoE of layer ``l - 1`` leaves this rank's unpooler output as a column
block ``[b, H / tp]`` BF16 (block ``rank`` of the row). The boundary of layer
``l`` gathers the blocks of every rank through a Lamport mailbox in NVLink
multicast memory (``MailboxSlot``), then runs the dual norm over the full row::

    b = branch * (post_weight + post_gain_center)  # or branch
    stream = alpha * stream + (beta * b) * rsqrt(mean(b**2) + post_eps)
    out = bf16(
        stream * rsqrt(mean(stream**2) + next_eps) * (next_weight + next_gain_center)
    )

with layer ``l - 1``'s post_ffn_norm and residual gates and layer ``l``'s
pre_qkv_proj_norm (model.norm at the final boundary, after the last layer).
The stream (FP32 [b, H]) is updated in place; ``aux`` receives it when layer
``l`` is an aux layer of the drafter.

Two kernels per call, as the stock fused path: the publisher (or the unpooler's
zero-copy epilogue, which writes the same bytes) multicasts this rank's block
into slot ``[NEXT_STAGE, rank]`` of every rank's ``[3, tp, capacity, H / tp]``
mailbox and sets ``ACTIVE_STAGE``; the consumer reads each 8-column fragment
from the one rank that owns it, re-arms it, rotates ``NEXT_STAGE`` and runs
the dual norm. Each boundary has its own mailbox and stage words.

The consumer releases its PDL dependents right after ``griddepcontrol.wait``,
before the poll, so the next kernel (the QKV front loading its weights) starts
while the gather waits on the other ranks. A dependent still waits for this
whole grid in its own ``griddepcontrol.wait``. Like the stock consumer it
loads the stream and the parameters before its wait: no kernel between the
stream's last writer and this one may trigger its dependents before its own
wait.

Numerics: the row reduction decides the last bits. Where the stock forward
runs the fused all-gather (the previous MoE defers its column gather and the
layer is not an aux layer), the consumer uses that op's serial reduction and
is bit-identical to it. Where the stock forward gathers the columns first and
runs the standalone dual norm (an aux layer, or a MoE that does not defer the
gather), it uses the standalone kernel's tree reduction and is bit-identical
to that. The aux state is the updated stream; the stock forward computes its
aux state with a separate gated-residual norm, which differs in the last bits.
"""

import functools
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, const_expr
from cutlass.cute.runtime import make_fake_compact_tensor
from torch import nn

from ..common.dual_norm import (
    CLUSTER_N,
    NSLOT,
    SMEM_BYTES,
    THREADS,
    DualNormConstants,
    boundary_constants,
    dual_norm_fragment,
    serial_row_sum,
    tree_row_sum,
)
from ..common.dual_norm import H as HIDDEN
from ..common.host import assert_not_capturing
from ..common.mailbox import SPIN_TIMEOUT_NS, LamportMailbox, poll_owned_fragment
from ..common.policy import MAILBOX_CAPACITY
from ..common.primitives import (
    LAMPORT_GENERATIONS,
    VEC_BF16,
    bf16x8_to_packed_u32x4,
    current_cu_stream,
    load_bf16x8_as_f32,
    load_f32x8,
    make_fake_dynamic_compact_tensor,
    packed_u32x4_to_bf16x8,
    store_f32x8,
    store_global_u32x4,
    to_cute,
    to_cute_dynamic,
)

# The executor's decode batch (runtime/providers.py BATCHES): the rows of the
# build-time warm-up and of the stock gather decision.
WARM_UP_ROWS = 8


class MailboxSlot(NamedTuple):
    """One boundary's slot in its symmetric-memory Lamport buffer.

    The buffer is ``[LAMPORT_GENERATIONS, tp, capacity, H / tp]`` BF16 on every
    rank; row ``t`` of rank ``r``'s block in generation ``g`` starts at element
    ``((g * tp + r) * capacity + t) * row_stride``. The generation a writer
    uses is ``NEXT_STAGE`` of the mailbox's stage words (``BoundaryMailbox.
    stage_state``) and the writer sets ``ACTIVE_STAGE`` to it; ``phase`` is
    the caller's label of the slot and does not select the generation.
    """

    # Every rank's buffer (unicast), in rank order.
    peers: tuple[int, ...]
    # The multicast address of the buffer.
    multicast: int
    # This rank's first column in a gathered [b, H] row.
    column_offset: int
    # Elements between the rows of one rank's block (H / tp).
    row_stride: int
    phase: int


class BoundaryMailbox:
    """The mailbox of one boundary: a ``LamportMailbox`` of ``[b, H / tp]``
    column blocks and its publisher, built collectively (every rank, the same
    order) before CUDA graph capture. Raises on every rank without NVLink
    multicast."""

    def __init__(
        self,
        group,
        *,
        tp: int,
        rank: int,
        device: torch.device,
        layer: int,
        capacity: int = MAILBOX_CAPACITY,
        enable_pdl: bool = True,
    ) -> None:
        assert_not_capturing("BoundaryMailbox")
        if HIDDEN % (tp * VEC_BF16):
            raise ValueError("H must split into 8-column fragments per rank")
        self.layer, self.tp, self.rank = layer, tp, rank
        self.capacity, self.width = capacity, HIDDEN // tp
        self.lamport = LamportMailbox(
            group,
            tp=tp,
            rank=rank,
            capacity=capacity,
            width=self.width,
            device=device,
        )
        # The stock fused all-gather's publisher tuning: 128 threads, the
        # stage released after the store.
        self.publisher = self.lamport.compile_publish(enable_pdl=enable_pdl)

    @property
    def tensor(self) -> torch.Tensor:
        """This rank's ``[3, tp, capacity, H / tp]`` BF16 buffer."""
        return self.lamport.tensor

    @property
    def stage_state(self) -> torch.Tensor:
        """Int32 [2]: ``NEXT_STAGE`` (set by the consumer), ``ACTIVE_STAGE``
        (set by the writer)."""
        return self.lamport.stage_state

    def slot(self, layer: int, phase: int) -> MailboxSlot:
        if layer != self.layer:
            raise ValueError(f"this is the mailbox of layer {self.layer}, not {layer}")
        if not 0 <= phase < LAMPORT_GENERATIONS:
            raise ValueError(f"phase must be in [0, {LAMPORT_GENERATIONS})")
        return MailboxSlot(
            peers=self.lamport.peers,
            multicast=self.lamport.multicast,
            column_offset=self.rank * self.width,
            row_stride=self.width,
            phase=phase,
        )

    def check(self, slot: MailboxSlot) -> None:
        if slot.multicast != self.lamport.multicast or slot.peers != self.lamport.peers:
            raise ValueError(f"the slot is not the mailbox of layer {self.layer}")

    def publish(self, cols: torch.Tensor, slot: MailboxSlot) -> None:
        """Multicast this rank's ``[b, H / tp]`` BF16 block into every rank's
        copy (what the zero-copy epilogue writes). Graph-capturable."""
        self.check(slot)
        rows = cols.shape[0]
        if (
            cols.dtype != torch.bfloat16
            or cols.shape != (rows, self.width)
            or not cols.is_contiguous()
            or rows > self.capacity
        ):
            raise ValueError(
                f"expected a contiguous [b <= {self.capacity}, {self.width}] BF16 block"
            )
        if rows:
            self.lamport.publish(self.publisher, cols)


def all_gather_fragment(
    branch,
    stream,
    alpha,
    beta,
    post_gamma,
    next_gamma,
    post_eps: float,
    next_eps: float,
    row_sum,
):
    """``dual_norm_fragment`` in the form of the public all-gather dual norm
    (all_gather_dual_norm.py at the sha above): FP32 constants, the zero-init
    reductions and the gated add written as one expression after the first row
    sum. The serial consumer must use it to be bitwise to that op; the tree
    consumer keeps the standalone kernel's form."""
    b = branch
    if post_gamma is not None:
        b = b * post_gamma
    post_sum = row_sum(
        (b * b).reduce(
            cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
        ),
        0,
    )
    post_inv_rms = cute.math.rsqrt(
        post_sum * Float32(1.0 / HIDDEN) + Float32(post_eps), fastmath=True
    )
    if alpha is None:
        updated = stream + (beta * b) * post_inv_rms
    else:
        updated = alpha * stream + (beta * b) * post_inv_rms
    next_sum = row_sum(
        (updated * updated).reduce(
            cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
        ),
        1,
    )
    next_inv_rms = cute.math.rsqrt(
        next_sum * Float32(1.0 / HIDDEN) + Float32(next_eps), fastmath=True
    )
    return updated, updated * next_inv_rms * next_gamma


class _AllGatherDualNormKernel:
    """Grid (b, CLUSTER_N), cluster (1, CLUSTER_N, 1), THREADS threads; thread
    ``(cluster rank, tid)`` owns columns ``8 * (cluster rank * THREADS + tid)``
    and the 7 after it."""

    def __init__(
        self,
        *,
        tp: int,
        capacity: int,
        constants: DualNormConstants,
        tree: bool,
        aux: bool,
        enable_pdl: bool,
        early_trigger: bool,
        timeout_ns: int,
    ) -> None:
        if constants.alpha_scalar is not None:
            raise ValueError("the boundary takes the channel-wise stream gate")
        self.tp, self.capacity = tp, capacity
        self.hidden_local = HIDDEN // tp
        self.constants = constants
        self.tree, self.aux = tree, aux
        self.enable_pdl = enable_pdl
        self.early_trigger = enable_pdl and early_trigger
        self.timeout_ns = timeout_ns
        # Tree: two mbarriers and a [4, 2 * NSLOT] FP32 buffer; serial: one
        # FP32 slot per (CTA, warp) for each of the two row statistics.
        self.smem_bytes = SMEM_BYTES if tree else 2 * NSLOT * 4

    @cute.jit
    def __call__(
        self,
        mailbox: cute.Tensor,
        stage_state: cute.Tensor,
        stream_rows: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        post_weight: cute.Tensor,
        next_weight: cute.Tensor,
        out: cute.Tensor,
        aux: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            mailbox,
            stage_state,
            stream_rows,
            alpha,
            beta,
            post_weight,
            next_weight,
            out,
            aux,
        ).launch(
            grid=(m, CLUSTER_N, 1),
            block=(THREADS, 1, 1),
            cluster=(1, CLUSTER_N, 1),
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mailbox: cute.Tensor,
        stage_state: cute.Tensor,
        stream_rows: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        post_weight: cute.Tensor,
        next_weight: cute.Tensor,
        out: cute.Tensor,
        aux: cute.Tensor,
    ) -> None:
        constants = self.constants
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        fragment = cluster_rank * THREADS + tidx
        column = Int64(fragment) * VEC_BF16
        row_element = Int64(token) * HIDDEN + column

        smem = cutlass.utils.SmemAllocator()
        if const_expr(self.tree):
            mbar = smem.allocate_array(cutlass.Int64, num_elems=2)
            buf = smem.allocate_tensor(
                Float32,
                cute.make_layout((4, NSLOT * 2), stride=(1, 4)),
                byte_alignment=16,
            )
            if tidx == 0:
                cute.arch.mbarrier_init(mbar, 1)
                cute.arch.mbarrier_init(mbar + 1, 1)
            cute.arch.mbarrier_init_fence()
            row_sum = tree_row_sum(buf, mbar, tidx, lane, warp, cluster_rank)
        else:
            post_slots = smem.allocate_array(Float32, NSLOT)
            next_slots = smem.allocate_array(Float32, NSLOT)
            row_sum = serial_row_sum(post_slots, next_slots, cluster_rank, warp, lane)
        # Every peer must have started before the row sums store into its
        # shared memory; the matching wait sits just before the first one.
        cute.arch.cluster_arrive_relaxed()

        # The gates, weights and stream do not depend on the publisher.
        alpha_values = load_f32x8(alpha, column)
        beta_values = load_f32x8(beta, column)
        next_gamma = load_bf16x8_as_f32(next_weight, column) + Float32(
            constants.next_gain_center
        )
        post_gamma = None
        if const_expr(constants.post_gain_center is not None):
            post_gamma = load_bf16x8_as_f32(post_weight, column) + Float32(
                constants.post_gain_center
            )
        stream_values = load_f32x8(stream_rows, row_element)

        if const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
            if const_expr(self.early_trigger):
                cute.arch.griddepcontrol_launch_dependents()

        branch_packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        poll_owned_fragment(
            mailbox,
            stage_state,
            branch_packed,
            token,
            fragment,
            self.tp,
            self.capacity,
            self.hidden_local,
            self.enable_pdl and not self.early_trigger,
            self.timeout_ns,
        )
        branch = packed_u32x4_to_bf16x8(branch_packed.load()).to(Float32)

        cute.arch.cluster_wait()
        if const_expr(self.tree):
            updated, result = dual_norm_fragment(
                branch,
                stream_values,
                alpha_values,
                beta_values,
                post_gamma,
                next_gamma,
                constants.post_eps,
                constants.next_eps,
                row_sum,
            )
        else:
            updated, result = all_gather_fragment(
                branch,
                stream_values,
                alpha_values,
                beta_values,
                post_gamma,
                next_gamma,
                constants.post_eps,
                constants.next_eps,
                row_sum,
            )
        if const_expr(self.tree):
            # No CTA may exit while a peer still reads its shared memory.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        store_f32x8(stream_rows, row_element, updated)
        if const_expr(self.aux):
            store_f32x8(aux, row_element, updated)
        store_global_u32x4(
            Int64((out.iterator + row_element).toint()),
            bf16x8_to_packed_u32x4(result.to(BFloat16)),
        )


def compile_all_gather_dual_norm(
    *,
    tp: int,
    capacity: int,
    constants: DualNormConstants,
    tree: bool,
    aux: bool,
    enable_pdl: bool = True,
    early_trigger: bool = True,
    timeout_ns: int = SPIN_TIMEOUT_NS,
):
    """Compile the consumer for one boundary's constants (before capture).

    Call it as ``compiled(mailbox, stage_state, stream, alpha, beta,
    post_weight, next_weight, out, aux, Int32(b), cu_stream)``: the flattened
    ``[3, tp, capacity, H / tp]`` BF16 mailbox, its Int32 [2] stage words,
    stream FP32 [b * H] (in place), alpha and beta FP32 [H], the weights BF16
    [H] (post_weight unused without a post weight), out BF16 [b * H] and aux
    FP32 [b * H] (unused without ``aux``).
    """
    weight = make_fake_compact_tensor(BFloat16, (HIDDEN,), assumed_align=16)
    gate = make_fake_compact_tensor(Float32, (HIDDEN,), assumed_align=32)
    rows_f32 = make_fake_dynamic_compact_tensor(
        Float32, alignment=32, divisibility=HIDDEN
    )
    rows_bf16 = make_fake_dynamic_compact_tensor(
        BFloat16, alignment=16, divisibility=HIDDEN
    )
    elements = LAMPORT_GENERATIONS * capacity * HIDDEN
    return cute.compile(
        _AllGatherDualNormKernel(
            tp=tp,
            capacity=capacity,
            constants=constants,
            tree=tree,
            aux=aux,
            enable_pdl=enable_pdl,
            early_trigger=early_trigger,
            timeout_ns=timeout_ns,
        ),
        make_fake_compact_tensor(BFloat16, (elements,), assumed_align=16),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        rows_f32,
        gate,
        gate,
        weight,
        weight,
        rows_bf16,
        rows_f32,
        Int32(capacity),
        current_cu_stream(),
    )


# The consumer takes the mailbox and the weights as arguments, so the boundaries
# with the same constants share one compiled kernel.
_cached_consumer = functools.cache(compile_all_gather_dual_norm)


def _vector(tensor: torch.Tensor, dtype: torch.dtype, name: str) -> torch.Tensor:
    if tensor.shape != (HIDDEN,) or tensor.dtype != dtype or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous [{HIDDEN}] {dtype} tensor")
    return tensor


def _rows(tensor: torch.Tensor, rows: int, dtype: torch.dtype, name: str) -> None:
    if (
        tensor.shape != (rows, HIDDEN)
        or tensor.dtype != dtype
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be a contiguous [{rows}, {HIDDEN}] {dtype}")


class AllGatherDualNorm:
    """The consumer of one boundary with its constants and weights bound and its
    kernel compiled (build time). ``mailbox`` is a ``BoundaryMailbox`` (or any
    object with its ``tp``, ``capacity``, ``tensor`` and ``stage_state``).

    ``run`` is graph-capturable and allocates nothing; every rank of the group
    must call it once per publish, with the same row count.
    """

    def __init__(
        self,
        mailbox,
        constants: DualNormConstants,
        *,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        post_weight: torch.Tensor | None,
        next_weight: torch.Tensor,
        tree: bool,
        aux: bool,
        enable_pdl: bool = True,
        early_trigger: bool = True,
        timeout_ns: int = SPIN_TIMEOUT_NS,
    ) -> None:
        assert_not_capturing("AllGatherDualNorm")
        if (post_weight is None) != (constants.post_gain_center is None):
            raise ValueError("post_weight must be given exactly with a post gain")
        self.capacity, self.tree, self.aux = mailbox.capacity, tree, aux
        self._compiled = _cached_consumer(
            tp=mailbox.tp,
            capacity=mailbox.capacity,
            constants=constants,
            tree=tree,
            aux=aux,
            enable_pdl=enable_pdl,
            early_trigger=early_trigger,
            timeout_ns=timeout_ns,
        )
        next_weight = _vector(next_weight, torch.bfloat16, "next_weight")
        self._mailbox = to_cute(mailbox.tensor.flatten(), 16)
        self._stage_state = to_cute(mailbox.stage_state, 4)
        self._alpha = to_cute(_vector(alpha, torch.float32, "alpha"), 32)
        self._beta = to_cute(_vector(beta, torch.float32, "beta"), 32)
        self._next_weight = to_cute(next_weight, 16)
        self._post_weight = (
            to_cute(_vector(post_weight, torch.bfloat16, "post_weight"), 16)
            if post_weight is not None
            else self._next_weight
        )

    def run(
        self,
        stream: torch.Tensor,
        out: torch.Tensor,
        aux: torch.Tensor | None = None,
    ) -> None:
        rows = stream.shape[0]
        if rows > self.capacity:
            raise ValueError(f"at most {self.capacity} rows")
        _rows(stream, rows, torch.float32, "stream")
        _rows(out, rows, torch.bfloat16, "out")
        if (aux is not None) != self.aux:
            raise ValueError(f"aux is {'required' if self.aux else 'not built'}")
        if aux is not None:
            _rows(aux, rows, torch.float32, "aux")
        if rows == 0:
            return
        stream_rows = to_cute_dynamic(stream.flatten(), 32, divisibility=HIDDEN)
        self._compiled(
            self._mailbox,
            self._stage_state,
            stream_rows,
            self._alpha,
            self._beta,
            self._post_weight,
            self._next_weight,
            to_cute_dynamic(out.flatten(), 16, divisibility=HIDDEN),
            (
                to_cute_dynamic(aux.flatten(), 32, divisibility=HIDDEN)
                if aux is not None
                else stream_rows
            ),
            Int32(rows),
            current_cu_stream(),
        )


class BoundaryOperands(NamedTuple):
    """What one stock boundary hands the dual norm."""

    constants: DualNormConstants
    alpha: torch.Tensor
    beta: torch.Tensor
    post_weight: torch.Tensor | None
    next_weight: torch.Tensor


def boundary_operands(post_norm, connection, next_norm) -> BoundaryOperands:
    """Constants and weights of the stock boundary ``post_norm`` (the previous
    layer's post_ffn_norm) -> ``connection`` (its add_ffn) -> ``next_norm``
    (this layer's pre_qkv_proj_norm, or model.norm), under the stock fused
    path's preconditions."""
    if (
        getattr(post_norm, "variance_size_override", None) is not None
        or getattr(next_norm, "variance_size_override", None) is not None
    ):
        raise ValueError("the boundary norms must reduce over the whole row")
    if next_norm.weight is None:
        raise ValueError("the next norm has no weight")
    constants = boundary_constants(post_norm, connection, next_norm)
    return BoundaryOperands(
        constants=constants,
        alpha=connection.alpha,
        beta=connection.beta,
        post_weight=post_norm.weight if post_norm.has_weight else None,
        next_weight=next_norm.weight,
    )


def stock_gathers(previous_layer: nn.Module, aux: bool, rows: int) -> bool:
    """Whether the stock forward runs this boundary as the fused all-gather
    (serial reduction) at ``rows`` tokens: the previous layer defers its
    unpooler's column gather and this layer is not an aux layer (the stock
    forward gathers the columns before an aux state and then runs the
    standalone dual norm, tree reduction)."""
    return not aux and bool(previous_layer._defers_column_gather(rows))


class GatherBoundary:
    """The input boundary (a ``BoundaryProvider``) of decoder layer
    ``layer_idx >= 1``: gathers the previous layer's unpooler blocks from this
    boundary's mailbox and runs the dual norm into the layer's
    pre_qkv_proj_norm."""

    def __init__(
        self, index: int, mailbox: BoundaryMailbox, kernel: AllGatherDualNorm
    ) -> None:
        self.index, self.mailbox, self.kernel = index, mailbox, kernel

    @classmethod
    def unsupported(cls, vllm_config) -> str | None:
        # NVLink multicast is checked where the mailbox is built, on every rank.
        return None

    @classmethod
    def build(cls, ctx, layer: nn.Module | None) -> "GatherBoundary":
        assert layer is not None and layer.layer_idx > ctx.backbone.start_layer
        previous = ctx.backbone.layers[layer.layer_idx - 1]
        next_norm = layer.self_attn.pre_qkv_proj_norm
        return cls._build(ctx, previous, next_norm, layer.layer_idx)

    @classmethod
    def _build(cls, ctx, previous: nn.Module, next_norm, index: int):
        if ctx.backbone.default_dtype != torch.bfloat16:
            raise ValueError("the boundary writes BF16 activations")
        aux = index in ctx.aux_layers
        operands = boundary_operands(
            previous.post_ffn_norm, previous.add_ffn, next_norm
        )
        mailbox = BoundaryMailbox(
            ctx.group, tp=ctx.tp, rank=ctx.rank, device=ctx.device, layer=index
        )
        kernel = AllGatherDualNorm(
            mailbox,
            operands.constants,
            alpha=operands.alpha,
            beta=operands.beta,
            post_weight=operands.post_weight,
            next_weight=operands.next_weight,
            tree=not stock_gathers(previous, aux, WARM_UP_ROWS),
            aux=aux,
        )
        provider = cls(index, mailbox, kernel)
        # Every rank compiles before any rank publishes: the warm-up's poll
        # traps after SPIN_TIMEOUT_NS, which a peer's compile could exceed.
        dist.barrier(group=ctx.group)
        provider.warm_up(ctx.device)
        return provider

    def warm_up(self, device: torch.device) -> None:
        """One publish and gather of zeros at the decode batch, on every rank,
        so no module load happens under capture."""
        rows = WARM_UP_ROWS
        cols = torch.zeros(
            rows, self.mailbox.width, dtype=torch.bfloat16, device=device
        )
        stream = torch.zeros(rows, HIDDEN, dtype=torch.float32, device=device)
        out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)
        aux = torch.empty_like(stream) if self.kernel.aux else None
        slot = self.mailbox.slot(self.index, 0)
        self.mailbox.publish(cols, slot)
        self.run(slot, stream, out, aux)
        torch.accelerator.synchronize(device)

    def prepare_kv(self) -> None:
        pass

    def run(
        self,
        slot: MailboxSlot,
        stream: torch.Tensor,
        out: torch.Tensor,
        aux: torch.Tensor | None,
    ) -> None:
        self.mailbox.check(slot)
        self.kernel.run(stream, out, aux)


class FinalBoundary(GatherBoundary):
    """The final boundary, after the last layer, into model.norm."""

    @classmethod
    def build(cls, ctx, layer: nn.Module | None) -> "FinalBoundary":
        backbone = ctx.backbone
        end = backbone.end_layer
        return cls._build(ctx, backbone.layers[end - 1], backbone.norm, end)
