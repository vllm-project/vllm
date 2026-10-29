# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
# The shared-only publish kernel, the mailbox allocation and the owned-fragment
# poll are adapted from FlashInfer (flashinfer/comm/mnnvl_cutedsl/kernel_ll/
# device_kernels.py, protocol.py and column_gather.py,
# flashinfer-ai/flashinfer@14ba7aaa84718abe7a5648184fa9ad15362e2880,
# Apache-2.0); the spins are bounded here.
# FlashInfer NOTICE: "FlashInfer / Copyright 2025-2026 NVIDIA / Copyright
# 2023-2026 FlashInfer community (https://flashinfer.ai/)".

"""Lamport mailboxes in symmetric memory.

A mailbox is ``[LAMPORT_GENERATIONS, tp, capacity, width]`` BF16 on every rank
of a TP group, mapped for NVLink multicast. A publisher multicasts its
``[m, width]`` rows into slot ``[stage, rank]`` of every rank's copy; readers
poll their own copy in 16-byte fragments until no fragment holds BF16 negative
zero (the sentinel; publishers store -0.0 as +0.0), then re-arm what they read.
A publisher can also store rows into one rank's copy alone through ``peers``
(the TP tail's route rows); ``poll_at`` reads slots at any fixed offsets.

Every spin is bounded: a poll that sees no data for ``SPIN_TIMEOUT_NS`` traps,
so a rank that diverges fails the launch instead of hanging the group.
"""

import functools
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_compact_tensor

from .primitives import (
    ACTIVE_STAGE,
    LAMPORT_GENERATIONS,
    NEGATIVE_ZERO_BF16_BITS,
    NEXT_STAGE,
    QUAD_BF16,
    VEC_BF16,
    current_cu_stream,
    f32_to_bf16_bits,
    fragment_has_negative_zero,
    global_timer_ns,
    load_global_bf16_as_f32,
    load_global_u32x2,
    load_global_u32x4,
    load_volatile_u32,
    make_fake_dynamic_compact_tensor,
    sanitize_negative_zero_u32x2,
    sanitize_negative_zero_u32x4,
    shuffle_sync_idx_u32,
    stmc_bf16x2,
    stmc_bf16x4,
    stmc_bf16x8,
    store_global_u32,
    store_lamport_sentinel_u32x4,
    to_cute,
    to_cute_dynamic,
    trap,
)

# A poll that has seen no data for this long traps. Every rank launches the
# same step within milliseconds; the bound only turns a divergence into an error.
SPIN_TIMEOUT_NS = 60_000_000_000


class LamportMailbox:
    """One TP group's ``[3, tp, capacity, width]`` BF16 mailbox and stage state.

    Construction is collective over ``group`` and raises on every rank when the
    buffer has no multicast mapping. ``peers`` are the ranks' buffer addresses
    in rank order and ``multicast`` the multicast address of the whole buffer.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        tp: int,
        rank: int,
        capacity: int,
        width: int,
        device: torch.device,
    ) -> None:
        if dist.get_world_size(group) != tp or dist.get_rank(group) != rank:
            raise ValueError("process group does not match tp and rank")
        if width % VEC_BF16:
            raise ValueError("width must be a multiple of 8 BF16 elements")
        self.tp, self.rank, self.capacity, self.width = tp, rank, capacity, width
        self.tensor = symm_mem.empty(
            (LAMPORT_GENERATIONS, tp, capacity, width),
            dtype=torch.bfloat16,
            device=device,
        )
        handle = symm_mem.rendezvous(self.tensor, group)
        self.multicast = int(handle.multicast_ptr or 0)
        if not self.multicast or self.multicast % 16:
            raise RuntimeError(
                "Muse Spark specialized decode needs NVLink multicast memory"
            )
        self.peers = tuple(int(pointer) for pointer in handle.buffer_ptrs)
        # Every slot starts as the sentinel, BF16 negative zero.
        self.tensor.view(torch.int16).fill_(-32768)
        self.stage_state = torch.zeros(2, dtype=torch.int32, device=device)
        torch.accelerator.synchronize(device)
        dist.barrier(group=group)
        self.view = to_cute(self.tensor.flatten(), 16)
        self.stage_view = to_cute(self.stage_state, 4)

    def compile_publish(
        self,
        *,
        threads: int = 128,
        release_before_store: bool = False,
        enable_pdl: bool = True,
    ):
        """The shared-only publisher of this mailbox (call before capture); one
        compile per launch shape, shared by every mailbox of that shape."""
        return _cached_publish(
            hidden=self.width,
            tp=self.tp,
            rank=self.rank,
            capacity_m=self.capacity,
            elements_per_thread=VEC_BF16,
            threads=threads,
            release_before_store=release_before_store,
            enable_pdl=enable_pdl,
        )

    def publish(self, publisher, rows: torch.Tensor) -> None:
        """Multicast ``rows`` [m, width] BF16 into slot [NEXT_STAGE, rank]."""
        publisher(
            to_cute_dynamic(rows.flatten(), 16, divisibility=self.width),
            self.stage_view,
            Int64(self.multicast),
            Int32(rows.shape[0]),
            current_cu_stream(),
        )


@cute.jit
def spin_deadline(deadline: Int64, timeout_ns: cutlass.Constexpr[int]) -> Int64:
    """Call once per poll that found no data: arms the deadline on the first
    call (zero means unarmed) and traps once it has passed."""
    now = global_timer_ns()
    if deadline == Int64(0):
        deadline = now + Int64(timeout_ns)
    elif now > deadline:
        trap()
    return deadline


@cute.jit
def rearm(pointer: cute.Pointer) -> None:
    """Store the sentinel over one 16-byte fragment."""
    store_lamport_sentinel_u32x4(Int64(pointer.toint()))


@cute.jit
def fetch(values: cute.Tensor, i: cutlass.Constexpr[int], pointer: cute.Pointer):
    """Load fragment ``i`` into ``values[i]``; returns its dirty flag at bit ``i``."""
    got = load_global_u32x4(pointer, volatile=True)
    for k in cutlass.range_constexpr(4):
        values[i, k] = got[k]
    dirty = Uint32(0)
    if fragment_has_negative_zero(got):
        dirty = Uint32(1 << i)
    return dirty


@cute.jit
def _poll(
    base: cute.Pointer,
    stride: Int64,
    offsets: cutlass.Constexpr[tuple],
    timeout_ns: cutlass.Constexpr[int],
):
    """Lamport-poll the 16-byte fragments at base + offsets[i] * stride
    (elements) into a [len(offsets), 4] u32 register tensor. All loads issue
    before any check and only dirty fragments re-poll; traps after
    ``timeout_ns`` without data."""
    count = len(offsets)
    values = cute.make_rmem_tensor(cute.make_layout((count, 4)), Uint32)
    pending = Uint32(0)
    for i in cutlass.range_constexpr(count):
        pending = pending | fetch(values, i, base + Int64(offsets[i]) * stride)
    deadline = Int64(0)
    while pending != Uint32(0):
        deadline = spin_deadline(deadline, timeout_ns)
        again = Uint32(0)
        for i in cutlass.range_constexpr(count):
            if ((pending >> Uint32(i)) & Uint32(1)) != Uint32(0):
                again = again | fetch(values, i, base + Int64(offsets[i]) * stride)
        pending = again
    return values


def poll(
    base: cute.Pointer,
    stride: Int64,
    count: int,
    timeout_ns: int = SPIN_TIMEOUT_NS,
):
    """``count`` fragments at base + i * stride (elements, ``stride`` at run
    time), polled as ``_poll`` does: a [count, 4] u32 register tensor."""
    return _poll(base, stride, tuple(range(count)), timeout_ns)


def poll_at(
    base: cute.Pointer,
    offsets: tuple,
    timeout_ns: int = SPIN_TIMEOUT_NS,
):
    """The fragments at base + offsets[i] (elements, compile-time), polled as
    ``_poll`` does: a [len(offsets), 4] u32 register tensor."""
    return _poll(base, Int64(1), offsets, timeout_ns)


@cute.jit
def poll_owned_fragment(
    mailbox: cute.Tensor,
    stage_state: cute.Tensor,
    packed: cute.Tensor,
    token: Int32,
    fragment: Int32,
    tp: cutlass.Constexpr[int],
    capacity_m: cutlass.Constexpr[int],
    hidden_local: cutlass.Constexpr[int],
    release_dependents: cutlass.Constexpr[bool],
    timeout_ns: cutlass.Constexpr[int] = SPIN_TIMEOUT_NS,
) -> None:
    """Column all-gather read: wait for this token's 8 global columns
    ``fragment`` from the one rank that owns them, store the BF16x8 fragment
    into ``packed`` and re-arm the slot; one thread per launch advances the
    stage. Releases PDL dependents after the poll when ``release_dependents``
    (a kernel that triggers earlier passes False)."""
    local_fragments = hidden_local // VEC_BF16
    active_stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
    source_rank = fragment // local_fragments
    local_fragment = fragment % local_fragments
    source_element = (
        (Int64(active_stage) * tp + Int64(source_rank)) * capacity_m + Int64(token)
    ) * hidden_local + Int64(local_fragment) * VEC_BF16
    source_pointer = cute.make_ptr(
        BFloat16,
        (mailbox.iterator + source_element).llvm_ptr,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    deadline = Int64(0)
    # Always true; a runtime value keeps the poll a device loop.
    dirty = fragment >= Int32(0)
    while dirty:
        value = load_global_u32x4(source_pointer, volatile=True)
        dirty = fragment_has_negative_zero(value)
        packed.store(value)
        if dirty:
            deadline = spin_deadline(deadline, timeout_ns)

    if token == 0 and fragment == 0:
        store_global_u32(
            stage_state.iterator + NEXT_STAGE,
            (active_stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
        )
    if cutlass.const_expr(release_dependents):
        cute.arch.griddepcontrol_launch_dependents()
    store_lamport_sentinel_u32x4(Int64(source_pointer.toint()))


class _SharedOnlyPublishDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        tp: int,
        rank: int,
        capacity_m: int,
        elements_per_thread: int,
        threads: int,
        release_before_store: bool,
        enable_pdl: bool,
    ) -> None:
        if elements_per_thread not in (1, QUAD_BF16, VEC_BF16):
            raise ValueError("elements_per_thread must be 1, 4, or 8")
        if hidden <= 0 or hidden % elements_per_thread:
            raise ValueError("hidden must divide evenly across thread fragments")
        self.hidden = hidden
        self.tp = tp
        self.rank = rank
        self.capacity_m = capacity_m
        self.elements_per_thread = elements_per_thread
        self.threads = threads
        self.fragments = hidden // elements_per_thread
        self.ctas_per_token = math.ceil(self.fragments / threads)
        self.release_before_store = release_before_store
        self.enable_pdl = enable_pdl

    @cute.jit
    def __call__(
        self,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            shared_output,
            stage_state,
            contribution_mailbox_multicast_address,
        ).launch(
            grid=(m * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        token = block // self.ctas_per_token
        cta_in_token = block % self.ctas_per_token
        fragment = cta_in_token * self.threads + tidx

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        if cutlass.const_expr(self.elements_per_thread == 1):
            bits = Uint32(0)
            if fragment < self.fragments:
                element = Int64(token) * self.hidden + Int64(fragment)
                bits = f32_to_bf16_bits(
                    load_global_bf16_as_f32(
                        Int64((shared_output.iterator + element).toint())
                    )
                )
                if bits == Uint32(NEGATIVE_ZERO_BF16_BITS):
                    bits = Uint32(0)
            partner_bits = shuffle_sync_idx_u32(
                bits,
                Int32(cute.arch.lane_idx() | Int32(1)),
            )
            packed_word = bits | (partner_bits << Uint32(16))
        elif cutlass.const_expr(self.elements_per_thread == QUAD_BF16):
            packed_words = cute.make_rmem_tensor(cute.make_layout((2,)), Uint32)
            packed_words.fill(Uint32(0))
            if fragment < self.fragments:
                element = Int64(token) * self.hidden + Int64(fragment) * QUAD_BF16
                pointer = cute.make_ptr(
                    BFloat16,
                    (shared_output.iterator + element).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=8,
                )
                packed_words.store(
                    sanitize_negative_zero_u32x2(load_global_u32x2(pointer))
                )
        else:
            packed_words = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
            packed_words.fill(Uint32(0))
            if fragment < self.fragments:
                element = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
                pointer = cute.make_ptr(
                    BFloat16,
                    (shared_output.iterator + element).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                packed_words.store(
                    sanitize_negative_zero_u32x4(load_global_u32x4(pointer))
                )

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        if cutlass.const_expr(self.release_before_store):
            if block == 0 and tidx == 0:
                store_global_u32(
                    stage_state.iterator + ACTIVE_STAGE,
                    stage,
                )
            cute.arch.barrier()
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()

        if cutlass.const_expr(self.elements_per_thread == 1):
            if (cute.arch.lane_idx() & Int32(1)) == Int32(
                0
            ) and fragment < self.fragments:
                mailbox_element = (
                    (Int64(stage) * self.tp + self.rank) * self.capacity_m
                    + Int64(token)
                ) * self.hidden + Int64(fragment)
                stmc_bf16x2(
                    contribution_mailbox_multicast_address + mailbox_element * 2,
                    packed_word,
                )
        elif cutlass.const_expr(self.elements_per_thread == QUAD_BF16):
            if fragment < self.fragments:
                mailbox_element = (
                    (Int64(stage) * self.tp + self.rank) * self.capacity_m
                    + Int64(token)
                ) * self.hidden + Int64(fragment) * QUAD_BF16
                stmc_bf16x4(
                    contribution_mailbox_multicast_address + mailbox_element * 2,
                    packed_words,
                )
        else:
            if fragment < self.fragments:
                mailbox_element = (
                    (Int64(stage) * self.tp + self.rank) * self.capacity_m
                    + Int64(token)
                ) * self.hidden + Int64(fragment) * VEC_BF16
                stmc_bf16x8(
                    contribution_mailbox_multicast_address + mailbox_element * 2,
                    packed_words,
                )

        if cutlass.const_expr(not self.release_before_store):
            if block == 0 and tidx == 0:
                store_global_u32(
                    stage_state.iterator + ACTIVE_STAGE,
                    stage,
                )
            cute.arch.barrier()
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()


def compile_shared_only_publish(
    *,
    hidden: int,
    tp: int,
    rank: int,
    capacity_m: int,
    elements_per_thread: int,
    threads: int,
    release_before_store: bool,
    enable_pdl: bool,
):
    """Compile the publisher that multicasts a local ``[m, hidden]`` BF16 tensor
    into slot [NEXT_STAGE, rank] and then sets ACTIVE_STAGE."""
    device_kernel = _SharedOnlyPublishDeviceKernel(
        hidden=hidden,
        tp=tp,
        rank=rank,
        capacity_m=capacity_m,
        elements_per_thread=elements_per_thread,
        threads=threads,
        release_before_store=release_before_store,
        enable_pdl=enable_pdl,
    )
    return cute.compile(
        device_kernel,
        make_fake_dynamic_compact_tensor(BFloat16, alignment=16, divisibility=hidden),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        Int64(0),
        Int32(capacity_m),
        current_cu_stream(),
    )


# The publisher takes the mailbox (stage words, multicast address) as arguments,
# so every mailbox of one shape shares one compiled kernel.
_cached_publish = functools.cache(compile_shared_only_publish)
