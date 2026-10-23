# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A TP all-gather over NVLink multicast with a Lamport mailbox, for the
specialized decode.

The specialized decode gathers small tensors across the TP group on its
critical path: each dense indexer layer of a row-sharded decode step gathers
its top-k rows (and the candidate source its candidate blocks), and every step
gathers the vocab-parallel logits. A few hundred kilobytes at most, so the
latency of the collective, not its bandwidth, is what counts.

vLLM's custom all-gather has a Lamport kernel for NVLink multicast
(``mnnvl_lamport_all_gather``), but it never waits for the rank's own
multicast copy: it takes its own slot from registers, and a copy that lands
after the next call has cleaned its stage stays there. A later gather of
another size lays the slots out differently and can read that word as a
peer's payload. A stress test of back-to-back gathers of alternating sizes in
CUDA graphs hits this hundreds of times in a few million gathers. This module
is a gather of the same shape that cannot leave a word behind.

The mailbox is symmetric memory with a multicast mapping, laid out as
16-byte packs::

    mailbox[generation][rank][pack]    3 generations x TP ranks x capacity

A call runs one kernel, one thread a 16-byte pack of the rank's input:

1. Publish: the thread stores its pack through the multicast address into
   every rank's ``mailbox[g][rank][pack]``. A word equal to the Lamport
   sentinel (``0x80000000``: FP32 negative zero, INT32_MIN) goes out as 0.
2. Collect: it polls ``mailbox[g][r][pack]`` of every rank ``r``, its own
   included, until no word of it is the sentinel; writes each rank's pack to
   the output, rank-major as NCCL lays out a dim-0 gather; and stores the
   sentinel back into the slots it read. Every slot a call writes is read
   after it has landed and reset by the same call, so no call leaves a word
   for a later one, whatever the sizes.
3. Generation: ``state[0]`` is the call's generation. Each CTA reads it once
   and counts itself in ``state[1]``; the last CTA to arrive advances the
   generation and clears the count. A rank can be at most one call ahead of
   a peer (finishing a call needs every rank's payload), so a rank writing
   the next generation never touches the slots a slower peer still reads;
   three generations, as vLLM's and FlashInfer's Lamport kernels use, leave
   one more in between.

The kernel triggers its PDL dependents once it has published. The output
holds the sanitized words on every rank, its own slot included, so all ranks
see the same bits. The caller must issue calls in the same order on every
rank and on one stream: calls on one instance must not overlap.
"""

from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.distributed import get_tp_group

from .primitives import current_cu_stream

THREADS = 128
PACK_BYTES = 16
GENERATIONS = 3
SENTINEL = 0x80000000
# state words
GENERATION = 0
ARRIVALS = 1


# The PTX helpers below follow the inline-asm wrappers of FlashInfer's
# cute_dsl_primitives.py (see the notice in primitives.py).
def _asm(result, operands, text, constraints, *, loc, ip):
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _load_u32x4(address: Int64, volatile: bool, *, loc=None, ip=None):
    """One 16-byte pack as four words; volatile for the Lamport polls."""
    opcode = "ld.volatile.global.v4.u32" if volatile else "ld.global.v4.u32"
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        f"{opcode} {{$0, $1, $2, $3}}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip))
        for index in range(4)
    )


@dsl_user_op
def _store_u32x4(
    address: Int64,
    w0: Uint32,
    w1: Uint32,
    w2: Uint32,
    w3: Uint32,
    multicast: bool,
    *,
    loc=None,
    ip=None,
) -> None:
    """One 16-byte pack, through a multicast address (``multimem.st``, which
    moves the bits unchanged) or to a local one."""
    opcode = (
        "multimem.st.relaxed.sys.global.v4.f32" if multicast else "st.global.v4.u32"
    )
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip)]
        + [word.ir_value(loc=loc, ip=ip) for word in (w0, w1, w2, w3)],
        f"{opcode} [$0], {{$1, $2, $3, $4}};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _load_volatile_u32(address: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _store_u32(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.global.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _arrive(address: Int64, *, loc=None, ip=None) -> Uint32:
    """Add 1 to the arrival count (acquire-release, GPU scope); returns the
    count before."""
    return Uint32(
        _asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip), Uint32(1).ir_value(loc=loc, ip=ip)],
            "atom.add.acq_rel.gpu.global.u32 $0, [$1], $2;",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


class _AllGatherKernel:
    """One thread a 16-byte pack of the rank's input, ``THREADS`` a CTA."""

    def __init__(self, rank: int, world_size: int, rank_packs: int) -> None:
        self.rank = rank
        self.world_size = world_size
        self.rank_packs = rank_packs

    @cute.jit
    def __call__(
        self,
        inp: Int64,
        out: Int64,
        mailbox: Int64,
        multicast: Int64,
        state: Int64,
        packs: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """Launch on ``stream``; addresses are byte addresses, ``packs`` the
        rank's input size in 16-byte packs."""
        self.kernel(inp, out, mailbox, multicast, state, packs).launch(
            grid=((packs + THREADS - 1) // THREADS, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        inp: Int64,
        out: Int64,
        mailbox: Int64,
        multicast: Int64,
        state: Int64,
        packs: Int32,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        grid, _, _ = cute.arch.grid_dim()
        pack = block * THREADS + tidx
        active = pack < packs
        smem = utils.SmemAllocator()
        shared_generation = smem.allocate_tensor(Uint32, cute.make_layout(1), 4)

        cute.arch.griddepcontrol_wait()

        # One read of the generation a CTA, before the CTA counts itself: the
        # last CTA to arrive can then advance it while the others still run.
        if tidx == 0:
            current = _load_volatile_u32(state + GENERATION * 4)
            shared_generation[0] = current
            if _arrive(state + ARRIVALS * 4) == Uint32(grid) - Uint32(1):
                _store_u32(
                    state + GENERATION * 4,
                    (current + Uint32(1)) % Uint32(GENERATIONS),
                )
                _store_u32(state + ARRIVALS * 4, Uint32(0))
        cute.arch.barrier()
        generation = shared_generation[0]
        # Pack offset of rank 0's slot of this pack in this generation.
        base = Int64(generation) * Int64(self.world_size * self.rank_packs) + Int64(
            pack
        )

        if active:
            w0, w1, w2, w3 = _load_u32x4(inp + Int64(pack) * PACK_BYTES, False)
            if w0 == Uint32(SENTINEL):
                w0 = Uint32(0)
            if w1 == Uint32(SENTINEL):
                w1 = Uint32(0)
            if w2 == Uint32(SENTINEL):
                w2 = Uint32(0)
            if w3 == Uint32(SENTINEL):
                w3 = Uint32(0)
            slot = base + Int64(self.rank * self.rank_packs)
            _store_u32x4(multicast + slot * PACK_BYTES, w0, w1, w2, w3, True)
        cute.arch.griddepcontrol_launch_dependents()

        # Every rank's slot, this rank's own included: its multicast copy must
        # land before the slot is reset, or it would outlive the reset.
        values = cute.make_rmem_tensor(cute.make_layout((self.world_size, 4)), Uint32)
        values.fill(Uint32(0))
        dirty = active
        while dirty:
            dirty = False
            for source in cutlass.range_constexpr(self.world_size):
                slot = base + Int64(source * self.rank_packs)
                p0, p1, p2, p3 = _load_u32x4(mailbox + slot * PACK_BYTES, True)
                dirty = (
                    dirty
                    | (p0 == Uint32(SENTINEL))
                    | (p1 == Uint32(SENTINEL))
                    | (p2 == Uint32(SENTINEL))
                    | (p3 == Uint32(SENTINEL))
                )
                values[source, 0] = p0
                values[source, 1] = p1
                values[source, 2] = p2
                values[source, 3] = p3
        if active:
            sentinel = Uint32(SENTINEL)
            for source in cutlass.range_constexpr(self.world_size):
                position = Int64(source) * Int64(packs) + Int64(pack)
                _store_u32x4(
                    out + position * PACK_BYTES,
                    values[source, 0],
                    values[source, 1],
                    values[source, 2],
                    values[source, 3],
                    False,
                )
                slot = base + Int64(source * self.rank_packs)
                _store_u32x4(
                    mailbox + slot * PACK_BYTES,
                    sentinel,
                    sentinel,
                    sentinel,
                    sentinel,
                    False,
                )


class LamportAllGather:
    """TP all-gather along dim 0 over NVLink multicast, for inputs of up to
    ``rank_capacity_bytes`` a rank (a multiple of 16 bytes).

    Construction is collective over the TP group: every rank compiles the
    kernel and allocates its mailbox, and it raises when NVLink multicast is
    unavailable. ``accepts`` tells whether a tensor fits, from its size
    alone, so every rank decides alike.
    """

    def __init__(self, *, rank_capacity_bytes: int, device: torch.device) -> None:
        if rank_capacity_bytes <= 0 or rank_capacity_bytes % PACK_BYTES:
            raise ValueError("the capacity must be a positive multiple of 16 bytes")
        group = get_tp_group().device_group
        self.world_size = world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        self.rank_capacity_bytes = rank_capacity_bytes
        rank_packs = rank_capacity_bytes // PACK_BYTES
        device = torch.device(device)
        with torch.accelerator.device_index(device.index):
            # cute.compile returns an untyped JIT executor.
            self._kernel: Any = cute.compile(
                _AllGatherKernel(rank, world_size, rank_packs),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int64(0),
                Int32(1),
                make_fake_stream(),
            )
            self._mailbox = symm_mem.empty(
                (GENERATIONS * world_size * rank_packs * PACK_BYTES // 4,),
                dtype=torch.int32,
                device=device,
            )
            self._mailbox_handle = symm_mem.rendezvous(self._mailbox, group)
            multicast = int(self._mailbox_handle.multicast_ptr or 0)
            if not multicast or multicast % PACK_BYTES:
                raise RuntimeError("NVLink multicast mapping is unavailable")
            self._multicast = multicast
            # Every slot starts as the sentinel.
            self._mailbox.fill_(-(2**31))
            self._state = torch.zeros(2, dtype=torch.int32, device=device)
            torch.accelerator.synchronize()
        dist.barrier(group=group)

    def accepts(self, inp: torch.Tensor) -> bool:
        """Whether ``inp`` fits: 1 to ``rank_capacity_bytes`` bytes, a
        multiple of 16."""
        return 0 < inp.nbytes <= self.rank_capacity_bytes and not inp.nbytes % 16

    def __call__(self, inp: torch.Tensor) -> torch.Tensor:
        """Every rank's ``inp`` concatenated along dim 0 in rank order, on the
        current stream. Words equal to ``0x80000000`` arrive as 0, on every
        rank (FP32 negative zero becomes positive zero)."""
        if not self.accepts(inp):
            raise ValueError(
                f"the all-gather takes 16-byte multiples of up to "
                f"{self.rank_capacity_bytes} bytes a rank, not {inp.nbytes}"
            )
        if not inp.is_contiguous() or inp.data_ptr() % PACK_BYTES:
            # The kernel loads 16-byte packs.
            inp = inp.clone(memory_format=torch.contiguous_format)
        out = torch.empty(
            (inp.shape[0] * self.world_size, *inp.shape[1:]),
            dtype=inp.dtype,
            device=inp.device,
        )
        self._kernel(
            Int64(inp.data_ptr()),
            Int64(out.data_ptr()),
            Int64(self._mailbox.data_ptr()),
            Int64(self._multicast),
            Int64(self._state.data_ptr()),
            Int32(inp.nbytes // PACK_BYTES),
            current_cu_stream(),
        )
        return out
