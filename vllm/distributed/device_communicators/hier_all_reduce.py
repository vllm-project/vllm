# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Island-aware hierarchical allreduce for multi-island PCIe topologies.

Targets boxes like 2x4 A100/A800 PCIe (two PIX islands bridged by the CPU
interconnect), where the NCCL 8-GPU ring pays the cross-socket latency on
every hop. Strategy per rank, all inside one Triton kernel launch:

  1. publish: copy the local input into this rank's IPC-shared slot,
     release-fence, bump the phase-A flag.
  2. island reduce: spin on the island peers' phase-A flags, then read the
     island's slots directly over P2P and reduce -> island partial, publish
     to the partial slot with a phase-B flag.
  3. cross exchange: spin on the counterpart rank's phase-B flag (the rank
     with the same island-local index in the other island), read its partial
     over P2P, add, write the final result. Cross-socket traffic is exactly
     one message per rank instead of the ring's repeated crossings.

Latency-bound small messages only (decode hidden states); large payloads
should stay on NCCL. Buffers are IPC-registered once; flags use monotonically
increasing sequence tokens so no zeroing is needed between calls.
"""

import ctypes
import glob
import os
from collections.abc import Sequence

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_MAX_ELEMS = 256 * 1024  # 512KB bf16 cap; decode messages are ~8-512KB
_MAX_CTA = 16
_NUM_WARPS = 8
# One-shot moves 4n remote bytes in 2 sync rounds, two-shot 7n/4 in 3 and
# crosses the slow inter-island link with only n/island_size. Latency wins
# below this size, bandwidth above it; crossover measured on 2x4 A800 PCIe
# (32KB: 45.6 vs 47.3us one-shot favoured, 48KB: 54.3 vs 49.6 two-shot).
_TWO_SHOT_MIN_ELEMS = 24 * 1024


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


_cudart = None


def _find_cudart() -> str:
    """Locate libcudart across install layouts: the CUDA runtime may come
    from the pip wheel, from torch's bundled libs, or from a system CUDA
    install (as in most container images)."""
    # If torch already loaded it, reuse exactly that library.
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                path = line.split()[-1] if " /" in line else ""
                if os.path.basename(path).startswith("libcudart.so"):
                    return path
    except OSError:
        pass
    tdir = os.path.dirname(torch.__file__)
    for pattern in (
        os.path.join(
            os.path.dirname(tdir), "nvidia", "cuda_runtime", "lib", "libcudart.so*"
        ),
        os.path.join(tdir, "lib", "libcudart*.so*"),
    ):
        found = glob.glob(pattern)
        if found:
            return found[0]
    from ctypes.util import find_library

    name = find_library("cudart")
    if name:
        return name
    raise RuntimeError(
        "hierarchical allreduce could not locate libcudart; it needs the CUDA "
        "runtime to map peer buffers via cudaIpcOpenMemHandle"
    )


def _get_cudart():
    """The shared buffers are cudaMalloc'd and IPC-opened via ctypes: torch
    opens IPC handles inside the OWNER's device context, so the mapping is
    never made peer-accessible to the importing device's kernels."""
    global _cudart
    if _cudart is None:
        lib = ctypes.CDLL(_find_cudart())
        lib.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        lib.cudaMemset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        lib.cudaIpcGetMemHandle.argtypes = [
            ctypes.POINTER(_IpcHandle),
            ctypes.c_void_p,
        ]
        lib.cudaIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            _IpcHandle,
            ctypes.c_uint,
        ]
        lib.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        lib.cudaFree.argtypes = [ctypes.c_void_p]
        _cudart = lib
    return _cudart


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed: cudaError {rc}")


@triton.jit
def _fence_sys(FENCE_LANES: tl.constexpr):
    """System-scope acq_rel fence, executed by every thread in the CTA.

    PCIe has no native remote atomics, so signalling is a plain remote store
    bracketed by this fence. Applying the asm to a tensor rather than a
    scalar is what makes every thread issue it: a fence executed by one
    thread does not order another thread's stores, so a scalar (single-lane)
    fence would leave the other lanes' data unordered against the flag.
    """
    tl.inline_asm_elementwise(
        "fence.acq_rel.sys; mov.u32 $0, $1;",
        "=r,r",
        [tl.zeros((FENCE_LANES,), dtype=tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _hier_all_reduce_kernel(
    inp_ptr,
    out_ptr,
    ptrs_ptr,  # [world] int64 device pointers to each rank's data slot
    partial_ptrs_ptr,  # [world] int64 pointers to each rank's partial slot
    flag_ptrs_ptr,  # [world] int64 pointers to each rank's flag array
    rank: tl.constexpr,
    island_base: tl.constexpr,  # first rank of this island
    island_size: tl.constexpr,
    counterpart: tl.constexpr,  # same-index rank in the other island
    WORLD: tl.constexpr,
    numel,
    token_ptr,  # [MAX_CTA] device int32 sequence counters (cudagraph-safe)
    MAX_ELEMS: tl.constexpr,
    FENCE_LANES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Each CTA owns a disjoint chunk with its own flag row and sequence
    # counter; chunks synchronize independently, so multiple CTAs keep
    # multiple PCIe read streams in flight (single-CTA queue depth is the
    # bandwidth bottleneck at >=32KB).
    pid = tl.program_id(0)
    ncta = tl.num_programs(0)
    token = tl.atomic_add(token_ptr + pid, 1, sem="relaxed") + 1
    chunk = tl.cdiv(numel, ncta)
    start = pid * chunk
    end = tl.minimum(numel, start + chunk)
    fbase = pid * 2 * WORLD

    # Double-buffer data/partial slots by token parity: a rank can lap a
    # slow peer by one call, but launch-granularity stream ordering plus
    # the flag waits make a two-call lap (same-parity reuse) impossible
    # before the peer's read.
    buf_off = (token % 2) * MAX_ELEMS
    my_slot = tl.cast(tl.load(ptrs_ptr + rank), tl.pointer_type(tl.bfloat16)) + buf_off
    my_partial = (
        tl.cast(tl.load(partial_ptrs_ptr + rank), tl.pointer_type(tl.bfloat16))
        + buf_off
    )
    my_flags = tl.cast(tl.load(flag_ptrs_ptr + rank), tl.pointer_type(tl.int32))

    # Phase A: publish this chunk locally, then signal island peers.
    for off in range(start, end, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < end
        x = tl.load(inp_ptr + offs, mask=mask, other=0.0)
        tl.store(my_slot + offs, x, mask=mask)
    _fence_sys(FENCE_LANES)
    tl.debug_barrier()
    for i in tl.static_range(island_size):
        peer = island_base + i
        if peer != rank:
            peer_flags = tl.cast(
                tl.load(flag_ptrs_ptr + peer), tl.pointer_type(tl.int32)
            )
            tl.store(peer_flags + fbase + 0 * WORLD + rank, token)

    # Wait for the island peers' phase-A signals for this chunk.
    for i in tl.static_range(island_size):
        peer = island_base + i
        if peer != rank:
            while (
                tl.atomic_add(
                    my_flags + fbase + 0 * WORLD + peer,
                    0,
                    sem="acquire",
                    scope="sys",
                )
                < token
            ):
                pass
    tl.debug_barrier()
    _fence_sys(FENCE_LANES)

    # Phase B: island reduce over P2P reads, publish partial, signal the
    # cross-island counterpart. Partials are bf16 to halve cross-island
    # bytes; accumulation stays fp32.
    for off in range(start, end, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < end
        acc = tl.load(my_slot + offs, mask=mask, other=0.0).to(tl.float32)
        for i in tl.static_range(island_size):
            peer = island_base + i
            if peer != rank:
                peer_slot = (
                    tl.cast(
                        tl.load(ptrs_ptr + peer),
                        tl.pointer_type(tl.bfloat16),
                    )
                    + buf_off
                )
                acc += tl.load(peer_slot + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(my_partial + offs, acc.to(tl.bfloat16), mask=mask)
    _fence_sys(FENCE_LANES)
    tl.debug_barrier()
    cp_flags = tl.cast(tl.load(flag_ptrs_ptr + counterpart), tl.pointer_type(tl.int32))
    tl.store(cp_flags + fbase + 1 * WORLD + rank, token)

    # Phase C: wait for the counterpart's phase-B signal for this chunk,
    # then do the single cross-island exchange.
    while (
        tl.atomic_add(
            my_flags + fbase + 1 * WORLD + counterpart,
            0,
            sem="acquire",
            scope="sys",
        )
        < token
    ):
        pass
    tl.debug_barrier()
    _fence_sys(FENCE_LANES)
    cp_partial = (
        tl.cast(
            tl.load(partial_ptrs_ptr + counterpart),
            tl.pointer_type(tl.bfloat16),
        )
        + buf_off
    )
    for off in range(start, end, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < end
        acc = tl.load(my_partial + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.load(cp_partial + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=mask)


@triton.jit
def _hier_two_shot_kernel(
    inp_ptr,
    out_ptr,
    ptrs_ptr,  # [world] int64 pointers to each rank's data slot
    partial_ptrs_ptr,  # [world] pointers to each rank's island-partial slot
    gather_ptrs_ptr,  # [world] pointers to each rank's global-shard slot
    flag_ptrs_ptr,  # [world] pointers to each rank's flag array
    rank: tl.constexpr,
    island_base: tl.constexpr,
    island_idx: tl.constexpr,  # this rank's index inside its island
    island_size: tl.constexpr,
    counterpart: tl.constexpr,
    WORLD: tl.constexpr,
    numel,
    token_ptr,  # [MAX_CTA] device int32 sequence counters
    MAX_ELEMS: tl.constexpr,
    MAX_CTA: tl.constexpr,
    FENCE_LANES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Two-shot variant: reduce-scatter, one cross-island shard exchange,
    then allgather. Moves 7n/4 remote bytes against the one-shot kernel's
    4n, and only n/island_size of that crosses the slow inter-island link
    (vs n) -- the win grows with message size.

    Partitioning is a stripe: CTA k owns slice k of *every* shard, so a
    reader CTA always waits on the same CTA index of its peers and no
    cross-CTA barrier is needed.
    """
    pid = tl.program_id(0)
    ncta = tl.num_programs(0)
    token = tl.atomic_add(token_ptr + pid, 1, sem="relaxed") + 1
    buf_off = (token % 2) * MAX_ELEMS

    shard = tl.cdiv(numel, island_size)
    ss = tl.cdiv(shard, ncta)
    my_start = island_idx * shard + pid * ss
    my_end = tl.minimum(tl.minimum(island_idx * shard + shard, my_start + ss), numel)

    my_data = tl.cast(tl.load(ptrs_ptr + rank), tl.pointer_type(tl.bfloat16)) + buf_off
    my_partial = (
        tl.cast(tl.load(partial_ptrs_ptr + rank), tl.pointer_type(tl.bfloat16))
        + buf_off
    )
    my_gather = (
        tl.cast(tl.load(gather_ptrs_ptr + rank), tl.pointer_type(tl.bfloat16)) + buf_off
    )
    my_flags = tl.cast(tl.load(flag_ptrs_ptr + rank), tl.pointer_type(tl.int32))

    # Phase A: publish this CTA's stripe of every shard -- exactly the bytes
    # each island peer reads for the shard it owns.
    for s in tl.static_range(island_size):
        start = s * shard + pid * ss
        end = tl.minimum(tl.minimum(s * shard + shard, start + ss), numel)
        for off in range(start, end, BLOCK):
            offs = off + tl.arange(0, BLOCK)
            mask = offs < end
            x = tl.load(inp_ptr + offs, mask=mask, other=0.0)
            tl.store(my_data + offs, x, mask=mask)
    _fence_sys(FENCE_LANES)
    tl.debug_barrier()
    for i in tl.static_range(island_size):
        peer = island_base + i
        if peer != rank:
            pf = tl.cast(tl.load(flag_ptrs_ptr + peer), tl.pointer_type(tl.int32))
            tl.store(pf + (0 * WORLD + rank) * MAX_CTA + pid, token)

    for i in tl.static_range(island_size):
        peer = island_base + i
        if peer != rank:
            while (
                tl.atomic_add(
                    my_flags + (0 * WORLD + peer) * MAX_CTA + pid,
                    0,
                    sem="acquire",
                    scope="sys",
                )
                < token
            ):
                pass
    tl.debug_barrier()
    _fence_sys(FENCE_LANES)

    # Phase B: reduce-scatter -- sum the island over the shard this rank owns.
    for off in range(my_start, my_end, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < my_end
        acc = tl.load(my_data + offs, mask=mask, other=0.0).to(tl.float32)
        for i in tl.static_range(island_size):
            peer = island_base + i
            if peer != rank:
                pd = (
                    tl.cast(tl.load(ptrs_ptr + peer), tl.pointer_type(tl.bfloat16))
                    + buf_off
                )
                acc += tl.load(pd + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(my_partial + offs, acc.to(tl.bfloat16), mask=mask)
    _fence_sys(FENCE_LANES)
    tl.debug_barrier()
    cpf = tl.cast(tl.load(flag_ptrs_ptr + counterpart), tl.pointer_type(tl.int32))
    tl.store(cpf + (1 * WORLD + rank) * MAX_CTA + pid, token)

    # Phase C: single cross-island exchange, shard-sized. The counterpart
    # holds the same island-local index, hence the same shard offsets.
    while (
        tl.atomic_add(
            my_flags + (1 * WORLD + counterpart) * MAX_CTA + pid,
            0,
            sem="acquire",
            scope="sys",
        )
        < token
    ):
        pass
    tl.debug_barrier()
    _fence_sys(FENCE_LANES)
    cp_partial = (
        tl.cast(
            tl.load(partial_ptrs_ptr + counterpart),
            tl.pointer_type(tl.bfloat16),
        )
        + buf_off
    )
    for off in range(my_start, my_end, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < my_end
        acc = tl.load(my_partial + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.load(cp_partial + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(my_gather + offs, acc.to(tl.bfloat16), mask=mask)
    _fence_sys(FENCE_LANES)
    tl.debug_barrier()
    for i in tl.static_range(island_size):
        peer = island_base + i
        if peer != rank:
            pf = tl.cast(tl.load(flag_ptrs_ptr + peer), tl.pointer_type(tl.int32))
            tl.store(pf + (2 * WORLD + rank) * MAX_CTA + pid, token)

    # Phase D: allgather the finished shards from the island.
    for i in tl.static_range(island_size):
        peer = island_base + i
        if peer != rank:
            while (
                tl.atomic_add(
                    my_flags + (2 * WORLD + peer) * MAX_CTA + pid,
                    0,
                    sem="acquire",
                    scope="sys",
                )
                < token
            ):
                pass
    tl.debug_barrier()
    _fence_sys(FENCE_LANES)
    for s in tl.static_range(island_size):
        peer = island_base + s
        pg = (
            tl.cast(tl.load(gather_ptrs_ptr + peer), tl.pointer_type(tl.bfloat16))
            + buf_off
        )
        start = s * shard + pid * ss
        end = tl.minimum(tl.minimum(s * shard + shard, start + ss), numel)
        for off in range(start, end, BLOCK):
            offs = off + tl.arange(0, BLOCK)
            mask = offs < end
            tl.store(
                out_ptr + offs,
                tl.load(pg + offs, mask=mask, other=0.0),
                mask=mask,
            )


class HierarchicalAllReduce:
    """Two-level island-aware allreduce over IPC-shared buffers.

    Args:
        group: torch.distributed group covering all ranks on this node.
        device: this rank's CUDA device.
        islands: rank partition, e.g. [[0,1,2,3],[4,5,6,7]]. Exactly two
            islands of equal size are supported.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        islands: Sequence[Sequence[int]],
    ) -> None:
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        assert len(islands) == 2 and len(islands[0]) == len(islands[1]), (
            "HierarchicalAllReduce supports exactly two equal islands"
        )
        self.islands = [list(i) for i in islands]
        me = self.rank
        self.island_idx = 0 if me in self.islands[0] else 1
        island = self.islands[self.island_idx]
        other = self.islands[1 - self.island_idx]
        self.island_base = min(island)
        self.island_size = len(island)
        self.counterpart = other[island.index(me)]

        rt = _get_cudart()
        _check(rt.cudaSetDevice(ctypes.c_int(device.index)), "cudaSetDevice")
        self._own = []
        self._opened = []
        # data slots (bf16), partial slots (fp32) — double-buffered; flags
        # [phase(2) x writer(world)] int32
        data_ptr = self._alloc(rt, 2 * _MAX_ELEMS * 2)
        partial_ptr = self._alloc(rt, 2 * _MAX_ELEMS * 2)
        gather_ptr = self._alloc(rt, 2 * _MAX_ELEMS * 2)
        flags_ptr = self._alloc(rt, _MAX_CTA * 2 * self.world_size * 4)
        flags2_ptr = self._alloc(rt, 3 * self.world_size * _MAX_CTA * 4)
        self._token_ctr = torch.zeros(_MAX_CTA, dtype=torch.int32, device=device)
        self._token_ctr2 = torch.zeros(_MAX_CTA, dtype=torch.int32, device=device)

        self._data_ptrs = self._exchange_ptrs(rt, data_ptr)
        self._partial_ptrs = self._exchange_ptrs(rt, partial_ptr)
        self._gather_ptrs = self._exchange_ptrs(rt, gather_ptr)
        self._flag_ptrs = self._exchange_ptrs(rt, flags_ptr)
        self._flag2_ptrs = self._exchange_ptrs(rt, flags2_ptr)

    def _alloc(self, rt, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        _check(rt.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
        _check(rt.cudaMemset(ptr, 0, nbytes), "cudaMemset")
        self._own.append(ptr.value)
        return ptr.value

    def _exchange_ptrs(self, rt, local_ptr: int) -> torch.Tensor:
        """Share a raw device allocation with all ranks via CUDA IPC and
        return a device tensor of every rank's pointer. Handles are opened
        with the importing device current so lazy peer access covers our
        kernels' direct loads/stores."""
        handle = _IpcHandle()
        _check(
            rt.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(local_ptr)),
            "cudaIpcGetMemHandle",
        )
        objs: list = [None] * self.world_size
        dist.all_gather_object(objs, (self.rank, bytes(handle)), group=self.group)
        ptrs = torch.zeros(self.world_size, dtype=torch.int64, device=self.device)
        for rank, hbytes in objs:
            if rank == self.rank:
                ptrs[rank] = local_ptr
                continue
            h = _IpcHandle.from_buffer_copy(hbytes)
            peer_ptr = ctypes.c_void_p()
            _check(
                rt.cudaIpcOpenMemHandle(ctypes.byref(peer_ptr), h, ctypes.c_uint(1)),
                "cudaIpcOpenMemHandle",
            )
            self._opened.append(peer_ptr.value)
            ptrs[rank] = peer_ptr.value
        return ptrs

    def __del__(self):
        try:
            rt = _get_cudart()
            for ptr in getattr(self, "_opened", []):
                rt.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))
            for ptr in getattr(self, "_own", []):
                rt.cudaFree(ctypes.c_void_p(ptr))
        except Exception:
            pass

    def should_use(self, inp: torch.Tensor) -> bool:
        return (
            inp.dtype == torch.bfloat16
            and inp.is_contiguous()
            and inp.numel() <= _MAX_ELEMS
            and inp.numel() % self.island_size == 0
        )

    def all_reduce(self, inp: torch.Tensor, out: torch.Tensor | None = None):
        if out is None:
            out = torch.empty_like(inp)
        numel = inp.numel()
        if numel >= _TWO_SHOT_MIN_ELEMS:
            ncta = min(_MAX_CTA, max(1, numel // (self.island_size * 1024)))
            _hier_two_shot_kernel[(ncta,)](
                inp.view(-1),
                out.view(-1),
                self._data_ptrs,
                self._partial_ptrs,
                self._gather_ptrs,
                self._flag2_ptrs,
                rank=self.rank,
                island_base=self.island_base,
                island_idx=self.rank - self.island_base,
                island_size=self.island_size,
                counterpart=self.counterpart,
                WORLD=self.world_size,
                numel=numel,
                token_ptr=self._token_ctr2,
                MAX_ELEMS=_MAX_ELEMS,
                MAX_CTA=_MAX_CTA,
                FENCE_LANES=_NUM_WARPS * 32,
                BLOCK=min(4096, triton.next_power_of_2(numel)),
                num_warps=_NUM_WARPS,
            )
            return out
        ncta = min(_MAX_CTA, max(1, (numel + 4095) // 4096))
        _hier_all_reduce_kernel[(ncta,)](
            inp.view(-1),
            out.view(-1),
            self._data_ptrs,
            self._partial_ptrs,
            self._flag_ptrs,
            rank=self.rank,
            island_base=self.island_base,
            island_size=self.island_size,
            counterpart=self.counterpart,
            WORLD=self.world_size,
            numel=numel,
            token_ptr=self._token_ctr,
            MAX_ELEMS=_MAX_ELEMS,
            FENCE_LANES=_NUM_WARPS * 32,
            BLOCK=min(8192, triton.next_power_of_2(numel)),
            num_warps=_NUM_WARPS,
        )
        return out
