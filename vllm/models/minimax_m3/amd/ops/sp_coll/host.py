# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pools and collectives on the sp_coll kernels: per rank ``sets x 2`` IPC-mapped
slots, one set per DBO microbatch. Two slots suffice: slot k is rewritten for phase
c + 2 only after every peer published c + 1, i.e. finished pulling phase c."""

import ctypes
import functools
import math

import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.models.minimax_m3.amd.ops.flydsl_utils import run_compiled as _run_compiled

from . import kernel as _kernel

logger = init_logger(__name__)

WORLD = _kernel.WORLD
SLOTS = 2
_ALIGN = 256


@functools.cache
def _kernels():
    k = _kernel
    return (
        k.compile_publish(),
        k.compile_fence_publish(),
        k.compile_pull(),
        k.compile_pull_unpack(),
        k.compile_sum4(),
    )


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


@functools.cache
def _hip():
    lib = ctypes.CDLL("libamdhip64.so")
    lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
    lib.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _IpcHandle,
        ctypes.c_uint,
    ]
    return lib


class _DevicePointer:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": (nbytes,),
            "typestr": "|u1",
            "strides": None,
            "version": 3,
        }


def _alloc_pool(nbytes: int, device: torch.device) -> tuple[int, bytes, torch.Tensor]:
    """-> (pointer, IPC handle bytes, uint8 tensor); never freed."""
    hip = _hip()
    ptr = ctypes.c_void_p()
    with torch.cuda.device(device):
        rc = hip.hipMalloc(ctypes.byref(ptr), nbytes)
    assert rc == 0, f"hipMalloc({nbytes}) failed: {rc}"
    handle = _IpcHandle()
    rc = hip.hipIpcGetMemHandle(ctypes.byref(handle), ptr)
    assert rc == 0, f"hipIpcGetMemHandle failed: {rc}"
    pool = torch.as_tensor(_DevicePointer(ptr.value, nbytes), device=device)
    assert pool.data_ptr() == ptr.value
    # raw bytes: a c_char field reads back only up to its first NUL
    raw = ctypes.string_at(ctypes.addressof(handle), ctypes.sizeof(handle))
    return ptr.value, raw, pool


def _open_pool(handle: bytes, device: torch.device) -> int:
    hip = _hip()
    ptr = ctypes.c_void_p()
    h = _IpcHandle()
    assert len(handle) == ctypes.sizeof(h)
    ctypes.memmove(ctypes.addressof(h), handle, len(handle))
    with torch.cuda.device(device):
        rc = hip.hipIpcOpenMemHandle(ctypes.byref(ptr), h, 1)  # lazy peer access
    assert rc == 0, f"hipIpcOpenMemHandle failed: {rc}"
    return ptr.value


def _exchange(obj, rank: int, world: int, group) -> list:
    """Every rank's ``obj``. Broadcasts: gloo's ``all_gather_object`` fails under
    inference mode (the profile run)."""
    out = [[None] for _ in range(world)]
    out[rank][0] = obj
    ranks = sorted(dist.get_process_group_ranks(group=group))
    for i, r in enumerate(ranks):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]


class Pools:
    """One rank's pool and flags plus the peers'; ``slot_bytes`` fits the largest
    phase."""

    def __init__(
        self,
        rank: int,
        world: int,
        group,
        device: torch.device,
        slot_bytes: int,
        sets: int = 2,
    ):
        assert world == WORLD, world
        self.rank, self.world, self.device = rank, world, device
        self.sets, self.slot_bytes = sets, slot_bytes
        nbytes = sets * SLOTS * slot_bytes
        ptr, handle, self.pool = _alloc_pool(nbytes, device)
        flag_ptr, flag_handle = ops.allocate_shared_buffer_and_handle(_ALIGN)
        peers = _exchange((handle, flag_handle), rank, world, group)
        self.pool_ptrs = [
            ptr if r == rank else _open_pool(h, device)
            for r, (h, _) in enumerate(peers)
        ]
        self.flag_ptrs = [
            flag_ptr if r == rank else ops.open_mem_handle(h)
            for r, (_, h) in enumerate(peers)
        ]
        self.seq = [0] * sets
        self.phase = [0] * sets
        publish, fence_publish, pull, pull_unpack, sum4 = _kernels()
        self.fence_publish, self.pull, self.sum4 = fence_publish, pull, sum4
        self.pull_unpack = pull_unpack
        stream = torch.cuda.current_stream(device)
        # zero every set's flag word and ticket counter (base + 4)
        for s in range(sets):
            base = self.flag_ptrs[rank] + _kernel.FLAG_STRIDE * s
            _run_compiled(publish, base, 0, stream)
            _run_compiled(publish, base + 4, 0, stream)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        logger.info(
            "sp_coll: %d sets x %d slots x %.0f MB pools, FlyDSL collectives on",
            sets,
            SLOTS,
            slot_bytes / 2**20,
        )

    def slot(self, set_id: int) -> tuple[int, int]:
        """Next slot of ``set_id``: (its byte offset in every pool, flag offset)."""
        k = self.phase[set_id] % SLOTS
        self.phase[set_id] += 1
        return (set_id * SLOTS + k) * self.slot_bytes, _kernel.FLAG_STRIDE * set_id

    def next_seq(self, set_id: int) -> int:
        self.seq[set_id] += 1
        return self.seq[set_id]


class Collectives:
    """One phase's collectives in the next slot of ``set_id``. ``alloc`` gives
    producers views of the slot that the collectives read in place; inputs that do
    not fit fall back to ``rccl`` (same sizes on every rank, so all ranks agree)."""

    def __init__(self, pools: Pools, set_id: int):
        self.p = pools
        self.set_id = set_id
        self.base, self.flag_off = pools.slot(set_id)
        self.off = 0
        self.stream: torch.cuda.Stream | None = None
        self.rccl = None

    def bind(self, stream: torch.cuda.Stream, rccl) -> "Collectives":
        self.stream, self.rccl = stream, rccl
        return self

    def alloc(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor | None:
        """A view of the slot for a producer to write; None when it does not fit."""
        n = math.prod(shape) * dtype.itemsize
        off = self._take(n)
        if off is None:
            return None
        return self.p.pool[off : off + n].view(dtype).view(shape)

    def _take(self, nbytes: int) -> int | None:
        """Reserve ``nbytes`` of the slot (256 B aligned); None when it does not fit."""
        off = (self.off + _ALIGN - 1) // _ALIGN * _ALIGN
        if off + nbytes > self.p.slot_bytes:
            return None
        self.off = off + nbytes
        return self.base + off

    def _stage(self, x: torch.Tensor) -> int | None:
        """``x``'s offset in every pool: in place if ``alloc`` placed it, else
        copied in; None when it does not fit."""
        n = x.numel() * x.element_size()
        start = self.p.pool.data_ptr() + self.base
        if start <= x.data_ptr() < start + self.p.slot_bytes:
            assert x.data_ptr() + n <= start + self.p.slot_bytes
            return x.data_ptr() - self.p.pool.data_ptr()
        off = self._take(n)
        if off is None:
            return None
        dst = self.p.pool[off : off + n].view(x.dtype).view(x.shape)
        dst.copy_(x)
        return off

    def _publish_wait(self) -> None:
        """Fence + publish this rank's slot; the pulls wait for the peers' flags."""
        p = self.p
        self.seq = p.next_seq(self.set_id)
        me_flag = p.flag_ptrs[p.rank] + self.flag_off
        _run_compiled(p.fence_publish, me_flag, self.seq, self.stream)

    def _pull(self, dst: torch.Tensor, off: int, block: int, src_off: int) -> None:
        p = self.p
        srcs = [ptr + off for ptr in p.pool_ptrs]
        flags = [f + self.flag_off for f in p.flag_ptrs]
        _run_compiled(
            p.pull,
            dst.data_ptr(),
            *srcs,
            *flags,
            block,
            src_off,
            self.seq,
            p.rank,
            self.stream,
        )

    def all_gather(self, outs: list[torch.Tensor], ins: list[torch.Tensor]) -> None:
        """Each ``[rows, ...]`` shard into its ``[padded, ...]`` output."""
        ins = [i.contiguous() for i in ins]
        offs = [self._stage(i) for i in ins]
        if any(o is None for o in offs):
            self.rccl.all_gather(outs, ins)
            return
        self._publish_wait()
        for o, i, off in zip(outs, ins, offs):
            assert o.is_contiguous() and o.dtype == i.dtype
            block = i.numel() * i.element_size()
            assert o.numel() * o.element_size() == WORLD * block, (o.shape, i.shape)
            self._pull(o, off, block, 0)

    def all_to_all(self, out: torch.Tensor, x: torch.Tensor) -> None:
        """Row block r of ``x`` to rank r; row block r of ``out`` from rank r."""
        x = x.contiguous()
        n = x.numel() * x.element_size()
        assert n % WORLD == 0 and out.is_contiguous()
        assert out.numel() * out.element_size() == n
        off = self._stage(x)
        if off is None:
            self.rccl.all_to_all(out, x)
            return
        self._publish_wait()
        block = n // WORLD
        self._pull(out, off, block, self.p.rank * block)

    def all_to_all_unpack(
        self, x: torch.Tensor, d: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``all_to_all`` of packed rows ``[WORLD * rows, d + d / 32]``, unpacked:
        (e4m3 codes ``[rows, WORLD * d]``, scales ``[rows, WORLD * d / 32]``), or
        None when ``x`` does not fit the slot."""
        x = x.contiguous()
        rows = x.shape[0] // WORLD
        assert x.dtype == torch.uint8 and x.shape == (WORLD * rows, d + d // 32)
        assert d % 32 == 0 and d % 16 == 0
        n = x.numel()
        off = self._stage(x)
        if off is None:
            return None
        self._publish_wait()
        codes = torch.empty((rows, WORLD * d), dtype=torch.uint8, device=x.device)
        scales = torch.empty(
            (rows, WORLD * d // 32), dtype=torch.uint8, device=x.device
        )
        p = self.p
        block = n // WORLD
        srcs = [ptr + off for ptr in p.pool_ptrs]
        flags = [f + self.flag_off for f in p.flag_ptrs]
        _run_compiled(
            p.pull_unpack,
            codes.data_ptr(),
            scales.data_ptr(),
            *srcs,
            *flags,
            block,
            p.rank * block,
            self.seq,
            p.rank,
            rows,
            d,
            self.stream,
        )
        return codes.view(torch.float8_e4m3fn), scales

    def reduce_scatter(self, out: torch.Tensor, x: torch.Tensor) -> None:
        """``[padded, H]`` partial sums -> this rank's ``[rows, H]`` sums (fp32,
        rank order)."""
        assert x.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
        x = x.contiguous()
        n = x.numel() * x.element_size()
        block = n // WORLD
        assert out.is_contiguous() and out.numel() * out.element_size() == block
        off = self._stage(x)
        if off is None:
            self.rccl.reduce_scatter(out, x)
            return
        self._publish_wait()
        # allocated and read on this stream only
        staging = torch.empty(n, dtype=torch.uint8, device=x.device)
        self._pull(staging, off, block, self.p.rank * block)
        srcs = [staging.data_ptr() + r * block for r in range(WORLD)]
        _run_compiled(self.p.sum4, out.data_ptr(), *srcs, block, self.stream)
