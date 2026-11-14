# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP transport on CUDA copy engines, synchronized on the GPU (one NVLink
domain).

Each rank shares one buffer holding its flags and a receive region
laid out like its persistent intermediate tensors. Per stage boundary, step N:

- sender: once the previous peer copy is done, pack the outputs into a local
  send buffer; on a side stream, wait `empty >= N - 1`, peer-copy into the
  receiver's region and write `full = N`.
- receiver: wait `full >= N`, copy the region into the persistent tensors
  and write `empty = N`.

Each flag lives on the GPU that waits on it and the peer writes it with stream
memory ops, so neither SMs nor the CPU wait on the transfer. Metadata goes over
the CPU group.

The waits omit CU_STREAM_WAIT_VALUE_FLUSH, which x86 hosts do not support. As
in NCCL's zero-SM copy-engine path, this relies on NVLink delivering the peer
copy before the flag write that follows it, so all PP GPUs must be NVLink peers.

With TP or PCP > 1, each (PCP, TP) position's PP chain is its own transport,
sending to the same position in the next stage: a stage's TP ranks hold the
same outputs and each sends all of them, and each PCP rank sends its own
tokens.

A chain on one node shares its receive buffers through CUDA IPC. A chain that
spans nodes (multi-node NVLink, e.g. GB200/GB300 NVL72) shares them through
FABRIC handles, which need an IMEX channel; mapping a peer's FABRIC memory only
succeeds within its NVLink domain. Stream memory ops cannot write another
node's memory, so there a flag is written to local staging and peer-copied,
after the data on the same stream.
"""

import contextlib
from collections import deque
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from vllm.distributed.pp_transport import PPTransport, check_intermediate_tensors
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator

# Flags at bytes 0 (full) and 8 (empty) of each shared buffer, and staging for
# the flags this rank writes to its peers at bytes 16 (their full) and 24 (their
# empty); data from here.
_DATA_OFFSET = 256


def _check(result) -> Any:
    err, *rest = result if isinstance(result, tuple) else (result,)
    if int(err) != 0:
        raise RuntimeError(f"CUDA call failed: {err}")
    return rest[0] if len(rest) == 1 else rest


class _Region(NamedTuple):
    """One tensor's region in a send or receive buffer."""

    offset: int
    row_bytes: int
    max_rows: int
    dtype: torch.dtype
    row_shape: tuple[int, ...]


def _layout(persistent: dict[str, torch.Tensor]) -> dict[str, _Region]:
    layout, offset = {}, 0
    for key, t in persistent.items():
        row_bytes = t[0].numel() * t.element_size()
        layout[key] = _Region(
            offset, row_bytes, t.shape[0], t.dtype, tuple(t.shape[1:])
        )
        offset += (t.shape[0] * row_bytes + 255) // 256 * 256
    return layout


def _nbytes(layout: dict[str, _Region]) -> int:
    return max(
        (r.offset + r.max_rows * r.row_bytes for r in layout.values()), default=0
    )


def _views(
    buf: torch.Tensor, layout: dict[str, _Region], rows: dict[str, int]
) -> dict[str, torch.Tensor]:
    """The first `rows[key]` rows of each key's region of `buf`."""
    out = {}
    for key, n in rows.items():
        r = layout[key]
        region = buf[r.offset : r.offset + n * r.row_bytes]
        out[key] = region.view(r.dtype).view(n, *r.row_shape)
    return out


def _device_bytes(ptr: int, nbytes: int) -> torch.Tensor:
    cai = dict(shape=(nbytes,), typestr="|u1", data=(ptr, False), version=3)
    return torch.as_tensor(SimpleNamespace(__cuda_array_interface__=cai), device="cuda")


def _wait_flag(stream: torch.cuda.Stream, addr: int, at_least: int) -> None:
    from cuda.bindings import driver as cu

    _check(
        cu.cuStreamWaitValue64(
            cu.CUstream(stream.cuda_stream),
            cu.CUdeviceptr(addr),
            cu.cuuint64_t(at_least),
            int(cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ),
        )
    )


def _write_flag(stream: torch.cuda.Stream, addr: int, value: int) -> None:
    from cuda.bindings import driver as cu

    # Default flags: a system-scope fence orders the stream's earlier writes
    # (the peer copies) before this one.
    _check(
        cu.cuStreamWriteValue64(
            cu.CUstream(stream.cuda_stream),
            cu.CUdeviceptr(addr),
            cu.cuuint64_t(value),
            0,
        )
    )


def _signal(
    stream: torch.cuda.Stream, addr: int, value: int, staging: int | None
) -> None:
    """Write a peer's flag after the stream's earlier work; through `staging`
    (local) and a peer copy when `addr` is on another node."""
    if staging is None:
        _write_flag(stream, addr, value)
        return
    from cuda.bindings import runtime as cudart

    _write_flag(stream, staging, value)
    kind = cudart.cudaMemcpyKind.cudaMemcpyDefault
    _check(cudart.cudaMemcpyAsync(addr, staging, 8, kind, stream.cuda_stream))


class CopyEngineUnavailable(RuntimeError):
    """Copy engines are not usable. Raised on every rank alike."""


def _ipc_share(nbytes: int) -> tuple[int, Any, Callable[[], None]]:
    """A zeroed buffer on this GPU: its pointer, the handle a peer opens it
    with, and its release."""
    from vllm import _custom_ops as ops

    ptr, handle = ops.allocate_shared_buffer_and_handle(nbytes)
    return ptr, handle, lambda: ops.free_shared_buffer(ptr)


def _ipc_open(handle: Any, nbytes: int) -> tuple[int, Callable[[], None]]:
    from cuda.bindings import runtime as cudart

    from vllm import _custom_ops as ops

    ptr = ops.open_mem_handle(handle)
    return ptr, lambda: _check(cudart.cudaIpcCloseMemHandle(ptr))


def _fabric_map(handle: Any, size: int) -> int:
    """Map a cuMem allocation for this GPU to a new address."""
    from cuda.bindings import driver as cu

    ptr = int(_check(cu.cuMemAddressReserve(size, 0, 0, 0)))
    try:
        _check(cu.cuMemMap(ptr, size, 0, handle, 0))
        access = cu.CUmemAccessDesc()
        access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = torch.accelerator.current_device_index()
        access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _check(cu.cuMemSetAccess(ptr, size, [access], 1))
    except Exception:
        _fabric_unmap(ptr, size)
        raise
    return ptr


def _fabric_unmap(ptr: int, size: int) -> None:
    from cuda.bindings import driver as cu

    # Not checked: also undoes a partial `_fabric_map`.
    cu.cuMemUnmap(ptr, size)
    cu.cuMemAddressFree(ptr, size)


def _fabric_share(nbytes: int) -> tuple[int, Any, Callable[[], None]]:
    """As `_ipc_share`, through a FABRIC handle."""
    from cuda.bindings import driver as cu

    from vllm.utils.vmm_driver import imex_channel_available

    if not imex_channel_available():
        raise RuntimeError("no IMEX channel for FABRIC handles")
    fabric = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = torch.accelerator.current_device_index()
    prop.requestedHandleTypes = fabric
    granularity = _check(cu.cuMemGetAllocationGranularity(prop, 0))
    size = -(-nbytes // granularity) * granularity
    handle = _check(cu.cuMemCreate(size, prop, 0))
    try:
        ptr = _fabric_map(handle, size)
    except Exception:
        cu.cuMemRelease(handle)
        raise

    def release() -> None:
        _fabric_unmap(ptr, size)
        _check(cu.cuMemRelease(handle))

    try:
        exported = _check(cu.cuMemExportToShareableHandle(handle, fabric, 0))
        _check(cu.cuMemsetD8(ptr, 0, size))
        torch.accelerator.synchronize()
    except Exception:
        release()
        raise
    return ptr, (bytes(exported.data), size), release


def _fabric_open(handle: Any, nbytes: int) -> tuple[int, Callable[[], None]]:
    """As `_ipc_open`, for a buffer from `_fabric_share`."""
    from cuda.bindings import driver as cu

    exported, size = handle
    fabric = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    imported = _check(cu.cuMemImportFromShareableHandle(exported, fabric))
    try:
        ptr = _fabric_map(imported, size)
    finally:
        # The mapping keeps the memory alive.
        cu.cuMemRelease(imported)
    return ptr, lambda: _fabric_unmap(ptr, size)


class CopyEngineTransport(PPTransport):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        pp_group: "GroupCoordinator",
        persistent: dict[str, torch.Tensor] | None,
    ) -> None:
        """Collective over the PP, TP and PCP groups. Raises `CopyEngineUnavailable`
        on every rank alike, with nothing left allocated, if copy engines are
        not usable on some rank."""
        from vllm.distributed.parallel_state import (
            get_pcp_group,
            get_tp_group,
            in_the_same_node_as,
        )
        from vllm.platforms import current_platform

        # The same on every rank, so this raises before any collective.
        if not current_platform.is_cuda():
            raise CopyEngineUnavailable("PP copy engines need CUDA")

        # Anything that can differ between ranks runs inside `step`, and the
        # ranks share its outcome before going on, so they all agree. That
        # includes the other chains over TP (a stage's TP ranks must all use
        # copy engines or none, since NCCL's receive all-gathers over TP) and
        # PCP, so every rank makes the same choice.
        rank, world = pp_group.rank_in_group, pp_group.world_size
        other_chains = [
            g for g in (get_tp_group(), get_pcp_group()) if g.world_size > 1
        ]
        one_node = all(in_the_same_node_as(pp_group.cpu_group, source_rank=0))
        share, open_peer = _ipc_share, _ipc_open
        if not one_node:
            share, open_peer = _fabric_share, _fabric_open
        # Releases this rank's buffers and peer mappings, in setup order.
        releases: list[Callable[[], None]] = []

        def step(fn) -> list[Any]:
            out = error = None
            try:
                out = fn()
            except Exception as e:
                error = repr(e)
            gathered: list[Any] = [None] * world
            torch.distributed.all_gather_object(
                gathered, (out, error), group=pp_group.cpu_group
            )
            errors = [e for _, e in gathered if e]
            # In turn over TP then PCP, so each rank sees every chain's errors.
            for group in other_chains:
                group_errors: list[Any] = [None] * group.world_size
                torch.distributed.all_gather_object(
                    group_errors, errors[0] if errors else None, group=group.cpu_group
                )
                errors = errors or [e for e in group_errors if e]
            if errors:
                # Best effort: every rank must still raise the same error.
                for release in reversed(releases):
                    with contextlib.suppress(Exception):
                        release()
                raise CopyEngineUnavailable(
                    f"PP copy engines are not usable: {errors[0]}"
                )
            return [o for o, _ in gathered]

        def allocate_recv() -> tuple[Any, ...]:
            import cuda.bindings.driver as cu

            layout = _layout(persistent) if persistent is not None else {}
            device = torch.accelerator.current_device_index()
            attr = cu.CUdevice_attribute
            memops = attr.CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS
            dev = _check(cu.cuDeviceGet(device))
            if not _check(cu.cuDeviceGetAttribute(memops, dev)):
                raise RuntimeError(f"GPU {device} lacks 64-bit stream memory ops")
            # Zeroed before the handle is shared.
            ptr, handle, release = share(_DATA_OFFSET + _nbytes(layout))
            releases.append(release)
            phys = current_platform.visible_device_id_to_physical_device_id(device)
            return ptr, handle, layout, phys

        ranks = step(allocate_recv)
        ptr, _, layout, _ = ranks[rank]

        def open_rank(r: int) -> int:
            peer_ptr, release = open_peer(
                ranks[r][1], _DATA_OFFSET + _nbytes(ranks[r][2])
            )
            releases.append(release)
            return peer_ptr

        def connect() -> tuple[int | None, int | None, int | None]:
            # Across nodes, only GPUs in one NVLink domain map each other's
            # FABRIC memory, so mapping it checks the same.
            if one_node and not current_platform.is_fully_connected(
                [r[3] for r in ranks]
            ):
                raise RuntimeError("PP GPUs are not all NVLink-connected")
            prev_ptr = next_ptr = send_ptr = None
            if not pp_group.is_first_rank:
                prev_ptr = open_rank(rank - 1)
            if not pp_group.is_last_rank:
                next_ptr = open_rank(rank + 1)
                # Outside the torch allocator, so memory profiling counts it.
                send_ptr, _, release = _ipc_share(_nbytes(ranks[rank + 1][2]))
                releases.append(release)
            return prev_ptr, next_ptr, send_ptr

        prev_ptr, next_ptr, send_ptr = step(connect)[rank]

        # Local staging for flags written to another node; see `_signal`.
        self.stage_full = None if one_node else ptr + 16
        self.stage_empty = None if one_node else ptr + 24
        if not pp_group.is_first_rank:
            assert persistent is not None and prev_ptr is not None
            self.src = rank - 1
            self.persistent = persistent
            self.recv_layout = layout
            self.recv_buf = _device_bytes(ptr + _DATA_OFFSET, _nbytes(layout))
            self.full = ptr
            self.peer_empty = prev_ptr + 8
        if not pp_group.is_last_rank:
            assert next_ptr is not None and send_ptr is not None
            self.dst = rank + 1
            self.send_layout = ranks[rank + 1][2]
            self.send_specs = {
                k: (r.dtype, r.row_shape) for k, r in self.send_layout.items()
            }
            self.send_buf = _device_bytes(send_ptr, _nbytes(self.send_layout))
            self.empty = ptr + 8
            self.peer_recv_buf = next_ptr + _DATA_OFFSET
            self.peer_full = next_ptr
            self.stream = torch.cuda.Stream()
            self.seq = 0
            # The send buffer's last peer copy.
            self.peer_done: torch.cuda.Event | None = None
            # Metadata sends with their step's peer_done.
            self.pending: deque[tuple[Any, torch.cuda.Event]] = deque()

    def recv(self, pp_group: "GroupCoordinator") -> IntermediateTensors:
        """This step's tensors, as views of the persistent tensors. The wait
        for the previous stage and the copy are queued on the current stream
        when `.tensors` is first read, so GPU work queued before then (input
        preparation) overlaps the transfer, as with NCCL."""
        from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors

        meta = pp_group.recv_object(src=self.src)
        src = _views(self.recv_buf, self.recv_layout, meta["rows"])
        out = {k: self.persistent[k][: v.shape[0]] for k, v in src.items()}

        def copy_in() -> None:
            stream = torch.cuda.current_stream()
            _wait_flag(stream, self.full, meta["seq"])
            for key, view in src.items():
                out[key].copy_(view)
            _signal(stream, self.peer_empty, meta["seq"], self.stage_empty)

        return AsyncIntermediateTensors(out, comm_postprocess=[copy_in])

    def send(self, tensors: dict, pp_group: "GroupCoordinator") -> None:
        from cuda.bindings import runtime as cudart

        check_intermediate_tensors(tensors, self.send_specs)
        rows = {k: t.shape[0] for k, t in tensors.items()}
        self.seq += 1
        meta_work = pp_group.isend_object(dict(seq=self.seq, rows=rows), dst=self.dst)

        # Main stream: pack once the previous peer copy is done.
        main = torch.cuda.current_stream()
        if self.peer_done is not None:
            main.wait_event(self.peer_done)
        for key, dst in _views(self.send_buf, self.send_layout, rows).items():
            dst.copy_(tensors[key])
        packed = torch.cuda.Event()
        packed.record(main)

        # Side stream: peer copy once the receive region is empty, then publish.
        self.stream.wait_event(packed)
        _wait_flag(self.stream, self.empty, self.seq - 1)
        for key, n in rows.items():
            r = self.send_layout[key]
            if n:
                _check(
                    cudart.cudaMemcpyAsync(
                        self.peer_recv_buf + r.offset,
                        self.send_buf.data_ptr() + r.offset,
                        n * r.row_bytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDefault,
                        self.stream.cuda_stream,
                    )
                )
        self.peer_done = torch.cuda.Event()
        self.peer_done.record(self.stream)
        _signal(self.stream, self.peer_full, self.seq, self.stage_full)

        # gloo sends complete only in wait(), which blocks until the peer
        # receives. Step k+1's copy waited for empty >= k, so k was received.
        self.pending.append((meta_work, self.peer_done))
        while len(self.pending) > 1 and self.pending[1][1].query():
            self.pending.popleft()[0].wait()
