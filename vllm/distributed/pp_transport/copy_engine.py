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

The platform's flag ops and peer copy come from `_CudaCopyEngines`, or on ROCm
(gfx950) from `rocm_sdma.SdmaCopyEngines`, which uses HIP's stream memory ops
and SDMA engines.

A chain on one node shares its receive buffers through CUDA (or HIP) IPC. A
chain that spans nodes (multi-node NVLink, e.g. GB200/GB300 NVL72; CUDA only)
shares them through FABRIC handles, which need an IMEX channel; mapping a peer's
FABRIC memory only succeeds within its NVLink domain. Stream memory ops cannot
write another node's memory, so there a flag is written to local staging and
peer-copied, after the data on the same stream.
"""

import contextlib
import ctypes
from collections import deque
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from vllm.distributed.pp_transport import PPTransport, check_intermediate_tensors
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.distributed.pp_transport.rocm_sdma import SdmaCopyEngines

# Flags at bytes 0 (full) and 8 (empty) of each shared buffer, and at 16 the
# local staging for the flag this rank writes to its previous stage's empty when
# that stage is on another node; the copy engines' own words from
# _ENGINE_OFFSET, data from _DATA_OFFSET.
# For heterogeneous PP fan-out (num_dsts > 1), each receiver gets its own
# 8-byte empty flag and _ENGINE_STRIDE-byte engine staging slot.
_ENGINE_OFFSET = 64
_ENGINE_STRIDE = 192
_DATA_OFFSET = 256


def _header_layout(num_dsts: int) -> tuple[list[int], list[int], int]:
    """Return `(empty_offsets, engine_offsets, data_offset)` for a rank feeding
    `num_dsts` downstream PP peers.

    For `num_dsts <= 1` (uniform PP, 1-to-1, or fan-in), this matches the
    legacy fixed layout (`empty=8`, `engine=64`, `data=256`).
    """
    if num_dsts <= 1:
        return [8], [_ENGINE_OFFSET], _DATA_OFFSET
    empty_offsets = [32 + 8 * i for i in range(num_dsts)]
    engine_base = ((32 + 8 * num_dsts + 63) // 64) * 64
    engine_offsets = [engine_base + _ENGINE_STRIDE * i for i in range(num_dsts)]
    data_offset = max(
        _DATA_OFFSET,
        ((engine_base + _ENGINE_STRIDE * num_dsts + 255) // 256) * 256,
    )
    return empty_offsets, engine_offsets, data_offset


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


class CopyEngineUnavailable(RuntimeError):
    """Copy engines are not usable. Raised on every rank alike."""


def _close_peer(ptr: int) -> None:
    # CudaRTLibrary calls HIP's IPC close on ROCm.
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    CudaRTLibrary().cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))


class _CudaCopyEngines:
    """CUDA's flag ops (stream memory ops) and, for a sender, its peer copy to
    the next stage (memcpys on copy engines)."""

    @staticmethod
    def check_device(device: int) -> None:
        import cuda.bindings.driver as cu

        attr = cu.CUdevice_attribute
        memops = attr.CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS
        dev = _check(cu.cuDeviceGet(device))
        if not _check(cu.cuDeviceGetAttribute(memops, dev)):
            raise RuntimeError(f"GPU {device} lacks 64-bit stream memory ops")

    @staticmethod
    def wait_flag(stream: torch.cuda.Stream, addr: int, at_least: int) -> None:
        from cuda.bindings import driver as cu

        result = cu.cuStreamWaitValue64(
            cu.CUstream(stream.cuda_stream),
            cu.CUdeviceptr(addr),
            cu.cuuint64_t(at_least),
            int(cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ),
        )
        _check(result)

    @staticmethod
    def write_flag(stream: torch.cuda.Stream, addr: int, value: int) -> None:
        from cuda.bindings import driver as cu

        # Default flags: a system-scope fence orders the stream's earlier writes
        # (the peer copies) before this one.
        result = cu.cuStreamWriteValue64(
            cu.CUstream(stream.cuda_stream),
            cu.CUdeviceptr(addr),
            cu.cuuint64_t(value),
            0,
        )
        _check(result)

    @classmethod
    def write_remote_flag(
        cls, stream: torch.cuda.Stream, addr: int, value: int, staging: int
    ) -> None:
        """As `write_flag`, for `addr` on another node, which stream memory ops
        cannot write: through `staging` (local) and a peer copy."""
        from cuda.bindings import runtime as cudart

        cls.write_flag(stream, staging, value)
        kind = cudart.cudaMemcpyKind.cudaMemcpyDefault
        _check(cudart.cudaMemcpyAsync(addr, staging, 8, kind, stream.cuda_stream))

    def __init__(
        self,
        send_buf: int,
        peer_recv_buf: int,
        peer_full: int,
        words: int,
        peer_pci: tuple[int, int, int],
        cross_node: bool,
    ) -> None:
        """`words`: this rank's shared-buffer bytes for the staging of `full`
        when the peer is on another node (`cross_node`); `peer_pci` is for SDMA
        and unused here."""
        self.send_buf = send_buf
        self.peer_recv_buf = peer_recv_buf
        self.peer_full = peer_full
        self.staging = words if cross_node else None

    def copy(
        self, stream: torch.cuda.Stream, layout: dict, rows: dict, seq: int
    ) -> None:
        """Queue on `stream` the copy of each key's first `rows` rows into the
        peer's receive region, then the write of `full = seq`."""
        from cuda.bindings import runtime as cudart

        for key, n in rows.items():
            r = layout[key]
            if n:
                result = cudart.cudaMemcpyAsync(
                    self.peer_recv_buf + r.offset,
                    self.send_buf + r.offset,
                    n * r.row_bytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDefault,
                    stream.cuda_stream,
                )
                _check(result)
        if self.staging is None:
            self.write_flag(stream, self.peer_full, seq)
        else:
            self.write_remote_flag(stream, self.peer_full, seq, self.staging)


def _ipc_share(nbytes: int) -> tuple[int, Any, Callable[[], None]]:
    """A zeroed buffer on this GPU: its pointer, the handle a peer opens it
    with, and its release."""
    from vllm import _custom_ops as ops

    ptr, handle = ops.allocate_shared_buffer_and_handle(nbytes)
    return ptr, handle, lambda: ops.free_shared_buffer(ptr)


def _ipc_open(handle: Any, nbytes: int) -> tuple[int, Callable[[], None]]:
    from vllm import _custom_ops as ops

    ptr = ops.open_mem_handle(handle)
    return ptr, lambda: _close_peer(ptr)


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
            HeterogeneousPPGroupCoordinator,
            get_pcp_group,
            get_tp_group,
            in_the_same_node_as,
            pp_boundary_dst_tp_ranks,
            pp_boundary_src_tp_rank,
        )

        # The same on every rank, so this raises before any collective.
        engines: type[_CudaCopyEngines] | type[SdmaCopyEngines]
        if current_platform.is_cuda():
            engines = _CudaCopyEngines
        elif current_platform.is_rocm():
            from vllm.distributed.pp_transport import rocm_sdma

            engines = rocm_sdma.SdmaCopyEngines
        else:
            raise CopyEngineUnavailable("PP copy engines need CUDA or ROCm")
        self.engines = engines

        stage_rank = pp_group.rank_in_group
        if isinstance(pp_group, HeterogeneousPPGroupCoordinator):
            # Heterogeneous PP's cpu_group already spans every rank in the
            # replica, so one all_gather_object shares handles and errors across
            # all stages and TP widths simultaneously.
            comm_world = pp_group.cpu_group.size()
            my_idx = pp_group.rank
            other_chains: list[GroupCoordinator] = []

            prev_idx: int | None = None
            prev_slot_idx = 0
            if not pp_group.is_first_rank:
                prev_stage = stage_rank - 1
                prev_tp_size = pp_group.stage_tp_sizes[prev_stage]
                cur_tp_size = pp_group.stage_tp_sizes[stage_rank]
                src_tp = pp_boundary_src_tp_rank(
                    pp_group.stage_tp_rank, prev_tp_size, cur_tp_size
                )
                prev_idx = pp_group.stage_offsets[prev_stage] + src_tp
                src_dsts = pp_boundary_dst_tp_ranks(src_tp, prev_tp_size, cur_tp_size)
                prev_slot_idx = src_dsts.index(pp_group.stage_tp_rank)

            next_indices: list[int] = []
            if not pp_group.is_last_rank:
                next_stage = stage_rank + 1
                cur_tp_size = pp_group.stage_tp_sizes[stage_rank]
                next_tp_size = pp_group.stage_tp_sizes[next_stage]
                dst_tps = pp_boundary_dst_tp_ranks(
                    pp_group.stage_tp_rank, cur_tp_size, next_tp_size
                )
                next_indices = [pp_group.stage_offsets[next_stage] + r for r in dst_tps]
        else:
            comm_world = pp_group.world_size
            my_idx = stage_rank
            other_chains = [
                g for g in (get_tp_group(), get_pcp_group()) if g.world_size > 1
            ]
            prev_idx = None if pp_group.is_first_rank else stage_rank - 1
            prev_slot_idx = 0
            next_indices = [] if pp_group.is_last_rank else [stage_rank + 1]

        empty_offsets, engine_offsets, my_data_offset = _header_layout(
            len(next_indices)
        )

        # Anything that can differ between ranks runs inside `step`, and the
        # ranks share its outcome before going on, so they all agree. That
        # includes the other chains over TP (a stage's TP ranks must all use
        # copy engines or none, since NCCL's receive all-gathers over TP) and
        # PCP, so every rank makes the same choice.
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
            gathered: list[Any] = [None] * comm_world
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
            if not one_node and engines is not _CudaCopyEngines:
                raise RuntimeError("PP SDMA copies need one node")
            layout = _layout(persistent) if persistent is not None else {}
            device = torch.accelerator.current_device_index()
            engines.check_device(device)
            # Zeroed before the handle is shared.
            ptr, handle, release = share(my_data_offset + _nbytes(layout))
            releases.append(release)
            phys = current_platform.visible_device_id_to_physical_device_id(device)
            props = torch.cuda.get_device_properties(device)
            pci = (props.pci_domain_id, props.pci_bus_id, props.pci_device_id)
            return ptr, handle, layout, phys, pci, my_data_offset, empty_offsets

        ranks = step(allocate_recv)
        ptr, _, layout, _, _, data_offset, _ = ranks[my_idx]

        def open_rank(r: int) -> int:
            peer_data_offset = ranks[r][5]
            peer_ptr, release = open_peer(
                ranks[r][1], peer_data_offset + _nbytes(ranks[r][2])
            )
            releases.append(release)
            return peer_ptr

        def connect() -> tuple[Any, ...]:
            # Across nodes, only GPUs in one NVLink domain map each other's
            # FABRIC memory, so mapping it checks the same.
            if one_node and not current_platform.is_fully_connected(
                [r[3] for r in ranks]
            ):
                raise RuntimeError("PP GPUs are not all directly connected")
            prev_ptr = send_ptr = None
            copiers: list[_CudaCopyEngines | SdmaCopyEngines] = []
            if prev_idx is not None:
                prev_ptr = open_rank(prev_idx)
            if next_indices:
                first_layout = ranks[next_indices[0]][2]
                if any(ranks[d][2] != first_layout for d in next_indices[1:]):
                    raise RuntimeError(
                        "Heterogeneous PP fan-out receivers have mismatched "
                        "intermediate tensor layouts"
                    )
                # Outside the torch allocator, so memory profiling counts it.
                send_ptr, _, release = _ipc_share(_nbytes(first_layout))
                releases.append(release)
                for i, d in enumerate(next_indices):
                    next_ptr = open_rank(d)
                    peer_data_offset = ranks[d][5]
                    copiers.append(
                        engines(
                            send_ptr,
                            next_ptr + peer_data_offset,
                            next_ptr,
                            ptr + engine_offsets[i],
                            ranks[d][4],
                            not one_node,
                        )
                    )
            return prev_ptr, send_ptr, copiers

        prev_ptr, send_ptr, self.copiers = step(connect)[my_idx]

        # Local staging for the empty flag written to another node.
        self.stage_empty = None if one_node else ptr + 16
        if prev_idx is not None:
            assert persistent is not None and prev_ptr is not None
            self.src = stage_rank - 1
            self.persistent = persistent
            self.recv_layout = layout
            self.recv_buf = _device_bytes(ptr + data_offset, _nbytes(layout))
            self.full = ptr
            peer_empty_offsets = ranks[prev_idx][6]
            self.peer_empty = prev_ptr + peer_empty_offsets[prev_slot_idx]
        if next_indices:
            assert send_ptr is not None and self.copiers
            self.dst = stage_rank + 1
            self.send_layout = ranks[next_indices[0]][2]
            self.send_specs = {
                k: (r.dtype, r.row_shape) for k, r in self.send_layout.items()
            }
            self.send_buf = _device_bytes(send_ptr, _nbytes(self.send_layout))
            self.empty_addrs = [ptr + off for off in empty_offsets]
            # One side stream per downstream peer so fan-out copies overlap
            # across NVLink links without head-of-line blocking.
            self.streams = [torch.cuda.Stream() for _ in self.copiers]
            self.seq = 0
            # The send buffer's last peer copies (one event per downstream peer).
            self.peer_done: list[torch.cuda.Event] = []
            # Metadata sends with their step's peer_done events.
            self.pending: deque[tuple[Any, list[torch.cuda.Event]]] = deque()

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
            self.engines.wait_flag(stream, self.full, meta["seq"])
            for key, view in src.items():
                out[key].copy_(view)
            if self.stage_empty is None:
                self.engines.write_flag(stream, self.peer_empty, meta["seq"])
            else:
                _CudaCopyEngines.write_remote_flag(
                    stream, self.peer_empty, meta["seq"], self.stage_empty
                )

        return AsyncIntermediateTensors(out, comm_postprocess=[copy_in])

    def send(self, tensors: dict, pp_group: "GroupCoordinator") -> None:
        if not self.copiers:
            # Fan-in boundary (src_tp > dst_tp): redundant replicated sender.
            return
        check_intermediate_tensors(tensors, self.send_specs)
        rows = {k: t.shape[0] for k, t in tensors.items()}
        self.seq += 1
        meta_work = pp_group.isend_object(dict(seq=self.seq, rows=rows), dst=self.dst)

        # Main stream: pack once the previous step's peer copies are all done.
        main = torch.cuda.current_stream()
        for ev in self.peer_done:
            main.wait_event(ev)
        for key, dst in _views(self.send_buf, self.send_layout, rows).items():
            dst.copy_(tensors[key])
        packed = torch.cuda.Event()
        packed.record(main)

        # Side streams: peer copy once each receiver's region is empty, then publish.
        done_events: list[torch.cuda.Event] = []
        for stream, empty_addr, copier in zip(
            self.streams, self.empty_addrs, self.copiers
        ):
            stream.wait_event(packed)
            self.engines.wait_flag(stream, empty_addr, self.seq - 1)
            copier.copy(stream, self.send_layout, rows, self.seq)
            ev = torch.cuda.Event()
            ev.record(stream)
            done_events.append(ev)
        self.peer_done = done_events

        # gloo sends complete only in wait(), which blocks until the peer
        # receives. Step k+1's copy waited for empty >= k, so k was received.
        self.pending.append((meta_work, done_events))
        while len(self.pending) > 1 and all(ev.query() for ev in self.pending[1][1]):
            self.pending.popleft()[0].wait()
