# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP transport on CUDA copy engines, synchronized on the GPU (one node).

Each rank shares one CUDA IPC buffer holding its flags and a receive region
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

The platform's flag ops and peer copy come from `_CudaCopyEngines`, or on ROCm
(gfx950) from `rocm_sdma.SdmaCopyEngines`, which uses HIP's stream memory ops
and SDMA engines.
"""

import contextlib
import ctypes
from collections import deque
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from vllm.distributed.pp_transport import PPTransport
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.distributed.pp_transport.rocm_sdma import SdmaCopyEngines

# Flags at bytes 0 (full) and 8 (empty) of each shared buffer, the copy
# engines' own words from _ENGINE_OFFSET, data from _DATA_OFFSET.
_ENGINE_OFFSET = 64
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

    def __init__(
        self,
        send_buf: int,
        peer_recv_buf: int,
        peer_full: int,
        words: int,
        peer_pci: tuple[int, int],
    ) -> None:
        """`words` and `peer_pci` are for SDMA and unused here."""
        self.send_buf = send_buf
        self.peer_recv_buf = peer_recv_buf
        self.peer_full = peer_full

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
        self.write_flag(stream, self.peer_full, seq)


class CopyEngineTransport(PPTransport):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        pp_group: "GroupCoordinator",
        persistent: dict[str, torch.Tensor] | None,
    ) -> None:
        """Collective over the PP group. Raises `CopyEngineUnavailable` on every
        rank alike, with nothing left allocated, if copy engines are not usable
        on some rank."""
        from vllm import _custom_ops as ops
        from vllm.distributed.parallel_state import in_the_same_node_as

        # The same on every rank, so these raise before any collective.
        parallel_config = vllm_config.parallel_config
        engines: type[_CudaCopyEngines] | type[SdmaCopyEngines]
        if current_platform.is_cuda():
            engines = _CudaCopyEngines
        elif current_platform.is_rocm():
            from vllm.distributed.pp_transport import rocm_sdma

            engines = rocm_sdma.SdmaCopyEngines
        else:
            raise CopyEngineUnavailable("PP copy engines need CUDA or ROCm")
        self.engines = engines
        if (
            parallel_config.tensor_parallel_size != 1
            or parallel_config.prefill_context_parallel_size != 1
        ):
            raise CopyEngineUnavailable("PP copy engines need TP=1 and PCP=1")
        if not all(in_the_same_node_as(pp_group.cpu_group, source_rank=0)):
            raise CopyEngineUnavailable("PP copy engines need one node")

        # Anything that can differ between ranks runs inside `step`, and the
        # ranks share its outcome before going on, so they all agree.
        rank, world = pp_group.rank_in_group, pp_group.world_size
        owned: list[int] = []  # Buffers this rank allocated.
        opened: list[int] = []  # Peer buffers this rank mapped.

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
            if errors:
                # Best effort: every rank must still raise the same error.
                with contextlib.suppress(Exception):
                    for p in opened:
                        _close_peer(p)
                    for p in owned:
                        ops.free_shared_buffer(p)
                raise CopyEngineUnavailable(
                    f"PP copy engines are not usable: {errors[0]}"
                )
            return [o for o, _ in gathered]

        def allocate_recv() -> tuple[Any, ...]:
            layout = _layout(persistent) if persistent is not None else {}
            device = torch.accelerator.current_device_index()
            engines.check_device(device)
            # Zeroed before the handle is shared.
            ptr, handle = ops.allocate_shared_buffer_and_handle(
                _DATA_OFFSET + _nbytes(layout)
            )
            owned.append(ptr)
            phys = current_platform.visible_device_id_to_physical_device_id(device)
            props = torch.cuda.get_device_properties(device)
            pci = (props.pci_bus_id, props.pci_device_id)
            return ptr, handle, layout, phys, pci

        ranks = step(allocate_recv)
        ptr, _, layout, _, _ = ranks[rank]

        def connect() -> tuple[Any, ...]:
            if not current_platform.is_fully_connected([r[3] for r in ranks]):
                raise RuntimeError("PP GPUs are not all directly connected")
            prev_ptr = next_ptr = send_ptr = None
            copier: _CudaCopyEngines | SdmaCopyEngines | None = None
            if not pp_group.is_first_rank:
                prev_ptr = ops.open_mem_handle(ranks[rank - 1][1])
                opened.append(prev_ptr)
            if not pp_group.is_last_rank:
                next_ptr = ops.open_mem_handle(ranks[rank + 1][1])
                opened.append(next_ptr)
                # Outside the torch allocator, so memory profiling counts it.
                send_ptr, _ = ops.allocate_shared_buffer_and_handle(
                    _nbytes(ranks[rank + 1][2])
                )
                owned.append(send_ptr)
                copier = engines(
                    send_ptr,
                    next_ptr + _DATA_OFFSET,
                    next_ptr,
                    ptr + _ENGINE_OFFSET,
                    ranks[rank + 1][4],
                )
            return prev_ptr, send_ptr, copier

        prev_ptr, send_ptr, self.copier = step(connect)[rank]

        if not pp_group.is_first_rank:
            assert persistent is not None and prev_ptr is not None
            self.src = rank - 1
            self.persistent = persistent
            self.recv_layout = layout
            self.recv_buf = _device_bytes(ptr + _DATA_OFFSET, _nbytes(layout))
            self.full = ptr
            self.peer_empty = prev_ptr + 8
        if not pp_group.is_last_rank:
            assert send_ptr is not None and self.copier is not None
            self.dst = rank + 1
            self.send_layout = ranks[rank + 1][2]
            self.send_buf = _device_bytes(send_ptr, _nbytes(self.send_layout))
            self.empty = ptr + 8
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
            self.engines.wait_flag(stream, self.full, meta["seq"])
            for key, view in src.items():
                out[key].copy_(view)
            self.engines.write_flag(stream, self.peer_empty, meta["seq"])

        return AsyncIntermediateTensors(out, comm_postprocess=[copy_in])

    def send(self, tensors: dict, pp_group: "GroupCoordinator") -> None:
        rows = {k: t.shape[0] for k, t in tensors.items()}
        for key, t in tensors.items():
            r = self.send_layout.get(key)
            if r is None or (t.dtype, tuple(t.shape[1:])) != (r.dtype, r.row_shape):
                raise RuntimeError(
                    f"PP send {key!r} {t.dtype} {tuple(t.shape)} does not match "
                    "the next stage's intermediate tensors"
                )
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
        self.engines.wait_flag(self.stream, self.empty, self.seq - 1)
        self.copier.copy(self.stream, self.send_layout, rows, self.seq)
        self.peer_done = torch.cuda.Event()
        self.peer_done.record(self.stream)

        # gloo sends complete only in wait(), which blocks until the peer
        # receives. Step k+1's copy waited for empty >= k, so k was received.
        self.pending.append((meta_work, self.peer_done))
        while len(self.pending) > 1 and self.pending[1][1].query():
            self.pending.popleft()[0].wait()
