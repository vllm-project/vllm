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
"""

from collections import deque
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from vllm.distributed.pp_transport import PPTransport
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator

# Flags at bytes 0 (full) and 8 (empty) of each shared buffer, data from here.
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


class CopyEngineTransport(PPTransport):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        pp_group: "GroupCoordinator",
        persistent: dict[str, torch.Tensor] | None,
    ) -> None:
        """Collective over the PP group. Raises on every rank alike if copy
        engines are not usable."""
        from vllm import _custom_ops as ops
        from vllm.distributed.parallel_state import in_the_same_node_as
        from vllm.platforms import current_platform

        # The same on every rank, so these raise before any collective.
        parallel_config = vllm_config.parallel_config
        if not current_platform.is_cuda():
            raise RuntimeError("PP copy engines need CUDA")
        if (
            parallel_config.tensor_parallel_size != 1
            or parallel_config.prefill_context_parallel_size != 1
        ):
            raise RuntimeError("PP copy engines need TP=1 and PCP=1")
        import cuda.bindings.driver as cu

        if not all(in_the_same_node_as(pp_group.cpu_group, source_rank=0)):
            raise RuntimeError("PP copy engines need all PP stages on one node")

        rank, world = pp_group.rank_in_group, pp_group.world_size
        layout = _layout(persistent) if persistent is not None else {}
        device = torch.accelerator.current_device_index()
        ptr = handle = error = None
        try:
            attr = cu.CUdevice_attribute
            memops = attr.CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS
            dev = _check(cu.cuDeviceGet(device))
            if not _check(cu.cuDeviceGetAttribute(memops, dev)):
                raise RuntimeError(f"GPU {device} lacks 64-bit stream memory ops")
            # Zeroed before the handle is shared.
            ptr, handle = ops.allocate_shared_buffer_and_handle(
                _DATA_OFFSET + _nbytes(layout)
            )
        except Exception as e:
            error = repr(e)
        phys = current_platform.visible_device_id_to_physical_device_id(device)
        ranks: list[Any] = [None] * world
        torch.distributed.all_gather_object(
            ranks, (handle, layout, error, phys), group=pp_group.cpu_group
        )
        errors = [r[2] for r in ranks if r[2]]
        if not current_platform.is_fully_connected([r[3] for r in ranks]):
            errors.append("PP GPUs are not all NVLink-connected")
        if errors:
            raise RuntimeError(f"PP copy engines are not usable: {errors[0]}")
        assert ptr is not None

        if not pp_group.is_first_rank:
            assert persistent is not None
            prev_ptr = ops.open_mem_handle(ranks[rank - 1][0])
            self.src = rank - 1
            self.persistent = persistent
            self.recv_layout = layout
            self.recv_buf = _device_bytes(ptr + _DATA_OFFSET, _nbytes(layout))
            self.full = ptr
            self.peer_empty = prev_ptr + 8
        if not pp_group.is_last_rank:
            next_ptr = ops.open_mem_handle(ranks[rank + 1][0])
            self.dst = rank + 1
            self.send_layout = ranks[rank + 1][1]
            nbytes = _nbytes(self.send_layout)
            # Outside the torch allocator, so memory profiling counts it as is.
            send_ptr, _ = ops.allocate_shared_buffer_and_handle(nbytes)
            self.send_buf = _device_bytes(send_ptr, nbytes)
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
        when ``.tensors`` is first read, so GPU work queued before then (input
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
            _write_flag(stream, self.peer_empty, meta["seq"])

        return AsyncIntermediateTensors(out, comm_postprocess=[copy_in])

    def send(self, tensors: dict, pp_group: "GroupCoordinator") -> None:
        from cuda.bindings import runtime as cudart

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
        _write_flag(self.stream, self.peer_full, self.seq)

        # gloo sends complete only in wait(), which blocks until the peer
        # receives. Step k+1's copy waited for empty >= k, so k was received.
        self.pending.append((meta_work, self.peer_done))
        while len(self.pending) > 1 and self.pending[1][1].query():
            self.pending.popleft()[0].wait()
