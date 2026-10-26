# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP transport on CUDA copy engines, synchronized on the GPU (one node).

Each PP rank allocates its flags and recv region in one CUDA IPC buffer (the
custom all-reduce ops), plus a local send region if it sends. Per stage
boundary, step N:

- sender: once the previous peer copy is done, copy the outputs into its send
  region; on a side stream, wait ``empty >= N - 1``, peer-copy into the
  receiver's recv region and write ``full = N``.
- receiver: wait ``full >= N``, copy the recv region into the persistent
  intermediate tensors and write ``empty = N``.

Each flag lives on the GPU that waits on it; the peer writes it with stream
memory ops. Neither SMs nor the CPU wait on the transfer. Metadata goes over
the CPU group.

Flags are awaited with cuStreamWaitValue64(GEQ), without
CU_STREAM_WAIT_VALUE_FLUSH (unsupported on x86 hosts). FLUSH's documented use
case is exactly this data-then-flag pattern, so dropping it rests on
precedent and measurement rather than a doc guarantee: NCCL ships the same
no-FLUSH scheme for zero-SM RMA, NVLink-gated like setup below
(src/rma/rma_ce.cc, src/ce_coll.cc), and per-tensor checksum sweeps through
saturation on x86 B300 showed zero mismatches.
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


def _check(result) -> Any:
    err, *rest = result if isinstance(result, tuple) else (result,)
    if int(err) != 0:
        raise RuntimeError(f"CUDA call failed: {err}")
    return rest[0] if len(rest) == 1 else rest


class _Region(NamedTuple):
    """One key's region in the send and recv buffers."""

    offset: int
    row_bytes: int
    max_rows: int
    dtype: torch.dtype
    row_shape: tuple[int, ...]


def _require_stream_memops(device: int) -> None:
    from cuda.bindings import driver as cu

    attr = cu.CUdevice_attribute
    dev = _check(cu.cuDeviceGet(device))
    name = "CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS"
    if not _check(cu.cuDeviceGetAttribute(getattr(attr, name), dev)):
        raise RuntimeError(f"device {device} lacks {name}")


def _wait_flag(stream: torch.cuda.Stream, addr: int, at_least: int) -> None:
    from cuda.bindings import driver as cu

    flags = cu.CUstreamWaitValue_flags
    _check(
        cu.cuStreamWaitValue64(
            cu.CUstream(stream.cuda_stream),
            cu.CUdeviceptr(addr),
            cu.cuuint64_t(at_least),
            int(flags.CU_STREAM_WAIT_VALUE_GEQ),
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


def _views(
    buf: torch.Tensor, layout: dict[str, _Region], rows: dict[str, int]
) -> dict[str, torch.Tensor]:
    """The first ``rows[key]`` rows of each key's region of ``buf``."""
    out = {}
    for key, n in rows.items():
        r = layout[key]
        region = buf[r.offset : r.offset + n * r.row_bytes]
        out[key] = region.view(r.dtype).view(n, *r.row_shape)
    return out


def _device_bytes(ptr: int, nbytes: int) -> torch.Tensor:
    cai = dict(shape=(nbytes,), typestr="|u1", data=(ptr, False), version=3)
    return torch.as_tensor(SimpleNamespace(__cuda_array_interface__=cai), device="cuda")


def _nbytes(layout: dict[str, _Region]) -> int:
    return max(r.offset + r.max_rows * r.row_bytes for r in layout.values())


class CopyEngineTransport(PPTransport):
    """Sends to the next stage and receives from the previous one, whichever
    this rank has."""

    def __init__(
        self,
        rank: int,
        buf: int,
        persistent: dict[str, torch.Tensor] | None,
        recv_layout: dict[str, _Region] | None,
        prev_buf: int | None,
        send_layout: dict[str, _Region] | None,
        next_buf: int | None,
    ) -> None:
        from vllm import _custom_ops as ops

        if recv_layout is not None:
            assert persistent is not None and prev_buf is not None
            self.src = rank - 1
            self.persistent = persistent
            self.recv_layout = recv_layout
            self.recv_buf = _device_bytes(buf + 256, _nbytes(recv_layout))
            self.full = buf
            self.peer_empty = prev_buf + 8
        if send_layout is not None:
            assert next_buf is not None
            self.dst = rank + 1
            self.send_layout = send_layout
            nbytes = _nbytes(send_layout)
            # Outside the torch allocator, so memory profiling counts it as is.
            send_ptr, _ = ops.allocate_shared_buffer_and_handle(nbytes)
            self.send_buf = _device_bytes(send_ptr, nbytes)
            self.empty = buf + 8
            self.peer_recv_buf = next_buf + 256
            self.peer_full = next_buf
            self.stream = torch.cuda.Stream()
            self.seq = 0
            # The send buffer's last peer copy.
            self.peer_done: torch.cuda.Event | None = None
            # Metadata sends with their step's peer_done.
            self._pending: deque[tuple[Any, torch.cuda.Event]] = deque()

    def recv(self, pp_group: "GroupCoordinator") -> IntermediateTensors:
        """This step's tensors, as views of the persistent tensors."""
        meta = pp_group.recv_object(src=self.src)
        stream = torch.cuda.current_stream()
        _wait_flag(stream, self.full, meta["seq"])
        out: dict[str, Any] = dict(meta["others"])
        received = _views(self.recv_buf, self.recv_layout, meta["rows"])
        for key, src in received.items():
            out[key] = self.persistent[key][: src.shape[0]].copy_(src)
        _write_flag(stream, self.peer_empty, meta["seq"])
        return IntermediateTensors(out)

    def send(self, tensors: dict[str, Any], pp_group: "GroupCoordinator") -> None:
        from cuda.bindings import runtime as cudart

        sent = {k: v for k, v in tensors.items() if isinstance(v, torch.Tensor)}
        others = {k: v for k, v in tensors.items() if not isinstance(v, torch.Tensor)}
        rows = {k: t.shape[0] for k, t in sent.items()}
        for key, t in sent.items():
            r = self.send_layout[key]
            if (
                t.dtype != r.dtype
                or tuple(t.shape[1:]) != r.row_shape
                or rows[key] > r.max_rows
            ):
                raise RuntimeError(
                    f"PP send {key!r} {t.dtype} {tuple(t.shape)} != its buffer"
                )
        self.seq += 1
        meta_work = pp_group.isend_object(
            dict(seq=self.seq, rows=rows, others=others), dst=self.dst
        )

        # Main stream: pack once the previous peer copy is done.
        main = torch.cuda.current_stream()
        if self.peer_done is not None:
            main.wait_event(self.peer_done)
        for key, dst in _views(self.send_buf, self.send_layout, rows).items():
            dst.copy_(sent[key])
        packed = torch.cuda.Event()
        packed.record(main)

        # Side stream: peer copy once the recv buffer is empty, then publish.
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
        self._pending.append((meta_work, self.peer_done))
        while len(self._pending) > 1 and self._pending[1][1].query():
            self._pending.popleft()[0].wait()

    @classmethod
    def create(
        cls,
        vllm_config: "VllmConfig",
        pp_group: "GroupCoordinator",
        persistent: dict[str, torch.Tensor] | None,
        profile_send: dict[str, Any] | None,
    ) -> tuple["CopyEngineTransport | None", str | None]:
        """Collective over the PP group; every rank returns a transport, or
        None and the same reason."""
        import vllm.envs as envs
        from vllm import _custom_ops as ops
        from vllm.distributed.device_communicators.all_reduce_utils import (
            gpu_p2p_access_check,
        )
        from vllm.distributed.parallel_state import in_the_same_node_as
        from vllm.platforms import current_platform

        # Same on every rank, so these return before any collective.
        parallel_config = vllm_config.parallel_config
        if not current_platform.is_cuda():
            return None, "not CUDA"
        if not vllm_config.use_v2_model_runner:
            return None, "needs the V2 model runner"
        if parallel_config.pipeline_parallel_size <= 1:
            return None, "no pipeline parallelism"
        if (
            parallel_config.tensor_parallel_size != 1
            or parallel_config.prefill_context_parallel_size != 1
        ):
            return None, "needs TP=1 and PCP=1"
        try:
            import cuda.bindings  # noqa: F401
        except ImportError:
            return None, "cuda-python is not installed"
        if not all(in_the_same_node_as(pp_group.cpu_group, source_rank=0)):
            return None, "PP stages span several nodes"

        rank, world = pp_group.rank_in_group, pp_group.world_size
        first, last = pp_group.is_first_rank, pp_group.is_last_rank

        def all_gather(obj: Any) -> list[Any]:
            out: list[Any] = [None] * world
            torch.distributed.all_gather_object(out, obj, group=pp_group.cpu_group)
            return out

        def can_p2p(src: int, dst: int) -> bool:
            """As custom all-reduce: trust the driver unless
            VLLM_SKIP_P2P_CHECK=0."""
            if envs.VLLM_SKIP_P2P_CHECK:
                visible = current_platform.logical_device_id_to_visible_device_id
                return torch.cuda.can_device_access_peer(visible(src), visible(dst))
            return gpu_p2p_access_check(src, dst)

        layout: dict[str, _Region] = {}
        nbytes = 0
        if not first and persistent is not None:
            for key, t in persistent.items():
                row_bytes = t[0].numel() * t.element_size()
                layout[key] = _Region(
                    nbytes, row_bytes, t.shape[0], t.dtype, tuple(t.shape[1:])
                )
                nbytes += (t.shape[0] * row_bytes + 255) // 256 * 256
        # The profile run's outputs: every send has these keys, dtypes and
        # row shapes.
        sent = None
        if not last and profile_send is not None:
            sent = {
                k: (t.dtype, tuple(t.shape[1:]))
                for k, t in profile_send.items()
                if isinstance(t, torch.Tensor)
            }
        # Flags (full at byte 0, empty at byte 8), then the recv region at
        # 256; zeroed before the handle is shared.
        ptr = handle = error = None
        try:
            _require_stream_memops(torch.accelerator.current_device_index())
            ptr, handle = ops.allocate_shared_buffer_and_handle(256 + nbytes)
        except Exception as e:
            error = repr(e)

        # Every rank sees every boundary (P2P from the same driver report or
        # probe cache), so all decide alike. Nothing is opened before that,
        # so a fallback only frees each rank's own buffer.
        phys = current_platform.visible_device_id_to_physical_device_id(
            torch.accelerator.current_device_index()
        )
        ranks = all_gather((handle, layout, sent, pp_group.local_rank, error, phys))
        errors = [r[4] for r in ranks if r[4]]
        # The no-FLUSH flag waits rest on NVLink's in-order remote-write
        # delivery (NCCL's zero-SM RMA path is likewise NVLink-only).
        if not current_platform.is_fully_connected([r[5] for r in ranks]):
            errors.append("PP group's GPUs are not NVLink-connected")
        for (_, _, sent_r, dev, _, _), (_, next_layout, _, next_dev, _, _) in zip(
            ranks, ranks[1:]
        ):
            expected = {k: (r.dtype, r.row_shape) for k, r in next_layout.items()}
            if sent_r is None:
                errors.append("a sender has no profile-run outputs")
            elif sent_r != expected:
                errors.append(f"send layout {sent_r} != receive layout {expected}")
            # Unconditional: the probe's first call is a world-group barrier.
            if not (can_p2p(dev, next_dev) and can_p2p(next_dev, dev)):
                errors.append(f"no P2P access between GPUs {dev} and {next_dev}")
        if errors:
            if ptr is not None:
                ops.free_shared_buffer(ptr)
            return None, errors[0]

        assert ptr is not None
        return (
            cls(
                rank,
                ptr,
                persistent,
                None if first else layout,
                None if first else ops.open_mem_handle(ranks[rank - 1][0]),
                None if last else ranks[rank + 1][1],
                None if last else ops.open_mem_handle(ranks[rank + 1][0]),
            ),
            None,
        )
