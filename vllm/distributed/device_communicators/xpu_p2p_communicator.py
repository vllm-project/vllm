# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os
import socket
import struct
import tempfile
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

from .xpu_communicator import XpuCommunicator

logger = init_logger(__name__)

try:
    from vllm_xpu_kernels import p2p as xpu_p2p
except ImportError:
    # A vllm_xpu_kernels release without the p2p ops; all_reduce stays on oneCCL.
    xpu_p2p = None  # type: ignore[assignment]


class XpuP2pCommunicator(XpuCommunicator):
    """Level Zero IPC all-reduce for small messages in 2-rank single-node TP.

    Each rank maps its peer's staging region once, over a Level Zero IPC
    handle whose dma-buf fd travels over a unix socket. Per call, one kernel
    from vllm_xpu_kernels stages, handshakes in device memory and reduces; the
    host only enqueues, so the call can be recorded into an XPU graph. Larger
    inputs, other collectives and groups where setup fails use oneCCL.
    """

    # Crossover against oneCCL with its SYCL kernels on 2x Arc Pro B70
    # (Qwen3.8-27B TP=2 decode under XPU graph): this path is faster up to
    # 40 KiB, level at 80 KiB and slower beyond. Also the staging slot size.
    _MAX_BYTES = 64 * 1024

    # A single two-operand add rounds once, which is bit-identical to fp32
    # accumulation for these dtypes.
    _SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

    # Bounds every wait on the peer during setup, so a rank that dies raises
    # here instead of hanging the other.
    _TIMEOUT_S = 60.0

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        use_all2all: bool = False,
    ):
        super().__init__(
            cpu_group, device, device_group, unique_name, use_all2all=use_all2all
        )
        self._ready = False
        self._peer_base: int | None = None
        try:
            self._open()
        except Exception:
            logger.warning(
                "Failed to set up p2p all-reduce for %s, using oneCCL",
                unique_name,
                exc_info=True,
            )
            self._close()
        if self._ready:
            logger.info("Using Level Zero IPC p2p all-reduce for %s", unique_name)

    def _agree(self, step: str, fn: Callable[[], Any]) -> tuple[bool, Any]:
        # Every rank must learn of a failure anywhere before the next step, or
        # its peer would wait on a socket or a device handshake forever.
        ok, result = False, None
        try:
            result, ok = fn(), True
        except Exception:
            logger.warning(
                "p2p all-reduce %s failed for %s",
                step,
                self.unique_name,
                exc_info=True,
            )
        oks: list[bool | None] = [None] * self.world_size
        dist.all_gather_object(oks, ok, group=self.cpu_group)
        return all(oks), result

    def _open(self) -> None:
        from torch.distributed.distributed_c10d import _world

        from vllm.distributed.parallel_state import in_the_same_node_as

        # The first three are the same on every rank, so the collective in the
        # last one runs on all ranks or none.
        if (
            xpu_p2p is None
            or self.world_size != 2
            or _world.pg_map.get(self.cpu_group, None) is None
            or not all(in_the_same_node_as(self.cpu_group, source_rank=0))
        ):
            logger.debug(
                "p2p all-reduce needs vllm_xpu_kernels with the p2p ops and a "
                "2-rank single-node process group; %s uses oneCCL",
                self.unique_name,
            )
            return

        def export():
            # A zeroed allocation of its own: the peer writes into it, which
            # must not share memory torch's caching allocator hands out.
            with torch.xpu.device(self.device):
                self._region = xpu_p2p.alloc_region(self._MAX_BYTES)
            return xpu_p2p.export_handle(self._region.data_ptr())

        ok, exported = self._agree("export", export)
        if not ok:
            return
        handle, fd, offset = exported
        payload = struct.pack("<Q", offset) + handle.numpy().tobytes()
        data, self._peer_fd = self._exchange(payload, fd)

        def open_peer():
            peer_off = struct.unpack_from("<Q", data)[0]
            peer = xpu_p2p.open_handle(
                torch.frombuffer(bytearray(data[8:]), dtype=torch.uint8),
                self._peer_fd,
                peer_off,
            )
            self._peer_base = peer - peer_off
            return peer

        ok, peer = self._agree("handshake", open_peer)
        if not ok:
            self._close()
            return

        # Both ranks hold a mapping now, so the export reference has done its
        # job. The driver may close the exported fd here, so it is never
        # closed separately.
        try:
            xpu_p2p.release_handle(handle)
        except Exception:
            logger.warning(
                "Level Zero IPC handle release failed for %s",
                self.unique_name,
                exc_info=True,
            )

        self._my_region = xpu_p2p.as_fptr(self._region.data_ptr())
        self._peer_region = xpu_p2p.as_fptr(peer)

        # Opening a handle can succeed where device-to-device access does not.
        def smoke_test():
            t = torch.full((64,), float(self.rank_in_group + 1), device=self.device)
            out = self._p2p_all_reduce(t)
            torch.xpu.synchronize()
            if not bool((out == 3.0).all()):
                raise RuntimeError(f"expected 3.0, got {out[:4].tolist()}")

        ok, _ = self._agree("smoke test", smoke_test)
        if ok:
            self._ready = True
        else:
            self._close()

    def _exchange(self, payload: bytes, fd: int) -> tuple[bytes, int]:
        # The dma-buf fd is process-local, so it has to travel as a real file
        # descriptor over a unix socket.
        if self.rank_in_group == 0:
            path: list[str | None] = [
                os.path.join(
                    tempfile.gettempdir(),
                    f"vllm_xpu_p2p_{os.getpid()}_"
                    f"{self.unique_name.replace(':', '_')}.sock",
                )
            ]
        else:
            path = [None]
        dist.broadcast_object_list(path, src=self.ranks[0], group=self.cpu_group)
        sock_path = path[0]
        assert sock_path is not None
        srv = None
        try:
            if self.rank_in_group == 0:
                srv = socket.socket(socket.AF_UNIX)
                srv.settimeout(self._TIMEOUT_S)
                srv.bind(sock_path)
                srv.listen(1)
                dist.barrier(group=self.cpu_group)
                conn, _ = srv.accept()
            else:
                dist.barrier(group=self.cpu_group)
                conn = socket.socket(socket.AF_UNIX)
                conn.settimeout(self._TIMEOUT_S)
                conn.connect(sock_path)
            with conn:
                # accept() does not inherit the listener's timeout.
                conn.settimeout(self._TIMEOUT_S)
                socket.send_fds(conn, [payload], [fd])
                data, fds, _, _ = socket.recv_fds(conn, 1024, 1)
        finally:
            if srv is not None:
                srv.close()
                os.unlink(sock_path)
        return data, fds[0]

    def _p2p_all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        input_ = input_.contiguous()
        output = torch.empty_like(input_)
        torch.ops._xpu_C.xpu_p2p_all_reduce(
            output, input_, self._my_region, self._peer_region, self._MAX_BYTES
        )
        return output

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if (
            self._ready
            and input_.dtype in self._SUPPORTED_DTYPES
            and input_.nbytes <= self._MAX_BYTES
        ):
            return self._p2p_all_reduce(input_)
        return super().all_reduce(input_)

    def _close(self) -> None:
        self._ready = False
        if self._peer_base is not None:
            with contextlib.suppress(Exception):
                xpu_p2p.close_handle(self._peer_base)
                os.close(self._peer_fd)
            self._peer_base = None

    def destroy(self):
        self._close()
        super().destroy()
