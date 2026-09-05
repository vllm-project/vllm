# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import mmap
import os
import socket
import struct
import tempfile
import time
import uuid

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

from .xpu_communicator import XpuCommunicator

logger = init_logger(__name__)

try:
    from vllm_xpu_kernels import p2p as xpu_p2p
except ImportError:
    # A vllm_xpu_kernels release predating the p2p ops; the paths below then
    # degrade to oneCCL like any other unavailable-path case.
    xpu_p2p = None  # type: ignore[assignment]

_IPC_OPS = (
    "xpu_ipc_export_handle",
    "xpu_ipc_release_handle",
    "xpu_ipc_open_handle",
    "xpu_ipc_close_handle",
    "xpu_p2p_memcpy",
    "xpu_p2p_queue_sync",
)


def _ops_available(*names: str) -> bool:
    """Whether this vllm_xpu_kernels build registers all of `names`."""
    return xpu_p2p is not None and all(
        hasattr(torch.ops._xpu_C, name) for name in names
    )


class XpuP2pCommunicator(XpuCommunicator):
    """Host-synchronized Level Zero IPC all-reduce for 2-rank single-node TP.

    oneCCL costs 85-91 Level Zero API calls per collective, a fixed
    per-invocation overhead that dominates decode-sized messages. Instead,
    each rank exposes a device staging buffer to its peer via a Level Zero
    IPC handle (a dma-buf fd passed once at init over a unix socket with
    SCM_RIGHTS) and reads the peer's buffer directly over PCIe: measured on
    2x Arc B70, a 10 KiB bf16 all-reduce drops from 56.0us (oneCCL) to
    17.2us. The per-call handshake runs on the host: publish a sequence
    number in a small /dev/shm control page, spin until the peer catches
    up. There is no Python API for cross-process XPU memory sharing, so the
    Level Zero interop lives in vllm_xpu_kernels as the xpu_ipc_* ops.

    This class is the fallback layer beneath XpuP2pDevCommunicator (which
    moves the handshake into a device kernel) and also serves messages too
    large for the device kernel's data path. If IPC setup fails on any
    rank, all_reduce quietly degrades to oneCCL; other collectives always
    use oneCCL.
    """

    # Only measured up to 1 MiB (54.1us vs 294.8us oneCCL). Larger
    # messages are unmeasured, so hand them to oneCCL.
    MAX_BYTES = 1 << 20

    # A single two-operand add rounds once, which is bit-identical to fp32
    # accumulation for these dtypes; integer all-reduces (and fp64 etc.)
    # fall back to oneCCL rather than risk rounding.
    _SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

    # Control page: one page of /dev/shm holding a uint64 sequence number
    # per rank, each on its own cache line. Signaling only - no data is
    # staged through host memory.
    _CTRL_BYTES = 4096
    _SEQ_STRIDE = 64

    # The peer normally arrives within microseconds (ranks run the same
    # collective sequence), so spin first for minimum latency. Past 1ms
    # something is off (peer descheduled, GPU hiccup): yield instead of
    # burning the core. Past the timeout the peer is almost certainly dead
    # or diverged - raising beats a silent deadlock.
    _SPIN_S = 0.001
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
        self._p2p_ready = False
        self._mm: mmap.mmap | None = None
        self._peer_alloc = None
        try:
            self._open_p2p()
        except Exception:
            logger.warning(
                "Failed to set up p2p all-reduce for %s, using oneCCL",
                unique_name,
                exc_info=True,
            )
            self._close_p2p()
        if self._p2p_ready:
            logger.info("Using Level Zero IPC p2p all-reduce for %s", unique_name)

    def _all_ranks_ok(self, ok: bool) -> bool:
        oks = [None] * self.world_size
        dist.all_gather_object(oks, ok, group=self.cpu_group)
        return all(oks)

    def _open_p2p(self) -> None:
        from torch.distributed.distributed_c10d import _world

        if _world.pg_map.get(self.cpu_group, None) is None:
            # Stateless groups cannot run the object broadcasts below.
            logger.debug(
                "p2p all-reduce disabled for %s: stateless process group",
                self.unique_name,
            )
            return
        if self.world_size != 2:
            # The double-buffered protocol below is written (and validated)
            # for exactly two ranks.
            logger.debug(
                "p2p all-reduce disabled for %s: world_size=%d, only 2 is supported",
                self.unique_name,
                self.world_size,
            )
            return

        from vllm.distributed.parallel_state import in_the_same_node_as

        if not all(in_the_same_node_as(self.cpu_group, source_rank=0)):
            logger.debug(
                "p2p all-reduce disabled for %s: ranks span multiple nodes",
                self.unique_name,
            )
            return
        if not os.access("/dev/shm", os.W_OK):
            logger.debug(
                "p2p all-reduce disabled for %s: /dev/shm is not writable",
                self.unique_name,
            )
            return
        if not _ops_available(*_IPC_OPS):
            # Every rank runs the same wheel, so this is as symmetric as the
            # checks above and needs no all-ranks agreement to return on.
            logger.debug(
                "p2p all-reduce disabled for %s: this vllm_xpu_kernels build "
                "does not provide the Level Zero IPC ops",
                self.unique_name,
            )
            return
        self._open_ctrl_page()

        ok = False
        try:
            # Every op below enqueues on the queue behind torch's current
            # stream, so they order against torch ops with no extra
            # synchronization and no queue for this class to hold.
            #
            # Fixed device buffers make the IPC handle exchange a one-time
            # cost; activations are staged into them per call, one slot per
            # sequence parity.
            self._stage = torch.empty(
                2 * self.MAX_BYTES + self._extra_shared_bytes(),
                dtype=torch.uint8,
                device=self.device,
            )
            self._recv = torch.empty(
                self.MAX_BYTES, dtype=torch.uint8, device=self.device
            )
            handle_t, fd, offset = xpu_p2p.export_handle(self._stage.data_ptr())
            handle = handle_t.numpy().tobytes()
            ok = True
        except Exception:
            logger.warning(
                "Level Zero IPC setup failed for %s",
                self.unique_name,
                exc_info=True,
            )
        if not self._all_ranks_ok(ok):
            return

        # Exchange (handle, offset, dma-buf fd) over a unix socket: the fd
        # inside a ze_ipc_mem_handle_t is process-local and must travel via
        # SCM_RIGHTS to be valid on the other side.
        if self.rank_in_group == 0:
            spath: list[str | None] = [
                os.path.join(
                    tempfile.gettempdir(),
                    f"vllm_xpu_p2p_{os.getpid()}_"
                    f"{self.unique_name.replace(':', '_')}.sock",
                )
            ]
        else:
            spath = [None]
        dist.broadcast_object_list(spath, src=self.ranks[0], group=self.cpu_group)
        sock_path = spath[0]
        assert sock_path is not None
        # Bounded like every other wait on the peer: a rank that dies
        # between the barrier and the exchange must raise here (degrading
        # to oneCCL via the caller), not hang this rank forever.
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
                # accept() does not inherit the listener's timeout
                conn.settimeout(self._TIMEOUT_S)
                socket.send_fds(conn, [struct.pack("<Q", offset) + handle], [fd])
                data, fds, _, _ = socket.recv_fds(conn, 1024, 1)
        finally:
            if srv is not None:
                srv.close()
                os.unlink(sock_path)

        ok = False
        try:
            peer_off = struct.unpack_from("<Q", data)[0]
            self._peer_fd = fds[0]
            # bytearray: torch.frombuffer warns on the read-only buffer a
            # bytes object exposes.
            peer_ptr = xpu_p2p.open_handle(
                torch.frombuffer(bytearray(data[8:]), dtype=torch.uint8),
                fds[0],
                peer_off,
            )
            self._peer_alloc = peer_ptr - peer_off  # base, for close_handle
            self._peer_stage = peer_ptr
            ok = True
        except Exception:
            logger.warning(
                "Level Zero IPC handshake failed for %s",
                self.unique_name,
                exc_info=True,
            )
        # Confirm both sides opened before the smoke test: it spins on the
        # peer's sequence number, and a dead peer would mean a 60s stall.
        if not self._all_ranks_ok(ok):
            self._close_p2p()
            return

        # Both ranks hold a live mapping now, which is the earliest point
        # this can run: the release drops the export reference the peer's
        # open resolves against. Without it the exporter keeps its dma-buf
        # fd and the driver's export bookkeeping for the life of the
        # process. The driver may close that fd here, so `fd` is never
        # closed separately. Cleanup only - a failure leaves a bounded leak
        # rather than a broken path, so it is logged, not degraded on.
        try:
            xpu_p2p.release_handle(handle_t)
        except Exception:
            logger.warning(
                "Level Zero IPC handle release failed for %s",
                self.unique_name,
                exc_info=True,
            )

        # Smoke-test the actual data path before trusting it: opening the
        # handle can succeed even when device-to-device access does not
        # work.
        ok = False
        try:
            t = torch.full((64,), float(self.rank_in_group + 1), device=self.device)
            ok = bool((self._p2p_all_reduce(t) == 3.0).all())
        except Exception:
            logger.warning(
                "Level Zero IPC p2p smoke test failed for %s",
                self.unique_name,
                exc_info=True,
            )
        if self._all_ranks_ok(ok):
            self._p2p_ready = True
        else:
            self._close_p2p()

    def _open_ctrl_page(self) -> None:
        # Rank 0 picks a name unique to this engine (pid + random) so
        # multiple vLLM instances on one host never collide.
        if self.rank_in_group == 0:
            name: list[str | None] = [
                f"vllm_xpu_p2p_{os.getpid()}_{uuid.uuid4().hex[:8]}_"
                f"{self.unique_name.replace(':', '_')}"
            ]
        else:
            name = [None]
        dist.broadcast_object_list(name, src=self.ranks[0], group=self.cpu_group)
        shm_name = name[0]
        assert shm_name is not None
        path = os.path.join("/dev/shm", shm_name)
        if self.rank_in_group == 0:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, self._CTRL_BYTES)
        dist.barrier(group=self.cpu_group)
        if self.rank_in_group != 0:
            fd = os.open(path, os.O_RDWR)
        self._mm = mmap.mmap(fd, self._CTRL_BYTES)
        os.close(fd)
        dist.barrier(group=self.cpu_group)
        # Both ranks hold the mapping now, so unlink immediately: the page
        # lives until the mappings are gone, and /dev/shm stays clean even
        # if a process is SIGKILLed.
        if self.rank_in_group == 0:
            os.unlink(path)
        self._seq = 0
        self._my_seq_off = self.rank_in_group * self._SEQ_STRIDE
        self._peer_seq_off = (1 - self.rank_in_group) * self._SEQ_STRIDE

    def _extra_shared_bytes(self) -> int:
        """Extra bytes to append to the exported staging region.

        Zero here; the device-sync variant appends its own slots and signal
        pages, sized from what its kernels report.
        """
        return 0

    def _wait_peer(self, seq: int) -> None:
        mm = self._mm
        assert mm is not None  # only called while _p2p_ready
        off = self._peer_seq_off
        if struct.unpack_from("<Q", mm, off)[0] >= seq:
            return
        start = time.monotonic()
        while struct.unpack_from("<Q", mm, off)[0] < seq:
            waited = time.monotonic() - start
            if waited > self._TIMEOUT_S:
                raise RuntimeError(
                    f"p2p all-reduce ({self.unique_name}): peer rank did not "
                    f"reach sequence {seq} within {self._TIMEOUT_S}s"
                )
            if waited > self._SPIN_S:
                os.sched_yield()

    def _p2p_all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # Sequence parity double-buffers the staging slots, so no release
        # barrier is needed: before this rank can reuse a parity slot (op
        # s+2), the peer must have signalled s+1, which it only does after
        # its op s - including every read of this rank's slot - completed
        # on its in-order queue.
        seq = self._seq + 1
        parity = seq & 1
        nbytes = input_.nbytes
        off = parity * self.MAX_BYTES

        my_slot = self._stage[off : off + nbytes].view(input_.dtype)
        my_slot.view(input_.shape).copy_(input_)
        # Host-synchronize before signaling so the peer's read sees the
        # staged data. The op waits on the current stream's queue itself,
        # unlike torch.xpu.synchronize(), which adds ~19us per call.
        xpu_p2p.queue_sync()
        mm = self._mm
        assert mm is not None  # only called while _p2p_ready
        struct.pack_into("<Q", mm, self._my_seq_off, seq)
        self._wait_peer(seq)

        xpu_p2p.memcpy(self._recv.data_ptr(), self._peer_stage + off, nbytes)
        peer_slot = self._recv[:nbytes].view(input_.dtype)
        output = torch.empty_like(input_)
        # In-order queue: the add is sequenced after the p2p copy. A single
        # two-operand add rounds once, which is bit-identical to fp32
        # accumulation for the supported dtypes.
        torch.add(
            my_slot.view(input_.shape),
            peer_slot.view(input_.shape),
            out=output,
        )
        self._seq = seq
        return output

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if (
            self._p2p_ready
            and input_.dtype in self._SUPPORTED_DTYPES
            and input_.nbytes <= self.MAX_BYTES
        ):
            return self._p2p_all_reduce(input_)
        return super().all_reduce(input_)

    def _close_p2p(self) -> None:
        self._p2p_ready = False
        if self._peer_alloc is not None:
            try:
                xpu_p2p.close_handle(self._peer_alloc)
                os.close(self._peer_fd)
            except Exception:
                pass
            self._peer_alloc = None
        if self._mm is not None:
            self._mm.close()
            self._mm = None

    def destroy(self):
        self._close_p2p()
        super().destroy()
