# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mirror worker slots of a multi-node CPU offload region onto the head node.

The tiering scheduler and its secondary tiers run in the engine core, which
shares a node with group rank 0 (the head node), and address only that node's
shared region. Workers on other nodes transfer between their GPU and a compact
node-local region that holds only their own slot. A store is reported only
after the worker's slot of every stored chunk is written into the head region,
and a load starts its DMA only after the worker's slot of every loaded chunk is
fetched from it. The scheduler counts a transfer complete once every worker
reports it, so a secondary tier never reads a chunk before all its slots
arrived.

Slot copies run on a background thread; loads go ahead of stores. A copy that
still fails after its retries is reported as a failed transfer. Before a failed
push is reported, a fence raises the rank's sequence on the head, so none of
its delayed attempts can overwrite a chunk the scheduler reassigned; if the
fence fails too, the store is never reported and the worker raises.

The head serves copies over ZMQ on its ``VLLM_HOST_IP`` and admits connections
only from the remote workers' addresses. Requests carry a per-engine token and
a per-client sequence number. None of this is cryptographic: the mirror needs
an isolated network.
"""

import contextlib
import hmac
import ipaddress
import itertools
import json
import queue
import secrets
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import zmq
from zmq.auth.thread import ThreadAuthenticator

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.node_topology import MirrorTopology
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

logger = init_logger(__name__)

_REQUEST_TIMEOUT_MS = 30_000
_REQUEST_ATTEMPTS = 3
# Each request moves at most this many slot bytes, which bounds the transient
# copies on both ends: host memory is shared with the GPU on unified-memory
# systems.
_MAX_REQUEST_BYTES = 32 * 1024 * 1024
_MAX_HEADER_BYTES = 1024 * 1024
_SEND_TIMEOUT_MS = 1_000
_RECEIVE_QUEUE = 8
_SHUTDOWN_TIMEOUT_S = 60.0
_OPS = ("put", "get", "ping", "fence")
_ZAP_DOMAIN = b"vllm-kv-offload-mirror"


@dataclass(frozen=True)
class SlotLayout:
    """Byte range of each worker slot inside every chunk row of a region."""

    num_chunks: int
    row_stride: int
    size: int

    @property
    def num_slots(self) -> int:
        return self.row_stride // self.size

    def rows(self, region: SharedOffloadRegion) -> np.ndarray:
        assert region.mmap_obj is not None
        flat = np.frombuffer(region.mmap_obj, dtype=np.uint8)
        return flat[: self.num_chunks * self.row_stride].reshape(
            self.num_chunks, self.row_stride
        )

    def read(self, rows: np.ndarray, chunk_ids: np.ndarray, slot: int) -> memoryview:
        start = slot * self.size
        return np.ascontiguousarray(rows[chunk_ids, start : start + self.size]).data

    def write(
        self, rows: np.ndarray, chunk_ids: np.ndarray, slot: int, payload
    ) -> None:
        start = slot * self.size
        data = np.frombuffer(payload, dtype=np.uint8)
        rows[chunk_ids, start : start + self.size] = data.reshape(
            len(chunk_ids), self.size
        )


def _bracket(host: str) -> str:
    return f"[{host}]" if ":" in host else host


def _zmq_endpoint(host: str, port: int | None = None) -> str:
    host = _bracket(host)
    return f"tcp://{host}" if port is None else f"tcp://{host}:{port}"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class RegionMirrorServer:
    """Serves slot reads and writes of the head region to remote workers.

    Only connections from ``allowed_hosts`` complete their handshake, so
    other hosts cannot queue messages. Requests carry an id, the engine token,
    a sequence number, an operation (``put``, ``get`` or ``ping``), the remote
    rank and chunk ids. A ``put`` older than one already applied for its rank
    is rejected, so a delayed retry cannot overwrite a chunk reassigned since.
    """

    def __init__(
        self,
        region: SharedOffloadRegion,
        layout: SlotLayout,
        host: str,
        token: str,
        remote_ranks: frozenset[int],
        *,
        allowed_hosts: frozenset[str],
        port: int = 0,
        max_request_bytes: int = _MAX_REQUEST_BYTES,
    ):
        if not allowed_hosts:
            raise ValueError("The region mirror needs the remote workers' hosts")
        self._rows: np.ndarray | None = layout.rows(region)
        self._layout = layout
        self._token = token.encode()
        self._max_chunks = max(1, max_request_bytes // layout.size)
        self._last_put_seq: dict[int, tuple[str, int]] = {}
        # Epochs of clients a rank has since replaced.
        self._retired_epochs: dict[int, set[str]] = {}
        self._rejected = 0
        self._rejected_logged_at = 0.0
        # Only ranks off the head node own slots that clients may move.
        self._remote_ranks = frozenset(
            rank for rank in remote_ranks if 0 < rank < layout.num_slots
        )
        self._context = zmq.Context()
        # ZAP admits connections by peer address before any message is read.
        self._auth = ThreadAuthenticator(self._context)
        self._auth.start()
        self._auth.allow(*allowed_hosts)
        try:
            self._socket = self._context.socket(zmq.ROUTER)
        except Exception:
            self._auth.stop()
            self._context.term()
            raise
        try:
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.setsockopt(zmq.ZAP_DOMAIN, _ZAP_DOMAIN)
            self._socket.setsockopt(zmq.RCVHWM, _RECEIVE_QUEUE)
            # A request carries at least one whole slot, however large.
            self._socket.setsockopt(
                zmq.MAXMSGSIZE,
                self._max_chunks * layout.size + _MAX_HEADER_BYTES,
            )
            if ":" in host:
                self._socket.setsockopt(zmq.IPV6, 1)
            if port:
                self._socket.bind(_zmq_endpoint(host, port))
                self.port = port
            else:
                self.port = self._socket.bind_to_random_port(_zmq_endpoint(host))
        except Exception:
            self._socket.close()
            self._auth.stop()
            self._context.term()
            raise
        self.address = _zmq_endpoint(host, self.port)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._serve, name="kv-offload-region-mirror", daemon=True
        )
        self._thread.start()

    def _handle(self, frames: list[bytes]) -> list[bytes]:
        if not frames or len(frames) > 2:
            raise ValueError("unexpected frame count")
        header, *payload = frames
        if len(header) > _MAX_HEADER_BYTES:
            raise ValueError("header too large")
        request = json.loads(header)
        request_id = str(request["id"]).encode()
        if not hmac.compare_digest(str(request["token"]).encode(), self._token):
            raise PermissionError("invalid token")
        op, rank, seq = request["op"], request["rank"], request["seq"]
        epoch = request.get("epoch")
        if op not in _OPS:
            raise ValueError(f"unknown operation {op!r}")
        # Head-node ranks write the head region themselves.
        if not _is_int(rank) or rank not in self._remote_ranks:
            raise ValueError(f"invalid rank {rank!r}")
        if not _is_int(seq):
            raise ValueError(f"invalid sequence number {seq!r}")
        if not isinstance(epoch, str):
            raise ValueError("missing client epoch")
        retired = self._retired_epochs.setdefault(rank, set())
        if epoch not in retired:
            # A restarted client starts a new epoch and its own sequence; the
            # late requests of the client it replaces are stale.
            last_epoch, _ = self._last_put_seq.setdefault(rank, (epoch, -1))
            if epoch != last_epoch:
                retired.add(last_epoch)
                self._last_put_seq[rank] = (epoch, -1)
        if op == "ping":
            return [request_id, b"ok"]
        last_epoch, last_seq = self._last_put_seq[rank]
        stale = epoch in retired or (epoch == last_epoch and seq <= last_seq)
        if op == "fence":
            # Raise the sequence so that older puts of this rank are rejected.
            if not stale:
                self._last_put_seq[rank] = (epoch, seq)
            return [request_id, b"ok"]
        chunks = request["chunks"]
        if (
            not isinstance(chunks, list)
            or not 0 < len(chunks) <= self._max_chunks
            or not all(_is_int(c) and 0 <= c < self._layout.num_chunks for c in chunks)
        ):
            raise ValueError("invalid chunk ids")
        chunk_ids = np.asarray(chunks, dtype=np.int64)
        assert self._rows is not None
        if op == "get":
            return [request_id, b"ok", self._layout.read(self._rows, chunk_ids, rank)]
        if len(payload) != 1 or len(payload[0]) != len(chunks) * self._layout.size:
            raise ValueError("payload size does not match the chunk ids")
        if stale:
            raise ValueError(f"stale put {seq} from rank {rank}")
        self._last_put_seq[rank] = (epoch, seq)
        self._layout.write(self._rows, chunk_ids, rank, payload[0])
        return [request_id, b"ok"]

    def _receive(self) -> tuple[bytes, list[bytes]]:
        """Receive one request, keeping at most a header and one payload frame.

        Frames past those are dropped; a message with more is rejected.
        """
        identity = self._socket.recv()
        frames: list[bytes] = []
        dropped = False
        while self._socket.getsockopt(zmq.RCVMORE):
            frame = self._socket.recv()
            if len(frames) < 2:
                frames.append(frame)
            else:
                dropped = True
        if dropped:
            frames = [frames[0], b"", b""]  # rejected as too many frames
        return identity, frames

    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while not self._stop.is_set():
            try:
                if not poller.poll(200):
                    continue
                identity, frames = self._receive()
            except zmq.ZMQError:
                if self._stop.is_set():
                    return
                logger.exception("KV offload region mirror receive failed")
                self._stop.wait(1.0)
                continue
            try:
                reply = self._handle(frames)
            except Exception as e:
                # At most one warning every 10 s, so rejected requests from an
                # admitted host cannot flood the log.
                self._rejected += 1
                now = time.monotonic()
                if now - self._rejected_logged_at >= 10.0:
                    logger.warning(
                        "Rejected %d KV offload region mirror request(s); last: %s",
                        self._rejected,
                        e,
                    )
                    self._rejected = 0
                    self._rejected_logged_at = now
                request_id = b""
                with contextlib.suppress(Exception):
                    request_id = str(json.loads(frames[0])["id"]).encode()
                reply = [request_id, b"error"]
            try:
                self._socket.send_multipart([identity, *reply])
            except zmq.ZMQError:
                logger.exception("KV offload region mirror reply failed")

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join()
        self._socket.close()
        self._auth.stop()
        self._context.term()
        self._rows = None


class RegionMirrorClient:
    """Copies one remote worker's slot between its region and the head region.

    The local region holds only this worker's slot (``local_layout``); in the
    head region the slot is the worker's rank. Copies are split into requests
    of at most ``max_request_bytes``. Each request is retried on a fresh
    socket after a timeout, and replies are matched by request id, so a late
    reply to an abandoned attempt is dropped. ``stop`` aborts waiting requests.
    Connections leave from ``source_host``, the address the server admits.
    """

    def __init__(
        self,
        region: SharedOffloadRegion,
        local_layout: SlotLayout,
        rank: int,
        address: str,
        token: str,
        *,
        timeout_ms: int = _REQUEST_TIMEOUT_MS,
        attempts: int = _REQUEST_ATTEMPTS,
        max_request_bytes: int = _MAX_REQUEST_BYTES,
        source_host: str | None = None,
    ):
        self._rows: np.ndarray | None = local_layout.rows(region)
        self._layout = local_layout
        self._rank = rank
        self._address = (
            address
            if source_host is None
            else f"tcp://{_bracket(source_host)}:0;{address.removeprefix('tcp://')}"
        )
        self._token = token
        self._timeout_ms = timeout_ms
        self._attempts = attempts
        self._chunks_per_request = max(1, max_request_bytes // local_layout.size)
        self._seq = itertools.count()
        self._epoch = uuid.uuid4().hex
        self._stopped = threading.Event()
        self._stop_reason = "is shutting down"
        self._context = zmq.Context()
        self._socket: zmq.Socket | None = None

    def _connect(self) -> zmq.Socket:
        if self._socket is None:
            socket = self._context.socket(zmq.DEALER)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDTIMEO, _SEND_TIMEOUT_MS)
            if self._address.startswith("tcp://["):
                socket.setsockopt(zmq.IPV6, 1)
            socket.connect(self._address)
            self._socket = socket
        return self._socket

    def _reset(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _request(self, op: str, chunk_ids: list[int], *payload) -> list:
        for attempt in range(1, self._attempts + 1):
            if self._stopped.is_set():
                raise RuntimeError(f"KV offload mirror {self._stop_reason}")
            request_id = uuid.uuid4().hex
            header = json.dumps(
                {
                    "id": request_id,
                    "token": self._token,
                    "seq": next(self._seq),
                    "epoch": self._epoch,
                    "op": op,
                    "rank": self._rank,
                    "chunks": chunk_ids,
                }
            ).encode()
            socket = self._connect()
            deadline = time.monotonic() + self._timeout_ms / 1000
            try:
                socket.send_multipart([header, *payload], copy=False)
            except zmq.Again:
                deadline = 0.0  # the head accepts no data: next attempt
            while (
                remaining := deadline - time.monotonic()
            ) > 0 and not self._stopped.is_set():
                if not socket.poll(min(int(remaining * 1000) + 1, 200)):
                    continue
                reply = socket.recv_multipart(copy=False)
                if reply[0].bytes != request_id.encode():
                    continue  # reply to an abandoned attempt
                if reply[1].bytes != b"ok":
                    raise RuntimeError(f"head rejected KV offload mirror {op}")
                return reply
            if self._stopped.is_set():
                break
            logger.warning(
                "KV offload mirror %s to %s timed out (attempt %d/%d)",
                op,
                self._address,
                attempt,
                self._attempts,
            )
            self._reset()
        if self._stopped.is_set():
            raise RuntimeError(f"KV offload mirror {self._stop_reason}")
        raise RuntimeError(
            f"KV offload mirror {op} to {self._address} failed after "
            f"{self._attempts} attempts"
        )

    @property
    def chunks_per_request(self) -> int:
        return self._chunks_per_request

    def _batches(self, chunk_ids: np.ndarray):
        for start in range(0, len(chunk_ids), self._chunks_per_request):
            yield chunk_ids[start : start + self._chunks_per_request]

    def ping(self) -> None:
        self._request("ping", [])

    def fence(self) -> None:
        """Make the head reject every put this client sent before."""
        self._request("fence", [])

    def push(self, chunk_ids: np.ndarray) -> None:
        assert self._rows is not None
        for batch in self._batches(chunk_ids):
            payload = self._layout.read(self._rows, batch, 0)
            self._request("put", batch.tolist(), payload)

    def pull(self, chunk_ids: np.ndarray) -> None:
        assert self._rows is not None
        for batch in self._batches(chunk_ids):
            reply = self._request("get", batch.tolist())
            self._layout.write(self._rows, batch, 0, reply[2].buffer)

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def stop(self, reason: str = "is shutting down") -> None:
        """Make waiting and later requests fail at once."""
        self._stop_reason = reason
        self._stopped.set()

    def shutdown(self) -> None:
        self._reset()
        self._context.term()
        self._rows = None


@dataclass
class _Copy:
    job_id: int
    chunk_ids: np.ndarray
    result: TransferResult | None = None
    load_specs: tuple[LoadStoreSpec, GPULoadStoreSpec] | None = None
    error: BaseException | None = None
    done: threading.Event = field(default_factory=threading.Event)
    # Chunks already copied; copies move one request at a time.
    copied: int = 0


# Loads go first, but a waiting store runs after this many consecutive loads,
# so a stream of loads cannot keep stored chunks pinned indefinitely.
_LOADS_PER_STORE = 4


class RegionMirrorWorker(OffloadingWorker):
    """CPU offloading worker whose region slots are mirrored on the head node.

    On the head node it only owns the mirror server. On other nodes a
    background thread pushes the slots of completed stores before they are
    reported and pulls the slots of loads before their DMA is submitted.
    ``wait`` covers the jobs' GPU transfers, which is what GPU block reuse
    needs; a store's chunk stays pinned until its push is reported.
    """

    def __init__(
        self,
        inner: OffloadingWorker,
        *,
        client: RegionMirrorClient | None = None,
        server: RegionMirrorServer | None = None,
    ):
        self._inner = inner
        self._client = client
        self._server = server
        self._pending_stores: dict[int, np.ndarray] = {}
        self._copies: dict[int, _Copy] = {}
        self._ready: list[TransferResult] = []
        self._cv = threading.Condition()
        self._loads: deque[tuple[Callable, _Copy]] = deque()
        self._stores: deque[tuple[Callable, _Copy]] = deque()
        self._load_streak = 0
        # Failed copies are logged at most once every 10 s.
        self._failed_since_log = 0
        self._failed_logged_at = 0.0
        self._batch_chunks = client.chunks_per_request if client else 0
        self._stopping = False
        # Set when a failed push could not be fenced; ends the engine.
        self._unfenced: RuntimeError | None = None
        self._completed: queue.Queue[_Copy] = queue.Queue()
        self._thread: threading.Thread | None = None
        if client is not None:
            self._thread = threading.Thread(
                target=self._copy_loop, name="kv-offload-mirror-io", daemon=True
            )
            self._thread.start()

    def _next_copy(self) -> tuple[Callable, _Copy] | None:
        with self._cv:
            while not (self._stopping or self._loads or self._stores):
                self._cv.wait()
            if self._stopping:
                return None
            if self._loads and (
                self._load_streak < _LOADS_PER_STORE or not self._stores
            ):
                self._load_streak += 1
                return self._loads.popleft()
            self._load_streak = 0
            return self._stores.popleft()

    def _copy_loop(self) -> None:
        while (item := self._next_copy()) is not None:
            copy_fn, copy = item
            end = min(copy.copied + self._batch_chunks, len(copy.chunk_ids))
            try:
                copy_fn(copy.chunk_ids[copy.copied : end])
            except BaseException as e:
                copy.error = e
                if copy.load_specs is None:
                    self._fence_failed_push()
            copy.copied = end
            if copy.error is None and end < len(copy.chunk_ids):
                # The rest goes back to the front of its queue, so a load
                # queued meanwhile waits for one request, not the whole copy.
                with self._cv:
                    queue_ = self._loads if copy.load_specs else self._stores
                    queue_.appendleft(item)
                continue
            # Queue before signalling: wait() drains the queue once signalled.
            self._completed.put(copy)
            copy.done.set()

    def _fence_failed_push(self) -> None:
        """Keep a failed push's delayed attempts from landing later.

        The scheduler drops a failed store's chunks and may reassign them; a
        delayed attempt applied after that would overwrite another key's
        slot. If the fence fails, the store is never reported and
        get_finished raises: the head may still apply such an attempt.
        """
        assert self._client is not None
        if self._client.stopped:
            return
        try:
            self._client.fence()
        except Exception as e:
            logger.exception("KV offload mirror could not fence a failed push")
            self._client.stop("stopped: a failed push could not be fenced")
            self._unfenced = RuntimeError(
                "KV offload mirror could not fence a failed push; a delayed "
                "attempt could overwrite a reused CPU chunk on the head node"
            )
            self._unfenced.__cause__ = e

    def _enqueue(self, copy_fn: Callable, copy: _Copy) -> None:
        self._copies[copy.job_id] = copy
        with self._cv:
            (self._loads if copy.load_specs else self._stores).append((copy_fn, copy))
            self._cv.notify()

    def _finish_copy(self, copy: _Copy) -> None:
        """Account a completed copy: submit a load's DMA or report a store.

        A failed copy reports its job as failed: the scheduler drops a failed
        store's chunks, and a failed load's blocks become load errors.
        """
        del self._copies[copy.job_id]
        if copy.error is not None:
            self._failed_since_log += 1
            now = time.monotonic()
            if now - self._failed_logged_at >= 10.0:
                logger.warning(
                    "%d KV offload mirror copies failed; last: job %d: %s",
                    self._failed_since_log,
                    copy.job_id,
                    copy.error,
                )
                self._failed_since_log = 0
                self._failed_logged_at = now
            self._ready.append(TransferResult(job_id=copy.job_id, success=False))
            return
        if copy.load_specs is not None:
            src_spec, dst_spec = copy.load_specs
            if not self._inner.submit_load(copy.job_id, src_spec, dst_spec):
                raise RuntimeError(
                    f"KV offload load job {copy.job_id} could not be submitted"
                )
        else:
            assert copy.result is not None
            self._ready.append(copy.result)

    def _drain_completed(self) -> None:
        while True:
            try:
                copy = self._completed.get_nowait()
            except queue.Empty:
                return
            self._finish_copy(copy)

    def _collect_inner(self) -> None:
        """Queue pushes for finished store DMAs; keep other results to report."""
        assert self._client is not None
        for result in self._inner.get_finished():
            chunk_ids = self._pending_stores.pop(result.job_id, None)
            if chunk_ids is None or not result.success:
                self._ready.append(result)
                continue
            self._enqueue(self._client.push, _Copy(result.job_id, chunk_ids, result))

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        if not self._inner.submit_store(job_id, src_spec, dst_spec):
            return False
        if self._client is not None:
            assert isinstance(dst_spec, CPULoadStoreSpec)
            self._pending_stores[job_id] = np.unique(dst_spec.chunk_ids)
        return True

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        if self._client is None:
            return self._inner.submit_load(job_id, src_spec, dst_spec)
        assert isinstance(src_spec, CPULoadStoreSpec)
        copy = _Copy(
            job_id, np.unique(src_spec.chunk_ids), load_specs=(src_spec, dst_spec)
        )
        self._enqueue(self._client.pull, copy)
        return True

    def get_finished(self) -> list[TransferResult]:
        if self._client is None:
            return self._inner.get_finished()
        self._drain_completed()
        if self._unfenced is not None:
            raise self._unfenced
        self._collect_inner()
        results, self._ready = self._ready, []
        return results

    def wait(self, job_ids: set[int]) -> None:
        if self._client is not None:
            # A pending load's DMA must exist before the inner wait.
            for job_id in job_ids:
                copy = self._copies.get(job_id)
                if copy is not None and copy.load_specs is not None:
                    while not copy.done.wait(1.0):
                        assert self._thread is not None
                        if not self._thread.is_alive():
                            raise RuntimeError(
                                "KV offload mirror thread stopped with a load pending"
                            )
            self._drain_completed()
        self._inner.wait(job_ids)

    def shutdown(self) -> None:
        stuck = False
        try:
            if self._thread is not None:
                assert self._client is not None
                # Pending copies cannot complete once the engine stops, and the
                # head may be gone: fail them instead of retrying each one.
                self._client.stop()
                with self._cv:
                    self._stopping = True
                    self._cv.notify()
                self._thread.join(_SHUTDOWN_TIMEOUT_S)
                if self._thread.is_alive():
                    # The thread still owns the client's socket and region
                    # views, so neither can be released under it.
                    logger.warning(
                        "KV offload mirror thread did not stop in time; its "
                        "region stays mapped"
                    )
                    stuck = True
                else:
                    self._client.shutdown()
            if self._server is not None:
                self._server.shutdown()
        finally:
            # The mirror's region views are gone, so the region can close.
            if not stuck:
                self._inner.shutdown()


def _all_gather(topology: MirrorTopology, value: Any) -> list[Any]:
    import torch.distributed as dist

    values: list[Any] = [None] * topology.group.world_size
    dist.all_gather_object(values, value, group=topology.group.cpu_group)
    return values


def _raise_if_failed(
    statuses: list[Any], own_error: BaseException | None, what: str
) -> None:
    if own_error is not None:
        raise own_error
    failed = {rank: status for rank, status in enumerate(statuses) if status}
    if failed:
        raise RuntimeError(f"KV offload {what} failed on {failed}")


def gather_construction_errors(
    topology: MirrorTopology, error: BaseException | None
) -> None:
    """Raise on every worker if any worker failed to build its offloading worker.

    Collective over the engine's workers, so a failure on one rank cannot leave
    the others blocked in a later collective.
    """
    statuses = _all_gather(topology, None if error is None else repr(error))
    _raise_if_failed(statuses, error, "worker construction")


def attach_region_mirror(
    worker: OffloadingWorker,
    region: SharedOffloadRegion,
    head_layout: SlotLayout,
    local_layout: SlotLayout,
    topology: MirrorTopology,
    *,
    port: int = 0,
    timeout_ms: int = _REQUEST_TIMEOUT_MS,
    attempts: int = _REQUEST_ATTEMPTS,
) -> OffloadingWorker:
    """Connect this worker's region to the head region through the mirror.

    Collective over the engine's workers: every rank takes part in the same
    exchanges, and any failure (missing address, server bind, client
    handshake) raises on all of them. On failure the caller still owns
    ``worker``.

    Args:
        worker: The node-local CPU offloading worker.
        region: The worker's shared region.
        head_layout: Slot geometry of the head node's region.
        local_layout: Slot geometry of this worker's region.
        topology: This worker's placement.
        port: Server port on the head node; 0 picks a free one.
        timeout_ms: Wait for one request attempt, in milliseconds.
        attempts: Attempts per request before a copy fails.

    """
    from vllm.utils.network_utils import get_ip

    host = None
    error: BaseException | None = None
    try:
        # The mirror listens and connects on the deployment's chosen network,
        # and the head admits only the remote workers' addresses.
        if not envs.VLLM_HOST_IP:
            raise RuntimeError(
                "Set VLLM_HOST_IP: the KV offload region mirror listens and "
                "connects on it when secondary tiers span several nodes."
            )
        # ZAP matches peer addresses literally, so compare normalized IPs.
        address = ipaddress.ip_address(get_ip())
        if address.is_unspecified:
            raise RuntimeError(
                f"VLLM_HOST_IP={address} is not a host address; the KV offload "
                "region mirror needs one on the isolated network."
            )
        host = str(address)
    except Exception as e:
        error = e
    hosts = _all_gather(topology, (None if error is None else repr(error), host))
    _raise_if_failed([status for status, _ in hosts], error, "mirror address")
    assert host is not None

    server = None
    endpoint = None
    if topology.rank == 0:
        try:
            token = secrets.token_hex(16)
            server = RegionMirrorServer(
                region,
                head_layout,
                host,
                token,
                topology.remote_ranks,
                allowed_hosts=frozenset(
                    hosts[rank][1] for rank in topology.remote_ranks
                ),
                port=port,
            )
            endpoint = (server.address, token)
        except Exception as e:
            error = e
    try:
        statuses = _all_gather(
            topology, (None if error is None else repr(error), endpoint)
        )
        _raise_if_failed([status for status, _ in statuses], error, "mirror server")
        address, token = statuses[0][1]
        client = None
        error = None
        if not topology.on_head:
            try:
                client = RegionMirrorClient(
                    region,
                    local_layout,
                    topology.rank,
                    address,
                    token,
                    source_host=host,
                    timeout_ms=timeout_ms,
                    attempts=attempts,
                )
                client.ping()
            except Exception as e:
                error = e
        try:
            pings = _all_gather(topology, None if error is None else repr(error))
            _raise_if_failed(pings, error, "mirror handshake")
        except BaseException:
            if client is not None:
                client.shutdown()
            raise
    except BaseException:
        if server is not None:
            server.shutdown()
        raise
    if client is None:
        return RegionMirrorWorker(worker, server=server)
    logger.info(
        "KV offload worker %d mirrors its region slots through %s",
        topology.rank,
        address,
    )
    return RegionMirrorWorker(worker, client=client)
