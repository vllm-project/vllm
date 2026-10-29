# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mirror worker slots of a multi-node CPU offload region onto the head node.

The tiering scheduler and its secondary tiers run in the engine core, which
shares a node with worker rank 0 (the head node), and address only that node's
shared region. A worker on another node allocates no CPU region: its copy plan
addresses its slot of each chunk in the head region, and it moves its GPU
blocks straight to and from there with one-sided NIXL transfers (RDMA WRITE and
READ), which the head serves without taking part: it only registers its region.
A store is reported only after the worker's slot of every stored chunk is in
the head region, and a load writes the worker's GPU blocks before the request
runs. The scheduler counts a transfer complete once every worker reports it,
so a secondary tier never reads a chunk before all its slots arrived.

The transfers need GPUDirect RDMA to run at network speed; without it UCX
emulates them through host memory.

Copies run on a background thread, split into transfers of about
``_MAX_TRANSFER_BYTES``; loads go ahead of stores, so a load waits for at most
one transfer of a large store. A copy that fails is reported as a failed
transfer. A transfer that ends in no terminal state may still land later, onto
a chunk or block that was reassigned, so it stops the engine instead.

Registered memory is reachable by any peer on the RDMA network that holds the
head's metadata: the mirror needs an isolated network.
"""

import functools
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.gpu_worker import (
    CopyPlanner,
    CPUOffloadingWorker,
    TensorRows,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

logger = init_logger(__name__)

_TRANSFER_TIMEOUT_S = 30.0
_SHUTDOWN_TIMEOUT_S = 60.0
# Bytes of one NIXL transfer; a load queued meanwhile goes between them.
_MAX_TRANSFER_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class SlotLayout:
    """Byte range of each worker slot inside every chunk row of a region."""

    num_chunks: int
    row_stride: int
    size: int

    def base_address(self, region: SharedOffloadRegion) -> int:
        assert region.mmap_obj is not None
        return np.frombuffer(region.mmap_obj, dtype=np.uint8).ctypes.data


@dataclass(frozen=True)
class MirrorTopology:
    """Placement of this worker relative to the head node (worker rank 0)."""

    group: Any
    rank: int
    on_head: bool
    # Ranks whose workers are not on the head node.
    remote_ranks: frozenset[int]


def mirror_topology() -> MirrorTopology | None:
    """Locate this worker relative to the head node.

    Collective over the world group, which holds exactly the engine's workers
    when there is one data parallel replica. Returns None when every worker
    shares the head node.
    """
    from vllm.distributed.parallel_state import get_world_group, in_the_same_node_as

    group = get_world_group()
    same_node = in_the_same_node_as(group.cpu_group, source_rank=0)
    if all(same_node):
        return None
    return MirrorTopology(
        group=group,
        rank=group.rank_in_group,
        on_head=same_node[group.rank_in_group],
        remote_ranks=frozenset(r for r, local in enumerate(same_node) if not local),
    )


class MirrorStuckError(RuntimeError):
    """A slot transfer reached no terminal state and may still land."""


def _is_remote_disconnect(error: Exception) -> bool:
    """Whether a NIXL raise ends the transfer: its peer is gone.

    Any other raise says nothing about the transfer itself (see the p2p
    tier's ``_is_terminal_state_error``).
    """
    name = type(error).__name__
    return name == "nixlRemoteDisconnectError" or "NIXL_ERR_REMOTE_DISCONNECT" in str(
        error
    )


def _new_agent(name: str):
    from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config

    if NixlWrapper is None:
        raise RuntimeError(
            "The KV offload region mirror needs NIXL when secondary tiers span "
            "several nodes; install the nixl package."
        )
    agent = NixlWrapper(name, nixl_agent_config(backends=[]))
    agent.create_backend("UCX", {})
    return agent


class RegionMirrorHead:
    """Registers the head node's region for the remote workers' transfers."""

    def __init__(self, region: SharedOffloadRegion, layout: SlotLayout, name: str):
        self._agent = _new_agent(name)
        self.base = layout.base_address(region)
        self._reg = self._agent.register_memory(
            [(self.base, layout.num_chunks * layout.row_stride, 0, "")], "DRAM"
        )
        self._peers: list[str] = []

    @property
    def metadata(self) -> bytes:
        return self._agent.get_agent_metadata()

    def add_peers(self, metadata: list[bytes]) -> None:
        for peer in metadata:
            self._peers.append(self._agent.add_remote_agent(peer))

    def shutdown(self) -> None:
        for peer in self._peers:
            self._agent.remove_remote_agent(peer)
        self._agent.deregister_memory(self._reg)


class RegionMirrorClient:
    """Moves one remote worker's GPU blocks to and from its slot of the head
    region.

    Each transfer takes (GPU address, head region address, size) triples.
    ``stop`` aborts waiting transfers.
    """

    def __init__(
        self,
        gpu_ranges: list[tuple[int, int, int]],
        head_metadata: bytes,
        name: str,
        *,
        timeout_s: float = _TRANSFER_TIMEOUT_S,
    ):
        self._timeout_s = timeout_s
        self._stopped = threading.Event()
        self._stop_reason = "is shutting down"
        self._agent = _new_agent(name)
        self._head = self._agent.add_remote_agent(head_metadata)
        self._device = gpu_ranges[0][2]
        self._reg = self._agent.register_memory(
            [(addr, size, dev, "") for addr, size, dev in gpu_ranges], "VRAM"
        )

    @property
    def metadata(self) -> bytes:
        return self._agent.get_agent_metadata()

    def _run(self, handle: Any, what: str) -> None:
        """Post a transfer and wait for its terminal state."""
        try:
            state = self._agent.transfer(handle)
            deadline = time.monotonic() + self._timeout_s
            while state not in ("DONE", "ERR"):
                if self._stopped.is_set() or time.monotonic() > deadline:
                    handle = None  # it may still land: never release it
                    raise MirrorStuckError(
                        f"KV offload mirror {what} did not finish within "
                        f"{self._timeout_s} s"
                    )
                time.sleep(0.0001)
                state = self._agent.check_xfer_state(handle)
        except MirrorStuckError:
            raise
        except Exception as e:
            if not _is_remote_disconnect(e):
                handle = None
                raise MirrorStuckError(
                    f"KV offload mirror {what} state is unknown: {e}"
                ) from e
            raise RuntimeError(f"KV offload mirror {what} failed: {e}") from e
        finally:
            if handle is not None:
                self._agent.release_xfer_handle(handle)
        if state == "ERR":
            raise RuntimeError(f"KV offload mirror {what} failed")

    def _transfer(
        self, op: str, gpu: np.ndarray, head: np.ndarray, sizes: np.ndarray
    ) -> None:
        if self._stopped.is_set():
            raise RuntimeError(f"KV offload mirror {self._stop_reason}")
        local_descs = self._agent.get_xfer_descs(
            [(int(a), int(s), self._device) for a, s in zip(gpu, sizes)], "VRAM"
        )
        remote_descs = self._agent.get_xfer_descs(
            [(int(a), int(s), 0) for a, s in zip(head, sizes)], "DRAM"
        )
        handle = self._agent.initialize_xfer(op, local_descs, remote_descs, self._head)
        self._run(handle, f"{op} of {len(sizes)} blocks")

    def write(self, gpu: np.ndarray, head: np.ndarray, sizes: np.ndarray) -> None:
        self._transfer("WRITE", gpu, head, sizes)

    def read(self, gpu: np.ndarray, head: np.ndarray, sizes: np.ndarray) -> None:
        self._transfer("READ", gpu, head, sizes)

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def stop(self, reason: str = "is shutting down") -> None:
        """Make waiting and later transfers fail at once."""
        self._stop_reason = reason
        self._stopped.set()

    def shutdown(self) -> None:
        self._agent.remove_remote_agent(self._head)
        self._agent.deregister_memory(self._reg)


@dataclass
class _Copy:
    job_id: int
    # Run one at a time, in order; loads queued meanwhile go between them.
    steps: list[Callable[[], None]]
    is_load: bool
    result: TransferResult
    error: BaseException | None = None
    done: threading.Event = field(default_factory=threading.Event)
    next_step: int = 0


# Loads go first, but a waiting store runs after this many consecutive loads,
# so a stream of loads cannot keep stored blocks pinned indefinitely.
_LOADS_PER_STORE = 4


class RegionMirrorHeadWorker(OffloadingWorker):
    """Worker rank 0's CPU offloading worker, keeping the head region registered
    for the remote workers until it shuts down."""

    def __init__(self, inner: CPUOffloadingWorker, head: RegionMirrorHead):
        self._inner = inner
        self._head = head

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        return self._inner.submit_store(job_id, src_spec, dst_spec)

    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
        expedited: bool = False,
    ) -> bool:
        return self._inner.submit_load(job_id, src_spec, dst_spec, expedited=expedited)

    def get_finished(self) -> list[TransferResult]:
        return self._inner.get_finished()

    def wait(self, job_ids: set[int]) -> None:
        self._inner.wait(job_ids)

    def shutdown(self) -> None:
        try:
            self._head.shutdown()
        finally:
            self._inner.shutdown()


def slot_planners(
    kv_caches: CanonicalKVCaches,
    blocks_per_chunk: int,
    layout: SlotLayout,
    slot_base: int,
) -> tuple[CopyPlanner, CopyPlanner]:
    """Store and load copy planners between this worker's GPU blocks and its
    slot of the head region, which starts at ``slot_base``.

    The slot holds each KV tensor's chunk page in turn, as on the head node.
    """
    gpu_rows = [
        TensorRows.of(t.tensor.view(torch.int8).view(-1, t.page_size_bytes))
        for t in kv_caches.tensors
    ]
    cpu_rows = []
    offset = 0
    for rows in gpu_rows:
        size = rows.size * blocks_per_chunk
        cpu_rows.append(
            TensorRows(slot_base + offset, layout.row_stride, size, layout.num_chunks)
        )
        offset += size
    if offset > layout.size:
        raise ValueError(
            f"KV offload worker needs {offset} bytes per chunk, but its slot of "
            f"the head region holds {layout.size}"
        )
    plan = functools.partial(
        CopyPlanner, gpu_rows, cpu_rows, blocks_per_chunk, kv_caches.group_data_refs
    )
    return plan(gpu_to_cpu=True), plan(gpu_to_cpu=False)


class RegionMirrorWorker(OffloadingWorker):
    """Offloading worker on a node other than the head's.

    A background thread copies the GPU blocks of each transfer to or from this
    worker's slot of the head region, so ``wait`` covers those copies.
    """

    def __init__(
        self,
        store_planner: CopyPlanner,
        load_planner: CopyPlanner,
        client: RegionMirrorClient,
        *,
        max_transfer_bytes: int = _MAX_TRANSFER_BYTES,
    ):
        self._store_planner = store_planner
        self._load_planner = load_planner
        self._client = client
        self._max_transfer_bytes = max_transfer_bytes
        self._copies: dict[int, _Copy] = {}
        self._ready: list[TransferResult] = []
        self._cv = threading.Condition()
        self._loads: deque[_Copy] = deque()
        self._stores: deque[_Copy] = deque()
        self._load_streak = 0
        # Failed copies are logged at most once every 10 s.
        self._failed_since_log = 0
        self._failed_logged_at = 0.0
        self._stopping = False
        # Set when a transfer may still land; ends the engine.
        self._stuck: MirrorStuckError | None = None
        self._completed: queue.Queue[_Copy] = queue.Queue()
        self._thread = threading.Thread(
            target=self._copy_loop, name="kv-offload-mirror-io", daemon=True
        )
        self._thread.start()

    def _next_copy(self) -> _Copy | None:
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
        while (copy := self._next_copy()) is not None:
            try:
                copy.steps[copy.next_step]()
            except MirrorStuckError as e:
                # A late write could land on a reassigned chunk or block:
                # report nothing more and stop the engine.
                logger.error("KV offload mirror transfer is stuck: %s", e)
                self._client.stop("stopped: a transfer may still land")
                self._stuck = e
                return
            except BaseException as e:
                copy.error = e
            copy.next_step += 1
            if copy.error is None and copy.next_step < len(copy.steps):
                # The rest goes back to the front of its queue, so a load
                # queued meanwhile waits for one transfer, not the whole copy.
                with self._cv:
                    (self._loads if copy.is_load else self._stores).appendleft(copy)
                continue
            # Queue before signalling: wait() drains the queue once signalled.
            self._completed.put(copy)
            copy.done.set()

    def _enqueue(
        self,
        job_id: int,
        copy_fn: Callable,
        gpu: np.ndarray,
        head: np.ndarray,
        sizes: np.ndarray,
        is_load: bool,
    ) -> None:
        # The copy reads or writes GPU blocks that the compute stream may
        # still be writing (a store's KV, a load's zeroing), so it waits.
        event = torch.Event()
        event.record()
        result = TransferResult(
            job_id=job_id, success=True, transfer_size=int(sizes.sum())
        )
        result.transfer_time = 0.0

        def step(gpu: np.ndarray, head: np.ndarray, sizes: np.ndarray) -> None:
            event.synchronize()
            start = time.perf_counter()
            copy_fn(gpu, head, sizes)
            result.transfer_time += time.perf_counter() - start

        # Each transfer holds the blocks whose end falls in its byte window.
        window = (np.cumsum(sizes) - 1) // self._max_transfer_bytes
        bounds = np.flatnonzero(np.diff(window)) + 1
        steps = [
            functools.partial(step, *batch)
            for batch in zip(*(np.split(a, bounds) for a in (gpu, head, sizes)))
        ]
        copy = _Copy(job_id, steps, is_load, result)
        self._copies[job_id] = copy
        with self._cv:
            (self._loads if is_load else self._stores).append(copy)
            self._cv.notify()

    def _finish_copy(self, copy: _Copy) -> None:
        """Report a completed copy.

        A failed copy reports its job as failed: the scheduler drops a failed
        store's chunks, and a failed load's blocks become load errors.
        """
        del self._copies[copy.job_id]
        if copy.error is None:
            self._ready.append(copy.result)
            return
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

    def _drain_completed(self) -> None:
        while True:
            try:
                copy = self._completed.get_nowait()
            except queue.Empty:
                return
            self._finish_copy(copy)

    def _raise_if_stuck(self) -> None:
        if self._stuck is not None:
            raise RuntimeError(
                "KV offload mirror stopped: a slot transfer may still land on a "
                "reused CPU chunk or GPU block"
            ) from self._stuck

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        gpu, head, sizes = self._store_planner.copy_ops(src_spec, dst_spec)
        self._enqueue(job_id, self._client.write, gpu, head, sizes, is_load=False)
        return True

    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
        expedited: bool = False,
    ) -> bool:
        head, gpu, sizes = self._load_planner.copy_ops(src_spec, dst_spec)
        self._enqueue(job_id, self._client.read, gpu, head, sizes, is_load=True)
        return True

    def get_finished(self) -> list[TransferResult]:
        self._drain_completed()
        self._raise_if_stuck()
        results, self._ready = self._ready, []
        return results

    def wait(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            copy = self._copies.get(job_id)
            if copy is None:
                continue
            while not copy.done.wait(1.0):
                self._raise_if_stuck()
                if not self._thread.is_alive():
                    raise RuntimeError(
                        "KV offload mirror thread stopped with a copy pending"
                    )
        self._drain_completed()

    def shutdown(self) -> None:
        # Pending copies cannot complete once the engine stops, and the head
        # may be gone: fail them instead of retrying each one.
        self._client.stop()
        with self._cv:
            self._stopping = True
            self._cv.notify()
        self._thread.join(_SHUTDOWN_TIMEOUT_S)
        if self._thread.is_alive() or self._stuck is not None:
            # A transfer may still use the registered GPU memory.
            logger.warning(
                "KV offload mirror did not stop cleanly; its GPU memory stays "
                "registered"
            )
            return
        self._client.shutdown()


def _all_gather(topology: MirrorTopology, value: Any) -> list[Any]:
    import torch.distributed as dist

    values: list[Any] = [None] * topology.group.world_size
    dist.all_gather_object(values, value, group=topology.group.cpu_group)
    return values


def _gather_raising(
    topology: MirrorTopology, error: BaseException | None, value: Any, what: str
) -> list[Any]:
    """All-gather ``value``; raise on every worker if any worker failed."""
    gathered = _all_gather(topology, (None if error is None else repr(error), value))
    if error is not None:
        raise error
    failed = {rank: status for rank, (status, _) in enumerate(gathered) if status}
    if failed:
        raise RuntimeError(f"KV offload {what} failed on {failed}")
    return [value for _, value in gathered]


def gather_construction_errors(
    topology: MirrorTopology, error: BaseException | None
) -> None:
    """Raise on every worker if any worker failed to build its offloading worker.

    Collective over the engine's workers, so a failure on one rank cannot leave
    the others blocked in a later collective.
    """
    _gather_raising(topology, error, None, "worker construction")


def attach_region_mirror(
    worker: CPUOffloadingWorker | None,
    region: SharedOffloadRegion | None,
    kv_caches: CanonicalKVCaches,
    blocks_per_chunk: int,
    head_layout: SlotLayout,
    topology: MirrorTopology,
    engine_id: str,
    *,
    timeout_s: float = _TRANSFER_TIMEOUT_S,
) -> OffloadingWorker:
    """Connect this worker to the head region through the mirror.

    Collective over the engine's workers: every rank takes part in the same
    exchanges, and any failure raises on all of them. Workers whose head
    layouts differ are rejected. On failure the caller still owns ``worker``.

    Args:
        worker: The CPU offloading worker on the head node; None elsewhere.
        region: Its shared region; None off the head node.
        kv_caches: This worker's GPU KV caches.
        blocks_per_chunk: GPU blocks per offloaded chunk.
        head_layout: Slot geometry of the head node's region.
        topology: This worker's placement.
        engine_id: Names the NIXL agents of this engine.
        timeout_s: Wait for one transfer before the mirror counts it stuck.

    """
    name = f"kv-offload-mirror-{engine_id}-{topology.rank}"
    head = None
    client: RegionMirrorClient | None = None
    planners: tuple[CopyPlanner, CopyPlanner] | None = None
    error: BaseException | None = None
    try:
        endpoint = None
        if topology.rank == 0:
            assert region is not None
            try:
                head = RegionMirrorHead(region, head_layout, name)
                endpoint = (head.metadata, head.base)
            except Exception as e:
                error = e
        gathered = _gather_raising(
            topology, error, (endpoint, head_layout), "mirror head registration"
        )
        head_metadata, head_base = gathered[0][0]
        # Each worker derives the layout from its own KV size; one that differs
        # (uneven pipeline stages) would address the wrong bytes of every row.
        mismatched = {
            rank: layout
            for rank, (_, layout) in enumerate(gathered)
            if layout != gathered[0][1]
        }
        if mismatched:
            raise ValueError(
                "The KV offload mirror needs the same CPU region geometry on "
                f"every worker; rank 0 has {gathered[0][1]}, other ranks have "
                f"{mismatched}"
            )

        metadata = None
        if not topology.on_head:
            try:
                planners = slot_planners(
                    kv_caches,
                    blocks_per_chunk,
                    head_layout,
                    head_base + topology.rank * head_layout.size,
                )
                client = RegionMirrorClient(
                    [
                        (
                            t.tensor.data_ptr(),
                            t.tensor.numel() * t.tensor.element_size(),
                            t.tensor.device.index or 0,
                        )
                        for t in kv_caches.tensors
                    ],
                    head_metadata,
                    name,
                    timeout_s=timeout_s,
                )
                metadata = client.metadata
            except Exception as e:
                error = e
        peers = _gather_raising(topology, error, metadata, "mirror registration")
        if head is not None:
            try:
                head.add_peers([peers[rank] for rank in sorted(topology.remote_ranks)])
            except Exception as e:
                error = e
        _gather_raising(topology, error, None, "mirror peer registration")
    except BaseException:
        if client is not None and not client.stopped:
            client.shutdown()
        if head is not None:
            head.shutdown()
        raise
    if client is not None:
        assert planners is not None
        logger.info(
            "KV offload worker %d mirrors its GPU blocks into the head node's "
            "region over NIXL",
            topology.rank,
        )
        return RegionMirrorWorker(*planners, client)
    assert worker is not None
    return worker if head is None else RegionMirrorHeadWorker(worker, head)
