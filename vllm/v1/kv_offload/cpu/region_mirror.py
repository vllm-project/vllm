# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mirror worker slots of a multi-node CPU offload region onto the head node.

The tiering scheduler and its secondary tiers run in the engine core, which
shares a node with worker rank 0 (the head node), and address only that node's
shared region. A worker on another node moves its own slot of each chunk
between its GPU and the head region with one-sided NIXL transfers (RDMA WRITE
and READ), which the head serves without taking part: it only registers its
region. A store is reported only after the worker's slot of every stored chunk
is in the head region, and a load reads it from there before the request runs.
The scheduler counts a transfer complete once every worker reports it, so a
secondary tier never reads a chunk before all its slots arrived.

Two sources are supported (``mirror_source``):

- ``cpu``: the worker keeps a compact node-local region holding only its slot,
  moves it to and from the GPU with the regular CPU offloading DMA, and
  mirrors it to the head region.
- ``gpu``: the worker transfers straight between its GPU blocks and the head
  region. It needs GPUDirect RDMA; without it UCX emulates such transfers
  through host memory, several times slower than ``cpu``.

Copies run on a background thread; loads go ahead of stores. A copy that fails
is reported as a failed transfer. A transfer that ends in no terminal state may
still land later, onto a chunk or block that was reassigned, so it stops the
engine instead.

Registered memory is reachable by any peer on the RDMA network that holds the
head's metadata: the mirror needs an isolated network.
"""

import ctypes
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
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

logger = init_logger(__name__)

MIRROR_SOURCES = ("auto", "cpu", "gpu")
_TRANSFER_TIMEOUT_S = 30.0
# Each transfer moves at most this many slot bytes, so a long store waits for
# at most one transfer before a load queued behind it starts.
_MAX_TRANSFER_BYTES = 256 * 1024 * 1024
_SHUTDOWN_TIMEOUT_S = 60.0
# The cpu source moves host memory only; without this UCX also wires up rc_gda
# GPU lanes (see the p2p tier's NixlTransport).
_HOST_ONLY_UCX_PARAMS = {"engine_config": "TLS=^rc_gda"}
_CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116
_CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124


def gpu_direct_rdma_supported(device: int) -> bool:
    """Whether a NIC may read and write this GPU's memory directly.

    True when the driver reports GPUDirect RDMA or dma-buf export for the
    device. Unified-memory parts such as GB10 report neither.
    """
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        handle = ctypes.c_int()
        if cuda.cuInit(0) or cuda.cuDeviceGet(ctypes.byref(handle), device):
            return False
        for attribute in (
            _CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED,
            _CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
        ):
            value = ctypes.c_int()
            if (
                not cuda.cuDeviceGetAttribute(ctypes.byref(value), attribute, handle)
                and value.value
            ):
                return True
    except OSError:
        pass
    return False


@dataclass(frozen=True)
class SlotLayout:
    """Byte range of each worker slot inside every chunk row of a region."""

    num_chunks: int
    row_stride: int
    size: int

    @property
    def num_slots(self) -> int:
        return self.row_stride // self.size

    def base_address(self, region: SharedOffloadRegion) -> int:
        assert region.mmap_obj is not None
        return np.frombuffer(region.mmap_obj, dtype=np.uint8).ctypes.data

    def slot_descs(self, base: int, slot: int) -> list[tuple[int, int, int]]:
        """One NIXL descriptor per chunk, covering ``slot`` of its row."""
        offset = base + slot * self.size
        return [
            (offset + chunk * self.row_stride, self.size, 0)
            for chunk in range(self.num_chunks)
        ]


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


def _new_agent(name: str, ucx_params: dict[str, str] | None = None):
    from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config

    if NixlWrapper is None:
        raise RuntimeError(
            "The KV offload region mirror needs NIXL when secondary tiers span "
            "several nodes; install the nixl package."
        )
    agent = NixlWrapper(name, nixl_agent_config(backends=[]))
    agent.create_backend("UCX", dict(ucx_params or {}))
    return agent


class _MirrorClientBase:
    """A remote worker's NIXL agent, connected to the head region."""

    def __init__(
        self,
        name: str,
        head_metadata: bytes,
        timeout_s: float,
        ucx_params: dict[str, str] | None,
    ):
        self._timeout_s = timeout_s
        self._stopped = threading.Event()
        self._stop_reason = "is shutting down"
        self._agent = _new_agent(name, ucx_params)
        self._head = self._agent.add_remote_agent(head_metadata)
        self._regs: list[Any] = []

    @property
    def metadata(self) -> bytes:
        return self._agent.get_agent_metadata()

    def _run(self, handle: Any, what: str) -> None:
        """Post a prepared transfer and wait for its terminal state."""
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

    def _check_running(self) -> None:
        if self._stopped.is_set():
            raise RuntimeError(f"KV offload mirror {self._stop_reason}")

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def stop(self, reason: str = "is shutting down") -> None:
        """Make waiting and later transfers fail at once."""
        self._stop_reason = reason
        self._stopped.set()

    def shutdown(self) -> None:
        self._agent.remove_remote_agent(self._head)
        for reg in self._regs:
            self._agent.deregister_memory(reg)


class RegionMirrorHead:
    """Registers the head node's region for the remote workers' transfers."""

    def __init__(
        self,
        region: SharedOffloadRegion,
        layout: SlotLayout,
        name: str,
        ucx_params: dict[str, str] | None = None,
    ):
        self._agent = _new_agent(name, ucx_params)
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


class RegionMirrorClient(_MirrorClientBase):
    """Moves one remote worker's slot between its region and the head region.

    The local region holds only this worker's slot (``local_layout``); in the
    head region the slot is the worker's rank. Copies are split into transfers
    of at most ``max_transfer_bytes``. ``stop`` aborts waiting transfers.
    """

    def __init__(
        self,
        region: SharedOffloadRegion,
        local_layout: SlotLayout,
        head_layout: SlotLayout,
        rank: int,
        head_metadata: bytes,
        head_base: int,
        name: str,
        *,
        timeout_s: float = _TRANSFER_TIMEOUT_S,
        max_transfer_bytes: int = _MAX_TRANSFER_BYTES,
    ):
        super().__init__(name, head_metadata, timeout_s, _HOST_ONLY_UCX_PARAMS)
        self.chunks_per_transfer = max(1, max_transfer_bytes // local_layout.size)
        base = local_layout.base_address(region)
        self._regs.append(
            self._agent.register_memory(
                [(base, local_layout.num_chunks * local_layout.row_stride, 0, "")],
                "DRAM",
            )
        )
        self._local = self._agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT",
            self._agent.get_xfer_descs(local_layout.slot_descs(base, 0), "DRAM"),
        )
        self._remote = self._agent.prep_xfer_dlist(
            self._head,
            self._agent.get_xfer_descs(head_layout.slot_descs(head_base, rank), "DRAM"),
        )

    def _transfer(self, op: str, chunk_ids: np.ndarray) -> None:
        self._check_running()
        ids = chunk_ids.tolist()
        handle = self._agent.make_prepped_xfer(op, self._local, ids, self._remote, ids)
        self._run(handle, f"{op} of {len(ids)} chunks")

    def push(self, chunk_ids: np.ndarray) -> None:
        self._transfer("WRITE", chunk_ids)

    def pull(self, chunk_ids: np.ndarray) -> None:
        self._transfer("READ", chunk_ids)

    def shutdown(self) -> None:
        self._agent.release_dlist_handle(self._local)
        self._agent.release_dlist_handle(self._remote)
        super().shutdown()


class GPUMirrorClient(_MirrorClientBase):
    """Moves one remote worker's GPU blocks straight to and from its slot of
    the head region.

    Addresses come from the CPU offloading worker's copy plan for a compact
    region laid out like ``local_layout`` at ``local_base``; each is mapped to
    the same byte of the worker's slot in the head region.
    """

    def __init__(
        self,
        gpu_ranges: list[tuple[int, int, int]],
        local_layout: SlotLayout,
        local_base: int,
        head_layout: SlotLayout,
        rank: int,
        head_metadata: bytes,
        head_base: int,
        name: str,
        *,
        timeout_s: float = _TRANSFER_TIMEOUT_S,
    ):
        super().__init__(name, head_metadata, timeout_s, None)
        self._device = gpu_ranges[0][2]
        self._regs.append(
            self._agent.register_memory(
                [(addr, size, dev, "") for addr, size, dev in gpu_ranges], "VRAM"
            )
        )
        self._local_base = local_base
        self._local_stride = local_layout.row_stride
        self._head_base = head_base + rank * head_layout.size
        self._head_stride = head_layout.row_stride

    def head_addresses(self, local_addresses: np.ndarray) -> np.ndarray:
        offsets = local_addresses - self._local_base
        chunks, within = np.divmod(offsets, self._local_stride)
        return self._head_base + chunks * self._head_stride + within

    def _transfer(
        self, op: str, gpu: np.ndarray, local: np.ndarray, sizes: np.ndarray
    ) -> None:
        self._check_running()
        head = self.head_addresses(local)
        local_descs = self._agent.get_xfer_descs(
            [(int(a), int(s), self._device) for a, s in zip(gpu, sizes)], "VRAM"
        )
        remote_descs = self._agent.get_xfer_descs(
            [(int(a), int(s), 0) for a, s in zip(head, sizes)], "DRAM"
        )
        handle = self._agent.initialize_xfer(op, local_descs, remote_descs, self._head)
        self._run(handle, f"{op} of {len(sizes)} blocks")

    def write(self, gpu: np.ndarray, local: np.ndarray, sizes: np.ndarray) -> None:
        self._transfer("WRITE", gpu, local, sizes)

    def read(self, gpu: np.ndarray, local: np.ndarray, sizes: np.ndarray) -> None:
        self._transfer("READ", gpu, local, sizes)


@dataclass
class _Copy:
    job_id: int
    # Run one at a time, in order; loads queued meanwhile go between them.
    steps: list[Callable[[], None]]
    is_load: bool
    # Reported once every step succeeded, unless on_success replaces it.
    result: TransferResult | None = None
    on_success: Callable[[], None] | None = None
    error: BaseException | None = None
    done: threading.Event = field(default_factory=threading.Event)
    next_step: int = 0


# Loads go first, but a waiting store runs after this many consecutive loads,
# so a stream of loads cannot keep stored chunks pinned indefinitely.
_LOADS_PER_STORE = 4


class RegionMirrorWorker(OffloadingWorker):
    """CPU offloading worker whose region slots are mirrored on the head node.

    On the head node it only keeps the head region registered. On other nodes
    a background thread runs the copies. With ``client`` (the cpu source) a
    store's slots are pushed after its DMA and before it is reported, and a
    load's slots are pulled before its DMA is submitted; ``wait`` covers the
    jobs' GPU transfers, which is what GPU block reuse needs. With
    ``gpu_client`` (the gpu source) the copies read and write the GPU blocks
    themselves, so ``wait`` covers them too.
    """

    def __init__(
        self,
        inner: CPUOffloadingWorker,
        *,
        client: RegionMirrorClient | None = None,
        gpu_client: GPUMirrorClient | None = None,
        head: RegionMirrorHead | None = None,
    ):
        assert client is None or gpu_client is None
        self._inner = inner
        self._client = client
        self._gpu_client = gpu_client
        self._mirror: _MirrorClientBase | None = client or gpu_client
        self._head = head
        self._pending_stores: dict[int, np.ndarray] = {}
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
        self._thread: threading.Thread | None = None
        if self._mirror is not None:
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
                assert self._mirror is not None
                logger.error("KV offload mirror transfer is stuck: %s", e)
                self._mirror.stop("stopped: a transfer may still land")
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

    def _enqueue(self, copy: _Copy) -> None:
        self._copies[copy.job_id] = copy
        with self._cv:
            (self._loads if copy.is_load else self._stores).append(copy)
            self._cv.notify()

    def _finish_copy(self, copy: _Copy) -> None:
        """Account a completed copy: report it, or run its follow-up.

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
        elif copy.on_success is not None:
            copy.on_success()
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

    def _raise_if_stuck(self) -> None:
        if self._stuck is not None:
            raise RuntimeError(
                "KV offload mirror stopped: a slot transfer may still land on a "
                "reused CPU chunk or GPU block"
            ) from self._stuck

    def _slot_steps(self, copy_fn: Callable, chunk_ids: np.ndarray) -> list:
        assert self._client is not None
        size = self._client.chunks_per_transfer
        return [
            functools.partial(copy_fn, chunk_ids[start : start + size])
            for start in range(0, len(chunk_ids), size)
        ]

    def _collect_inner(self) -> None:
        """Queue pushes for finished store DMAs; keep other results to report."""
        assert self._client is not None
        for result in self._inner.get_finished():
            chunk_ids = self._pending_stores.pop(result.job_id, None)
            if chunk_ids is None or not result.success:
                self._ready.append(result)
                continue
            steps = self._slot_steps(self._client.push, chunk_ids)
            self._enqueue(_Copy(result.job_id, steps, is_load=False, result=result))

    def _gpu_copy(
        self,
        job_id: int,
        copy_fn: Callable,
        gpu: np.ndarray,
        local: np.ndarray,
        sizes: np.ndarray,
        is_load: bool,
    ) -> None:
        # The copy reads or writes GPU blocks that the compute stream may
        # still be writing (a store's KV, a load's zeroing), so it waits.
        event = torch.Event()
        event.record()
        num_bytes = int(sizes.sum())
        result = TransferResult(job_id=job_id, success=True, transfer_size=num_bytes)

        def step() -> None:
            event.synchronize()
            start = time.perf_counter()
            copy_fn(gpu, local, sizes)
            result.transfer_time = time.perf_counter() - start

        self._enqueue(_Copy(job_id, [step], is_load=is_load, result=result))

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        if self._gpu_client is not None:
            gpu, local, sizes = self._inner.store_copy_ops(src_spec, dst_spec)
            self._gpu_copy(
                job_id, self._gpu_client.write, gpu, local, sizes, is_load=False
            )
            return True
        if not self._inner.submit_store(job_id, src_spec, dst_spec):
            return False
        if self._client is not None:
            assert isinstance(dst_spec, CPULoadStoreSpec)
            self._pending_stores[job_id] = np.unique(dst_spec.chunk_ids)
        return True

    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
        expedited: bool = False,
    ) -> bool:
        if self._gpu_client is not None:
            local, gpu, sizes = self._inner.load_copy_ops(src_spec, dst_spec)
            self._gpu_copy(job_id, self._gpu_client.read, gpu, local, sizes, True)
            return True
        if self._client is None:
            return self._inner.submit_load(
                job_id, src_spec, dst_spec, expedited=expedited
            )
        assert isinstance(src_spec, CPULoadStoreSpec)

        def submit_dma() -> None:
            if not self._inner.submit_load(
                job_id, src_spec, dst_spec, expedited=expedited
            ):
                raise RuntimeError(
                    f"KV offload load job {job_id} could not be submitted"
                )

        steps = self._slot_steps(self._client.pull, np.unique(src_spec.chunk_ids))
        self._enqueue(_Copy(job_id, steps, is_load=True, on_success=submit_dma))
        return True

    def get_finished(self) -> list[TransferResult]:
        if self._mirror is None:
            return self._inner.get_finished()
        self._drain_completed()
        self._raise_if_stuck()
        if self._client is not None:
            self._collect_inner()
        results, self._ready = self._ready, []
        return results

    def wait(self, job_ids: set[int]) -> None:
        if self._mirror is not None:
            # A pending load's DMA must exist before the inner wait; a gpu
            # source's copies are the GPU transfers themselves.
            for job_id in job_ids:
                copy = self._copies.get(job_id)
                if copy is None or not (copy.is_load or self._gpu_client):
                    continue
                while not copy.done.wait(1.0):
                    self._raise_if_stuck()
                    assert self._thread is not None
                    if not self._thread.is_alive():
                        raise RuntimeError(
                            "KV offload mirror thread stopped with a copy pending"
                        )
            self._drain_completed()
        self._inner.wait(job_ids)

    def shutdown(self) -> None:
        stuck = False
        try:
            if self._thread is not None:
                assert self._mirror is not None
                # Pending copies cannot complete once the engine stops, and the
                # head may be gone: fail them instead of retrying each one.
                self._mirror.stop()
                with self._cv:
                    self._stopping = True
                    self._cv.notify()
                self._thread.join(_SHUTDOWN_TIMEOUT_S)
                if self._thread.is_alive() or self._stuck is not None:
                    # A transfer may still use the registered memory.
                    logger.warning(
                        "KV offload mirror did not stop cleanly; its region "
                        "stays mapped"
                    )
                    stuck = True
                else:
                    self._mirror.shutdown()
            if self._head is not None:
                self._head.shutdown()
        finally:
            # The mirror's registrations are gone, so the region can close.
            if not stuck:
                self._inner.shutdown()


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


def resolve_mirror_source(source: str, device: int) -> str:
    """Resolve ``auto`` to ``gpu`` when the GPU supports GPUDirect RDMA."""
    if source not in MIRROR_SOURCES:
        raise ValueError(
            f"mirror_source must be one of {MIRROR_SOURCES}, got {source!r}"
        )
    if source == "auto":
        return "gpu" if gpu_direct_rdma_supported(device) else "cpu"
    return source


def attach_region_mirror(
    worker: CPUOffloadingWorker,
    region: SharedOffloadRegion,
    head_layout: SlotLayout,
    local_layout: SlotLayout,
    topology: MirrorTopology,
    engine_id: str,
    *,
    source: str = "auto",
    timeout_s: float = _TRANSFER_TIMEOUT_S,
) -> OffloadingWorker:
    """Connect this worker's region to the head region through the mirror.

    Collective over the engine's workers: every rank takes part in the same
    exchanges, and any failure raises on all of them. On failure the caller
    still owns ``worker``.

    Args:
        worker: The node-local CPU offloading worker.
        region: The worker's shared region.
        head_layout: Slot geometry of the head node's region.
        local_layout: Slot geometry of this worker's region.
        topology: This worker's placement.
        engine_id: Names the NIXL agents of this engine.
        source: ``auto``, ``cpu`` or ``gpu`` (see the module docstring).
        timeout_s: Wait for one transfer before the mirror counts it stuck.

    """
    name = f"kv-offload-mirror-{engine_id}-{topology.rank}"
    head = None
    client: RegionMirrorClient | GPUMirrorClient | None = None
    error: BaseException | None = None
    try:
        own_source = None
        if not topology.on_head:
            try:
                own_source = resolve_mirror_source(
                    source, torch.accelerator.current_device_index()
                )
            except Exception as e:
                error = e
        sources = _gather_raising(topology, error, own_source, "mirror source")
        gpu_source = "gpu" in sources
        # The head serves GPU memory only through UCX's GPU transports.
        head_ucx = None if gpu_source else _HOST_ONLY_UCX_PARAMS
        if topology.rank == 0:
            try:
                head = RegionMirrorHead(region, head_layout, name, head_ucx)
                endpoint = (head.metadata, head.base)
            except Exception as e:
                error, endpoint = e, None
        else:
            endpoint = None
        head_metadata, head_base = _gather_raising(
            topology, error, endpoint, "mirror head registration"
        )[0]

        metadata = None
        if not topology.on_head:
            try:
                if sources[topology.rank] == "gpu":
                    client = GPUMirrorClient(
                        worker.gpu_memory_ranges(),
                        local_layout,
                        local_layout.base_address(region),
                        head_layout,
                        topology.rank,
                        head_metadata,
                        head_base,
                        name,
                        timeout_s=timeout_s,
                    )
                else:
                    client = RegionMirrorClient(
                        region,
                        local_layout,
                        head_layout,
                        topology.rank,
                        head_metadata,
                        head_base,
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

        if isinstance(client, RegionMirrorClient):
            try:
                # The head region has no data yet, so reading one chunk's slot
                # proves the path without disturbing anything.
                client.pull(np.zeros(1, dtype=np.int64))
            except Exception as e:
                error = e
        _gather_raising(topology, error, None, "mirror handshake")
    except BaseException:
        if client is not None and not client.stopped:
            client.shutdown()
        if head is not None:
            head.shutdown()
        raise
    if client is None:
        return RegionMirrorWorker(worker, head=head)
    logger.info(
        "KV offload worker %d mirrors its region slots into the head node's "
        "region over NIXL (mirror_source=%s)",
        topology.rank,
        sources[topology.rank],
    )
    if isinstance(client, GPUMirrorClient):
        return RegionMirrorWorker(worker, gpu_client=client)
    return RegionMirrorWorker(worker, client=client)
