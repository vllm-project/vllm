# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IPC memory sessions and synchronous whole-block loads."""

import ctypes
import hashlib
import os
import secrets
import socket
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed import get_dp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.simple_kv_offload.cuda_mem_ops import _resolve_max_batch_descriptors

from .data import PeerHandshake, PeerMemory, PeerMetadata, PeerWorkerMetadata
from .native import CleanupQueue
from .session import AccessMode, Grant, Mover, PeerSession, Transfer
from .stats import PeerL1Stats

logger = init_logger(__name__)


class CompletionWatermark:
    """Only advance over consecutive fences recorded after all step writers."""

    def __init__(self):
        self._events: deque[tuple[int, Callable[[], bool]]] = deque()
        self._completed = 0
        self._reported = 0

    def record(self, step: int, query: Callable[[], bool]) -> None:
        self._events.append((step, query))

    def report(self) -> PeerWorkerMetadata | None:
        while self._events and self._events[0][1]():
            self._completed = self._events.popleft()[0]
        if self._completed > self._reported:
            self._reported = self._completed
            return PeerWorkerMetadata(self._completed)
        return None


def cache_segments(
    cache: KVCacheConfig,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Byte regions per group, preserving layer order and excluding padding."""
    result: list[list[tuple[int, int, int]]] = [[] for _ in cache.kv_cache_groups]
    for group_id, group in enumerate(cache.kv_cache_groups):
        for tensor in cache.kv_cache_tensors:
            for index, name in enumerate(tensor.layers):
                spec = group.kv_cache_specs.get(name)
                if spec is not None:
                    result[group_id].append(
                        (
                            tensor.offset + index * tensor.layer_stride,
                            tensor.block_stride,
                            spec.page_size_bytes,
                        )
                    )
    return tuple(tuple(regions) for regions in result)


class _CopyTransfer:
    def __init__(self, event: torch.cuda.Event):
        self.event = event

    def done(self) -> bool:
        return self.event.query()

    def failed_requests(self) -> set[str]:
        return set()

    def quiesced(self) -> bool:
        return self.done()


@dataclass
class CopyEvents:
    copy_start: torch.cuda.Event
    copy_done: torch.cuda.Event
    wait_start: torch.cuda.Event
    wait_done: torch.cuda.Event


class CopyMetrics:
    """Bound pending events; only read timestamps after the main wait completes."""

    def __init__(self, stats: PeerL1Stats, limit: int = 256):
        self.stats, self.limit = stats, limit
        self.pending: deque[CopyEvents] = deque()

    def volume(
        self, copies: list[tuple[int, int, int]], num_bytes: int | None = None
    ) -> None:
        if num_bytes is None:
            num_bytes = sum(length for _, _, length in copies)
        self.stats.inc("worker_load_steps")
        self.stats.inc("worker_copy_bytes", num_bytes)
        self.stats.inc("worker_copy_descriptors", len(copies))
        self.stats.observe("worker_copy_bytes_per_step", num_bytes)
        self.stats.observe("worker_copy_descriptors_per_step", len(copies))

    def begin(self) -> bool:
        self.drain()
        if len(self.pending) >= self.limit:
            self.stats.inc("worker_timing_dropped")
            return False
        return True

    def drain(self) -> None:
        while self.pending and self.pending[0].wait_done.query():
            events = self.pending.popleft()
            self.stats.observe(
                "worker_copy_gpu_seconds",
                max(0.0, events.copy_start.elapsed_time(events.copy_done)) / 1000,
            )
            self.stats.observe(
                "worker_main_wait_seconds",
                max(0.0, events.wait_start.elapsed_time(events.wait_done)) / 1000,
            )


class LoadStoreMover:
    """Submit a whole step with one DMA batch and completion fence."""

    def __init__(
        self,
        cleanup: CleanupQueue,
        stream: torch.cuda.Stream,
        *,
        stats: PeerL1Stats | None = None,
    ):
        if current_platform.is_rocm():
            check_rocm_copy_environment()
        self.cleanup, self.stream = cleanup, stream
        self.stats = stats if stats is not None else PeerL1Stats()
        self.metrics = CopyMetrics(self.stats)
        runtime = ctypes.CDLL(torch._C.__file__)
        name = (
            "hipLaunchHostFunc" if current_platform.is_rocm() else "cudaLaunchHostFunc"
        )
        function = getattr(runtime, name)
        function.argtypes = [ctypes.c_void_p] * 3
        function.restype = ctypes.c_int
        self._launch_host = function

    def submit(
        self,
        grants: list[Grant],
        copies: list[tuple[int, int, int]],
        *,
        num_bytes: int | None = None,
    ) -> _CopyTransfer:
        self.metrics.volume(copies, num_bytes)
        timing = self.metrics.begin()
        node = 0
        for g in reversed(grants):
            node = self.cleanup.create(
                g.endpoint, g.epoch, g.reader, g.sequence, next_node=node
            )
        # This fence includes prior drafter writes into recycled target blocks.
        previous = torch.cuda.Event()
        previous.record(torch.cuda.current_stream())
        done = torch.cuda.Event(enable_timing=timing)
        copy_start = torch.cuda.Event(enable_timing=True) if timing else None
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(previous)
            if copy_start is not None:
                copy_start.record(self.stream)
            max_batch = _resolve_max_batch_descriptors() or max(1, len(copies))
            for start in range(0, len(copies), max_batch):
                src, dst, sizes = zip(*copies[start : start + max_batch])
                ops.swap_blocks_batch(
                    torch.tensor(src, dtype=torch.int64),
                    torch.tensor(dst, dtype=torch.int64),
                    torch.tensor(sizes, dtype=torch.int64),
                )
            done.record(self.stream)
            if node:
                result = self._launch_host(
                    self.stream.cuda_stream, self.cleanup.callback_address, node
                )
                if result != 0:
                    raise RuntimeError(f"Peer L1 completion callback failed: {result}")
        main = torch.cuda.current_stream()
        if copy_start is not None:
            wait_start = torch.cuda.Event(enable_timing=True)
            wait_done = torch.cuda.Event(enable_timing=True)
            wait_start.record(main)
        main.wait_event(done)
        if copy_start is not None:
            wait_done.record(main)
            self.metrics.pending.append(
                CopyEvents(copy_start, done, wait_start, wait_done)
            )
        return _CopyTransfer(done)

    def get_stats(self) -> PeerL1Stats:
        self.metrics.drain()
        return self.stats.clone_and_reset()


def check_rocm_copy_environment() -> bool:
    """Warn unless the copy stream can get a high-priority hardware queue."""
    if os.environ.get("DEBUG_HIP_IGNORE_STREAM_PRIORITY") == "0" and (
        os.environ.get("ROCPROFILER_QUEUE_INTERPOSITION") == "1"
        or os.environ.get("HSA_TOOLS_DISABLE_REGISTER") == "1"
    ):
        return True
    logger.warning(
        "Peer L1 copies may share a normal-priority HIP queue with other "
        "streams and delay them. Launch with DEBUG_HIP_IGNORE_STREAM_PRIORITY=0 "
        "and ROCPROFILER_QUEUE_INTERPOSITION=1 before HIP initialization; "
        "legacy profiler interception drops hardware queue priority."
    )
    return False


class IPCSession:
    """Load/store reads are fail-stop; only copy completion releases a grant."""

    access_mode = AccessMode.LOAD_STORE
    cost: Literal["local", "fabric", "rdma"] = "local"

    def __init__(
        self,
        local_storage: torch.UntypedStorage,
        remote_storage: torch.UntypedStorage,
        cleanup: CleanupQueue,
        stream: torch.cuda.Stream,
        *,
        mover: LoadStoreMover | None = None,
    ):
        self.local_storage, self.mapping = local_storage, remote_storage
        self.mover: Mover = mover or LoadStoreMover(cleanup, stream)

    def submit(self, grant: Grant, copies: list[tuple[int, int, int]]) -> Transfer:
        return self.mover.submit([grant], copies)


class PeerWorker:
    def __init__(self, config: VllmConfig, cache: KVCacheConfig):
        self.config, self.cache = config, cache
        self.rank = config.parallel_config.data_parallel_rank
        self.cleanup = CleanupQueue()
        self._stats = PeerL1Stats()
        self._sessions: dict[int, PeerSession] = {}
        self._members: dict[int, PeerMemory] = {}
        self._watermark = CompletionWatermark()
        self._handshake: PeerHandshake | None = None
        self._stream: torch.cuda.Stream | None = None
        self._storage: torch.UntypedStorage | None = None
        self._mover: LoadStoreMover | None = None

    def register(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if not kv_caches:
            raise ValueError("Peer L1 requires a device KV cache")
        storages = {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage()
            for tensor in kv_caches.values()
        }
        if len(storages) != 1:
            raise ValueError("Peer L1 requires one shared device backing allocation")
        self._storage = next(iter(storages.values()))
        if self._storage.device.type != "cuda":
            raise ValueError("Peer L1 IPC requires CUDA or ROCm device memory")
        device = torch.accelerator.current_device_index()
        parallel = self.config.parallel_config
        world = parallel.data_parallel_size
        transfer = self.config.kv_transfer_config
        assert transfer is not None
        assert transfer.engine_id is not None
        instance = hashlib.sha256(transfer.engine_id.encode()).hexdigest()[:24]
        endpoint = f"vllm-peer-l1/{instance}/{self.rank}"
        signature = repr(
            (
                self.config.model_config.model,
                self.config.model_config.revision,
                self.config.model_config.dtype,
                self.cache.kv_cache_layout,
                [
                    (g.manager_config, g.kv_cache_specs)
                    for g in self.cache.kv_cache_groups
                ],
            )
        )
        compatibility = hashlib.sha256(signature.encode()).hexdigest()
        handles = {
            rank: self._storage._share_cuda_()
            for rank in range(world)
            if rank != self.rank
        }
        local = PeerMemory(
            self.rank,
            endpoint,
            secrets.randbits(63) + 1,
            socket.gethostname(),
            device,
            compatibility,
            self.cache.num_blocks,
            cache_segments(self.cache),
            handles,
        )
        gathered: list[PeerMemory | None] = [None] * world
        if world > 1:
            dist.all_gather_object(gathered, local, group=get_dp_group().cpu_group)
        else:
            gathered = [local]
        members = []
        for member in gathered:
            if not isinstance(member, PeerMemory):
                raise RuntimeError("Peer L1 received an invalid member description")
            members.append(member)
        self._stream = torch.cuda.Stream(device=device, priority=-1)
        self._mover = LoadStoreMover(self.cleanup, self._stream, stats=self._stats)
        for member in members:
            if member.host != local.host:
                raise ValueError("Peer L1 requires all DP workers on the same OS")
            if member.compatibility != compatibility:
                raise ValueError(
                    "Peer L1 KV geometry or model compatibility differs across ranks"
                )
            self._members[member.rank] = member
            if member.rank == self.rank:
                continue
            handle = member.handles[self.rank]
            storage = torch.UntypedStorage._new_shared_cuda(device, *handle[1:])
            self._sessions[member.rank] = IPCSession(
                self._storage, storage, self.cleanup, self._stream, mover=self._mover
            )
        # Workers retain imported handles until all readers leave at shutdown.
        self._handshake = PeerHandshake([replace(m, handles={}) for m in members])

    def handshake(self) -> PeerHandshake | None:
        return self._handshake

    def load(self, meta: PeerMetadata) -> None:
        assert self._storage is not None
        local = self._members[self.rank]
        local_base = self._storage.data_ptr()
        copies: list[tuple[int, int, int]] = []
        num_bytes = 0
        for job in meta.jobs:
            session = self._sessions[job.peer]
            remote = self._members[job.peer]
            remote_base = session.mapping.data_ptr()
            for group, source, target in job.blocks:
                if not (
                    0 < source < remote.num_blocks and 0 < target < local.num_blocks
                ):
                    raise RuntimeError("Peer L1 copy row is outside its memory session")
                for src, dst in zip(
                    remote.segments[group], local.segments[group], strict=True
                ):
                    src_offset, src_stride, length = src
                    dst_offset, dst_stride, dst_length = dst
                    if length != dst_length:
                        raise RuntimeError(
                            "Peer L1 source and destination pages differ"
                        )
                    num_bytes += length
                    source_ptr = remote_base + src_offset + source * src_stride
                    target_ptr = local_base + dst_offset + target * dst_stride
                    if copies and (
                        copies[-1][0] + copies[-1][2] == source_ptr
                        and copies[-1][1] + copies[-1][2] == target_ptr
                    ):
                        s, d, n = copies.pop()
                        copies.append((s, d, n + length))
                    else:
                        copies.append((source_ptr, target_ptr, length))
        if meta.jobs:
            assert self._mover is not None
            self._mover.submit(
                [job.grant for job in meta.jobs], copies, num_bytes=num_bytes
            )

    def save(self, meta: PeerMetadata) -> None:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self._watermark.record(meta.step, event.query)

    def report(self) -> PeerWorkerMetadata | None:
        return self._watermark.report()

    def get_stats(self) -> PeerL1Stats:
        if self._mover is not None:
            return self._mover.get_stats()
        return self._stats.clone_and_reset()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()
        self.cleanup.close()
        self._sessions.clear()
