# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-process IPC copies, RDMA dispatch and GPU write-completion reporting."""

import ctypes
import hashlib
import os
import secrets
import socket
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed import get_dp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.simple_kv_offload.cuda_mem_ops import _resolve_max_batch_descriptors

from .async_load import AsyncReceives
from .data import CopyJob, PeerHandshake, PeerMemory, PeerMetadata, PeerWorkerMetadata
from .native import CleanupQueue
from .registry import IPC_LEASE_MS, Domain
from .session import Grant, release_grant
from .stats import PeerL1Stats

logger = init_logger(__name__)

if TYPE_CHECKING:
    from .rdma import RdmaReadMover


def memory_description(
    config: VllmConfig, cache: KVCacheConfig, domain: Domain, device: int
) -> PeerMemory:
    """Describe rank-local memory without opening or importing device handles."""
    transfer = config.kv_transfer_config
    assert transfer is not None and transfer.engine_id is not None
    rank = config.parallel_config.data_parallel_rank
    instance = hashlib.sha256(transfer.engine_id.encode()).hexdigest()[:24]
    signature = repr(
        (
            domain.compatibility_id or config.model_config.model,
            config.model_config.revision,
            config.model_config.dtype,
            cache.kv_cache_layout,
            [(g.manager_config, g.kv_cache_specs) for g in cache.kv_cache_groups],
        )
    )
    return PeerMemory(
        rank,
        domain.endpoint(rank)
        if domain.instance_id
        else f"vllm-peer-l1/{instance}/{rank}",
        secrets.randbits(63) + 1,
        socket.gethostname(),
        device,
        hashlib.sha256(signature.encode()).hexdigest(),
        cache.num_blocks,
        cache_segments(cache),
        {},
        instance_id=domain.instance_id or instance,
        kv_compatibility_id=domain.compatibility_id,
        hash_settings=hashlib.sha256(
            repr(
                (
                    config.cache_config.prefix_caching_hash_algo,
                    os.environ.get("PYTHONHASHSEED", "fixed"),
                    resolve_kv_cache_block_sizes(cache, config),
                    cache.cache_hit_alignment_tokens,
                    config.num_hash_lookahead_tokens,
                    config.num_reprefillable_tokens,
                )
            ).encode()
        ).hexdigest(),
        restart_group=domain.instance_id or instance,
        lease_s=(domain.lease_ms if domain.instance_id else IPC_LEASE_MS) / 1000,
    )


def validated_members(
    gathered: list[PeerMemory | None], *, rank: int, implicit: bool
) -> list[PeerMemory]:
    """Validate the descriptions exchanged by one DP group."""
    members = []
    for member in gathered:
        if not isinstance(member, PeerMemory):
            raise RuntimeError("Peer L1 received an invalid member description")
        members.append(member)
    if {m.rank for m in members} != set(range(len(members))) or not 0 <= rank < len(
        members
    ):
        raise ValueError("Peer L1 requires a complete DP membership")
    if implicit:
        if any(m.kv_compatibility_id for m in members):
            raise ValueError(
                "Peer implicit and explicit instance modes differ across ranks"
            )
        # The gather defines this instance; engine IDs intentionally differ by DP rank.
        instance = next(m.instance_id for m in members if m.rank == 0)
        members = [
            replace(m, instance_id=instance, restart_group=instance) for m in members
        ]
    local = next(m for m in members if m.rank == rank)
    for member in members:
        if member.compatibility != local.compatibility:
            raise ValueError(
                "Peer L1 KV geometry or model compatibility differs across ranks"
            )
        if (
            member.instance_id != local.instance_id
            or member.hash_settings != local.hash_settings
            or member.restart_group != local.restart_group
        ):
            raise ValueError(
                "Peer instance identity or prefix hash settings differ across ranks"
            )
    return members


def exchange_handshake(
    local: PeerMemory,
    credential: bytes,
    *,
    implicit: bool,
    group: dist.ProcessGroup | None = None,
) -> PeerHandshake:
    """Exchange a credential only on the trusted memory-handle bootstrap."""
    proposed = credential or (secrets.token_bytes(32) if local.rank == 0 else b"")
    world = dist.get_world_size(group) if group is not None else 1
    gathered: list[tuple[PeerMemory, bytes] | None] = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, (local, proposed), group=group)
    else:
        gathered = [(local, proposed)]
    if any(item is None for item in gathered):
        raise RuntimeError("Peer L1 received an incomplete memory handshake")
    entries = [item for item in gathered if item is not None]
    members = validated_members(
        [member for member, _ in entries], rank=local.rank, implicit=implicit
    )
    secret = next(value for member, value in entries if member.rank == 0)
    if len(secret) != 32 or any(value and value != secret for _, value in entries):
        raise ValueError("Peer L1 credentials differ across DP ranks")
    return PeerHandshake(members, secret)


class CompletionWatermark:
    """Track consecutive completed steps on the worker's model execution thread.

    Callers record fences after all KV writers, including speculative drafts.
    """

    def __init__(self):
        self._events: deque[tuple[int, Callable[[], bool]]] = deque()
        self._completed = 0
        self._reported = 0

    def record(self, step: int, query: Callable[[], bool]) -> None:
        """Register a nonblocking fence query covering all writers in this step."""
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


class LoadStoreMover:
    """Submit IPC copies from the model thread onto a dedicated copy stream.

    A native host callback releases source grants only after the copies finish;
    stream waits order GPU work without blocking the model thread.
    """

    def __init__(
        self,
        cleanup: CleanupQueue,
        stream: torch.cuda.Stream,
    ):
        if current_platform.is_rocm():
            check_rocm_copy_environment()
        self.cleanup, self.stream = cleanup, stream
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
    ) -> torch.cuda.Event:
        """Order target reuse, copies and grant release; return a completion event.

        The current stream must cover all prior writers to the targets. Grants
        and imported source mappings must remain valid through completion.
        """
        node = 0
        for g in reversed(grants):
            node = self.cleanup.create(
                g.endpoint, g.epoch, g.reader, g.sequence, next_node=node
            )
        # A recycled target can still have speculative writes queued on the
        # model stream. Finish them before the copy stream overwrites that memory.
        previous = torch.cuda.Event()
        previous.record(torch.cuda.current_stream())
        done = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(previous)
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
        main.wait_event(done)
        return done


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


class PeerWorker:
    """Own rank-local KV mappings and copy dispatch in the GPU worker process.

    Model-thread hooks submit IPC work and poll receive results. The RDMA I/O
    thread posts reads only after the model thread's target-write fence.
    """

    def __init__(
        self, config: VllmConfig, cache: KVCacheConfig, *, domain: Domain | None = None
    ):
        self.config, self.cache = config, cache
        transfer = config.kv_transfer_config
        assert transfer is not None
        self.domain = (
            domain
            if domain is not None
            else Domain.from_options(transfer.kv_connector_extra_config)
        )
        if (
            self.domain.instance_id
            and config.cache_config.prefix_caching_hash_algo.startswith("xxhash")
            and not os.environ.get("PYTHONHASHSEED")
        ):
            raise ValueError("Cross-instance xxhash requires a shared PYTHONHASHSEED")
        self.rank = config.parallel_config.data_parallel_rank
        self.cleanup: CleanupQueue | None = None
        self._stats = PeerL1Stats()
        self._receives = AsyncReceives(self._release, self._stats, self._retire)
        self._rdma: RdmaReadMover | None = None
        self._async_jobs: tuple[CopyJob, ...] = ()
        self._completed_step = 0
        self._async_step_submitted = 0
        # Raw copy pointers do not retain imported IPC mappings. Keep them
        # alive until close() has synchronized the copy stream.
        self._imported_storages: dict[int | str, torch.UntypedStorage] = {}
        self._members: dict[int | str, PeerMemory] = {}
        self._watermark = CompletionWatermark()
        self._handshake: PeerHandshake | None = None
        self._stream: torch.cuda.Stream | None = None
        self._storage: torch.UntypedStorage | None = None
        self._mover: LoadStoreMover | None = None

    def register(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register backing storage and exchange handles on the worker startup thread.

        This can block on the native build, RDMA setup and the DP collective.
        All participating ranks must call with compatible KV allocations.
        """
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
        local = memory_description(self.config, self.cache, self.domain, device)
        if self.domain.instance_id:
            from .rdma import RdmaReadMover

            assert self.config.kv_transfer_config is not None
            self._rdma = RdmaReadMover(
                self.domain,
                self._storage,
                self.cache.num_blocks,
                local.segments,
                self._stats,
            )
        handles = {
            rank: self._storage._share_cuda_()
            for rank in range(world)
            if rank != self.rank
        }
        local = replace(
            local,
            handles=handles,
            session_kinds=("ipc", "rdma") if self._rdma is not None else ("ipc",),
            rdma=self._rdma.advertisement if self._rdma is not None else b"",
            rdma_transport=self._rdma.transport.name if self._rdma is not None else "",
        )
        handshake = exchange_handshake(
            local,
            self.domain.credential,
            implicit=not bool(self.domain.instance_id),
            group=get_dp_group().cpu_group if world > 1 else None,
        )
        members = handshake.members
        self.domain = replace(self.domain, credential=handshake.credential)
        self.cleanup = CleanupQueue(credential=handshake.credential)
        local = next(m for m in members if m.rank == self.rank)
        self._stream = torch.cuda.Stream(device=device, priority=-1)
        self._mover = LoadStoreMover(self.cleanup, self._stream)
        for member in members:
            if member.host != local.host and self._rdma is None:
                raise ValueError("Peer L1 requires all DP workers on the same OS")
            self._members[member.rank] = member
            if member.rank == self.rank:
                continue
            if member.host != local.host:
                if not member.rdma:
                    raise ValueError("Remote DP worker has no RDMA memory session")
                continue
            handle = member.handles[self.rank]
            storage = torch.UntypedStorage._new_shared_cuda(device, *handle[1:])
            self._imported_storages[member.rank] = storage
        self._handshake = PeerHandshake(
            [replace(m, handles={}) for m in members], handshake.credential
        )

    def handshake(self) -> PeerHandshake | None:
        return self._handshake

    def load(self, meta: PeerMetadata) -> None:
        """Enqueue IPC copies and save RDMA jobs on the model execution thread.

        Targets must already be allocated. IPC inserts a current-stream wait;
        RDMA submission waits until `save` records the step's last writer.
        """
        assert self._storage is not None
        local = self._members[self.rank]
        local_base = self._storage.data_ptr()
        copies: list[tuple[int, int, int]] = []
        self._async_jobs = tuple(
            job for job in meta.jobs if job.grant.lease_s is not None
        )
        for job in meta.jobs:
            if job.grant.lease_s is not None:
                continue
            num_bytes = 0
            storage = self._imported_storages[job.peer]
            remote = self._members[job.peer]
            remote_base = storage.data_ptr()
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
            self._stats.inc("worker_copy_bytes", num_bytes, transport="ipc")
        synchronous = [job.grant for job in meta.jobs if job.grant.lease_s is None]
        if synchronous:
            assert self._mover is not None
            self._mover.submit(synchronous, copies)

    def save(self, meta: PeerMetadata) -> None:
        """Fence all step writers and queue RDMA reads without waiting on the host.

        The caller must have ordered every writer, including speculative draft
        work, onto the current stream before this hook.
        """
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self._watermark.record(meta.step, event.query)
        for job in self._async_jobs:
            if meta.step <= self._async_step_submitted:
                continue
            assert self._rdma is not None
            read = self._receives.add(job, self.domain.timeout_s)
            self._rdma.submit(read, event.query)
        self._async_jobs = ()
        self._async_step_submitted = max(self._async_step_submitted, meta.step)

    def report(self) -> PeerWorkerMetadata | None:
        """Poll fences and return publication/quarantine updates without waiting."""
        watermark = self._watermark.report()
        if watermark is not None:
            self._completed_step = watermark.completed_step
        meta = self._receives.metadata(self._completed_step)
        if (
            watermark is not None
            or meta.quarantined
            or meta.drained
            or meta.failed_peers
            or meta.successful_peers
        ):
            return meta
        return None

    def transfer_results(self, finished_req_ids: set[str]):
        return self._receives.poll(finished_req_ids)

    def _release(self, grant: Grant) -> None:
        assert self.cleanup is not None
        release_grant(self.cleanup, grant)

    def _retire(self, identity: str) -> None:
        if self._rdma is not None:
            self._rdma.retire(identity)

    def get_stats(self) -> PeerL1Stats:
        return self._stats.clone_and_reset()

    def close(self) -> None:
        """Stop dispatch, synchronize IPC copies and release imported mappings.

        Call after model hooks stop. RDMA registrations and uncertain DMA
        resources may remain alive until process exit.
        """
        if self._rdma is not None:
            self._rdma.close()
        if self._stream is not None:
            self._stream.synchronize()
        if self.cleanup is not None:
            self.cleanup.close()
        self._imported_storages.clear()
