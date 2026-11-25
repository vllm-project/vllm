# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler policy and external block leases for peer L1 sharing."""

import os
import select
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from time import monotonic

import msgspec
import numpy as np

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.lookup import ExternalKVCacheLookup
from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    get_group_id,
    make_block_hash_with_group_id,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import KVCacheConfig, get_reusable_prefix_length
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from .data import CopyJob, PeerHandshake, PeerMemory, PeerMetadata, PeerWorkerMetadata
from .native import CleanupQueue, NativeOwner, PlacementReplica
from .registry import IPC_LEASE_MS, CandidateReplicas, Domain, Manifest, Resolver
from .session import (
    Grant,
    NativeOwnerChannel,
    OwnerBusy,
    OwnerDeferred,
    peer_key,
)
from .stats import PeerL1Stats

logger = init_logger(__name__)

PIN_WINDOW = 16384
PIN_CAP = 65536
RPC_TIMEOUT_MS = 50
BUSY_TIMEOUT_S = 1.0
RDMA_COOLDOWN_S = 1.0
RDMA_COOLDOWN_MAX_S = 30.0
RDMA_QUARANTINE_FRACTION = 1 / 16


@dataclass
class _Read:
    peer: int
    grant: Grant
    tokens: int
    positions: tuple[tuple[int, int], ...]
    job: CopyJob | None = None


@dataclass
class _Queued:
    peer: int
    sequence: int
    keys: list[bytes]
    tokens: int
    local_tokens: int
    positions: tuple[tuple[int, int], ...]
    acquired_before: float


@dataclass
class _Busy:
    deadline: float
    generations: dict[int, int]
    positions: int = 0
    deadline_reported: bool = False
    queued: _Queued | None = None
    terminal: bool = False


@dataclass
class _Cooldown:
    failures: int
    retry_after: float


class PeerScheduler:
    """Manage publication, source leases and target ownership on the scheduler thread.

    Only this thread mutates the block pool. Native threads serve owner RPCs
    and placement updates; the wakeup thread only requests engine work.
    """

    def __init__(
        self,
        config: VllmConfig,
        cache: KVCacheConfig,
        supported: tuple[int, ...],
        *,
        domain: Domain | None = None,
        window: int = PIN_WINDOW,
        pin_cap: int = PIN_CAP,
    ):
        self.config, self.cache, self.supported = config, cache, supported
        transfer = config.kv_transfer_config
        assert transfer is not None
        self.domain = (
            domain
            if domain is not None
            else Domain.from_options(transfer.kv_connector_extra_config)
        )
        self.rank = config.parallel_config.data_parallel_rank
        self.window, self.pin_cap = window, pin_cap
        self.timeout = RPC_TIMEOUT_MS
        self.busy_timeout = BUSY_TIMEOUT_S
        if min(window, pin_cap) <= 0:
            raise ValueError("Peer L1 window and pin cap must be positive")
        self.step = 0
        self._stats = PeerL1Stats()
        self._reserve = 0
        self._published: memoryview | None = None
        self._pending: dict[int, dict[BlockHashWithGroupId, int]] = {}
        self._quarantined: dict[str, tuple[KVCacheBlock, ...]] = {}
        self._quarantine_rows = 0
        self._rdma_disabled = False
        self._cooldowns: dict[str, _Cooldown] = {}
        self._cooldown_s = RDMA_COOLDOWN_S
        self._cooldown_max_s = RDMA_COOLDOWN_MAX_S
        self._quarantine_limit = int(
            max(0, cache.num_blocks - 1) * RDMA_QUARANTINE_FRACTION
        )
        self._reads: dict[str, _Read] = {}
        # Only allocated requests enter this map. True suppresses further RDMA
        # after failure; delayed worker results must not reinsert finished requests.
        self._rdma_admissions: dict[str, bool] = {}
        self._busy: dict[str, _Busy] = {}
        self._round_robin = 0
        self._pool: BlockPool | None = None
        self._manager: KVCacheManager | None = None
        self._owner: NativeOwner | None = None
        self._replica: PlacementReplica | CandidateReplicas | None = None
        self._local_replica: PlacementReplica | None = None
        self._local_count = 0
        self._resolver: Resolver | None = None
        self._remote_version = 0
        self._peer_memory: dict[int | str, PeerMemory] = {}
        self._cleanup: CleanupQueue | None = None
        self._channels: list[NativeOwnerChannel] = []
        self._remote_channels: dict[tuple[str, int], NativeOwnerChannel] = {}
        self._ranks: list[int | str] = []
        self._wake_stop = threading.Event()
        self._wake_thread: threading.Thread | None = None
        block_size, self.hash_size = resolve_kv_cache_block_sizes(cache, config)
        self.alignment = cache.cache_hit_alignment_tokens or block_size
        self._full_groups = tuple(
            group
            for group in supported
            if issubclass(
                cache.kv_cache_groups[group].manager_config.manager_cls,
                FullAttentionManager,
            )
        )
        self._lookup = ExternalKVCacheLookup(
            cache,
            max_model_len=config.model_config.max_model_len,
            max_in_flight_tokens=config.max_in_flight_tokens,
            scheduler_block_size=block_size,
            hash_block_size=self.hash_size,
            num_hash_lookahead_tokens=config.num_hash_lookahead_tokens,
        )

    def bind(self, manager: KVCacheManager) -> None:
        """Install the exclusive lease guard."""
        pool = manager.block_pool
        if pool.lease_guard is not None:
            raise ValueError(
                "Peer L1 requires exclusive ownership of the block lease guard"
            )
        self._manager, self._pool = manager, pool
        pool.lease_guard = self

    def handshake(self, metadata: PeerHandshake) -> None:
        """Start native services after the worker supplies memory and credentials."""
        if self._owner is not None:
            raise RuntimeError("Peer L1 cannot replace a live memory session")
        credential = self.domain.credential or metadata.credential
        if len(credential) != 32 or (
            metadata.credential and metadata.credential != credential
        ):
            raise ValueError(
                "Peer L1 requires the credential from its memory handshake"
            )
        self.domain = replace(self.domain, credential=credential)
        self._cleanup = CleanupQueue(self.timeout, credential=credential)
        local = next(member for member in metadata.members if member.rank == self.rank)
        assert self._pool is not None
        self._reserve = min(
            self.window, self._pool.free_block_queue.num_free_blocks // 9
        )
        self._owner = NativeOwner(
            local.endpoint,
            local.epoch,
            self.cache.num_blocks,
            window=self.window,
            pin_cap=self.pin_cap,
            initial_window=self._reserve,
            lease_ms=self.domain.lease_ms if self.domain.instance_id else IPC_LEASE_MS,
            credential=self.domain.credential,
        )
        self._published = self._owner.published_rows
        peers = [member for member in metadata.members if member.rank != self.rank]
        self._ranks = [member.rank for member in peers]
        self._local_count = len(peers)
        self._local_replica = PlacementReplica(
            [
                (m.endpoint, m.epoch, m.num_blocks * max(1, len(m.segments)))
                for m in peers
            ],
            credential=self.domain.credential,
        )
        self._replica = self._local_replica
        self._peer_memory = {m.rank: m for m in peers}
        self._channels = [
            NativeOwnerChannel(
                m.endpoint,
                m.epoch,
                local.epoch,
                self._cleanup,
                self.timeout,
                lease_s=m.lease_s if m.host != local.host else None,
            )
            for m in peers
        ]
        self._owner.describe(
            msgspec.json.encode(local),
            msgspec.json.encode(
                Manifest(
                    1, local.instance_id, tuple(metadata.members), len(metadata.members)
                )
            ),
        )
        if self.domain.instance_id and self.domain.seeds:
            self._resolver = Resolver(self.domain, local, self.timeout)
            self._resolver.start()

    def _adopt_remotes(self) -> None:
        if self._resolver is None or self._reads:
            return
        snapshot = self._resolver.snapshot()
        if snapshot is None or snapshot.version == self._remote_version:
            return
        for waiting in self._busy.values():
            if waiting.queued is not None and waiting.queued.peer >= self._local_count:
                self._cancel_queued(waiting)
            waiting.generations = {
                p: g for p, g in waiting.generations.items() if p < self._local_count
            }
        self._channels = self._channels[: self._local_count]
        self._ranks = self._ranks[: self._local_count]
        self._peer_memory = {
            k: v for k, v in self._peer_memory.items() if isinstance(k, int)
        }
        assert self._owner is not None and self._local_replica is not None
        assert self._cleanup is not None
        for member in snapshot.members:
            identity = f"{member.instance_id}/{member.rank}/{member.epoch}"
            self._ranks.append(identity)
            self._peer_memory[identity] = member
            key = member.endpoint, member.epoch
            channel = self._remote_channels.get(key)
            if channel is None:
                channel = NativeOwnerChannel(
                    member.endpoint,
                    member.epoch,
                    self._owner.epoch,
                    self._cleanup,
                    self.timeout,
                    lease_s=member.lease_s,
                )
                self._remote_channels[key] = channel
            self._channels.append(channel)
        self._replica = CandidateReplicas(
            self._local_replica, snapshot.replica, self._local_count
        )
        self._remote_version = snapshot.version
        self._resolver.acknowledge(snapshot.version)

    def set_engine_wakeup(self, callback: Callable[[], None]) -> None:
        """Install a callback safe to invoke from the eventfd watcher thread.

        The callback must enqueue engine work without accessing the block pool.
        """
        assert self._owner is not None
        fd = self._owner.wake_fd

        def watch():
            while not self._wake_stop.is_set():
                if select.select([fd], [], [], 0.1)[0]:
                    try:
                        os.eventfd_read(fd)
                    except BlockingIOError:
                        continue
                    if not self._wake_stop.is_set():
                        callback()

        self._wake_thread = threading.Thread(target=watch, daemon=True)
        self._wake_thread.start()
        if self.pending():
            callback()

    def on_cached(
        self,
        block: KVCacheBlock,
        key: BlockHashWithGroupId,
        *,
        num_tokens: int | None,
    ) -> None:
        """Stage a full-block alias until the worker reports its writes complete."""
        group = get_group_id(key)
        if (
            group not in self.supported
            or num_tokens is None
            or num_tokens % self.cache.kv_cache_groups[group].block_size
        ):
            return
        self._pending.setdefault(block.block_id, {})[key] = self.step + 1

    def on_uncached(self, block: KVCacheBlock, *, popped: bool) -> bool:
        """Remove aliases before reuse; return whether external pins prevent reuse.

        Adopt a pool reference for pinned in-use rows before local free, or
        withhold pinned free rows from allocation until a later absorb.
        """
        self._pending.pop(block.block_id, None)
        # Only this scheduler changes publication flags, so the unlocked zero
        # check is safe. Nonzero rows still require locked retirement for RPC pins.
        if (
            self._owner is None
            or self._published is None
            or not self._published[block.block_id]
        ):
            return False
        result = self._owner.retire_for_pool(
            block.block_id, withhold=block.ref_cnt == 0
        )
        # ABI result 2 transfers an unabsorbed owner pin into a pool reference.
        if result == 2:
            assert self._pool is not None
            self._pool.touch([block])
        pinned = bool(result)
        if pinned and block.ref_cnt == 0:
            self._reserve -= 1
            assert self._reserve >= 0
        return pinned

    def admission_reserve(self) -> int:
        return self._reserve

    @contextmanager
    def reset_scope(self):
        """Freeze new grants while the caller attempts a synchronous pool reset.

        The scheduler must not allocate or yield to another pool mutator inside
        this scope. The yielded flag says whether external ownership is clear.
        """
        if self._owner is None:
            yield False
            return
        ready = self._owner.freeze()
        ready = ready and not self._pending
        try:
            yield ready
        finally:
            assert self._pool is not None
            clear = ready and not len(self._pool.cached_block_hash_to_block)
            self._owner.thaw(clear=clear)

    def pending(self) -> bool:
        return bool(self._pending or (self._owner and self._owner.pending()))

    def _local(self, block_hash: BlockHash, group: int, local_tokens: int) -> bool:
        assert self._manager is not None
        pool = self._manager.coordinator.single_type_managers[group].block_pool
        blocks = pool.get_cached_block(block_hash, [group])
        return bool(
            blocks
            and blocks[0].block_hash_num_tokens is not None
            and blocks[0].block_hash_num_tokens <= local_tokens
        )

    def get_stats(self) -> PeerL1Stats:
        return self._stats.clone_and_reset()

    def _cancel_queued(self, waiting: _Busy | None) -> None:
        if waiting is not None and waiting.queued is not None:
            queued = waiting.queued
            self._channels[queued.peer].cancel_unsubmitted(queued.sequence)
            waiting.queued = None

    def _accept(
        self,
        request: Request,
        local_tokens: int,
        limit: int,
        peer: int,
        grant: Grant,
        length: int,
        positions: tuple[tuple[int, int], ...],
    ) -> tuple[int, bool]:
        self._decision(request, limit - local_tokens, length - local_tokens)
        self._stats.inc(
            "reader_tokens",
            length - local_tokens,
            transport="ipc" if grant.lease_s is None else "rdma",
        )
        self._stats.inc("reader_granted_rows", len(grant.rows))
        self._busy.pop(request.request_id, None)
        self._round_robin = peer + 1
        self._reads[request.request_id] = _Read(
            peer, grant, length - local_tokens, positions
        )
        return length - local_tokens, grant.lease_s is not None

    def _rdma_blocked(self, peer: int, request_id: str) -> bool:
        channel = self._channels[peer]
        if channel.lease_s is None:
            return False
        cooldown = self._cooldowns.get(channel.endpoint)
        return (
            self._rdma_admissions.get(request_id, False)
            or self._rdma_disabled
            or (cooldown is not None and monotonic() < cooldown.retry_after)
        )

    def _rdma_health_metrics(self) -> None:
        self._stats.gauge("reader_rdma_quarantine_disabled", int(self._rdma_disabled))

    def _deadline_fallback(
        self, request: Request, waiting: _Busy, eligible: int
    ) -> tuple[int, bool]:
        if not waiting.deadline_reported:
            self._stats.inc("reader_deadline_fallbacks")
            self._decision(
                request, eligible, 0, reason="deadline", positions=waiting.positions
            )
            self._cancel_queued(waiting)
            waiting.deadline_reported = True
        return 0, False

    def _decision(
        self,
        request: Request,
        eligible: int,
        hit: int,
        *,
        reason: str = "no_match",
        positions: int = 0,
    ) -> None:
        if not hit and eligible:
            logger.debug(
                "Peer L1 fallback request=%s positions=%d granted=0 reason=%s",
                request.request_id,
                positions,
                reason,
            )

    def _full_budget(self, length: int, local_tokens: int, budget: int) -> int:
        """Bound length by the rows full attention alone would have to load."""
        if not self._full_groups:
            return length
        low, high = local_tokens // self.alignment, length // self.alignment
        while low < high:
            middle = (low + high + 1) // 2
            tokens = middle * self.alignment
            required = sum(
                max(
                    0,
                    tokens // self.cache.kv_cache_groups[g].block_size
                    - local_tokens // self.cache.kv_cache_groups[g].block_size,
                )
                for g in self._full_groups
            )
            if required <= budget:
                low = middle
            else:
                high = middle - 1
        return low * self.alignment

    def match(self, request: Request, local_tokens: int) -> tuple[int | None, bool]:
        """Return extra prefix tokens and whether the worker must load asynchronously.

        `None` tokens asks the scheduler to retry a busy or deferred owner.
        Each owner RPC can block for `self.timeout` milliseconds; one lookup
        may contact multiple peers. A selected grant pins all requested rows.
        """
        self._adopt_remotes()
        self._stats.inc("reader_lookups")
        saved = self._reads.get(request.request_id)
        if saved is not None:
            return saved.tokens, saved.grant.lease_s is not None
        limit = (
            min(
                get_reusable_prefix_length(
                    request.num_tokens, self.config.num_hash_lookahead_tokens
                ),
                len(request.block_hashes) * self.hash_size,
            )
            // self.alignment
            * self.alignment
        )
        if self._replica is None or not self._channels:
            self._decision(request, max(0, limit - local_tokens), 0, reason="no_peers")
            return 0, False
        replica = self._replica
        if limit <= local_tokens:
            waiting = self._busy.get(request.request_id)
            self._cancel_queued(waiting)
            return 0, False
        waiting = self._busy.get(request.request_id)
        if waiting is not None and waiting.terminal:
            return 0, False
        if waiting is not None and waiting.queued is not None:
            queued = waiting.queued
            if self._rdma_blocked(queued.peer, request.request_id):
                self._cancel_queued(waiting)
                waiting.generations.pop(queued.peer, None)
            elif queued.local_tokens != local_tokens:
                self._cancel_queued(waiting)
                waiting.generations.clear()
            elif (
                replica.generation(queued.peer) <= waiting.generations[queued.peer]
                and monotonic() < waiting.deadline
            ):
                return None, False
            else:
                try:
                    grant = self._channels[queued.peer].acquire_exact(
                        queued.keys,
                        defer=True,
                        sequence=queued.sequence,
                        acquired_before=queued.acquired_before,
                    )
                except OwnerDeferred as deferred:
                    waiting.generations[queued.peer] = deferred.generation
                    if monotonic() >= waiting.deadline:
                        return self._deadline_fallback(
                            request, waiting, limit - local_tokens
                        )
                    return None, False
                if grant is not None:
                    return self._accept(
                        request,
                        local_tokens,
                        limit,
                        queued.peer,
                        grant,
                        queued.tokens,
                        queued.positions,
                    )
                self._cancel_queued(waiting)
                waiting.terminal = True
                self._decision(
                    request,
                    limit - local_tokens,
                    0,
                    reason="deferred_rejected",
                    positions=len(queued.positions),
                )
                return 0, False
        if waiting is not None and monotonic() >= waiting.deadline:
            return self._deadline_fallback(request, waiting, limit - local_tokens)
        available = []
        for peer in range(len(self._channels)):
            if waiting is not None and peer in waiting.generations:
                if replica.generation(peer) <= waiting.generations[peer]:
                    continue
                del waiting.generations[peer]
            available.append(peer)
        if not available:
            assert waiting is not None
            return None, False
        identities: dict[tuple[BlockHash, int], bytes] = {}
        local: dict[tuple[BlockHash, int], bool] = {}
        remote: dict[tuple[BlockHash, int], np.ndarray] = {}

        def identity(h: BlockHash, group: int) -> bytes:
            pair = (h, group)
            if pair not in identities:
                identities[pair] = peer_key(make_block_hash_with_group_id(h, group))
            return identities[pair]

        def is_local(h: BlockHash, group: int) -> bool:
            pair = (h, group)
            if pair not in local:
                local[pair] = self._local(h, group, local_tokens)
            return local[pair]

        # Full-attention rows bound candidate prefixes. Let the coordinator
        # choose sliding-window and speculative replay-boundary keys; requiring
        # every earlier window row would reject valid bounded-cache hits.
        full = []
        for group in self.supported:
            spec = self.cache.kv_cache_groups[group]
            if not issubclass(spec.manager_config.manager_cls, FullAttentionManager):
                continue
            size = spec.block_size
            for end in range((local_tokens // size + 1) * size, limit + 1, size):
                h = request.block_hashes[end // self.hash_size - 1]
                full.append((h, group, end - size))
        matrix = replica.query([identity(h, g) for h, g, _ in full])
        bounds = [limit] * len(self._channels)
        for (h, group, start), hits in zip(full, matrix):
            remote[h, group] = hits
            for peer, hit in enumerate(hits):
                if not hit and start < bounds[peer] and not is_local(h, group):
                    bounds[peer] = start
        selected = None
        order = sorted(
            available,
            key=lambda peer: (
                int(self._channels[peer].lease_s is not None),
                -bounds[peer],
                (peer - self._round_robin) % len(self._channels),
            ),
        )
        best = local_tokens
        fallback_reason, fallback_positions = "no_match", 0
        for peer in order:
            if self._rdma_blocked(peer, request.request_id):
                fallback_reason = (
                    "rdma_request_failed"
                    if self._rdma_admissions.get(request.request_id, False)
                    else "rdma_quarantine_budget"
                    if self._rdma_disabled
                    else "rdma_cooldown"
                )
                continue
            if bounds[peer] <= best:
                continue

            def presence(h: BlockHash, group: int, peer_index: int = peer) -> bool:
                if is_local(h, group):
                    return True
                pair = (h, group)
                if pair not in remote:
                    remote[pair] = replica.query([identity(h, group)])[0]
                return bool(remote[pair][peer_index])

            advertised_window = replica.window(peer)
            row_limit = min(self.window, self.pin_cap)
            length = self._full_budget(bounds[peer], local_tokens, row_limit)
            requested_positions = None
            budget_truncated = length < bounds[peer]
            while length > local_tokens:
                length, hashes = self._lookup.find_longest_cache_hit(
                    request.block_hashes,
                    length,
                    presence,
                    num_request_tokens=request.num_tokens,
                )
                positions = tuple(
                    (g, index)
                    for g, group_hashes in enumerate(hashes)
                    for index, h in enumerate(group_hashes)
                    if h is not None and not is_local(h, g)
                )
                if requested_positions is None:
                    requested_positions = len(positions)
                if len(positions) <= row_limit:
                    break
                length = (length // self.alignment - 1) * self.alignment
            granted_positions = len(positions) if length > local_tokens else 0
            fallback_positions = max(fallback_positions, requested_positions or 0)
            if budget_truncated or (
                requested_positions is not None
                and requested_positions > granted_positions
            ):
                reason = "window" if self.window <= self.pin_cap else "pin_cap"
                fallback_reason = reason
                self._stats.inc("reader_truncations", reason=reason)
                logger.debug(
                    "Peer L1 truncation request=%s granted=%d reason=%s",
                    request.request_id,
                    granted_positions,
                    reason,
                )
            if length <= best:
                continue
            if (
                self._channels[peer].lease_s is not None
                and length - best < self.domain.min_gain_pages * self.alignment
            ):
                continue
            keys = []
            for group, index in positions:
                block_hash = hashes[group][index]
                assert block_hash is not None
                keys.append(identity(block_hash, group))
            if keys:
                # Prune only after authorization succeeds: a stale best peer
                # must not hide the next complete candidate.
                try:
                    grant = self._channels[peer].acquire_exact(
                        keys,
                        defer=selected is None and len(keys) > advertised_window,
                    )
                except OwnerDeferred as deferred:
                    if waiting is None:
                        waiting = _Busy(monotonic() + self.busy_timeout, {})
                        self._busy[request.request_id] = waiting
                    waiting.generations[peer] = deferred.generation
                    waiting.positions = max(waiting.positions, len(positions))
                    waiting.queued = _Queued(
                        peer,
                        deferred.sequence,
                        keys,
                        length,
                        local_tokens,
                        positions,
                        deferred.acquired_before,
                    )
                    return None, False
                except OwnerBusy as busy:
                    if waiting is None:
                        waiting = _Busy(monotonic() + self.busy_timeout, {})
                        self._busy[request.request_id] = waiting
                    # A reply can reveal a smaller window than a stale replica.
                    # Retry once that generation (and its budget) is visible.
                    waiting.generations[peer] = busy.generation - int(
                        busy.window < advertised_window
                    )
                    waiting.positions = max(waiting.positions, len(positions))
                    continue
                if grant is not None:
                    if selected is not None:
                        _, previous_peer, _, previous_grant = selected
                        self._channels[previous_peer].cancel_unsubmitted(
                            previous_grant.sequence
                        )
                    best = length
                    selected = (length, peer, positions, grant)
                else:
                    fallback_reason = "authorization"
        if selected is not None:
            length, peer, positions, grant = selected
            return self._accept(
                request, local_tokens, limit, peer, grant, length, positions
            )
        if waiting is not None and waiting.generations:
            return None, False
        self._decision(
            request,
            limit - local_tokens,
            0,
            reason=fallback_reason,
            positions=fallback_positions,
        )
        self._busy.pop(request.request_id, None)
        return 0, False

    def allocated(self, request: Request, blocks: KVCacheBlocks, tokens: int) -> None:
        """Pair source rows with allocated targets, or cancel an unused hit."""
        read = self._reads.get(request.request_id)
        if read is None:
            return
        if tokens == 0 and read.job is None:
            self._reads.pop(request.request_id)
            self._channels[read.peer].cancel_unsubmitted(read.grant.sequence)
            return
        if tokens != read.tokens:
            raise RuntimeError("Peer L1 allocation differs from the acquired prefix")
        ids = blocks.get_block_ids()
        plan = tuple(
            (group, src, ids[group][index])
            for (group, index), src in zip(read.positions, read.grant.rows, strict=True)
            if ids[group][index] != 0
        )
        if len(plan) != len(read.positions):
            raise RuntimeError("Peer L1 required a null destination block")
        identity = self._ranks[read.peer]
        read.job = CopyJob(
            identity,
            read.grant,
            plan,
            request.request_id,
            self._peer_memory.get(identity) if read.grant.lease_s is not None else None,
        )
        if read.grant.lease_s is not None:
            self._rdma_admissions.setdefault(request.request_id, False)

    def build(self) -> PeerMetadata:
        """Emit copy jobs and reconcile native pins with pool references atomically.

        No pool allocation may interleave with the deferred-row snapshot,
        absorb actions and reserve update. Unallocated grants are cancelled.
        """
        self._rdma_health_metrics()
        jobs = []
        for read in self._reads.values():
            if read.job is None:
                self._channels[read.peer].cancel_unsubmitted(read.grant.sequence)
            else:
                jobs.append(read.job)
        self._reads.clear()
        self._adopt_remotes()
        if self._owner is not None:
            assert self._pool is not None
            touch, free, withheld = self._owner.absorb(
                free_blocks=self._pool.free_block_queue.num_free_blocks,
                in_use_rows=[
                    row
                    for row in self._owner.deferred_rows()
                    if self._pool.blocks[row].ref_cnt > 0
                ],
            )
            self._pool.touch([self._pool.blocks[i] for i in touch])
            self._pool.free_blocks(self._pool.blocks[i] for i in free)
            self._pool.return_withheld([self._pool.blocks[i] for i in withheld])
            physical_free = self._pool.free_block_queue.num_free_blocks
            owner_stats = self._owner.absorb_stats()
            if (
                min(self.window, physical_free // 9)
                > owner_stats["owner_effective_window"]
            ):
                self._owner.complete_absorb(physical_free)
                owner_stats = self._owner.absorb_stats()
            self._reserve = owner_stats["owner_reserve"]
            for key, kind, family, reason, stage in self._owner.stat_schema:
                value = owner_stats[key]
                if kind == "counter":
                    if value:
                        self._stats.inc(family, value, reason=reason, stage=stage)
                elif kind == "gauge":
                    self._stats.gauge(family, value)
                elif kind != "internal":
                    raise ValueError(f"Unknown peer L1 stat kind: {kind}")
        self.step += 1
        return PeerMetadata(self.step, tuple(jobs))

    def update(self, output: KVConnectorOutput) -> None:
        """Apply worker results before core releases failed receive targets.

        Quarantine takes extra target references independently of the request.
        Only aliases covered by the completed-step watermark are published.
        """
        for request_id in output.failed_recving or ():
            if request_id in self._rdma_admissions:
                self._rdma_admissions[request_id] = True
        meta = output.kv_connector_worker_meta
        if not isinstance(meta, PeerWorkerMetadata):
            return
        if self._resolver is not None:
            for endpoint in meta.failed_peers:
                self._resolver.invalidate(endpoint)
        for endpoint, acquired_before in meta.successful_peers.items():
            cooldown = self._cooldowns.get(endpoint)
            if (
                endpoint not in meta.failed_peers
                and cooldown is not None
                and acquired_before >= cooldown.retry_after
            ):
                del self._cooldowns[endpoint]
        for endpoint in meta.failed_peers:
            previous = self._cooldowns.get(endpoint)
            failures = 1 if previous is None else previous.failures + 1
            delay = min(
                self._cooldown_max_s, self._cooldown_s * 2 ** min(failures - 1, 30)
            )
            self._cooldowns[endpoint] = _Cooldown(failures, monotonic() + delay)
        assert self._pool is not None
        for transfer, quarantined_rows in meta.quarantined.items():
            if transfer not in self._quarantined:
                blocks = tuple(self._pool.blocks[row] for row in set(quarantined_rows))
                self._pool.touch(blocks)
                self._quarantined[transfer] = blocks
                self._quarantine_rows += len(blocks)
                logger.error(
                    "Peer L1 quarantined transfer=%s rows=%d; "
                    "retaining targets until drain",
                    transfer,
                    len(blocks),
                )
        for transfer in meta.drained:
            blocks = self._quarantined.pop(transfer, ())
            self._pool.free_blocks(blocks)
            self._quarantine_rows -= len(blocks)
        self._stats.gauge("reader_quarantine_refs", self._quarantine_rows)
        disabled = self._quarantine_rows > self._quarantine_limit
        if disabled and not self._rdma_disabled:
            logger.error(
                "Peer L1 RDMA disabled: quarantined rows=%d limit=%d; "
                "inspect RDMA failures",
                self._quarantine_rows,
                self._quarantine_limit,
            )
        self._rdma_disabled = disabled
        self._rdma_health_metrics()
        assert self._owner is not None
        keys, rows = [], []
        for row, pending in list(self._pending.items()):
            for key, step in list(pending.items()):
                if step <= meta.completed_step:
                    keys.append(peer_key(key))
                    rows.append(row)
                    del pending[key]
            if not pending:
                del self._pending[row]
        if keys:
            self._owner.publish(keys, rows)

    def finished(self, request_id: str) -> None:
        """Cancel unsubmitted grants; submitted copies keep their cleanup state."""
        self._rdma_admissions.pop(request_id, None)
        waiting = self._busy.pop(request_id, None)
        self._cancel_queued(waiting)
        read = self._reads.pop(request_id, None)
        if read is not None:
            self._channels[read.peer].cancel_unsubmitted(read.grant.sequence)

    def close(self) -> None:
        """Stop services after the caller has stopped engine hooks and pool work."""
        self._wake_stop.set()
        if self._wake_thread is not None:
            self._wake_thread.join()
        for request_id in self._reads.keys() | self._busy.keys():
            self.finished(request_id)
        if self._resolver is not None:
            self._resolver.close()
        if self._local_replica is not None:
            self._local_replica.close()
        if self._cleanup is not None:
            self._cleanup.close()
        if self._owner is not None:
            self._owner.close()
        if self._pool is not None:
            self._pool.lease_guard = None
