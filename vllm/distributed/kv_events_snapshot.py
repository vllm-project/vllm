# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dependency-preserving KV event snapshots.

Snapshots are replay programs for a fresh, private consumer index: source
BlockStored events in their original order, then BlockRemoved events to leave
exactly the live residency and reference counts. Source events are never split:
their token/hash alignment and extra keys retain their original meaning.

The recorder must observe the stream from its beginning. Missing metadata or
resource exhaustion disables snapshots rather than returning partial state.
"""

import queue
import threading
import time
from collections import Counter, deque
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import msgspec
import zmq

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
    KVEventBatch,
    ZmqEventPublisher,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    ExternalBlockHash,
    maybe_convert_block_hash,
)

logger = init_logger(__name__)
SNAPSHOT_UNAVAILABLE_SEQ = -2
_Scope = tuple[str | None, int | None, str | None, str | None]
_BlockKey = tuple[str | None, int | None, str | None, str | None, ExternalBlockHash]


@dataclass(eq=False)
class _Source:
    event: BlockStored
    dependencies: tuple[int, ...]
    cost: int
    references: int = 0
    orphaned: bool = False
    orphan_generation: int = 0


class KVCacheSnapshot:
    """Live references and the source events required to reconstruct them.

    A source is retained while a live key or another retained source needs it.
    Unreferenced source metadata is retained in a bounded FIFO so asynchronous
    GPU removal and CPU storage can transfer metadata across scheduler batches.
    """

    MAX_METADATA_BYTES = 256 * 1024 * 1024
    MAX_ORPHAN_METADATA_BYTES = 64 * 1024 * 1024
    MAX_REFERENCES = 1_000_000

    def __init__(self) -> None:
        self._live: dict[_BlockKey, tuple[int, int]] = {}
        self._sources: dict[int, _Source] = {}
        self._known: dict[ExternalBlockHash, dict[int, None]] = {}
        self._unused: deque[tuple[int, int]] = deque()
        self._next_id = 0
        self._metadata_bytes = 0
        self._orphan_metadata_bytes = 0
        self._references = 0

    def __len__(self) -> int:
        return len(self._live)

    @staticmethod
    def _keys(event: BlockStored) -> Iterator[_BlockKey]:
        return (
            (event.medium, event.group_idx, event.locality, event.ownership, h)
            for h in event.block_hashes
        )

    @staticmethod
    def _metadata_hash(h: ExternalBlockHash) -> ExternalBlockHash:
        # Offloading emits raw bytes even when GPU events use integer hashes.
        # Normalize lookup only; source events keep their original wire values.
        if isinstance(h, bytes):
            return maybe_convert_block_hash(BlockHash(h))
        return h

    def _dependency(self, h: ExternalBlockHash) -> int:
        sources = self._known.get(self._metadata_hash(h))
        if not sources:
            raise ValueError(f"Missing reconstruction metadata for block {h!r}")
        return next(reversed(sources))

    def _acquire(self, source_id: int) -> None:
        source = self._sources[source_id]
        if source.orphaned:
            source.orphaned = False
            self._orphan_metadata_bytes -= source.cost
        source.references += 1

    def apply(self, events: list[Any]) -> None:
        for event in events:
            if isinstance(event, BlockStored):
                self._store(event)
            elif isinstance(event, BlockRemoved):
                for h in event.block_hashes:
                    key = (
                        event.medium,
                        event.group_idx,
                        event.locality,
                        event.ownership,
                        h,
                    )
                    if key in self._live:
                        count, source = self._live[key]
                        self._references -= 1
                        if count > 1:
                            self._live[key] = (count - 1, source)
                        else:
                            del self._live[key]
                            self._release(source)
            elif isinstance(event, AllBlocksCleared):
                for key in list(self._live):
                    if key[0] in (MEDIUM_GPU, None):
                        count, source = self._live.pop(key)
                        self._references -= count
                        self._release(source)
            else:
                raise ValueError(f"Unsupported KV event: {type(event).__name__}")
        self._collect()

    def _store(self, event: BlockStored) -> None:
        if not event.block_hashes:
            return
        dependencies: set[int] = set()
        if event.parent_block_hash is not None:
            parent_sources = self._known.get(
                self._metadata_hash(event.parent_block_hash)
            )
            if parent_sources:
                dependencies.add(next(reversed(parent_sources)))
            else:
                dependencies.update(self._dependency(h) for h in event.block_hashes)
                event = msgspec.structs.replace(
                    event, parent_block_hash=None, token_ids=[]
                )
        elif not event.token_ids:
            dependencies.update(self._dependency(h) for h in event.block_hashes)
        # Account conservatively for decoded integers, containers and wire data.
        cost = 512 + 64 * (len(event.block_hashes) + len(event.token_ids))
        cost += len(msgspec.msgpack.encode(event))
        dependencies_tuple = tuple(sorted(dependencies))
        for dep in dependencies_tuple:
            self._acquire(dep)
        try:
            self._collect(self.MAX_METADATA_BYTES - cost)
            if self._metadata_bytes + cost > self.MAX_METADATA_BYTES:
                raise ValueError("Snapshot metadata budget exceeded")
            if self._references + len(event.block_hashes) > self.MAX_REFERENCES:
                raise ValueError("Snapshot reference budget exceeded")
        except Exception:
            for dep in dependencies_tuple:
                self._release(dep)
            raise
        source_id = self._next_id
        self._next_id += 1
        self._sources[source_id] = _Source(event, dependencies_tuple, cost)
        self._metadata_bytes += cost
        if event.token_ids:
            for h in event.block_hashes:
                self._known.setdefault(self._metadata_hash(h), {})[source_id] = None
        for key in self._keys(event):
            count = 0
            if key in self._live:
                count, old = self._live[key]
                self._release(old)
            self._live[key] = (count + 1, source_id)
            self._acquire(source_id)
            self._references += 1

    def _release(self, source_id: int) -> None:
        source = self._sources[source_id]
        source.references -= 1
        if source.references == 0 and not source.orphaned:
            source.orphaned = True
            source.orphan_generation += 1
            self._orphan_metadata_bytes += source.cost
            self._unused.append((source_id, source.orphan_generation))

    def _collect(self, metadata_limit: int | None = None) -> None:
        while self._unused and (
            self._orphan_metadata_bytes > self.MAX_ORPHAN_METADATA_BYTES
            or (metadata_limit is not None and self._metadata_bytes > metadata_limit)
        ):
            source_id, orphan_generation = self._unused.popleft()
            source = self._sources.get(source_id)
            if (
                source is None
                or source.references
                or not source.orphaned
                or source.orphan_generation != orphan_generation
            ):
                continue
            del self._sources[source_id]
            self._metadata_bytes -= source.cost
            self._orphan_metadata_bytes -= source.cost
            if source.event.token_ids:
                for h in {self._metadata_hash(h) for h in source.event.block_hashes}:
                    known = self._known[h]
                    known.pop(source_id, None)
                    if not known:
                        del self._known[h]
            for dep in source.dependencies:
                self._release(dep)

    def export(self, max_blocks_per_event: int = 1024) -> Iterator[KVCacheEvent]:
        """Replay intact source events, then correct excess residency.

        Only removals may be chunked. Splitting stores changes sparse or
        canonical-block mappings in consumers.
        """
        emitted: Counter[_BlockKey] = Counter()
        for source in self._sources.values():
            yield source.event
            emitted.update(self._keys(source.event))
        for key, (count, source_id) in self._live.items():
            source = self._sources[source_id]
            while emitted[key] < count:
                yield source.event
                emitted.update(self._keys(source.event))
        removals: dict[_Scope, list[ExternalBlockHash]] = {}
        for key, count in emitted.items():
            excess = count - self._live.get(key, (0, 0))[0]
            if excess:
                hashes = removals.setdefault(key[:4], [])
                for _ in range(excess):
                    hashes.append(key[4])
                    if len(hashes) == max_blocks_per_event:
                        yield BlockRemoved(
                            block_hashes=hashes,
                            medium=key[0],
                            group_idx=key[1],
                            locality=key[2],
                            ownership=key[3],
                        )
                        hashes = removals[key[:4]] = []
        for (medium, group, locality, ownership), hashes in removals.items():
            if hashes:
                yield BlockRemoved(
                    block_hashes=hashes,
                    medium=medium,
                    group_idx=group,
                    locality=locality,
                    ownership=ownership,
                )


class KVEventSnapshotRecorder:
    """Owns snapshot state and its ROUTER socket on one thread.

    Input is the immutable payload sent on PUB. Resource exhaustion invalidates
    the recorder for this publisher lifetime; live publishing continues.
    """

    EVENTS_PER_CHUNK = 256
    BLOCKS_PER_EVENT = 1024
    POLL_INTERVAL_MS = 20
    MAX_PENDING_BYTES = 64 * 1024 * 1024
    MAX_REPLY_BYTES = 256 * 1024 * 1024
    MAX_REQUESTS = 32

    def __init__(
        self, endpoint: str, data_parallel_rank: int, stream_id: bytes
    ) -> None:
        self._dp_rank = data_parallel_rank
        self._stream_id = stream_id
        self._inbox: queue.Queue[tuple[int, bytes]] = queue.Queue(maxsize=4096)
        self._pending_bytes = 0
        self._lock = threading.Lock()
        self._snapshot = KVCacheSnapshot()
        self._seq = -1
        self._failed = threading.Event()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self.endpoint: str | None = None
        self._thread = threading.Thread(
            target=self._run, args=(endpoint,), daemon=True, name="kv-snapshot-recorder"
        )
        self._thread.start()
        self._ready.wait(timeout=5)
        if self.endpoint is None:
            self.shutdown(timeout=1)
            raise RuntimeError(f"Unable to bind KV snapshot endpoint {endpoint}")

    def record(self, seq: int, payload: bytes) -> None:
        with self._lock:
            if self._failed.is_set() or self._stop.is_set():
                return
            if self._pending_bytes + len(payload) > self.MAX_PENDING_BYTES:
                self._failed.set()
                return
            try:
                self._inbox.put_nowait((seq, payload))
            except queue.Full:
                self._failed.set()
                return
            self._pending_bytes += len(payload)

    def shutdown(self, timeout: float) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    def _run(self, endpoint: str) -> None:
        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
        try:
            with zmq.Context.instance().socket(zmq.ROUTER) as router:
                router.setsockopt(zmq.LINGER, 0)
                router.setsockopt(zmq.SNDHWM, self.MAX_REQUESTS)
                router.setsockopt(zmq.SNDTIMEO, 100)
                self.endpoint = ZmqEventPublisher._bind(router, endpoint)
                self._ready.set()
                while not self._stop.is_set():
                    if router.poll(self.POLL_INTERVAL_MS):
                        self._serve(router, encoder, decoder)
                    else:
                        self._drain(decoder)
        except Exception:
            self._failed.set()
            logger.exception("KV snapshot service stopped; live publishing continues")
        finally:
            self._ready.set()

    def _drain(self, decoder: msgspec.msgpack.Decoder) -> None:
        # A finite FIFO cut: future arrivals cannot postpone this snapshot.
        for _ in range(self._inbox.qsize()):
            with self._lock:
                try:
                    seq, payload = self._inbox.get_nowait()
                except queue.Empty:
                    break
                self._pending_bytes -= len(payload)
            if self._failed.is_set():
                continue
            try:
                self._snapshot.apply(decoder.decode(payload).events)
                self._seq = seq
            except Exception:
                logger.exception("KV snapshot fold failed at sequence %d", seq)
                self._failed.set()
        if self._failed.is_set():
            self._snapshot = KVCacheSnapshot()

    def _serve(self, router, encoder, decoder) -> None:
        envelopes: list[list[bytes]] = []
        for _ in range(self.MAX_REQUESTS):
            if not router.poll(0):
                break
            frames = router.recv_multipart()
            try:
                envelopes.append(frames[: frames.index(b"") + 1])
            except ValueError:
                envelopes.append(frames[:1])
        self._drain(decoder)
        reply = [self._seq.to_bytes(8, "big", signed=True), self._stream_id]
        if not self._failed.is_set():
            try:
                size = 0
                for chunk in self._encode_chunks(encoder):
                    size += len(chunk)
                    if size > self.MAX_REPLY_BYTES:
                        raise ValueError("Snapshot response budget exceeded")
                    reply.append(chunk)
            except Exception:
                logger.exception("KV snapshot export failed")
                self._failed.set()
        if self._failed.is_set():
            reply = [
                SNAPSHOT_UNAVAILABLE_SEQ.to_bytes(8, "big", signed=True),
                self._stream_id,
            ]
        for envelope in envelopes:
            # A slow requester retries; it cannot hold the recorder thread.
            with suppress(zmq.Again):
                router.send_multipart(envelope + reply, flags=zmq.DONTWAIT, copy=False)

    def _encode_chunks(self, encoder: msgspec.msgpack.Encoder) -> Iterator[bytes]:
        ts = time.time()
        events: list[KVCacheEvent] = []
        for event in self._snapshot.export(self.BLOCKS_PER_EVENT):
            events.append(event)
            if len(events) == self.EVENTS_PER_CHUNK:
                yield encoder.encode(self._chunk(ts, events))
                events = []
                time.sleep(0)
        if events:
            yield encoder.encode(self._chunk(ts, events))

    def _chunk(self, ts: float, events: list[KVCacheEvent]) -> KVEventBatch:
        return KVEventBatch(
            ts=ts,
            events=events,  # type: ignore[arg-type]
            data_parallel_rank=self._dp_rank,
        )
