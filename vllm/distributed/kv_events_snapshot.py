# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live-state KV event snapshots.

A snapshot is a synthesized event program for a fresh, private consumer
index. It stores every block the consumer must be able to name after the cut,
parents first, removes them all again, which keeps what the consumer learned
about each hash, and then stores each live residency by hash with its exact
count.

State is per block, not per source event. A block's record holds its parent,
tokens and hash inputs. A record is retained while the block is resident in
any tier, while a retained record names it as parent, or while it is among the
most recently dead blocks, because an offload store can arrive after its GPU
copy was evicted. Retention therefore follows the live cache, not the event
history.

The recorder must observe the stream from its beginning. Missing metadata or
resource exhaustion disables snapshots rather than returning partial state.
"""

import queue
import threading
import time
from array import array
from collections import deque
from collections.abc import Iterator
from contextlib import suppress
from typing import Any

import msgspec
import zmq

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
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
_BlockKey = tuple[str | None, int | None, ExternalBlockHash]


class _Record:
    """Hash inputs of one block and the reasons it is retained."""

    __slots__ = (
        "parent",
        "tokens",
        "block_size",
        "extra_key",
        "lora_id",
        "lora_name",
        "group",
        "live",
        "children",
        "death",
    )

    def __init__(
        self,
        parent: ExternalBlockHash | None,
        tokens: array,
        event: BlockStored,
        extra_key: Any,
    ) -> None:
        self.parent = parent
        self.tokens = tokens
        self.block_size = event.block_size
        self.extra_key = extra_key
        self.lora_id = event.lora_id
        self.lora_name = event.lora_name
        self.group = event.group_idx
        # Live residency references across all scopes.
        self.live = 0
        # Retained records that name this block as their parent.
        self.children = 0
        # Ring generation while recently dead, else 0.
        self.death = 0


class KVCacheSnapshot:
    """Live residency and the block records needed to reconstruct it."""

    MAX_RECORDS = 1_000_000
    MAX_REFERENCES = 1_000_000
    # A block's offload store can complete after its GPU eviction: reusing a
    # block flushes its pending store in the same scheduler step, and the
    # completion is published with that step's or the next step's batch. Dead
    # records are kept for this many batches that carry events; more than
    # MAX_RING_BLOCKS of them within that window disables snapshots.
    RING_BATCHES = 16
    MAX_RING_BLOCKS = 65_536

    def __init__(self) -> None:
        self._live: dict[_BlockKey, int] = {}
        self._records: dict[ExternalBlockHash, _Record] = {}
        # group_idx -> (kv_cache_spec_kind, sliding window) as last announced.
        self._groups: dict[int | None, tuple[str | None, int | None]] = {}
        self._ring: deque[tuple[ExternalBlockHash, int, int]] = deque()
        self._ring_size = 0
        self._generation = 0
        self._batch = 0
        self._references = 0

    def __len__(self) -> int:
        return len(self._live)

    @staticmethod
    def _hash(h: ExternalBlockHash) -> ExternalBlockHash:
        # Offloading may emit raw bytes where GPU events use integer hashes.
        if isinstance(h, bytes):
            return maybe_convert_block_hash(BlockHash(h))
        return h

    def apply(self, events: list[Any]) -> None:
        """Fold one published batch. Retention is decided after the batch.

        Heartbeats carry no events and do not age the ring.
        """
        if not events:
            return
        self._batch += 1
        for event in events:
            if isinstance(event, BlockStored):
                self._store(event)
            elif isinstance(event, BlockRemoved):
                self._check_scope(event)
                for h in event.block_hashes:
                    self._remove((event.medium, event.group_idx, self._hash(h)))
            elif isinstance(event, AllBlocksCleared):
                for key in [k for k in self._live if k[0] in (MEDIUM_GPU, None)]:
                    count = self._live.pop(key)
                    self._references -= count
                    self._release_live(key[2], count)
            else:
                raise ValueError(f"Unsupported KV event: {type(event).__name__}")
        self._collect()

    @staticmethod
    def _check_scope(event: BlockStored | BlockRemoved) -> None:
        if event.locality is not None or event.ownership is not None:
            raise ValueError("Snapshot does not support block locality or ownership")

    def _store(self, event: BlockStored) -> None:
        if not event.block_hashes:
            return
        self._check_scope(event)
        if event.kv_cache_spec_kind is not None:
            self._groups[event.group_idx] = (
                event.kv_cache_spec_kind,
                event.kv_cache_spec_sliding_window,
            )
        hashes = [self._hash(h) for h in event.block_hashes]
        if self._references + len(hashes) > self.MAX_REFERENCES:
            raise ValueError("Snapshot reference budget exceeded")
        if event.token_ids:
            size = event.block_size
            if size <= 0 or len(event.token_ids) != size * len(hashes):
                raise ValueError("Snapshot requires dense block stores")
            parent = (
                None
                if event.parent_block_hash is None
                else self._hash(event.parent_block_hash)
            )
            extra = event.extra_keys
            for i, h in enumerate(hashes):
                tokens = array("I", event.token_ids[i * size : (i + 1) * size])
                record = self._records.get(h)
                if record is None:
                    if parent is not None and parent not in self._records:
                        raise ValueError(
                            f"Missing reconstruction metadata for parent {parent!r}"
                        )
                    if len(self._records) >= self.MAX_RECORDS:
                        raise ValueError("Snapshot record budget exceeded")
                    record = _Record(parent, tokens, event, extra[i] if extra else None)
                    self._records[h] = record
                    if parent is not None:
                        self._records[parent].children += 1
                elif (
                    record.parent,
                    record.tokens,
                    record.block_size,
                    record.extra_key,
                    record.lora_id,
                    record.lora_name,
                    record.group,
                ) != (
                    parent,
                    tokens,
                    size,
                    extra[i] if extra else None,
                    event.lora_id,
                    event.lora_name,
                    event.group_idx,
                ):
                    # One record per hash: a block hashed differently per
                    # group, or restated with other inputs, cannot be rebuilt.
                    raise ValueError(f"Conflicting reconstruction metadata for {h!r}")
                self._add_live((event.medium, event.group_idx, h))
                parent = h
        else:
            for h in hashes:
                record = self._records.get(h)
                if record is None:
                    raise ValueError(f"Missing reconstruction metadata for block {h!r}")
                if record.group != event.group_idx:
                    raise ValueError(f"Conflicting reconstruction metadata for {h!r}")
                self._add_live((event.medium, event.group_idx, h))

    def _add_live(self, key: _BlockKey) -> None:
        self._live[key] = self._live.get(key, 0) + 1
        self._references += 1
        record = self._records[key[2]]
        record.live += 1
        if record.death:
            record.death = 0
            self._ring_size -= 1

    def _remove(self, key: _BlockKey) -> None:
        count = self._live.get(key)
        if count is None:
            return
        if count > 1:
            self._live[key] = count - 1
        else:
            del self._live[key]
        self._references -= 1
        self._release_live(key[2], 1)

    def _release_live(self, h: ExternalBlockHash, count: int) -> None:
        record = self._records[h]
        record.live -= count
        if record.live == 0:
            self._generation += 1
            record.death = self._generation
            self._ring.append((h, self._generation, self._batch))
            self._ring_size += 1

    def _collect(self) -> None:
        oldest = self._batch - self.RING_BATCHES
        while self._ring and self._ring[0][2] <= oldest:
            h, generation, _ = self._ring.popleft()
            record = self._records.get(h)
            if record is None or record.death != generation:
                continue
            record.death = 0
            self._ring_size -= 1
            self._drop(h, record)
        if self._ring_size > self.MAX_RING_BLOCKS:
            raise ValueError("Snapshot ring budget exceeded")
        # Revived and dropped records leave stale entries behind.
        if len(self._ring) > 2 * self._ring_size + 1024:
            self._ring = deque(
                (h, g, b)
                for h, g, b in self._ring
                if (r := self._records.get(h)) is not None and r.death == g
            )

    def _drop(self, h: ExternalBlockHash, record: _Record) -> None:
        while not (record.live or record.children or record.death):
            del self._records[h]
            if record.parent is None:
                return
            h = record.parent
            record = self._records[h]
            record.children -= 1

    def export(
        self, max_blocks_per_event: int = 1024
    ) -> Iterator[BlockStored | BlockRemoved]:
        """Teach every retained block, clear it, then set live residency.

        Each retained block is stored once with its tokens, parents first, in
        its group's GPU scope, and removed there again; the consumer keeps
        what it learned about each hash. Token-less stores then bring every
        live scope to its exact count. Clearing before setting keeps a dead
        block from removing a live block the consumer keys identically.
        """
        children: dict[ExternalBlockHash, list[ExternalBlockHash]] = {}
        roots: list[ExternalBlockHash] = []
        for h, record in self._records.items():
            if record.parent is None:
                roots.append(h)
            else:
                children.setdefault(record.parent, []).append(h)

        def attrs(record: _Record) -> tuple:
            return (record.group, record.block_size, record.lora_id, record.lora_name)

        def segment(blocks: list[ExternalBlockHash]) -> BlockStored:
            first = self._records[blocks[0]]
            kind, window = self._groups.get(first.group, (None, None))
            tokens: list[int] = []
            extra: list[Any] = []
            for h in blocks:
                record = self._records[h]
                tokens.extend(record.tokens)
                extra.append(record.extra_key)
            return BlockStored(
                block_hashes=blocks,
                parent_block_hash=first.parent,
                token_ids=tokens,
                block_size=first.block_size,
                lora_id=first.lora_id,
                medium=MEDIUM_GPU,
                lora_name=first.lora_name,
                extra_keys=extra,
                group_idx=first.group,
                kv_cache_spec_kind=kind,
                kv_cache_spec_sliding_window=window,
            )

        stack = list(reversed(roots))
        while stack:
            h = stack.pop()
            blocks = [h]
            record = self._records[h]
            key = attrs(record)
            while True:
                kids = children.get(h, ())
                if len(kids) != 1:
                    stack.extend(reversed(kids))
                    break
                child = kids[0]
                child_record = self._records[child]
                if attrs(child_record) != key or len(blocks) == max_blocks_per_event:
                    stack.append(child)
                    break
                blocks.append(child)
                h = child
            yield segment(blocks)

        taught: dict[int | None, list[ExternalBlockHash]] = {}
        for h, record in self._records.items():
            taught.setdefault(record.group, []).append(h)
        for group, hashes in taught.items():
            for start in range(0, len(hashes), max_blocks_per_event):
                yield BlockRemoved(
                    block_hashes=hashes[start : start + max_blocks_per_event],
                    medium=MEDIUM_GPU,
                    group_idx=group,
                )

        # Every live scope reaches its exact count, one reference per round.
        refs: dict[tuple[str | None, int | None], list[tuple[ExternalBlockHash, int]]]
        refs = {}
        for (medium, group, h), count in self._live.items():
            refs.setdefault((medium, group), []).append((h, count))
        for (medium, group), owed in refs.items():
            round_ = 0
            while owed:
                hashes = [h for h, _ in owed]
                for start in range(0, len(hashes), max_blocks_per_event):
                    # The shape of an offloading placeholder store: the
                    # consumer resolves each hash and its group's kind.
                    yield BlockStored(
                        block_hashes=hashes[start : start + max_blocks_per_event],
                        parent_block_hash=None,
                        token_ids=[],
                        block_size=0,
                        lora_id=None,
                        medium=medium,
                        lora_name=None,
                        group_idx=group,
                    )
                round_ += 1
                owed = [(h, count) for h, count in owed if count > round_]


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
        events: list[BlockStored | BlockRemoved] = []
        for event in self._snapshot.export(self.BLOCKS_PER_EVENT):
            events.append(event)
            if len(events) == self.EVENTS_PER_CHUNK:
                yield encoder.encode(self._chunk(ts, events))
                events = []
                time.sleep(0)
        if events:
            yield encoder.encode(self._chunk(ts, events))

    def _chunk(
        self, ts: float, events: list[BlockStored | BlockRemoved]
    ) -> KVEventBatch:
        return KVEventBatch(
            ts=ts,
            events=events,  # type: ignore[arg-type]
            data_parallel_rank=self._dp_rank,
        )
