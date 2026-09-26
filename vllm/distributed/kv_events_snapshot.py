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

The recorder must observe the stream from its beginning. A block it cannot
rebuild is tainted: snapshots are unavailable while any retained record is
tainted, and become available again once none is. Lost input, unsupported
events and exhausted budgets disable snapshots until the publisher restarts.
Snapshots are never partial.
"""

import threading
import time
import uuid
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
        "tainted",
    )

    def __init__(self, group: int | None) -> None:
        # Without tokens the record is a placeholder for a hash whose inputs
        # were never seen.
        self.parent: ExternalBlockHash | None = None
        self.tokens = array("I")
        self.block_size = 0
        self.extra_key: Any = None
        self.lora_id: int | None = None
        self.lora_name: str | None = None
        self.group = group
        # Live residency references across all scopes.
        self.live = 0
        # Retained records that name this block as their parent.
        self.children = 0
        # Ring generation while recently dead, else 0.
        self.death = 0
        # The block cannot be rebuilt: a placeholder or conflicting inputs.
        self.tainted = False


class KVCacheSnapshot:
    """Live residency and the block records needed to reconstruct it."""

    MAX_RECORDS = 1_000_000
    MAX_REFERENCES = 1_000_000
    # A block's offload store can complete after its GPU eviction: reusing a
    # block flushes its pending store in the same scheduler step, and the
    # completion is published with that step's or the next step's batch. Dead
    # records are kept for this many batches that carry events, at most
    # MAX_RING_BLOCKS of them; beyond that the oldest leave early.
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
        # Retained records that cannot be rebuilt, and the first reason.
        self.tainted = 0
        self.taint_reason = ""
        # Removals of hashes without a record; consumers bootstrapped after
        # the record left fail on them once.
        self.forgotten_removals = 0
        # Dead records dropped before RING_BATCHES to keep the ring in budget.
        self.expired_early = 0

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
        size = event.block_size
        if (
            not event.token_ids
            or size <= 0
            or len(event.token_ids) != size * len(hashes)
        ):
            # Rebuilding a block needs its own token span.
            sparse = bool(event.token_ids)
            for h in hashes:
                record = self._records.get(h)
                if record is None:
                    record = self._new_record(h, event.group_idx)
                    self._taint(record, f"no reconstruction metadata for {h!r}")
                elif sparse or record.group != event.group_idx:
                    self._taint(
                        record, f"conflicting reconstruction metadata for {h!r}"
                    )
                self._add_live((event.medium, event.group_idx, h))
            return
        parent = (
            None
            if event.parent_block_hash is None
            else self._hash(event.parent_block_hash)
        )
        extra = event.extra_keys
        for i, h in enumerate(hashes):
            tokens = array("I", event.token_ids[i * size : (i + 1) * size])
            extra_key = extra[i] if extra else None
            record = self._records.get(h)
            if record is None:
                record = self._new_record(h, event.group_idx)
                self._fill(record, parent, tokens, event, extra_key)
            elif not record.tokens and record.group == event.group_idx:
                self._fill(record, parent, tokens, event, extra_key)
                record.tainted = False
                self.tainted -= 1
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
                extra_key,
                event.lora_id,
                event.lora_name,
                event.group_idx,
            ):
                # One record per hash: a block hashed differently per group,
                # or restated with other inputs, cannot be rebuilt.
                self._taint(record, f"conflicting reconstruction metadata for {h!r}")
            self._add_live((event.medium, event.group_idx, h))
            parent = h

    def _new_record(self, h: ExternalBlockHash, group: int | None) -> _Record:
        if len(self._records) >= self.MAX_RECORDS:
            raise ValueError("Snapshot record budget exceeded")
        record = self._records[h] = _Record(group)
        return record

    def _fill(
        self,
        record: _Record,
        parent: ExternalBlockHash | None,
        tokens: array,
        event: BlockStored,
        extra_key: Any,
    ) -> None:
        record.parent = parent
        record.tokens = tokens
        record.block_size = event.block_size
        record.extra_key = extra_key
        record.lora_id = event.lora_id
        record.lora_name = event.lora_name
        record.group = event.group_idx
        if parent is not None:
            parent_record = self._records.get(parent)
            if parent_record is None:
                parent_record = self._new_record(parent, event.group_idx)
                self._taint(parent_record, f"no reconstruction metadata for {parent!r}")
            parent_record.children += 1

    def _taint(self, record: _Record, reason: str) -> None:
        if record.tainted:
            return
        if not self.tainted:
            self.taint_reason = reason
        record.tainted = True
        self.tainted += 1

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
            if key[2] not in self._records:
                self.forgotten_removals += 1
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
        while self._ring and (
            self._ring[0][2] <= oldest or self._ring_size > self.MAX_RING_BLOCKS
        ):
            h, generation, batch = self._ring.popleft()
            record = self._records.get(h)
            if record is None or record.death != generation:
                continue
            if batch > oldest:
                self.expired_early += 1
            record.death = 0
            self._ring_size -= 1
            self._drop(h, record)
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
            if record.tainted:
                self.tainted -= 1
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
        Only an untainted snapshot can be exported.
        """
        assert not self.tainted
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

    Input is the immutable payload sent on PUB. While the snapshot is tainted,
    requests receive the unavailable sequence. If one did, `stream_id` changes
    when the taint clears, so consumers that fell back to live events alone
    bootstrap again. Lost input or exhausted budgets invalidate the recorder
    for this publisher lifetime; live publishing continues.
    """

    EVENTS_PER_CHUNK = 256
    BLOCKS_PER_EVENT = 1024
    POLL_INTERVAL_MS = 20
    MAX_PENDING_BATCHES = 4096
    MAX_PENDING_BYTES = 64 * 1024 * 1024
    # The publisher waits this long for room before the recorder gives up.
    MAX_RECORD_WAIT_S = 1.0
    MAX_REPLY_BYTES = 256 * 1024 * 1024
    MAX_REQUESTS = 32
    REPORT_INTERVAL_S = 60.0

    def __init__(self, endpoint: str, data_parallel_rank: int) -> None:
        self._dp_rank = data_parallel_rank
        # The publisher identity sent with every live frame and reply.
        self.stream_id = uuid.uuid4().bytes
        self._inbox: deque[tuple[int, bytes]] = deque()
        self._pending_bytes = 0
        self._room = threading.Condition()
        self._snapshot = KVCacheSnapshot()
        self._seq = -1
        self._tainted = False
        self._served_unavailable = False
        self._reported = (0, 0)
        self._next_report = 0.0
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
        """Queue a published batch, waiting a bounded time for room."""
        size = len(payload)
        with self._room:
            if not self._room.wait_for(
                lambda: (
                    self._failed.is_set() or self._stop.is_set() or self._fits(size)
                ),
                timeout=self.MAX_RECORD_WAIT_S,
            ):
                logger.error(
                    "KV snapshot recorder is %d batches behind at sequence %d; "
                    "snapshots are unavailable until the engine restarts",
                    len(self._inbox),
                    seq,
                )
                self._failed.set()
            if self._failed.is_set() or self._stop.is_set():
                return
            self._inbox.append((seq, payload))
            self._pending_bytes += size

    def _fits(self, size: int) -> bool:
        return not self._inbox or (
            len(self._inbox) < self.MAX_PENDING_BATCHES
            and self._pending_bytes + size <= self.MAX_PENDING_BYTES
        )

    def shutdown(self, timeout: float) -> None:
        self._stop.set()
        with self._room:
            self._room.notify_all()
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
        with self._room:
            cut = len(self._inbox)
        for _ in range(cut):
            with self._room:
                seq, payload = self._inbox.popleft()
                self._pending_bytes -= len(payload)
                self._room.notify()
            if self._failed.is_set():
                continue
            try:
                self._snapshot.apply(decoder.decode(payload).events)
            except Exception:
                logger.exception(
                    "KV snapshot fold failed at sequence %d; snapshots are "
                    "unavailable until the engine restarts",
                    seq,
                )
                self._failed.set()
                continue
            self._seq = seq
            self._report(seq)
        if self._failed.is_set():
            self._snapshot = KVCacheSnapshot()

    def _report(self, seq: int) -> None:
        snapshot = self._snapshot
        if bool(snapshot.tainted) != self._tainted:
            self._tainted = not self._tainted
            if self._tainted:
                logger.warning(
                    "KV snapshots unavailable from sequence %d: %s",
                    seq,
                    snapshot.taint_reason,
                )
            elif self._served_unavailable:
                self._served_unavailable = False
                self.stream_id = uuid.uuid4().bytes
                logger.info(
                    "KV snapshots available again from sequence %d under a new "
                    "publisher identity",
                    seq,
                )
            else:
                logger.info("KV snapshots available again from sequence %d", seq)
        counters = (snapshot.forgotten_removals, snapshot.expired_early)
        if counters != self._reported and time.monotonic() >= self._next_report:
            self._reported = counters
            self._next_report = time.monotonic() + self.REPORT_INTERVAL_S
            logger.warning(
                "KV snapshot recorder totals: %d removals of blocks without a "
                "record, %d dead blocks dropped early to keep the ring in budget",
                *counters,
            )

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
        reply = [self._seq.to_bytes(8, "big", signed=True), self.stream_id]
        if self._snapshot.tainted:
            self._served_unavailable = True
        elif not self._failed.is_set():
            try:
                size = 0
                for chunk in self._encode_chunks(encoder):
                    size += len(chunk)
                    if size > self.MAX_REPLY_BYTES:
                        raise ValueError("Snapshot response budget exceeded")
                    reply.append(chunk)
            except Exception:
                logger.exception(
                    "KV snapshot export failed; snapshots are unavailable until "
                    "the engine restarts"
                )
                self._failed.set()
        if self._failed.is_set() or self._snapshot.tainted:
            reply = [
                SNAPSHOT_UNAVAILABLE_SEQ.to_bytes(8, "big", signed=True),
                self.stream_id,
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
