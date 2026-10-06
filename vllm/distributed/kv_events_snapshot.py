# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache snapshots for routers that join a ZMQ KV event stream late.

A router builds its prefix index from KV events, and replay only keeps recent
batches. A snapshot gives a joining router the publisher's current cache
state, plus the sequence of the last live batch it covers.

How it works:

- The publisher thread hands every batch it publishes to
  `KVEventSnapshotRecorder.record()`, which queues it.
- The recorder thread folds queued batches into a `KVCacheSnapshot`: one
  record per block hash with its parent, tokens and hash inputs, and a live
  reference count per scope (medium, KV cache group, locality, ownership).
- On a request, the recorder folds everything queued so far and replies with
  the last folded sequence, the publisher identity, and event batches that
  rebuild the live state. The router applies them like live events, then
  follows the live stream from the next sequence.

A record stays while its block is resident in any scope, while a retained
record names it as parent, and for `RING_BATCHES` event-carrying batches after
its last residency ends, because an offload store can complete after its GPU
copy was evicted.

Snapshots are best effort. A block cannot be rebuilt when no store taught its
tokens: a token-less offload store that lands after the block's record left,
or a store that skips blocks (sliding window, Mamba) for a hash no other store
taught. A hash restated with other inputs, as by KV cache groups with
different block sizes, cannot be rebuilt either. The export leaves such blocks
and their descendants out, and routers learn them when they are stored again.

Any recorder error, including a full input queue or an exceeded limit, stops
the recorder: it logs, frees its state, and answers every request as
unavailable until the engine restarts. Live publishing is unaffected.
"""

import queue
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
    EventBatch,
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
SNAPSHOT_REQUEST = b"snapshot"
SNAPSHOT_UNAVAILABLE_SEQ = -2
# medium, group_idx, locality, ownership
_Scope = tuple[str | None, int | None, str | None, str | None]
_BlockKey = tuple[_Scope, ExternalBlockHash]


class _Record(msgspec.Struct, gc=False):  # type: ignore[call-arg]
    """Hash inputs of one block and the reasons it is retained.

    Records name other blocks by hash only, so they cannot form reference
    cycles and stay out of the cyclic garbage collector.
    """

    # The group whose GPU scope teaches the block while it is dead.
    group: int | None
    parent: ExternalBlockHash | None = None
    # Packed unsigned token ids. Without tokens the record is a placeholder
    # for a hash whose inputs were never seen.
    tokens: bytes = b""
    block_size: int = 0
    extra_key: Any = None
    # Offload stores can omit extra keys; they are compared once stated.
    extra_known: bool = False
    lora_id: int | None = None
    lora_name: str | None = None
    # Live residency references across all scopes.
    live: int = 0
    # Retained records that name this block as their parent.
    children: int = 0
    # Ring generation while recently dead, else 0.
    death: int = 0
    # The hash was restated with other inputs.
    conflict: bool = False


class KVCacheSnapshot:
    """Live residency and the block records needed to reconstruct it."""

    # A block's offload store can complete after its GPU eviction: reusing a
    # block flushes its pending store in the same scheduler step, and the
    # completion is published with that step's or the next step's batch. A
    # GPU-only prefix cache reset does not flush pending stores, so they can
    # land tens of batches later. Dead records are kept for this many batches
    # that carry events, at most MAX_RING_BLOCKS of them; beyond that the
    # oldest leave early.
    RING_BATCHES = 64
    MAX_RING_BLOCKS = 65_536

    def __init__(self, max_blocks: int = 1_000_000) -> None:
        # Limits block records and, separately, live residency references.
        self.max_blocks = max_blocks
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
                scope = self._scope(event)
                for h in event.block_hashes:
                    self._remove((scope, self._hash(h)))
            elif isinstance(event, AllBlocksCleared):
                for key in [k for k in self._live if k[0][0] in (MEDIUM_GPU, None)]:
                    count = self._live.pop(key)
                    self._references -= count
                    self._release_live(key[1], count)
            else:
                raise ValueError(f"Unsupported KV event: {type(event).__name__}")
        self._collect()

    @staticmethod
    def _scope(event: BlockStored | BlockRemoved) -> _Scope:
        return (event.medium, event.group_idx, event.locality, event.ownership)

    def _store(self, event: BlockStored) -> None:
        if not event.block_hashes:
            return
        if event.kv_cache_spec_kind is not None:
            self._groups[event.group_idx] = (
                event.kv_cache_spec_kind,
                event.kv_cache_spec_sliding_window,
            )
        scope = self._scope(event)
        hashes = [self._hash(h) for h in event.block_hashes]
        if self._references + len(hashes) > self.max_blocks:
            raise ValueError("Snapshot reference budget exceeded")
        size = event.block_size
        if (
            not event.token_ids
            or size <= 0
            or len(event.token_ids) != size * len(hashes)
        ):
            # A token-less offload store, or a store that skips null or masked
            # blocks (sliding window, Mamba), does not give each block its own
            # token span. It adds residency to blocks other stores taught.
            for h in hashes:
                if h not in self._records:
                    self._new_record(h, event.group_idx)
                self._add_live((scope, h))
            return
        parent = (
            None
            if event.parent_block_hash is None
            else self._hash(event.parent_block_hash)
        )
        packed = array("I", event.token_ids)
        width = size * packed.itemsize
        data = packed.tobytes()
        extra = event.extra_keys
        for i, h in enumerate(hashes):
            tokens = data[i * width : (i + 1) * width]
            extra_key = extra[i] if extra else None
            record = self._records.get(h)
            if record is None:
                record = self._new_record(h, event.group_idx)
                self._fill(record, parent, tokens, event, extra_key)
            elif not record.tokens:
                self._fill(record, parent, tokens, event, extra_key)
            elif (
                record.parent,
                record.tokens,
                record.block_size,
                record.lora_id,
                record.lora_name,
            ) != (parent, tokens, size, event.lora_id, event.lora_name) or (
                extra is not None
                and record.extra_known
                and record.extra_key != extra_key
            ):
                # One record per hash cannot hold two sets of inputs.
                record.conflict = True
            elif extra is not None and not record.extra_known:
                record.extra_key = extra_key
                record.extra_known = True
            self._add_live((scope, h))
            parent = h

    def _new_record(self, h: ExternalBlockHash, group: int | None) -> _Record:
        if len(self._records) >= self.max_blocks:
            raise ValueError("Snapshot record budget exceeded")
        record = self._records[h] = _Record(group)
        return record

    def _fill(
        self,
        record: _Record,
        parent: ExternalBlockHash | None,
        tokens: bytes,
        event: BlockStored,
        extra_key: Any,
    ) -> None:
        record.parent = parent
        record.tokens = tokens
        record.block_size = event.block_size
        record.extra_key = extra_key
        record.extra_known = event.extra_keys is not None
        record.lora_id = event.lora_id
        record.lora_name = event.lora_name
        record.group = event.group_idx
        if parent is not None:
            parent_record = self._records.get(parent)
            if parent_record is None:
                parent_record = self._new_record(parent, event.group_idx)
            parent_record.children += 1

    def _add_live(self, key: _BlockKey) -> None:
        self._live[key] = self._live.get(key, 0) + 1
        self._references += 1
        record = self._records[key[1]]
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
        self._release_live(key[1], 1)

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
            h, generation, _ = self._ring.popleft()
            record = self._records.get(h)
            if record is None or record.death != generation:
                continue
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
            if record.parent is None:
                return
            h = record.parent
            record = self._records[h]
            record.children -= 1

    def export(
        self, max_blocks_per_event: int = 1024
    ) -> Iterator[BlockStored | BlockRemoved]:
        """Teach every retained block, clear the dead, then set live residency.

        Each rebuildable block is stored once with its tokens, parents first:
        a live block in one of its live scopes, a dead block in its group's
        GPU scope. Dead blocks are removed again. Token-less stores then bring
        every live scope to its exact count, and a last removal drops the
        teaching store of each live block. The consumer counts references per
        scope and hash, so that removal never evicts, a live block keeps its
        engine hash resolvable throughout, and a dead block the consumer keys
        like a live one is removed before the live residency is set. Blocks
        that cannot be rebuilt, and their descendants, are left out.
        """
        # Most blocks have one child, so only branch points allocate a list.
        first_child: dict[ExternalBlockHash, ExternalBlockHash] = {}
        more_children: dict[ExternalBlockHash, list[ExternalBlockHash]] = {}
        roots: list[ExternalBlockHash] = []
        for h, record in self._records.items():
            if record.parent is None:
                roots.append(h)
            elif record.parent not in first_child:
                first_child[record.parent] = h
            else:
                more_children.setdefault(record.parent, []).append(h)
        live_scope: dict[ExternalBlockHash, _Scope] = {}
        for live_key in self._live:
            live_scope.setdefault(live_key[1], live_key[0])

        def buildable(h: ExternalBlockHash) -> bool:
            record = self._records[h]
            return bool(record.tokens) and not record.conflict

        def scope(h: ExternalBlockHash) -> _Scope:
            return live_scope.get(h) or (MEDIUM_GPU, self._records[h].group, None, None)

        def attrs(h: ExternalBlockHash) -> tuple:
            record = self._records[h]
            return (scope(h), record.block_size, record.lora_id, record.lora_name)

        def segment(blocks: list[ExternalBlockHash]) -> BlockStored:
            first = self._records[blocks[0]]
            medium, group, locality, ownership = scope(blocks[0])
            kind, window = self._groups.get(group, (None, None))
            tokens = array("I")
            extra: list[Any] = []
            for h in blocks:
                record = self._records[h]
                tokens.frombytes(record.tokens)
                extra.append(record.extra_key)
            return BlockStored(
                block_hashes=blocks,
                parent_block_hash=first.parent,
                token_ids=tokens.tolist(),
                block_size=first.block_size,
                lora_id=first.lora_id,
                medium=medium,
                lora_name=first.lora_name,
                extra_keys=extra,
                group_idx=group,
                kv_cache_spec_kind=kind,
                kv_cache_spec_sliding_window=window,
                locality=locality,
                ownership=ownership,
            )

        def removals(hashes: Iterator[ExternalBlockHash]) -> Iterator[BlockRemoved]:
            scoped: dict[_Scope, list[ExternalBlockHash]] = {}
            for h in hashes:
                scoped.setdefault(scope(h), []).append(h)
            for (medium, group, locality, ownership), blocks in scoped.items():
                for start in range(0, len(blocks), max_blocks_per_event):
                    yield BlockRemoved(
                        block_hashes=blocks[start : start + max_blocks_per_event],
                        medium=medium,
                        group_idx=group,
                        locality=locality,
                        ownership=ownership,
                    )

        included: set[ExternalBlockHash] = set()
        stack = [h for h in reversed(roots) if buildable(h)]
        while stack:
            h = stack.pop()
            blocks = [h]
            key = attrs(h)
            while True:
                included.add(h)
                child = first_child.get(h)
                if child is None:
                    break
                if h in more_children:
                    stack.extend(
                        c for c in reversed([child, *more_children[h]]) if buildable(c)
                    )
                    break
                if not buildable(child):
                    break
                if attrs(child) != key or len(blocks) == max_blocks_per_event:
                    stack.append(child)
                    break
                blocks.append(child)
                h = child
            yield segment(blocks)

        yield from removals(h for h in included if h not in live_scope)

        # Every live scope reaches its exact count, one reference per round.
        refs: dict[_Scope, list[tuple[ExternalBlockHash, int]]] = {}
        for (live_scope_key, h), count in self._live.items():
            if h in included:
                refs.setdefault(live_scope_key, []).append((h, count))
        for (medium, group, locality, ownership), owed in refs.items():
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
                        locality=locality,
                        ownership=ownership,
                    )
                round_ += 1
                owed = [(h, count) for h, count in owed if count > round_]

        yield from removals(h for h in live_scope if h in included)


class KVEventSnapshotRecorder:
    """Folds published batches on its own thread and serves snapshots.

    `publisher_id` identifies the publisher until it restarts. Any recorder
    error stops the recorder for the rest of the publisher's life, and live
    publishing continues.
    """

    # The publisher waits this long for queue room before the recorder stops.
    MAX_RECORD_WAIT_S = 1.0
    MAX_PENDING_BATCHES = 4096
    POLL_INTERVAL_MS = 20
    # Bounds replies queued for one requester that stopped reading.
    MAX_QUEUED_REPLIES = 4
    EVENTS_PER_CHUNK = 256
    BLOCKS_PER_EVENT = 1024

    def __init__(
        self,
        endpoint: str,
        data_parallel_rank: int,
        max_blocks: int = 1_000_000,
        max_response_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self._dp_rank = data_parallel_rank
        self._max_response_bytes = max_response_bytes
        self.publisher_id = uuid.uuid4().bytes
        self._inbox: queue.Queue[tuple[int, EventBatch]] = queue.Queue(
            self.MAX_PENDING_BATCHES
        )
        # None once the recorder has stopped.
        self._snapshot: KVCacheSnapshot | None = KVCacheSnapshot(max_blocks)
        self._seq = -1
        self._failed = threading.Event()
        self._stop = threading.Event()
        # Bound on the caller's thread so a bind error raises to the caller,
        # as ZmqEventPublisher does; the recorder thread owns it afterwards.
        self._router = zmq.Context.instance().socket(zmq.ROUTER)
        self._router.setsockopt(zmq.LINGER, 0)
        self._router.setsockopt(zmq.SNDHWM, self.MAX_QUEUED_REPLIES)
        try:
            self.endpoint = ZmqEventPublisher._bind(self._router, endpoint)
        except Exception:
            self._router.close()
            raise
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="kv-snapshot-recorder"
        )
        self._thread.start()

    def record(self, seq: int, batch: EventBatch) -> None:
        """Queue a published batch, waiting a bounded time for room."""
        if self._failed.is_set():
            return
        try:
            self._inbox.put((seq, batch), timeout=self.MAX_RECORD_WAIT_S)
        except queue.Full:
            logger.error(
                "KV snapshot recorder is %d batches behind at sequence %d; "
                "snapshots are unavailable until the engine restarts",
                self.MAX_PENDING_BATCHES,
                seq,
            )
            self._failed.set()

    def shutdown(self, timeout: float) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        with self._router as router:
            while not self._stop.is_set():
                requested = router.poll(self.POLL_INTERVAL_MS)
                # Fold before replying, so a snapshot covers every batch
                # published before the request, which the requester may
                # already hold from the live stream.
                self._drain()
                if requested:
                    self._serve(router)

    def _drain(self) -> None:
        # A finite cut: batches published while folding wait for the next call.
        for _ in range(self._inbox.qsize()):
            seq, batch = self._inbox.get_nowait()
            if self._snapshot is None:
                continue
            try:
                self._snapshot.apply(batch.events)
            except Exception:
                logger.exception(
                    "KV snapshot fold failed at sequence %d; snapshots are "
                    "unavailable until the engine restarts",
                    seq,
                )
                self._failed.set()
            else:
                self._seq = seq
        if self._failed.is_set():
            self._snapshot = None

    def _serve(self, router: zmq.Socket) -> None:
        # A REQ request arrives on the ROUTER as [peer identity, empty
        # delimiter, request]; the reply goes back under the same two frames.
        frames = router.recv_multipart()
        if len(frames) != 3 or frames[1] or frames[2] != SNAPSHOT_REQUEST:
            logger.warning(
                "Ignoring a malformed KV snapshot request of %d frames", len(frames)
            )
            return
        reply = [
            SNAPSHOT_UNAVAILABLE_SEQ.to_bytes(8, "big", signed=True),
            self.publisher_id,
        ]
        if self._snapshot is not None:
            try:
                chunks, size = [], 0
                for chunk in self._encode_chunks(self._snapshot):
                    size += len(chunk)
                    if size > self._max_response_bytes:
                        raise ValueError("Snapshot response budget exceeded")
                    chunks.append(chunk)
                reply[0] = self._seq.to_bytes(8, "big", signed=True)
                reply.extend(chunks)
            except Exception:
                logger.exception(
                    "KV snapshot export failed; snapshots are unavailable "
                    "until the engine restarts"
                )
                self._failed.set()
                self._snapshot = None
        # A requester that stopped reading retries; it cannot hold this thread.
        with suppress(zmq.Again):
            router.send_multipart(frames[:2] + reply, flags=zmq.DONTWAIT, copy=False)

    def _encode_chunks(self, snapshot: KVCacheSnapshot) -> Iterator[bytes]:
        encoder = msgspec.msgpack.Encoder()
        ts = time.time()
        events: list[BlockStored | BlockRemoved] = []
        for event in snapshot.export(self.BLOCKS_PER_EVENT):
            events.append(event)
            if len(events) == self.EVENTS_PER_CHUNK:
                yield encoder.encode(self._chunk(ts, events))
                events = []
                # Let the engine threads take the GIL between chunks.
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
