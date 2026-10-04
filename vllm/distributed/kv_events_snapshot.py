# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live-state KV event snapshots.

A snapshot is a synthesized event program that a consumer applies like live
events. It stores every block the consumer must be able to name after the cut,
parents first, removes the dead ones again, stores each live residency by hash
with its exact count, and then removes the first store of each live block.

State is per block, not per source event. A block's record holds its parent,
tokens and hash inputs. Residency is counted per scope: medium, KV cache group,
locality and ownership. A record is retained while the block is resident in
any scope, while a retained record names it as parent, or while it is among
the most recently dead blocks, because an offload store can arrive after its
GPU copy was evicted. Retention therefore follows the live cache, not the event
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
    # The block cannot be rebuilt: a placeholder or conflicting inputs.
    tainted: bool = False


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
                    placeholder = self._new_record(h, event.group_idx)
                    self._taint(placeholder, f"no reconstruction metadata for {h!r}")
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
                record.tainted = False
                self.tainted -= 1
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
                # One record per hash: a hash restated with other inputs, as by
                # KV cache groups with different block sizes, cannot be rebuilt.
                self._taint(record, f"conflicting reconstruction metadata for {h!r}")
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
        record = self._records[key[1]]
        record.live += 1
        if record.death:
            record.death = 0
            self._ring_size -= 1

    def _remove(self, key: _BlockKey) -> None:
        count = self._live.get(key)
        if count is None:
            if key[1] not in self._records:
                self.forgotten_removals += 1
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
        """Teach every retained block, clear the dead, then set live residency.

        Each retained block is stored once with its tokens, parents first: a
        live block in one of its live scopes, a dead block in its group's GPU
        scope. Dead blocks are removed again. Token-less stores then bring
        every live scope to its exact count, and a last removal drops the
        teaching store of each live block. The consumer counts references per
        scope and hash, so that removal never evicts, a live block keeps its
        engine hash resolvable throughout, and a dead block the consumer keys
        like a live one is removed before the live residency is set.
        Only an untainted snapshot can be exported.
        """
        assert not self.tainted
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

        stack = list(reversed(roots))
        while stack:
            h = stack.pop()
            blocks = [h]
            key = attrs(h)
            while True:
                child = first_child.get(h)
                if child is None:
                    break
                if h in more_children:
                    stack.extend(reversed([child, *more_children[h]]))
                    break
                if attrs(child) != key or len(blocks) == max_blocks_per_event:
                    stack.append(child)
                    break
                blocks.append(child)
                h = child
            yield segment(blocks)

        yield from removals(h for h in self._records if h not in live_scope)

        # Every live scope reaches its exact count, one reference per round.
        refs: dict[_Scope, list[tuple[ExternalBlockHash, int]]] = {}
        for (live_scope_key, h), count in self._live.items():
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

        yield from removals(iter(live_scope))


class KVEventSnapshotRecorder:
    """Owns snapshot state and its ROUTER socket on one thread.

    Input is the immutable payload sent on PUB. While the snapshot is tainted,
    requests receive the unavailable sequence and retry later. `stream_id`
    changes only with the publisher, so consumers that are following the live
    stream keep their state. Lost input or exhausted budgets invalidate the
    recorder for this publisher lifetime; live publishing continues.
    """

    # Reply framing, polling and backpressure. Deployment-dependent limits
    # are constructor arguments.
    EVENTS_PER_CHUNK = 256
    BLOCKS_PER_EVENT = 1024
    POLL_INTERVAL_MS = 20
    MAX_PENDING_BATCHES = 4096
    MAX_PENDING_BYTES = 64 * 1024 * 1024
    # The publisher waits this long for room before the recorder gives up.
    MAX_RECORD_WAIT_S = 1.0
    MAX_REQUESTS = 32
    REPORT_INTERVAL_S = 60.0

    def __init__(
        self,
        endpoint: str,
        data_parallel_rank: int,
        max_blocks: int = 1_000_000,
        max_response_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self._dp_rank = data_parallel_rank
        self._max_blocks = max_blocks
        self._max_response_bytes = max_response_bytes
        # The publisher identity sent with every live frame and reply.
        self.stream_id = uuid.uuid4().bytes
        self._inbox: deque[tuple[int, bytes]] = deque()
        self._pending_bytes = 0
        self._room = threading.Condition()
        self._snapshot = KVCacheSnapshot(max_blocks)
        self._seq = -1
        self._tainted = False
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
            self._snapshot = KVCacheSnapshot(self._max_blocks)

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
        if not (self._failed.is_set() or self._snapshot.tainted):
            try:
                size = 0
                for chunk in self._encode_chunks(encoder):
                    size += len(chunk)
                    if size > self._max_response_bytes:
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
