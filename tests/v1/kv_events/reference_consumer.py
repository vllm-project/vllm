# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A minimal KV-event consumer for end-to-end tests.

The consumer rebuilds the engine's prefix-cache state from the published
``KVEventBatch`` stream alone (live subscription plus replay) and predicts how
many prompt tokens a new request will find cached. The tests compare that
prediction with the ``cached_tokens`` the engine reports for the real request,
which is the contract every external router or indexer relies on.

The prediction mirrors ``KVCacheCoordinator.find_longest_cache_hit``: one
lookup per KV cache group (full attention, sliding window, Mamba), reconciled
to a single hit length, using only what the events carry (``group_idx``,
``kv_cache_spec_kind``, ``kv_cache_spec_sliding_window``, ``block_size``).
"""

import hashlib
import math
import time
from collections import Counter
from dataclasses import dataclass, field

import msgspec
import zmq

from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)

BlockHash = int | bytes
Tokens = tuple[int, ...]

FULL_ATTENTION_KINDS = frozenset(
    {
        "full_attention",
        "mla_attention",
        "sink_full_attention",
        "chunked_local_attention",
        # Publishers older than the spec-kind field: treat as full attention.
        None,
    }
)
SLIDING_WINDOW_KINDS = frozenset({"sliding_window", "sliding_window_mla"})
MAMBA_KINDS = frozenset({"mamba"})


@dataclass
class GroupState:
    """What the events taught us about one KV cache group."""

    block_size: int
    kind: str | None = None
    sliding_window: int | None = None
    resident: set[BlockHash] = field(default_factory=set)

    @property
    def is_full_attention(self) -> bool:
        return self.kind in FULL_ATTENTION_KINDS

    @property
    def is_sliding_window(self) -> bool:
        return self.kind in SLIDING_WINDOW_KINDS

    @property
    def is_mamba(self) -> bool:
        return self.kind in MAMBA_KINDS


class ReferenceIndex:
    """Prefix index of published blocks plus the resident set of every group.

    Only information carried by the events is used. A block hash is the hash
    of a token prefix (the engine's chained hash), so it is identified by the
    tokens before it: a ``BlockStored`` rooted at ``parent_block_hash`` tells
    us ``prefix(hash) = prefix(parent) + token_ids``. A block is considered
    cached in a group while a ``BlockStored`` for that group has been seen
    without a later ``BlockRemoved`` or ``AllBlocksCleared``.
    """

    def __init__(self) -> None:
        # Token prefix each published hash stands for, up to its block's end
        # (None when its parent was never published, so the prefix cannot be
        # reconstructed).
        self.prefix_of: dict[BlockHash, Tokens | None] = {}
        # Hash of every reconstructed prefix, keyed by the tokens the hash
        # covers: the prefix plus its lookahead tokens.
        self.hash_of_prefix: dict[Tokens, BlockHash] = {}
        # Tokens after each block that its hash covers, learned from the
        # events' ``lookahead_token_ids``: EAGLE-style drafters read past a
        # block's end, so the engine only reuses a block whose following
        # tokens match too (``VllmConfig.num_hash_lookahead_tokens``).
        self.lookahead = 0
        self.groups: dict[int, GroupState] = {}
        self.stats: Counter[str] = Counter()
        # Event sequences a consumer cannot interpret: a parent that was never
        # published, a hash re-published for a different prefix, token counts
        # that do not match the hashes, group metadata that changes.
        self.anomalies: list[str] = []
        # Events that are ambiguous but handled by convention (see
        # ``_apply_stored``); a few samples for the test report.
        self.notes: list[str] = []

    @property
    def resident(self) -> set[tuple[int, BlockHash]]:
        return {
            (group_idx, block_hash)
            for group_idx, group in self.groups.items()
            for block_hash in group.resident
        }

    def describe_groups(self) -> str:
        groups = ", ".join(
            f"group {idx}: {group.kind} block_size={group.block_size}"
            + (
                f" sliding_window={group.sliding_window}"
                if group.sliding_window
                else ""
            )
            for idx, group in sorted(self.groups.items())
        )
        if self.lookahead:
            groups += f"; hashes cover {self.lookahead} lookahead token(s)"
        return groups

    def apply_batch(
        self, events: list[BlockStored | BlockRemoved | AllBlocksCleared]
    ) -> None:
        """Apply one published batch.

        Hybrid models publish their groups in order, so a Mamba group's event
        can name a parent hash that the attention group only defines later in
        the same batch: such events are applied after the rest of the batch.
        """
        deferred: list[BlockStored] = []
        for event in events:
            if (
                isinstance(event, BlockStored)
                and event.parent_block_hash is not None
                and event.parent_block_hash not in self.prefix_of
            ):
                deferred.append(event)
                continue
            if isinstance(event, AllBlocksCleared):
                for stored in deferred:
                    self.apply(stored)
                deferred.clear()
            self.apply(event)
        for stored in deferred:
            self.apply(stored)

    def apply(self, event: BlockStored | BlockRemoved | AllBlocksCleared) -> None:
        if isinstance(event, BlockStored):
            self._apply_stored(event)
        elif isinstance(event, BlockRemoved):
            self._apply_removed(event)
        elif isinstance(event, AllBlocksCleared):
            self.prefix_of.clear()
            self.hash_of_prefix.clear()
            for group in self.groups.values():
                group.resident.clear()
            self.stats["cleared"] += 1
        else:
            self.anomalies.append(f"unknown event type {type(event).__name__}")

    def _group_for(self, event: BlockStored) -> GroupState:
        group_idx = event.group_idx if event.group_idx is not None else 0
        group = self.groups.get(group_idx)
        if group is None:
            group = GroupState(
                block_size=event.block_size,
                kind=event.kv_cache_spec_kind,
                sliding_window=event.kv_cache_spec_sliding_window,
            )
            self.groups[group_idx] = group
        elif (group.block_size, group.kind, group.sliding_window) != (
            event.block_size,
            event.kv_cache_spec_kind,
            event.kv_cache_spec_sliding_window,
        ):
            self.anomalies.append(
                f"group {group_idx} metadata changed: {group} vs {event}"
            )
        return group

    def _apply_stored(self, event: BlockStored) -> None:
        group = self._group_for(event)
        hashes = list(event.block_hashes)
        block_size = event.block_size
        num_tokens = len(event.token_ids)
        span_blocks = num_tokens // block_size
        # A single hash with fewer tokens than a block is a partial entry: the
        # prefix-chain hash at an interior boundary of a larger cache block.
        if len(hashes) == 1 and 0 < num_tokens < block_size:
            ends = [num_tokens]
        elif num_tokens % block_size or len(hashes) > span_blocks:
            self.anomalies.append(
                f"BlockStored with {len(hashes)} hashes, block_size {block_size} "
                f"but {num_tokens} token ids"
            )
            return
        elif len(hashes) < span_blocks:
            # Groups that keep only some blocks (Mamba state checkpoints,
            # sliding-window tails) skip the others, but the event still spans
            # every block and does not say which boundaries the hashes belong
            # to (vllm-project/vllm#44488). All a consumer can do is mark the
            # hashes resident and locate them through another group's dense
            # chain, which publishes the same prefix hashes.
            self.stats["sparse_stored" if hashes else "empty_stored"] += 1
            if len(self.notes) < 5:
                self.notes.append(
                    f"group {event.group_idx}: {len(hashes)} hashes for a "
                    f"{span_blocks}-block span of {num_tokens} tokens"
                )
            for block_hash in hashes:
                if block_hash in group.resident:
                    self.stats["stored_duplicate"] += 1
                else:
                    group.resident.add(block_hash)
                    self.stats["stored"] += 1
            return
        else:
            ends = [(index + 1) * block_size for index in range(len(hashes))]
        lookahead = self._lookahead_tokens(event, len(hashes))
        parent = event.parent_block_hash
        if parent is None:
            parent_prefix: Tokens | None = ()
        elif parent in self.prefix_of:
            parent_prefix = self.prefix_of[parent]
        else:
            self.anomalies.append(f"BlockStored parent {parent!r} was never published")
            parent_prefix = None
        for block_hash, end, after in zip(hashes, ends, lookahead):
            prefix = (
                None
                if parent_prefix is None
                else parent_prefix + tuple(event.token_ids[:end])
            )
            known = self.prefix_of.get(block_hash)
            if known is not None and prefix is not None and known != prefix:
                self.anomalies.append(
                    f"block {block_hash!r} re-published for a different prefix"
                )
            if prefix is not None or block_hash not in self.prefix_of:
                self.prefix_of[block_hash] = prefix
            if prefix is not None:
                self.hash_of_prefix[prefix + after] = block_hash
            if block_hash in group.resident:
                self.stats["stored_duplicate"] += 1
            else:
                group.resident.add(block_hash)
                self.stats["stored"] += 1

    def _lookahead_tokens(self, event: BlockStored, num_hashes: int) -> list[Tokens]:
        """The tokens after each stored block that its hash covers."""
        if event.lookahead_token_ids is None:
            if self.lookahead:
                self.anomalies.append("BlockStored without lookahead tokens")
            return [()] * num_hashes
        after = [tuple(tokens) for tokens in event.lookahead_token_ids]
        lengths = {len(tokens) for tokens in after}
        if len(after) != num_hashes or len(lengths) != 1:
            self.anomalies.append(
                f"BlockStored with {num_hashes} hashes but lookahead token "
                f"lists of lengths {[len(tokens) for tokens in after]}"
            )
        elif self.lookahead and lengths != {self.lookahead}:
            self.anomalies.append(
                f"lookahead changed from {self.lookahead} to {lengths.pop()} tokens"
            )
        else:
            self.lookahead = lengths.pop()
        return after

    def _apply_removed(self, event: BlockRemoved) -> None:
        if event.group_idx is None:
            groups = list(self.groups.values())
        else:
            group = self.groups.get(event.group_idx)
            groups = [group] if group is not None else []
        for block_hash in event.block_hashes:
            if block_hash not in self.prefix_of:
                self.anomalies.append(f"removed unknown block {block_hash!r}")
            removed = False
            for group in groups:
                if block_hash in group.resident:
                    group.resident.discard(block_hash)
                    removed = True
            self.stats["removed" if removed else "removed_absent"] += 1

    # ------------------------------------------------------------------
    # Prediction

    def _hash_at(self, token_ids: list[int], num_tokens: int) -> BlockHash | None:
        """The published hash of ``token_ids[:num_tokens]``. With lookahead
        hashes the engine can only compute it once the tokens after it are
        known, so a block ending too close to the prompt's end has none."""
        covered = num_tokens + self.lookahead
        if covered > len(token_ids):
            return None
        return self.hash_of_prefix.get(tuple(token_ids[:covered]))

    def _is_resident(self, group: GroupState, token_ids: list[int], end: int) -> bool:
        block_hash = self._hash_at(token_ids, end)
        return block_hash is not None and block_hash in group.resident

    def cached_prefix_blocks(
        self, token_ids: list[int], block_size: int, group_idx: int = 0
    ) -> int:
        """Number of leading full blocks of ``token_ids`` resident in a group."""
        group = self.groups.get(group_idx)
        if group is None:
            return 0
        count = 0
        while (count + 1) * block_size <= len(token_ids) and self._is_resident(
            group, token_ids, (count + 1) * block_size
        ):
            count += 1
        return count

    def predict_cached_tokens(
        self, token_ids: list[int], *, eagle_drop: bool = False
    ) -> int:
        """What ``cached_tokens`` the engine should report for ``token_ids``.

        ``eagle_drop`` is True when the engine runs EAGLE/EAGLE3/MTP
        speculative decoding. The drafter reads one token past the cached
        prefix, so the engine either drops the last matched block
        (``SpeculativeConfig.use_eagle_block_drop``) or, by default, hashes
        that token into each block; the events show which (lookahead tokens),
        and a lookahead hash needs no drop.
        """
        if not self.groups:
            return 0
        eagle_drop = eagle_drop and not self.lookahead
        block_sizes = [group.block_size for group in self.groups.values()]
        hash_block_size = min(block_sizes)
        scheduler_block_size = math.lcm(*block_sizes)
        # Fine-grained (hash-block) hits need an aligned Mamba group with a
        # larger block and every other group able to look up interior hashes;
        # sliding-window groups cannot unless their block is the hash block.
        fine_grained = any(
            group.is_mamba and group.block_size > hash_block_size
            for group in self.groups.values()
        ) and not any(
            group.is_sliding_window and group.block_size != hash_block_size
            for group in self.groups.values()
        )
        alignment = hash_block_size if fine_grained else scheduler_block_size
        # The engine always recomputes the last prompt token.
        max_hit = len(token_ids) - 1
        hashed_tokens = (
            max(0, len(token_ids) - self.lookahead) // hash_block_size * hash_block_size
        )

        def lookup(group: GroupState, max_length: int, drop: bool) -> int:
            if group.is_mamba:
                return self._mamba_hit(
                    group, token_ids, max_length, fine_grained, alignment
                )
            if group.is_sliding_window:
                return self._sliding_window_hit(
                    group, token_ids, max_length, alignment, drop
                )
            return self._full_attention_hit(
                group, token_ids, max_length, fine_grained, alignment, drop
            )

        groups = sorted(
            self.groups.items(), key=lambda item: not item[1].is_full_attention
        )
        if len(groups) == 1:
            return lookup(groups[0][1], max_hit, eagle_drop)

        spec_groups = {
            (group.kind, group.block_size, group.sliding_window) for _, group in groups
        }
        simple_hybrid = len(spec_groups) == 2 and groups[0][1].is_full_attention
        hit = max_hit
        full_attention_hits: dict[int, int] = {}
        while True:
            current = hit
            for group_idx, group in groups:
                if group_idx in full_attention_hits:
                    current = min(current, full_attention_hits[group_idx])
                    continue
                max_length = current
                drop = eagle_drop and not group.is_mamba
                if drop:
                    # EAGLE matches one extra unit past the candidate length
                    # and drops it afterwards.
                    margin = (
                        hash_block_size
                        if fine_grained
                        and group.is_full_attention
                        and group.block_size > hash_block_size
                        else group.block_size
                    )
                    max_length = min(current + margin, hashed_tokens)
                found = lookup(group, max_length, drop)
                if group.is_full_attention:
                    full_attention_hits[group_idx] = found
                current = found
            if current >= hit:
                break
            hit = current
            if simple_hybrid:
                break
        return hit

    def _full_attention_hit(
        self,
        group: GroupState,
        token_ids: list[int],
        max_length: int,
        fine_grained: bool,
        alignment: int,
        drop: bool,
    ) -> int:
        block_size = group.block_size
        hit = 0
        while hit + block_size <= max_length and self._is_resident(
            group, token_ids, hit + block_size
        ):
            hit += block_size
        if fine_grained:
            # Extend into the first non-full block through its interior hash
            # boundaries, longest first.
            hash_block_size = alignment
            upper = min(
                hit + block_size - hash_block_size,
                max_length // hash_block_size * hash_block_size,
            )
            for end in range(upper, hit, -hash_block_size):
                if self._is_resident(group, token_ids, end):
                    hit = end
                    break
        if drop and hit > 0:
            hit -= min(alignment, block_size)
        hit -= hit % alignment
        return hit

    def _sliding_window_hit(
        self,
        group: GroupState,
        token_ids: list[int],
        max_length: int,
        alignment: int,
        drop: bool,
    ) -> int:
        block_size = group.block_size
        assert group.sliding_window is not None
        # Enough contiguous blocks to cover the window; one more when the last
        # matched block is dropped for EAGLE.
        needed = -(-(group.sliding_window - 1) // block_size) + (1 if drop else 0)
        num_blocks = max_length // block_size
        run = 0
        hit_blocks: int | None = None
        # Right to left: the newest window that is fully cached wins.
        for i in range(num_blocks - 1, -1, -1):
            if self._is_resident(group, token_ids, (i + 1) * block_size):
                if run == 0 and block_size != alignment:
                    post_pop_blocks = i if drop else i + 1
                    if post_pop_blocks * block_size % alignment != 0:
                        continue
                run += 1
                if run >= needed:
                    hit_blocks = i + run
                    break
            else:
                run = 0
        if hit_blocks is None:
            # No full window: only a cached run starting at block 0 counts.
            hit_blocks = run
            while block_size != alignment and hit_blocks * block_size % alignment:
                hit_blocks -= 1
        if drop and hit_blocks > 0:
            hit_blocks -= 1
            while block_size != alignment and hit_blocks * block_size % alignment:
                hit_blocks -= 1
        return hit_blocks * block_size

    def _mamba_hit(
        self,
        group: GroupState,
        token_ids: list[int],
        max_length: int,
        fine_grained: bool,
        alignment: int,
    ) -> int:
        block_size = group.block_size
        if fine_grained:
            hash_block_size = alignment
            for end in range(
                max_length // hash_block_size * hash_block_size, 0, -hash_block_size
            ):
                if self._is_resident(group, token_ids, end):
                    return end
            return 0
        # A Mamba block is a state checkpoint: the newest cached one serves
        # the whole prefix before it.
        for i in range(max_length // block_size - 1, -1, -1):
            end = (i + 1) * block_size
            if block_size != alignment and end % alignment != 0:
                continue
            if self._is_resident(group, token_ids, end):
                return end
        return 0


def expected_cached_tokens(
    index: ReferenceIndex,
    token_ids: list[int],
    block_size: int | None = None,
    *,
    eagle_drop: bool = False,
) -> int:
    """What ``cached_tokens`` the engine should report for ``token_ids``.

    ``block_size`` is accepted for readability at the call sites; the index
    learns every group's block size from the events themselves.
    """
    del block_size
    return index.predict_cached_tokens(token_ids, eagle_drop=eagle_drop)


class EventStream:
    """ZMQ subscriber with the gap, replay, duplicate and restart handling a
    real consumer needs, feeding a :class:`ReferenceIndex`."""

    def __init__(
        self, pub_endpoint: str, replay_endpoint: str, topic: str = ""
    ) -> None:
        self.pub_endpoint = pub_endpoint
        self.replay_endpoint = replay_endpoint
        self.topic_bytes = topic.encode("utf-8")
        self.ctx = zmq.Context.instance()
        self.decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
        self.index = ReferenceIndex()
        self.last_seq = -1
        self.batches = 0
        self.replayed = 0
        self.restarts = 0
        self.duplicates = 0
        # Digest of every applied batch by sequence number, to tell a live
        # copy of a replayed batch from a restarted publisher reusing numbers.
        self.applied: dict[int, bytes] = {}
        # (first sequence we asked for, first sequence the replay returned)
        # for every replay that could not cover the whole gap.
        self.lost: list[tuple[int, int]] = []
        self.sub: zmq.Socket | None = None
        self.replay = self.ctx.socket(zmq.DEALER)
        self.replay.connect(replay_endpoint)
        self.connect()

    def connect(self) -> None:
        assert self.sub is None
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, self.topic_bytes)
        self.sub.connect(self.pub_endpoint)

    def disconnect(self) -> None:
        assert self.sub is not None
        self.sub.close(linger=0)
        self.sub = None

    def close(self) -> None:
        if self.sub is not None:
            self.disconnect()
        self.replay.close(linger=0)

    def _apply_batch(self, seq: int, payload: bytes) -> None:
        batch = self.decoder.decode(payload)
        self.index.apply_batch(batch.events)
        self.last_seq = seq
        self.batches += 1
        self.applied[seq] = hashlib.sha256(payload).digest()

    def resync(self, timeout_s: float = 5.0) -> None:
        """Ask the publisher for everything after ``last_seq``.

        Records a loss when the publisher no longer holds the first missing
        batch (its replay buffer is bounded), which is all a consumer can
        detect today: the replay simply starts at the oldest retained batch.
        """
        wanted = self.last_seq + 1
        self.replay.send_multipart([b"", wanted.to_bytes(8, "big")])
        first: int | None = None
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            if not self.replay.poll(int(remaining * 1000)):
                raise TimeoutError("replay did not finish in time")
            frames = self.replay.recv_multipart()
            if frames and frames[0] == b"":
                frames = frames[1:]
            if len(frames) != 3 or not frames[-1]:
                break  # end-of-replay marker
            _, seq_bytes, payload = frames
            seq = int.from_bytes(seq_bytes, "big")
            if seq <= self.last_seq:
                continue
            if first is None:
                first = seq
            self._apply_batch(seq, payload)
            self.replayed += 1
        if first is not None and first != wanted:
            self.lost.append((wanted, first))

    def _handle_live(self, seq: int, payload: bytes) -> None:
        if seq <= self.last_seq:
            if self.applied.get(seq) == hashlib.sha256(payload).digest():
                # The live copy of a batch a replay already delivered.
                self.duplicates += 1
                return
            # A different batch under a used sequence number: the engine
            # process was restarted and counts from zero again. Everything
            # learned from the old process is gone, and the new process may
            # already have published batches this socket missed while it was
            # reconnecting, so replay from the start.
            self.index = ReferenceIndex()
            self.last_seq = -1
            self.applied.clear()
            self.restarts += 1
            self.resync()
        elif seq > self.last_seq + 1:
            self.resync()
        if seq > self.last_seq:
            self._apply_batch(seq, payload)

    def drain(self, idle_ms: int = 500, timeout_s: float = 30.0) -> int:
        """Apply published batches until the stream is quiet for ``idle_ms``.

        Returns the number of live batches received.
        """
        assert self.sub is not None, "not connected"
        received = 0
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.sub.poll(idle_ms):
                return received
            topic, seq_bytes, payload = self.sub.recv_multipart()
            assert topic == self.topic_bytes
            self._handle_live(int.from_bytes(seq_bytes, "big"), payload)
            received += 1
        return received
