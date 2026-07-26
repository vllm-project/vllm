# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections import OrderedDict
from collections.abc import Iterable
from typing import TypedDict

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class _SessionStats(TypedDict):
    # `hits` is kept as float end-to-end: seeded from ghost scores at
    # insert time (fractional) and incremented on touch (integer). The
    # session-score math is float either way, so no truncation is
    # needed.
    hits: float
    last_touch: int
    prefix_depth: int


class SAECachePolicy(CachePolicy):
    """SAE (Session-Aware Eviction) cache policy.

    A conversation's KV blocks are only useful as a whole chain —
    evicting one block in the middle forces recomputation of everything
    after it. SAE groups blocks stored by the same request into a
    *session* and evicts session-worst-first, tail-first, so shared
    prefixes outlive their suffixes.

    Data Structures:
        blocks: Resident block table, key -> BlockStatus.
        session_keys: Keys owned by each session, in insertion order.
            Eviction reads the tail (``reversed(session_keys[sid])``).
        key_to_session: Reverse index, key -> session_id, for O(1)
            session lookup during touch.
        session_stats: Per-session ``{hits, last_touch, prefix_depth}``.
            Fields drive the session score — see Session Score.
        ghost_scores: Per-key float score. Persists after a key is
            evicted so a returning key carries its popularity into
            whichever new session it joins.
        pending_merge_pointers: ``req_id -> session_id``. Set by
            ``record_lookup`` on a resident hit, consumed by
            ``open_session``, dropped by ``on_request_finished``.
        logical_time: Monotonic tick, +1 per ``touch``. Drives the
            recency term.

    Session Score (see ``_session_score``):
        last_touch + hits * TOUCH_WEIGHT + prefix_depth_bonus(depth)

        Lower is worse. Three terms, one per ``session_stats`` field:
            ``last_touch``: ``logical_time`` at the most recent touch.
                The recency term — higher = more recent.
            ``hits``: cumulative touch count (+1 per ``touch`` call
                that hit at least one of the session's keys). The
                frequency term — higher = more frequent.
            ``prefix_depth``: count of the batch's keys already
                resident at session open time (fixed for the session's
                lifetime). The prefix-bonus term — shallower = larger
                bonus.

        Coefficients live on the class as ``_TOUCH_WEIGHT``,
        ``_PREFIX_DEPTH_BONUS_SCALE``, ``_PREFIX_DEPTH_HALF_LIFE``.

    Algorithm Flow (four manager hooks bracket each request):
        1. Cache lookup (``record_lookup``):
           Remembers per request the last-lookup-hit session (in
           ``pending_merge_pointers``), so step 2 can merge into it.
           On a miss, does nothing — no pointer set, so step 2 opens
           a fresh session.

        2. Cache insertion (``open_session`` -> ``insert*`` ->
           ``close_session``):
           The manager brackets each prepare_store batch's insert
           loop. If ``pending_merge_pointers[req_id]`` still points at
           a live session, ``open_session`` continues it; otherwise it
           opens a fresh one. ``close_session`` ends the batch.
           Fresh sessions made of previously-popular blocks are seeded
           from their ghost scores, avoiding a cold-start penalty.

        3. Cache touch (``touch``):
             - Bump each touched key's ghost score (weighted by
               position and whether the key was a hit or miss).
             - For each unique session touched, ``hits += 1`` and
               ``last_touch = logical_time`` — **one touch call = one
               hit per session**, regardless of how many of its keys
               were in the batch.
             - Every ``decay_interval`` calls: apply decay to session
               hits and ghost scores, and prune tiny non-resident
               ghost entries.

        4. Block eviction (``evict``):
             - **Admission gate**: a fresh session must score at
               least as high as the worst incumbent; a merge session
               bypasses the gate. Gate failure → return ``None`` and
               decline the store.
             - **Worst-first walk**: pull idle tail keys, worst-scoring
               session first, until ``n`` are collected. If fewer than
               ``n`` can be collected, return ``None`` and decline the
               store.

    Tunables (constructor kwargs, forwarded from the manager):
        decay_interval: ``touch`` calls per decay tick (~one ``touch``
            per request).
        decay_factor: Scale applied to session hits and ghost scores
            each tick.
        ghost_hit_weight: Ghost bump when the key is resident and has
            finished storing.
        ghost_miss_weight: Ghost bump when the key is missing or still
            storing.
        ghost_norm: Divisor when seeding a fresh session from ghosts.
    """

    # Session score coefficients. See _session_score.
    _PREFIX_DEPTH_BONUS_SCALE: float = 30000.0
    _PREFIX_DEPTH_HALF_LIFE: float = 8.0
    _TOUCH_WEIGHT: float = 1500.0

    # Ghost entries below this value are pruned during decay when the
    # underlying key is no longer resident.
    _GHOST_PRUNE_THRESHOLD: float = 0.01

    # Precomputed 1/log2(pos+2) table for position-weighted ghost bumps
    # in `touch`. Positions beyond the table fall through to a direct
    # computation.
    _POS_WEIGHT_TABLE_SIZE: int = 1024
    _POS_WEIGHTS = [1.0 / math.log2(i + 2.0) for i in range(_POS_WEIGHT_TABLE_SIZE)]

    def __init__(
        self,
        cache_capacity: int,
        *,
        decay_interval: int = 500,
        decay_factor: float = 0.9,
        ghost_hit_weight: float = 12.0,
        ghost_miss_weight: float = 1.0,
        ghost_norm: float = 12.0,
    ) -> None:
        self.decay_interval: int = decay_interval
        self.decay_factor: float = decay_factor
        self.ghost_hit_weight: float = ghost_hit_weight
        self.ghost_miss_weight: float = ghost_miss_weight
        self.ghost_norm: float = ghost_norm

        # Resident state.
        self.blocks: dict[OffloadKey, BlockStatus] = {}
        self.session_keys: dict[int, list[OffloadKey]] = {}
        self.key_to_session: dict[OffloadKey, int] = {}
        self.session_stats: dict[int, _SessionStats] = {}
        self.ghost_scores: dict[OffloadKey, float] = {}
        self.evictable_blocks: OrderedDict[OffloadKey, None] = OrderedDict()

        # Session lifecycle.
        self.logical_time: int = 0
        self.next_session_id: int = 0
        self.open_session_id: int | None = None
        self.open_session_is_merged: bool = False

        # Decay.
        self.lookups_since_decay: int = 0

        # Merge pointers, keyed per in-flight request. Value is the
        # session id of the request's most recent resident hit.
        self.pending_merge_pointers: dict[str, int] = {}

    # --- basic block access ---

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        # Pure existence check. See record_lookup for the merge-pointer
        # entry point used by scheduler-driven lookups.
        return self.blocks.get(key)

    @override
    def record_lookup(self, key: OffloadKey, req_context: ReqContext) -> None:
        # Only scheduler-driven lookups feed the session-merge pointer.
        # The manager's internal existence checks (prepare_store,
        # prepare_load, etc.) use `get` directly and must not affect
        # merge decisions — otherwise a shared-prefix hit in one
        # request would appear to "continue" whichever session owned
        # that prefix, collapsing many logical sessions into one.
        #
        # Any resident block counts (including HIT_PENDING blocks whose
        # store has not yet completed): the manager's "already stored?"
        # filter also considers them present, so a subsequent
        # prepare_store from the same request will legitimately merge
        # into this session.
        block = self.blocks.get(key)
        if block is None:
            return
        session_id = self.key_to_session.get(key)
        if session_id is None:
            return
        self.pending_merge_pointers[req_context.req_id] = session_id

    # --- session lifecycle ---

    @override
    def open_session(self, req_context: ReqContext, num_blocks_in_cache: int) -> None:
        # Called by the manager immediately before its insert loop.
        # If this request has a live merge pointer whose target session
        # is still resident, continue that session; otherwise open a
        # fresh session tagged with the batch's prefix depth.
        assert self.open_session_id is None, "open_session called with unclosed session"
        candidate = self.pending_merge_pointers.pop(req_context.req_id, None)
        merge_target: int | None = (
            candidate
            if candidate is not None and candidate in self.session_stats
            else None
        )

        if merge_target is not None:
            self.open_session_id = merge_target
            self.session_stats[merge_target]["last_touch"] = self.logical_time
            self.open_session_is_merged = True
        else:
            new_session_id = self.next_session_id
            self.next_session_id += 1
            self.open_session_id = new_session_id
            self.session_keys[new_session_id] = []
            self.session_stats[new_session_id] = _SessionStats(
                hits=0.0,
                last_touch=self.logical_time,
                prefix_depth=num_blocks_in_cache,
            )
            self.open_session_is_merged = False

    @override
    def close_session(self) -> None:
        self.open_session_id = None
        self.open_session_is_merged = False

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        assert self.open_session_id is not None, (
            "insert called without an open session — the manager must call "
            "open_session before its insert loop"
        )
        session_id = self.open_session_id
        self.session_keys[session_id].append(key)
        self.key_to_session[key] = session_id
        self.blocks[key] = block
        # Seed a fresh session's hits from each new key's ghost score
        # so blocks with prior popularity give their new session a
        # head start. Merged sessions already carry their earned hits.
        if not self.open_session_is_merged:
            seed = self.ghost_scores.get(key, 0.0) / self.ghost_norm
            self.session_stats[session_id]["hits"] += seed

    @override
    def remove(self, key: OffloadKey) -> None:
        block = self.blocks.pop(key, None)
        if block is None:
            return
        session_id = self.key_to_session.pop(key, None)
        if session_id is not None:
            keys = self.session_keys.get(session_id)
            if keys is not None and key in keys:
                keys.remove(key)
            if not keys:
                self.session_keys.pop(session_id, None)
                self.session_stats.pop(session_id, None)
        self.evictable_blocks.pop(key, None)

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        # Drop any lingering merge pointer for this request so the dict
        # cannot grow unboundedly across long runs.
        self.pending_merge_pointers.pop(req_context.req_id, None)

    # --- touch: session-hit accounting, ghost scoring, decay ---

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        self.logical_time += 1
        touched_sessions: set[int] = set()
        for position, key in enumerate(keys):
            block = self.blocks.get(key)
            is_hit = block is not None and block.is_ready
            bonus = self.ghost_hit_weight if is_hit else self.ghost_miss_weight
            self.ghost_scores[key] = (
                self.ghost_scores.get(key, 0.0)
                + self._position_weight(position) * bonus
            )
            session_id = self.key_to_session.get(key)
            if session_id is not None:
                touched_sessions.add(session_id)

        for session_id in touched_sessions:
            stats = self.session_stats[session_id]
            stats["hits"] = stats["hits"] + 1
            stats["last_touch"] = self.logical_time

        self.lookups_since_decay += 1
        if self.lookups_since_decay % self.decay_interval == 0:
            self._apply_decay()

    # --- eviction ---

    @override
    def evict(
        self,
        n: int,
        protected: set[OffloadKey],
        req_context: ReqContext,
        num_blocks_in_cache: int,
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []

        # Continuation stores (same request has a live merge pointer to a
        # still-resident session) bypass the admission gate. Fresh
        # sessions must beat the worst incumbent's score to be admitted.
        is_continuation = self._peek_merge_target(req_context.req_id) is not None
        if not is_continuation and not self._admission_allows_new_session(
            num_blocks_in_cache
        ):
            return None

        # Collect evictable candidates atomically. Walk sessions from
        # worst score to best, yielding idle keys from each session's
        # tail. State is only mutated once we know n candidates exist.
        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        session_id: int | None
        sessions_worst_first = sorted(
            self.session_stats.keys(), key=self._session_score
        )
        for session_id in sessions_worst_first:
            if len(candidates) == n:
                break
            keys = self.session_keys.get(session_id, [])
            for key in reversed(keys):
                if len(candidates) == n:
                    break
                if key in protected:
                    continue
                if key not in self.evictable_blocks:
                    continue
                block = self.blocks.get(key)
                if block is None or block.ref_cnt != 0:
                    continue
                candidates.append((key, block))

        if len(candidates) < n:
            return None

        # Commit: drop each candidate from all indices.
        for key, _ in candidates:
            session_id = self.key_to_session.pop(key, None)
            if session_id is not None:
                keys = self.session_keys.get(session_id, [])
                if key in keys:
                    keys.remove(key)
                if not keys:
                    self.session_keys.pop(session_id, None)
                    self.session_stats.pop(session_id, None)
            self.blocks.pop(key, None)
            self.evictable_blocks.pop(key, None)
        return candidates

    # --- clear + evictable set maintenance ---

    @override
    def clear(self) -> None:
        self.blocks.clear()
        self.session_keys.clear()
        self.key_to_session.clear()
        self.session_stats.clear()
        self.ghost_scores.clear()
        self.evictable_blocks.clear()
        self.logical_time = 0
        self.next_session_id = 0
        self.lookups_since_decay = 0
        self.open_session_id = None
        self.open_session_is_merged = False
        self.pending_merge_pointers.clear()

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        self.evictable_blocks[key] = None

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        self.evictable_blocks.pop(key, None)

    # --- internal helpers ---

    def _position_weight(self, position: int) -> float:
        if position < self._POS_WEIGHT_TABLE_SIZE:
            return self._POS_WEIGHTS[position]
        return 1.0 / math.log2(position + 2.0)

    def _prefix_depth_bonus(self, prefix_depth: int) -> float:
        return self._PREFIX_DEPTH_BONUS_SCALE / (
            1.0 + prefix_depth / self._PREFIX_DEPTH_HALF_LIFE
        )

    def _session_score(self, session_id: int) -> float:
        stats = self.session_stats[session_id]
        return (
            stats["last_touch"]
            + stats["hits"] * self._TOUCH_WEIGHT
            + self._prefix_depth_bonus(stats["prefix_depth"])
        )

    def _admission_allows_new_session(self, prefix_depth: int) -> bool:
        # A new session is admitted iff its baseline score (logical time
        # plus prefix-depth bonus) is at least as high as the worst
        # incumbent session's full score. Ghost scores are deliberately
        # not included in this comparison: they seed new session hits
        # but do not shift the gate.
        if not self.session_stats:
            return True
        baseline = self.logical_time + self._prefix_depth_bonus(prefix_depth)
        worst = min(self.session_stats.keys(), key=self._session_score)
        return baseline >= self._session_score(worst)

    def _peek_merge_target(self, req_id: str) -> int | None:
        # Non-consuming peek used by ``evict`` to decide whether a
        # store is a continuation. ``open_session`` pops the entry
        # when it actually consumes the merge.
        session_id = self.pending_merge_pointers.get(req_id)
        if session_id is None or session_id not in self.session_stats:
            return None
        return session_id

    def _apply_decay(self) -> None:
        # Scale session hit counters and ghost scores toward zero so
        # stale sessions and long-dormant keys don't accumulate
        # unbounded importance. Non-resident ghost entries that fall
        # below a small threshold are pruned to bound memory.
        for stats in self.session_stats.values():
            stats["hits"] = stats["hits"] * self.decay_factor
        for key in list(self.ghost_scores):
            decayed = self.ghost_scores[key] * self.decay_factor
            if key not in self.blocks and decayed < self._GHOST_PRUNE_THRESHOLD:
                del self.ghost_scores[key]
            else:
                self.ghost_scores[key] = decayed
