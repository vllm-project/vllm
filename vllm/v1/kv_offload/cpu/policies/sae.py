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
    hits: float
    last_touch: int
    prefix_depth: int


class SAECachePolicy(CachePolicy):
    """
    SAE (Session-Aware Eviction) cache policy.

    Groups newly-stored blocks into per-request sessions and biases
    eviction toward sessions with weaker recent access patterns.
    Per-key "ghost" scores persist across evictions so that a returning
    block still contributes its long-term popularity to whichever new
    session it enters.

    Data Structures:
        blocks: The resident block table (key -> BlockStatus).
        session_keys: Keys owned by each session, in insertion order.
        key_to_session: Reverse index for O(1) session lookup on touch.
        session_stats: Per-session {hits, last_touch, prefix_depth}.
            ``prefix_depth`` is the batch's prefix depth at session
            open time (number of already-cached blocks preceding the
            first newly stored key). Both ``hits`` and ``last_touch``
            grow with subsequent touches.
        ghost_scores: Per-key float score, decayed periodically. Persists
            after a key is evicted so a returning key still carries its
            popularity into the next session it joins.
        evictable_blocks: OrderedDict of keys with ref_cnt == 0,
            maintained by mark_evictable / mark_non_evictable for fast
            candidate scans during eviction.
        open_session_id: The session currently accepting inserts, or None.
            Opened by ``open_session`` and sealed by ``close_session``,
            both of which are called by the manager around each
            prepare_store batch's insert loop.
        pending_merge_pointers: Per-request merge candidate,
            ``req_id -> (session_id, hit_count)``. Set by
            ``record_lookup`` on a resident-block hit and consumed by
            ``open_session`` when the same request's store batch
            arrives. ``hit_count`` counts hits seen for the request
            since the entry was last consumed; it is diagnostic and
            not read by the merge decision.
        open_session_is_merged: Whether the currently-open session was
            entered via a merge (skips ghost-reseeding on further
            inserts into it) rather than freshly created.

    Algorithm Flow:
        1. Cache lookup (get / record_lookup):
           ``get`` is a pure existence check with no side effects.
           A separate hook, ``record_lookup``, signals that a lookup
           was scheduler-driven (not one of the manager's internal
           existence checks in prepare_store / prepare_load /
           complete_load / complete_store). ``record_lookup`` records
           ``req_id -> (session_id, hit_count)`` on any lookup that
           finds a resident block, so the same request's next store
           can merge into that session. Only the last-hit session is
           kept per request.

        2. Cache insertion (open_session -> insert* -> close_session):
           The manager brackets each prepare_store batch's insert loop
           with ``open_session(req_context, num_blocks_in_cache)`` and
           ``close_session()``. ``open_session`` either merges into a
           session recorded by a recent same-request lookup or opens a
           fresh session tagged with the batch's prefix depth.
           ``insert`` appends to the open session; for a
           freshly-opened session, ``hits`` is seeded from each key's
           ghost score (divided by ``ghost_norm``). ``close_session``
           truncates the seeded float ``hits`` to int so subsequent
           accounting stays integer.

        3. Cache touch (touch) — session-hit accounting, ghost
           scoring, and decay:
           Increments the logical clock, then walks the touched keys
           once: each key's residency (resident + ready = hit,
           otherwise miss) drives a ``ghost_hit_weight`` /
           ``ghost_miss_weight`` bump to its ghost score, weighted by
           the key's position within the touch batch. Separately,
           collects the set of unique sessions touched and bumps each
           one's ``hits`` and ``last_touch`` by one. Every
           ``decay_interval``-th call, session ``hits`` and ghost
           scores are scaled by ``decay_factor`` and non-resident
           ghost entries below a small threshold are pruned.

        4. Block eviction (evict) — admission gate + worst-first walk:
           If the current prepare_store batch continues an existing
           session (same request has a live merge pointer), the
           admission gate is skipped and eviction proceeds. Otherwise
           the gate compares the would-be new session's baseline
           (``logical_time + prefix_depth_bonus``) to the worst
           incumbent session's full score; if the baseline is lower,
           ``evict`` returns None to decline the store. Otherwise
           walks sessions in worst-first order, yielding idle
           (ref_cnt == 0, not ``protected``) keys from each session's
           tail until ``n`` are collected. If fewer than ``n`` are
           collectable, returns None without state changes.

    Session Score:
        session_score(s) = last_touch
                         + hits * TOUCH_WEIGHT
                         + PREFIX_DEPTH_BONUS_SCALE
                             / (1 + prefix_depth / PREFIX_DEPTH_HALF_LIFE)

        Lower score = worse. The admission gate compares the would-be
        new session's baseline
        (``logical_time + prefix_depth_bonus``) to the worst
        incumbent's full score. Ghost scores are deliberately excluded
        from the gate — they only seed a new session's initial hits.

    Tunables (constructor kwargs, forwarded from the manager):
        decay_interval: Number of ``touch`` calls between decay ticks
            (roughly one ``touch`` per request).
        decay_factor: Scale factor applied to session hits and ghost
            scores at each decay tick.
        ghost_hit_weight: Ghost score bump on resident+ready hits during
            touch.
        ghost_miss_weight: Ghost score bump on misses (block absent or
            not ready) during touch.
        ghost_norm: Divisor when seeding a new session's initial hits
            from its keys' ghost scores.
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

        # Merge pointers, keyed per in-flight request.
        self.pending_merge_pointers: dict[str, tuple[int, int]] = {}

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
        # We accept any resident block (including HIT_PENDING blocks
        # whose store has not yet completed) so that hit_count aligns
        # with the manager's "already stored?" filter used to derive
        # `num_blocks_in_cache`.
        block = self.blocks.get(key)
        if block is None:
            return
        session_id = self.key_to_session.get(key)
        if session_id is None:
            return
        req_id = req_context.req_id
        _, hit_count = self.pending_merge_pointers.get(req_id, (session_id, 0))
        self.pending_merge_pointers[req_id] = (session_id, hit_count + 1)

    # --- session lifecycle ---

    @override
    def open_session(self, req_context: ReqContext, num_blocks_in_cache: int) -> None:
        # Called by the manager immediately before its insert loop.
        # If this request has a live merge pointer whose target session
        # is still resident, continue that session; otherwise open a
        # fresh session tagged with the batch's prefix depth.
        assert self.open_session_id is None, "open_session called with unclosed session"
        entry = self.pending_merge_pointers.pop(req_context.req_id, None)
        merge_target: int | None = None
        if entry is not None:
            candidate_session, _ = entry
            if candidate_session in self.session_stats:
                merge_target = candidate_session

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
        # Truncate the seeded float `hits` to int so subsequent
        # accounting (which does `hits += 1` on touch) stays integer.
        if (
            self.open_session_id is not None
            and self.open_session_id in self.session_stats
        ):
            stats = self.session_stats[self.open_session_id]
            stats["hits"] = int(stats["hits"])
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
        entry = self.pending_merge_pointers.get(req_id)
        if entry is None:
            return None
        session_id, _ = entry
        if session_id not in self.session_stats:
            return None
        return session_id

    def _apply_decay(self) -> None:
        # Scale session hit counters and ghost scores toward zero so
        # stale sessions and long-dormant keys don't accumulate
        # unbounded importance. Non-resident ghost entries that fall
        # below a small threshold are pruned to bound memory.
        for stats in self.session_stats.values():
            stats["hits"] = int(stats["hits"] * self.decay_factor)
        for key in list(self.ghost_scores):
            decayed = self.ghost_scores[key] * self.decay_factor
            if key not in self.blocks and decayed < self._GHOST_PRUNE_THRESHOLD:
                del self.ghost_scores[key]
            else:
                self.ghost_scores[key] = decayed
