# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-Aware Eviction (SAE) cache policy for CPU offload.

Ported from the out-of-tree ``sae_kv_offload`` plugin package. See
``SAECachePolicy`` for the algorithm description and the documented
semantic differences from the reference.
"""

import math
from collections import OrderedDict
from collections.abc import Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class SAECachePolicy(CachePolicy):
    """
    SAE (Session-Aware Eviction) cache policy.

    Groups newly-stored blocks into sessions and biases eviction toward
    sessions with weaker recent access patterns. Per-key "ghost" scores
    persist across evictions so a session's baseline reflects the
    long-term popularity of the blocks it contains.

    Data Structures:
        _blocks: The resident block table (key -> BlockStatus).
        _sid_to_keys: Keys owned by each session, insertion-ordered.
        _key_to_sid: Reverse index for O(1) session lookup on touch.
        _sid_stats: Per-session {hits, last_touch, start_pos}.
        _key_ghost: Per-key float score, decayed periodically. Persists
            after a key is evicted so a returning key still carries its
            popularity into the next session.
        _evictable_keys: OrderedDict of keys with ref_cnt == 0,
            maintained by mark_evictable / mark_non_evictable for fast
            eviction candidate scans.
        _open_sid: The session currently accepting inserts, or None.
            Opened by `open_session` and sealed by `close_session`, both
            of which are called by the manager around each prepare_store
            batch's insert loop.
        _lookup_state: Per-req_id merge pointer, ``req_id ->
            (sid, hit_count)``. ``sid`` is the session id of the most
            recent resident hit; ``hit_count`` counts hits recorded
            for this req_id since the entry was popped. Set/updated
            by `record_lookup`. Consumed and removed by `open_session`
            when a same-req_id store batch arrives; the merge fires as
            long as ``sid`` is still resident (the reference's
            ``hit_count == start_pos`` positional check is dropped —
            see semantic difference #4).
        _merged_open_session: Whether the currently-open session was
            entered via a merge (skips ghost-reseeding on further inserts
            into it) rather than freshly created.

    Algorithm Flow:
        1. Cache lookup (get / record_lookup):
           `get` is a pure existence check with no side effects. The
           CPU manager calls it in five places, only one of which is a
           genuine scheduler-driven request lookup — the other four
           (prepare_store's "already stored?" filter, prepare_load,
           complete_load, complete_store) are bookkeeping. Merge-pointer
           setting is factored out into `record_lookup(key, req_id)`,
           which the manager calls only from its own `lookup` method
           (the scheduler entry point). Each lookup on an existing
           block (present in ``_blocks``, whether or not ``is_ready``)
           for a given req_id bumps that entry's ``hit_count``
           (initializing to 1 on first hit) and refreshes ``sid`` to
           the hit's owning session, so at the end of a
           scheduler-prefix pass the entry captures how many blocks of
           prefix hit and which session owns the last one. We
           deliberately count ``HIT_PENDING`` (stored-but-not-yet-ready)
           blocks here to align with ``prepare_store``'s
           already-stored filter, which excludes ALL existing blocks
           from ``keys_to_store``. If get() set the merge pointer,
           the "already stored?" filter's hits on prior sessions would
           cause every store to be merged into whichever session
           happened to own a prefix hit — collapsing the whole cache
           to one session at high hit rates.

        2. Cache insertion (open_session -> insert* -> close_session):
           The manager brackets each prepare_store batch's insert loop
           with `open_session(req_id, start_pos)` and `close_session()`.
           `open_session` decides whether to merge into the sid recorded
           by a recent same-req_id lookup (see semantic difference #4)
           or open a fresh session with the batch's real `start_pos`.
           `insert` simply appends to the open session; for a
           freshly-opened session, `hits` is seeded from
           `_key_ghost[key] / ghost_norm` on each insert. `close_session`
           truncates float `hits` to int, matching the reference's
           `initial_hits = int(ghost_sum / ghost_norm)` step.

        3. Cache touch (touch) - Session hit accounting + ghost
           scoring + decay:
           Increments the logical timer, then walks the touched keys
           once: each key's residency (resident + ready = hit, otherwise
           miss) drives a `ghost_hit_weight` / `ghost_miss_weight` bump
           to `_key_ghost[key]`, weighted by `_pos_weight(i)`. Separately,
           collects unique sids touched and bumps `hits += 1` and
           `last_touch` once per unique sid. Every `decay_interval`-th
           call, session `hits` are scaled by `decay_factor` and
           non-resident ghost entries below 0.01 are pruned.

        4. Block eviction (evict) - Admission gate + worst-first walk:
           If the current prepare_store batch is continuing (merging into)
           a session — same req_id and lookup within the last one timer
           step — the admission gate is skipped, matching the reference's
           `not is_merging` condition. Otherwise the gate compares the
           would-be new session's baseline
           `logical_timer + 30000/(1 + start_pos/8)` to the worst
           incumbent's full score (see _score below); if the baseline is
           lower, evict returns None declining the store. Otherwise walks
           sessions worst-first, yielding idle (ref_cnt == 0, not in
           `protected`) keys from each session's tail until `n` are
           collected. Returns None if fewer than `n` are collectable —
           no partial state changes.

    Session Score:
        _score(sid) = last_touch + hits*1500.0 + 30000.0/(1 + start_pos/8)
        Lower score = worse. The admission gate compares the would-be
        new session's baseline (`logical_timer + 30000/(1 + start_pos/8)`)
        to the worst incumbent's full score, deliberately excluding
        ghost scores from the gate — matching the reference algorithm's
        design.

    Tunables (constructor kwargs, forwarded from the manager):
        decay_interval: `touch` calls (~one per request) between decay
            ticks.
        decay_factor: Scale factor applied per decay tick.
        ghost_hit_weight: Ghost score bump on resident+ready hits.
        ghost_miss_weight: Ghost score bump on misses / non-ready.
        ghost_norm: Divisor when seeding new session hits from ghost
            scores.

    Semantic differences from the reference algorithm:
        1. Session boundaries are opened and closed explicitly by the
           manager (`open_session` / `close_session` calls) rather than
           reconstructed from a call sequence or a batch of block hashes
           handed to prepare_store. This makes the boundary independent
           of the surrounding call pattern (touch, remove, get) which
           the reference implicitly relied on.
        2. Ghost-score position weighting and hit/miss classification
           happen in `touch`, not `get`: the scheduler calls
           manager.lookup (get) one key at a time with no batch
           context, but calls touch once per request with the group's
           known key list. That list isn't necessarily the exact set
           `get` classified as hits this pass (it can include more, or
           fewer, keys — e.g. a sliding-window suffix), so `touch`
           reclassifies hit/miss per key from residency rather than
           trusting a precomputed hit count. Position within `touch`'s
           batch is a prefix position for full-attention groups, and a
           window-relative position for sliding-window groups (which
           touch a trimmed suffix).
        3. `start_pos` is threaded from the manager. The manager
           computes it as ``len(keys) - len(keys_to_store)`` in
           prepare_store — the number of already-resident prefix blocks
           preceding the first not-yet-stored key — matching the
           reference's ``bh_list.index(to_store[0])`` at
           ``sae-kv-offload/manager.py:208``. Passed to both `evict`
           (for the admission gate baseline) and `open_session` (for
           new-session stats), so gate math and session scoring both
           see the real prefix depth. Earlier port versions pinned
           start_pos = 0, which caused pos_bonus to cancel out of the
           gate comparison and the gate to refuse ~all stores once
           cache filled.
        4. Session continuation (`is_merging`) uses only per-`req_id`
           sid identity — the reference's ``_last_lookup_count ==
           start_pos`` positional check and its ``<= 1 timer`` window
           are both dropped. ``record_lookup(key, req_id)`` records
           ``req_id -> (sid, hit_count)`` in ``_lookup_state`` on any
           hit that finds a resident block, refreshing ``sid`` to the
           owning session of the most recent hit. ``evict`` and
           ``open_session`` merge into that sid as long as it is still
           in ``_sid_stats``. ``hit_count`` is kept in the tuple so
           the invariant "one entry per in-flight request" is easy to
           reason about and so future diagnostics can reconstruct
           drift, but it is not read by the merge decision.

           Rationale: the reference's design assumed lookup and store
           happen back-to-back inside one scheduler tick (vllm 0.18
           semantics), which vllm 0.23's batched scheduler no longer
           honors. Cache churn between a request's lookup and its own
           ``prepare_store`` — driven by concurrent traffic under high
           conversation counts — makes positional equality
           unreachable in practice, so requiring it disables merging
           entirely. Per-req_id state (instead of the reference's
           single global slot) is still required to prevent concurrent
           requests from overwriting each other's merge pointers
           between lookup and prepare_store. ``get`` deliberately does
           NOT touch ``_lookup_state`` — internal existence checks in
           prepare_store, prepare_load etc. must not imply "continue
           that session."
        5. Eviction is atomic; the reference is not. The reference's
           prepare_store frees blocks as it walks the worst-first
           order and only checks `needed > 0` after the walk
           finishes — so a walk that comes up short has already
           freed some real blocks before reporting failure. This
           `evict` instead collects candidates into a list first and
           only mutates state once `len(candidates) >= n` is
           confirmed, per the `CachePolicy.evict` contract ("if None
           is returned, no state changes are made"). Required by the
           interface, not a bug to port.
    """

    # Precomputed 1/log2(pos+2) table for _pos_weight, matching the reference.
    _POS_WEIGHTS = [1.0 / math.log2(i + 2.0) for i in range(1024)]

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
        self._cache_capacity = cache_capacity
        self._decay_interval = decay_interval
        self._decay_factor = decay_factor
        self._ghost_hit_weight = ghost_hit_weight
        self._ghost_miss_weight = ghost_miss_weight
        self._ghost_norm = ghost_norm

        self._blocks: dict[OffloadKey, BlockStatus] = {}
        self._sid_to_keys: dict[int, list[OffloadKey]] = {}
        self._key_to_sid: dict[OffloadKey, int] = {}
        self._sid_stats: dict[int, dict[str, float | int]] = {}
        self._key_ghost: dict[OffloadKey, float] = {}
        self._evictable_keys: OrderedDict[OffloadKey, None] = OrderedDict()

        self._logical_timer: int = 0
        self._sid_counter: int = 0
        self._lookup_count: int = 0
        self._open_sid: int | None = None
        # Per-req_id merge pointer, req_id -> (sid, hit_count). hit_count
        # counts resident+ready hits recorded for this req_id since the
        # entry was last popped, letting open_session enforce
        # `hit_count == start_pos` (see semantic difference #4).
        self._lookup_state: dict[str, tuple[int, int]] = {}
        self._merged_open_session: bool = False

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        # Pure existence check — no side effects. See `record_lookup` for
        # the merge-pointer entry point.
        return self._blocks.get(key)

    @override
    def record_lookup(self, key: OffloadKey, req_id: str) -> None:
        # Only scheduler-driven request lookups feed the session-merge
        # pointer, matching the reference algorithm which set its merge
        # pointer only inside its `lookup` method — not inside
        # `prepare_store`. Keyed by req_id so concurrent requests don't
        # overwrite each other's merge pointers between lookup and
        # prepare_store (see semantic difference #4).
        #
        # Each existing-block lookup bumps the entry's hit_count and
        # refreshes sid to the hit's owning session, so at the end of
        # the scheduler's prefix walk the entry captures
        # (last-hit sid, prefix depth). Any block present in
        # ``_blocks`` counts here — including one that is stored but
        # not yet is_ready (`HIT_PENDING` at the manager level). This
        # matches what ``prepare_store``'s "already stored?" filter
        # treats as "already present": it uses ``self._policy.get(key)
        # is None`` and thus excludes pending blocks from
        # ``keys_to_store``. If ``record_lookup`` filtered on
        # ``is_ready`` as well, our recorded ``hit_count`` would come
        # in below the actual prefix depth the scheduler and
        # ``prepare_store`` see — the positional check would then
        # fail with a large ``start_pos - hit_count`` drift and merges
        # would never fire (observed empirically in benchmark run
        # 1169193: `pos_drift_mean ≈ 67 blocks`).
        # A trailing miss (block is None) leaves the entry untouched,
        # matching the reference which stopped incrementing at the
        # first miss.
        block = self._blocks.get(key)
        if block is None:
            return
        sid = self._key_to_sid.get(key)
        if sid is None:
            return
        _, hit_count = self._lookup_state.get(req_id, (sid, 0))
        self._lookup_state[req_id] = (sid, hit_count + 1)

    def _pos_weight(self, pos: int) -> float:
        if pos < len(self._POS_WEIGHTS):
            return self._POS_WEIGHTS[pos]
        return 1.0 / math.log2(pos + 2.0)

    def _run_decay(self) -> None:
        for stats in self._sid_stats.values():
            stats["hits"] = int(stats["hits"] * self._decay_factor)
        for k in list(self._key_ghost):
            new_score = self._key_ghost[k] * self._decay_factor
            if k not in self._blocks and new_score < 0.01:
                del self._key_ghost[k]
            else:
                self._key_ghost[k] = new_score

    def _peek_merge_sid(self, req_id: str) -> int | None:
        """Return the merge-candidate sid for ``req_id`` if one is still
        resident. Non-consuming — used by ``evict`` to decide gate-skip.
        ``open_session`` pops the entry when it actually consumes it."""
        entry = self._lookup_state.get(req_id)
        if entry is None:
            return None
        sid, _ = entry
        if sid not in self._sid_stats:
            return None
        return sid

    @override
    def on_request_finished(self, req_id: str) -> None:
        self._lookup_state.pop(req_id, None)

    @override
    def open_session(self, req_id: str, start_pos: int) -> None:
        # Called by the manager before the insert loop of a prepare_store
        # batch. Consume any live merge candidate this req_id installed
        # via record_lookup and continue that session as long as its sid
        # is still resident. The reference's ``_last_lookup_count ==
        # start_pos`` positional check was dropped: under vllm 0.23's
        # batched scheduler, cache churn between a request's ``lookup``
        # and its ``prepare_store`` made positional equality essentially
        # unreachable, so requiring it disabled the merge machinery.
        # Per-req_id keying still prevents cross-request pointer leaks
        # (see semantic difference #4).
        assert self._open_sid is None, "open_session called with unclosed session"
        entry = self._lookup_state.pop(req_id, None)
        merge_sid: int | None = None
        if entry is not None:
            entry_sid, _ = entry
            if entry_sid in self._sid_stats:
                merge_sid = entry_sid
        if merge_sid is not None:
            self._open_sid = merge_sid
            self._sid_stats[merge_sid]["last_touch"] = self._logical_timer
            self._merged_open_session = True
        else:
            sid = self._sid_counter
            self._sid_counter += 1
            self._open_sid = sid
            self._sid_to_keys[sid] = []
            self._sid_stats[sid] = {
                "hits": 0.0,
                "last_touch": self._logical_timer,
                "start_pos": start_pos,
            }
            self._merged_open_session = False

    @override
    def close_session(self) -> None:
        """Close the currently-open session, truncating its float-accumulated
        ``hits`` to int so subsequent arithmetic stays integer (matching the
        reference's ``initial_hits = int(ghost_sum / ghost_norm)`` step).
        """
        if self._open_sid is not None and self._open_sid in self._sid_stats:
            stats = self._sid_stats[self._open_sid]
            stats["hits"] = int(stats["hits"])
        self._open_sid = None
        self._merged_open_session = False

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        assert self._open_sid is not None, (
            "insert called without an open session — the manager must call "
            "open_session before its insert loop"
        )
        sid = self._open_sid
        self._sid_to_keys[sid].append(key)
        self._key_to_sid[key] = sid
        self._blocks[key] = block
        if not self._merged_open_session:
            seed = self._key_ghost.get(key, 0.0) / self._ghost_norm
            self._sid_stats[sid]["hits"] += seed

    @override
    def remove(self, key: OffloadKey) -> None:
        block = self._blocks.pop(key, None)
        if block is None:
            return
        sid = self._key_to_sid.pop(key, None)
        if sid is not None:
            seq = self._sid_to_keys.get(sid)
            if seq is not None and key in seq:
                seq.remove(key)
            if not seq:
                self._sid_to_keys.pop(sid, None)
                self._sid_stats.pop(sid, None)
        self._evictable_keys.pop(key, None)

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        self._logical_timer += 1
        touched_sids: set[int] = set()
        for i, k in enumerate(keys):
            block = self._blocks.get(k)
            is_hit = block is not None and block.is_ready
            bonus = self._ghost_hit_weight if is_hit else self._ghost_miss_weight
            self._key_ghost[k] = (
                self._key_ghost.get(k, 0.0) + self._pos_weight(i) * bonus
            )
            sid = self._key_to_sid.get(k)
            if sid is not None:
                touched_sids.add(sid)
        for sid in touched_sids:
            stats = self._sid_stats[sid]
            stats["hits"] = stats["hits"] + 1
            stats["last_touch"] = self._logical_timer
        self._lookup_count += 1
        if self._lookup_count % self._decay_interval == 0:
            self._run_decay()

    def _score(self, sid: int) -> float:
        stats = self._sid_stats[sid]
        pos_bonus = 30000.0 / (1.0 + stats["start_pos"] / 8.0)
        freq_bonus = stats["hits"] * 1500.0
        return stats["last_touch"] + freq_bonus + pos_bonus

    def _admission_gate_allows(self, start_pos: int) -> bool:
        # Reference excludes ghost scores from the admission gate — the gate
        # compares the incumbent worst score to a bare timer+pos_bonus baseline
        # (see the reference's prepare_store comment "Block ghost scores are
        # intentionally NOT included here").
        if not self._sid_stats:
            return True
        new_score = self._logical_timer + 30000.0 / (1.0 + start_pos / 8.0)
        worst_sid = min(self._sid_stats.keys(), key=self._score)
        return new_score >= self._score(worst_sid)

    @override
    def evict(
        self,
        n: int,
        protected: set[OffloadKey],
        req_id: str,
        start_pos: int,
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []
        # Peek — don't consume. open_session will pop the entry when it
        # actually merges the store into the recorded session.
        is_merging = self._peek_merge_sid(req_id) is not None
        if not is_merging and not self._admission_gate_allows(start_pos):
            return None

        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        sorted_sids = sorted(self._sid_stats.keys(), key=self._score)
        for sid in sorted_sids:
            if len(candidates) == n:
                break
            seq = self._sid_to_keys.get(sid, [])
            for k in reversed(seq):
                if len(candidates) == n:
                    break
                if k in protected:
                    continue
                if k not in self._evictable_keys:
                    continue
                block = self._blocks.get(k)
                if block is None or block.ref_cnt != 0:
                    continue
                candidates.append((k, block))

        if len(candidates) < n:
            return None

        for k, _ in candidates:
            if k in self._key_to_sid:
                sid = self._key_to_sid.pop(k)
                sid_seq = self._sid_to_keys.get(sid, [])
                if k in sid_seq:
                    sid_seq.remove(k)
                if not sid_seq:
                    self._sid_to_keys.pop(sid, None)
                    self._sid_stats.pop(sid, None)
            self._blocks.pop(k, None)
            self._evictable_keys.pop(k, None)
        return candidates

    @override
    def clear(self) -> None:
        self._blocks.clear()
        self._sid_to_keys.clear()
        self._key_to_sid.clear()
        self._sid_stats.clear()
        self._key_ghost.clear()
        self._evictable_keys.clear()
        self._logical_timer = 0
        self._sid_counter = 0
        self._lookup_count = 0
        self._open_sid = None
        self._lookup_state.clear()
        self._merged_open_session = False

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        self._evictable_keys[key] = None

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        self._evictable_keys.pop(key, None)
