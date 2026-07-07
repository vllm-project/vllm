# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-Aware Eviction (SAE) cache policy for CPU offload.

Ported from the out-of-tree ``sae_kv_offload`` plugin package. See
``SAECachePolicy`` for the algorithm description and the five
documented semantic differences from the reference.
"""

import math
from collections import OrderedDict
from collections.abc import Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey
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
        _last_event: Most recent event kind; drives session-boundary
            detection.
        _last_hit_sid: sid of the most recent get() hit, or None.
            Consumed by the next insert() that opens a session, to decide
            whether to merge into it instead of starting fresh (see
            semantic difference #4).
        _merged_open_session: Whether the currently-open session was
            entered via a merge (skips ghost-reseeding on further inserts
            into it) rather than freshly created.

    Algorithm Flow:
        1. Cache lookup (get):
           Returns the block if resident. No bookkeeping — ghost
           scoring and decay live in `touch` (see below), since `get`
           is called from several internal existence checks
           (prepare_load, complete_load, prepare_store, complete_store)
           that aren't genuine request-driven cache accesses.

        2. Cache insertion (insert) - Session boundary detection:
           If the previous event was not another `insert`, first checks
           `_last_hit_sid` for a live merge candidate (see semantic
           difference #4); if none, opens a new session. Consecutive
           `insert`s join the open session (merged or fresh). For a
           freshly-opened session, `hits` is seeded from
           `_key_ghost[key] / ghost_norm` accumulated across the
           session's inserts, then truncated to int when the session
           closes; a merged session's `hits` is left untouched by
           inserts (only `last_touch` is bumped), matching the
           reference's is_merging behavior.

        3. Cache touch (touch) - Session hit accounting + ghost
           scoring + decay:
           Closes the open session. Increments the logical timer, then
           walks the touched keys once: each key's residency
           (resident + ready = hit, otherwise miss) drives a
           `ghost_hit_weight` / `ghost_miss_weight` bump to
           `_key_ghost[key]`, weighted by `_pos_weight(i)` where `i` is
           the key's index within this call's batch — earlier keys
           accumulate ghost faster, matching the reference's
           prefix-position weighting. Separately, collects the set of
           unique sids touched and bumps `hits += 1` and `last_touch`
           once per unique sid — a batch of 10 keys from the same
           session adds 1, not 10. Every `decay_interval`-th call
           (counted per `touch` invocation, i.e. per request), session
           `hits` are scaled by `decay_factor` (truncated to int,
           matching the reference) and non-resident ghost entries
           below 0.01 are pruned.

        4. Block eviction (evict) - Admission gate + worst-first walk:
           Closes the open session, then runs the admission gate unless
           `_last_hit_sid` names a live session (merging in progress —
           see semantic difference #4), in which case the gate is
           skipped entirely, matching the reference's `not is_merging`
           condition. Otherwise the gate compares the would-be new
           session's baseline score `logical_timer + 30000.0` to the
           worst incumbent's full score (see _score below); if the
           baseline is lower, evict returns None declining the store.
           Otherwise walks sessions worst-first, yielding idle
           (ref_cnt == 0, not in `protected`) keys from each session's
           tail until `n` are collected. Returns None if fewer than
           `n` are collectable — no partial state changes.

    Session Score:
        _score(sid) = last_touch + hits*1500.0 + 30000.0/(1 + start_pos/8)
        Lower score = worse. The admission gate compares the would-be
        new session's baseline (`logical_timer + 30000.0`) to the worst
        incumbent's full score, deliberately excluding ghost scores
        from the gate — matching the reference algorithm's design.

    Tunables (constructor kwargs, forwarded from the manager):
        decay_interval: `touch` calls (~one per request) between decay
            ticks.
        decay_factor: Scale factor applied per decay tick.
        ghost_hit_weight: Ghost score bump on resident+ready hits.
        ghost_miss_weight: Ghost score bump on misses / non-ready.
        ghost_norm: Divisor when seeding new session hits from ghost
            scores.

    Semantic differences from the reference algorithm:
        1. Session boundaries are reconstructed from the call sequence
           (insert after touch/evict/remove/clear opens a new session;
           consecutive inserts join it) rather than taken from a batch
           of block hashes handed to prepare_store.
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
        3. start_pos is always zero: the per-key insert interface has
           no batch to index into, so every session gets a fixed
           pos_bonus = 30000.0 in both _score and the admission gate
           baseline (internally consistent — the gate compares
           apples to apples).
        4. Session continuation (is_merging) is approximated by a
           single `_last_hit_sid` pointer instead of the reference's
           `_last_lookup_sid` / `_last_lookup_timer` / `_last_lookup_count`
           plus a `start_pos` equality check. The reference's version
           depends on batched lookup/prepare_store calls computing an
           integer position within one list; get/insert here are
           per-key with no batch, position, or request identity at
           all. `_last_hit_sid` is set by get() on a hit, left
           untouched on a miss (it must survive the trailing
           not-yet-stored-key checks in the manager's own
           "already stored?" filter), read by evict() to skip the
           gate, and consumed once by the next insert() that opens a
           session. This relies on get()/evict()/insert() running
           within one synchronous prepare_store() call for correctness
           rather than a numeric recency window, and — like the
           reference, which also has no request identity — can in
           principle merge into an unrelated session if an unrelated
           hit lands in between; this is accepted as the same class of
           fragility the reference already has, not solved here.
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
        self._last_event: str = "init"
        self._last_hit_sid: int | None = None
        self._merged_open_session: bool = False

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        block = self._blocks.get(key)
        if block is not None and block.is_ready:
            sid = self._key_to_sid.get(key)
            if sid is not None:
                self._last_hit_sid = sid
        self._last_event = "get"
        return block

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

    def _seal_open_session(self) -> None:
        """Close the currently-open session, truncating its float-accumulated
        ``hits`` to int so subsequent arithmetic stays integer (matching the
        reference's ``initial_hits = int(ghost_sum / ghost_norm)`` step).
        """
        if self._open_sid is not None and self._open_sid in self._sid_stats:
            stats = self._sid_stats[self._open_sid]
            stats["hits"] = int(stats["hits"])
        self._open_sid = None

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        if self._last_event != "insert" or self._open_sid is None:
            merge_sid = self._last_hit_sid
            self._last_hit_sid = None
            if merge_sid is not None and merge_sid in self._sid_stats:
                # Continuing a session a recent get() hit into — matching the
                # reference's is_merging: bump last_touch only, no fresh sid,
                # no ghost-seeded hits for the merged blocks.
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
                    "start_pos": 0,
                }
                self._merged_open_session = False
        sid = self._open_sid
        self._sid_to_keys[sid].append(key)
        self._key_to_sid[key] = sid
        self._blocks[key] = block
        if not self._merged_open_session:
            seed = self._key_ghost.get(key, 0.0) / self._ghost_norm
            self._sid_stats[sid]["hits"] += seed
        self._last_event = "insert"

    @override
    def remove(self, key: OffloadKey) -> None:
        block = self._blocks.pop(key, None)
        if block is None:
            self._seal_open_session()
            self._last_event = "remove"
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
        self._seal_open_session()
        self._last_event = "remove"

    @override
    def touch(self, keys: Iterable[OffloadKey]) -> None:
        self._seal_open_session()
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
        self._last_event = "touch"

    def _score(self, sid: int) -> float:
        stats = self._sid_stats[sid]
        pos_bonus = 30000.0 / (1.0 + stats["start_pos"] / 8.0)
        freq_bonus = stats["hits"] * 1500.0
        return stats["last_touch"] + freq_bonus + pos_bonus

    def _admission_gate_allows(self) -> bool:
        # Reference excludes ghost scores from the admission gate — the gate
        # compares the incumbent worst score to a bare timer+pos_bonus baseline
        # (see the reference's prepare_store comment "Block ghost scores are
        # intentionally NOT included here").
        if not self._sid_stats:
            return True
        new_score = self._logical_timer + 30000.0
        worst_sid = min(self._sid_stats.keys(), key=self._score)
        return new_score >= self._score(worst_sid)

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        # Captured before _seal_open_session() (which doesn't touch
        # _last_hit_sid) so a merge candidate set by a preceding get() this
        # same prepare_store() call still bypasses the gate, matching the
        # reference's `if not is_merging and needed > 0`.
        is_merging = (
            self._last_hit_sid is not None and self._last_hit_sid in self._sid_stats
        )
        self._seal_open_session()
        self._last_event = "evict"
        if n == 0:
            return []
        if not is_merging and not self._admission_gate_allows():
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
        self._last_event = "clear"
        self._last_hit_sid = None
        self._merged_open_session = False

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        self._evictable_keys[key] = None

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        self._evictable_keys.pop(key, None)
