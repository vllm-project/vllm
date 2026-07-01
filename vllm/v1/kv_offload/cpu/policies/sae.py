# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-Aware Eviction (SAE) cache policy for CPU offload.

Ported from the out-of-tree ``sae_kv_offload`` plugin package. Two
documented semantic differences from that reference:

1. Session boundaries are reconstructed from the call sequence
   (``insert`` after ``touch``/``evict``/``remove``/``clear`` opens a
   new session; consecutive ``insert`` calls join it) rather than
   taken from a batch of block hashes handed to ``prepare_store``.
2. Per-batch position weighting on ``get`` is dropped because the
   current scheduler calls ``manager.lookup`` one key at a time.
"""

from collections import OrderedDict
from collections.abc import Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class SAECachePolicy(CachePolicy):
    """Session-Aware Eviction cache policy.

    See the module docstring for the two semantic differences from
    the reference algorithm.
    """

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

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        block = self._blocks.get(key)
        if block is not None and block.is_ready:
            self._key_ghost[key] = (
                self._key_ghost.get(key, 0.0) + self._ghost_hit_weight
            )
        else:
            self._key_ghost[key] = (
                self._key_ghost.get(key, 0.0) + self._ghost_miss_weight
            )
        self._lookup_count += 1
        if self._lookup_count % self._decay_interval == 0:
            self._run_decay()
        self._last_event = "get"
        return block

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
            sid = self._sid_counter
            self._sid_counter += 1
            self._open_sid = sid
            self._sid_to_keys[sid] = []
            self._sid_stats[sid] = {
                "hits": 0.0,
                "last_touch": self._logical_timer,
                "start_pos": 0,
            }
        sid = self._open_sid
        self._sid_to_keys[sid].append(key)
        self._key_to_sid[key] = sid
        self._blocks[key] = block
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
        for k in keys:
            sid = self._key_to_sid.get(k)
            if sid is not None:
                touched_sids.add(sid)
        for sid in touched_sids:
            stats = self._sid_stats[sid]
            stats["hits"] = stats["hits"] + 1
            stats["last_touch"] = self._logical_timer
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
        self._seal_open_session()
        self._last_event = "evict"
        if n == 0:
            return []
        if not self._admission_gate_allows():
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

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        self._evictable_keys[key] = None

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        self._evictable_keys.pop(key, None)
