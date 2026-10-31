# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Collection, Sequence

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, get_offload_group_idx
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

# Demoted chunks rank below every LRU rank (whose first element counts up from
# 1), oldest demotion first.
_DEMOTED_RANK_BASE = -(1 << 62)


class MambaAwareLRUCachePolicy(LRUCachePolicy):
    """LRU that evicts superseded Mamba states first.

    A hybrid model offloads one state per Mamba group at the deepest boundary
    a request stores, its tip. Once a later request on the same block-hash
    chain stores a deeper state, the old tip only serves requests that branch
    off between the two, yet LRU refreshes it whenever a request loads it.

    When a request finishes, each cached Mamba chunk on its chain that lies
    below the deepest cached state and was stored as its writer's tip moves to
    the eviction end. Attention chunks, states stored below their writer's
    tip (such as shared-prefix junctions) and the deepest state keep their
    LRU rank. A request that reuses a demoted chunk gives it an LRU rank
    again. Without Mamba groups, this is plain LRU.
    """

    def __init__(
        self, cache_capacity: int, mamba_group_ids: Collection[int] = ()
    ) -> None:
        super().__init__(cache_capacity)
        self._mamba_group_ids = frozenset(mamba_group_ids)
        self._tips: set[OffloadKey] = set()
        self._num_demoted = 0

    @override
    def remove(self, key: OffloadKey) -> None:
        super().remove(key)
        self._tips.discard(key)

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        evicted = super().evict(n, protected)
        if evicted:
            self._tips.difference_update(key for key, _ in evicted)
        return evicted

    @override
    def clear(self) -> None:
        super().clear()
        self._tips.clear()

    @override
    def on_request_finished(
        self,
        key_groups: Sequence[Sequence[OffloadKey]],
        insertion_only_keys: set[OffloadKey],
        reused_keys: set[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        super().on_request_finished(
            key_groups, insertion_only_keys, reused_keys, req_context
        )
        if not self._mamba_group_ids:
            return
        # The positions span the request's whole chain, so this also finds
        # cached states that the request did not access in this tier.
        states = {
            key: position
            for key, position in req_context.get_offload_key_positions().items()
            if key in self.chunks
            and get_offload_group_idx(key) in self._mamba_group_ids
        }
        if not states:
            return
        new_states = [key for key in insertion_only_keys if key in states]
        if new_states:
            tip = max(states[key] for key in new_states)
            self._tips.update(key for key in new_states if states[key] == tip)
        deepest = max(states.values())
        for key, position in states.items():
            if position < deepest and key in self._tips and self._ranks[key][0] > 0:
                self._num_demoted += 1
                self._ranks[key] = (_DEMOTED_RANK_BASE + self._num_demoted, 0)
                if key in self._evictable:
                    self._push_evictable(key)
        self._maybe_compact_heap()
