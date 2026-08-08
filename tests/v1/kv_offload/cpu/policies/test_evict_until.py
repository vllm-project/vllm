"""
Focused unit tests for CachePolicy.evict_until() (LRU + ARC).

Covers: predicate failure (no mutation), predicate success (exact commit),
protected keys, ref_cnt filtering, ARC ghost-list semantics, and backward-
compatible evict(n) wrapper.
"""

from collections.abc import Iterable

import pytest

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy


def _k(i: int) -> OffloadKey:
    return make_offload_key(f"k{i:04d}".encode(), 0)


def _block(block_id: int, ref_cnt: int = 0) -> BlockStatus:
    b = BlockStatus(block_id)
    b.ref_cnt = ref_cnt
    return b


def _block_ids(policy):
    if hasattr(policy, "blocks"):
        return frozenset((b.block_id, b.ref_cnt) for b in policy.blocks.values())
    return frozenset((b.block_id, b.ref_cnt) for b in policy.t1.values()) | frozenset(
        (b.block_id, b.ref_cnt) for b in policy.t2.values()
    )


def _save_state(policy):
    blocks = _block_ids(policy)
    if isinstance(policy, ARCCachePolicy):
        meta = {
            "t1_keys": list(policy.t1.keys()),
            "t2_keys": list(policy.t2.keys()),
            "b1_keys": list(policy.b1.keys()),
            "b2_keys": list(policy.b2.keys()),
            "target_t1_size": policy.target_t1_size,
        }
    else:
        meta = {"evictable_keys": list(policy.evictable_blocks.keys())}
    return blocks, meta


def _assert_unchanged(before_blocks, before_meta, policy):
    assert before_blocks == _block_ids(policy)
    if isinstance(policy, ARCCachePolicy):
        assert list(policy.t1.keys()) == before_meta["t1_keys"]
        assert list(policy.t2.keys()) == before_meta["t2_keys"]
        assert list(policy.b1.keys()) == before_meta["b1_keys"]
        assert list(policy.b2.keys()) == before_meta["b2_keys"]
        assert policy.target_t1_size == before_meta["target_t1_size"]
    else:
        assert list(policy.evictable_blocks.keys()) == before_meta["evictable_keys"]


@pytest.fixture(params=["lru", "arc"])
def filled_policy(request):
    cap = 8
    pol = LRUCachePolicy(cap) if request.param == "lru" else ARCCachePolicy(cap)
    for i in range(4):
        pol.insert(_k(i), _block(i))
    return pol


@pytest.fixture(params=["lru", "arc"])
def mixed_refcnt_policy(request):
    cap = 8
    pol = LRUCachePolicy(cap) if request.param == "lru" else ARCCachePolicy(cap)
    for i in range(4):
        pol.insert(_k(i), _block(i, ref_cnt={0: 0, 1: 0, 2: 1, 3: -1}[i]))
    return pol


def test_predicate_failure_no_mutation(filled_policy):
    blocks, meta = _save_state(filled_policy)
    result = filled_policy.evict_until(can_fit=lambda c: False, protected=set())
    assert result is None
    _assert_unchanged(blocks, meta, filled_policy)


def test_commits_exact_prefix(filled_policy):
    collected = []

    def can_fit(c):
        collected.append(len(c))
        return len(c) == 2

    result = filled_policy.evict_until(can_fit, protected=set())
    assert result is not None
    assert len(result) == 2
    assert collected == [1, 2]
    assert filled_policy.get(_k(0)) is None
    assert filled_policy.get(_k(2)) is not None


def test_skips_protected(filled_policy):
    protected = {_k(0), _k(2)}
    result = filled_policy.evict_until(
        can_fit=lambda c: len(c) >= 2, protected=protected
    )
    assert result is not None
    assert len(result) == 2
    assert all(key not in protected for key, _ in result)
    assert all(filled_policy.get(k) is not None for k in protected)


def test_not_enough_unprotected_returns_none(filled_policy):
    assert (
        filled_policy.evict_until(lambda c: True, protected={_k(i) for i in range(4)})
        is None
    )


def test_skips_nonzero_refcnt(mixed_refcnt_policy):
    result = mixed_refcnt_policy.evict_until(lambda c: len(c) >= 2, protected=set())
    assert result is not None
    assert len(result) == 2
    assert mixed_refcnt_policy.get(_k(2)) is not None
    assert mixed_refcnt_policy.get(_k(3)) is not None


def test_arc_ghost_semantics():
    policy = ARCCachePolicy(8)
    for i in range(4):
        policy.insert(_k(i), _block(i))
    policy.touch([_k(2), _k(3)], None)
    policy.target_t1_size = 0.0

    result = policy.evict_until(can_fit=lambda c: len(c) == 2, protected=set())
    assert result is not None
    assert len(result) == 2
    for key, _ in result:
        assert key in policy.b1 or key in policy.b2


def test_arc_t1_selected():
    policy = ARCCachePolicy(8)
    for i in range(4):
        policy.insert(_k(i), _block(i))
    policy.touch([_k(2)], None)
    policy.target_t1_size = 1.0
    result = policy.evict_until(lambda c: len(c) == 1, protected=set())
    assert result is not None
    assert result[0][0] == _k(0)
    assert result[0][0] in policy.b1


def test_arc_t2_selected():
    policy = ARCCachePolicy(8)
    for i in range(4):
        policy.insert(_k(i), _block(i))
    policy.touch([_k(2), _k(3)], None)
    policy.target_t1_size = 3.0
    result = policy.evict_until(lambda c: len(c) == 1, protected=set())
    assert result is not None
    assert result[0][0] == _k(3)
    assert result[0][0] in policy.b2


def test_evict_wrapper_backward_compat(filled_policy):
    result = filled_policy.evict(2, set())
    assert result is not None
    assert len(result) == 2
    assert filled_policy.get(_k(0)) is None
    result = filled_policy.evict(10, set())
    assert result is None
    protected = {_k(3)}
    result = filled_policy.evict(1, protected=protected)
    assert result is not None
    assert result[0][0] != _k(3)
    assert filled_policy.evict(0, set()) == []


def test_lru_order():
    policy = LRUCachePolicy(8)
    for i in (3, 1, 2, 0):
        policy.insert(_k(i), _block(i))
    policy.touch([_k(0)], None)
    result = policy.evict_until(can_fit=lambda c: len(c) == 3, protected=set())
    assert [key for key, _ in result] == [_k(3), _k(1), _k(2)]


def test_empty_policy():
    for pcls in (LRUCachePolicy, ARCCachePolicy):
        assert pcls(8).evict_until(lambda c: True, set()) is None


def test_can_fit_not_called_when_blocked():
    for pcls in (LRUCachePolicy, ARCCachePolicy):
        policy = pcls(4)
        policy.insert(_k(0), _block(0, ref_cnt=1))
        calls = []

        def can_fit(candidates, calls=calls):
            calls.append(candidates)
            return False

        result = policy.evict_until(can_fit, set())
        assert result is None
        assert calls == []


def test_legacy_evict_only_subclass_instantiable():
    """External-style subclass implementing only abstract ``evict`` (not
    ``evict_until``) must remain instantiable.
    The default ``evict_until`` raises ``NotImplementedError``."""

    class LegacyPolicy(CachePolicy):
        """Minimal policy that only overrides legacy ``evict``."""

        def __init__(self, cache_capacity: int):
            self.capacity = cache_capacity
            self._blocks: dict[OffloadKey, BlockStatus] = {}
            self._order: list[OffloadKey] = []

        def get(self, key: OffloadKey) -> BlockStatus | None:
            return self._blocks.get(key)

        def insert(self, key: OffloadKey, block: BlockStatus) -> None:
            self._blocks[key] = block
            self._order.append(key)

        def remove(self, key: OffloadKey) -> None:
            self._blocks.pop(key, None)
            self._order[:] = [k for k in self._order if k != key]

        def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
            pass

        def clear(self) -> None:
            self._blocks.clear()
            self._order.clear()

        def evict(
            self, n: int, protected: set[OffloadKey]
        ) -> list[tuple[OffloadKey, BlockStatus]] | None:
            result: list[tuple[OffloadKey, BlockStatus]] = []
            for key in list(self._order):
                if key in protected or key not in self._blocks:
                    continue
                if self._blocks[key].ref_cnt != 0:
                    continue
                result.append((key, self._blocks[key]))
                if len(result) >= n:
                    break
            for key, _ in result:
                del self._blocks[key]
                self._order.remove(key)
            return result or None

    policy = LegacyPolicy(4)
    policy.insert(_k(0), _block(0))
    policy.insert(_k(1), _block(1))

    # legacy evict works
    r = policy.evict(1, set())
    assert r is not None
    assert r[0][0] == _k(0)

    # evict_until raises NotImplementedError
    with pytest.raises(NotImplementedError, match="variable-size compact eviction"):
        policy.evict_until(lambda c: True, set())
