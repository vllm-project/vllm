# SPDX-License-Identifier: Apache-2.0
"""Unit tests for FaultInjectL2Adapter.

The adapter is a decorator that drops a deterministic key subset at load to
simulate partial L2 retrieve failures. These tests wrap a real MockL2Adapter
and assert the load-result bitmap has exactly the expected bits cleared while
the lookup bitmap is left intact, using only public interface methods.
"""

# Standard
import select
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters.fault_inject_l2_adapter import (
    FaultInjectL2Adapter,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

N_KEYS = 8


def _object_key(chunk_id: int) -> ObjectKey:
    """Build an ObjectKey from a chunk id (test fixture)."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def _memory_obj(size: int = 256, fill_value: float = 1.0) -> TensorMemoryObj:
    """Build a filled TensorMemoryObj for use as a store/load buffer."""
    raw = torch.empty(size, dtype=torch.float32)
    raw.fill_(fill_value)
    meta = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw, meta, parent_allocator=None)


def _wait_fd(event_fd: int, timeout: float = 5.0) -> bool:
    """Wait up to ``timeout`` seconds for ``event_fd`` to signal, draining it.

    Returns True if the fd became readable before the timeout, False otherwise.
    """
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    if poll.poll(timeout * 1000):
        try:
            consume_fd(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


def _make_adapter(rate: float = 0.0, seed: int = 0, gap_indices=(), gap_tail_ratios=()):
    """Build a FaultInjectL2Adapter wrapping a fresh MockL2Adapter.

    Returns the ``(wrapper, inner)`` pair.
    """
    inner = MockL2Adapter(MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0))
    wrapper = FaultInjectL2Adapter(
        inner,
        rate=rate,
        seed=seed,
        gap_indices=tuple(gap_indices),
        gap_tail_ratios=tuple(gap_tail_ratios),
    )
    return wrapper, inner


def _store_all(adapter, keys):
    """Store one memory object per key and drain the completion event."""
    fd = adapter.get_store_event_fd()
    adapter.submit_store_task(keys, [_memory_obj() for _ in keys])
    assert _wait_fd(fd)
    adapter.pop_completed_store_tasks()


def _lookup_bitmap(adapter, keys):
    """Run a lookup-and-lock for ``keys`` and return its result bitmap."""
    fd = adapter.get_lookup_and_lock_event_fd()
    tid = adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
    assert _wait_fd(fd)
    # query_*_result is non-idempotent (returns non-None once); poll briefly.
    for _ in range(50):
        bm = adapter.query_lookup_and_lock_result(tid)
        if bm is not None:
            return bm
        time.sleep(0.01)
    raise AssertionError("lookup result never ready")


def _load_bitmap(adapter, keys):
    """Run a load for ``keys`` and return its result bitmap."""
    fd = adapter.get_load_event_fd()
    tid = adapter.submit_load_task(keys, [_memory_obj() for _ in keys])
    assert _wait_fd(fd)
    for _ in range(50):
        bm = adapter.query_load_result(tid)
        if bm is not None:
            return bm
        time.sleep(0.01)
    raise AssertionError("load result never ready")


# =============================================================================
# Pass-through (rate=0): no faults.
# =============================================================================


def test_rate_zero_is_passthrough():
    """rate=0 passes every key through unchanged (no drops)."""
    adapter, inner = _make_adapter(rate=0.0)
    try:
        keys = [_object_key(i) for i in range(N_KEYS)]
        _store_all(adapter, keys)
        lookup = _lookup_bitmap(adapter, keys)
        load = _load_bitmap(adapter, keys)
        assert lookup.popcount() == N_KEYS
        assert load.popcount() == N_KEYS
    finally:
        adapter.close()


# =============================================================================
# gap_indices: exact, deterministic drops.
# =============================================================================


def test_load_gap_cleared_lookup_intact():
    """A gapped load clears exactly those load bits, leaving lookup intact."""
    gap = {1, 4, 6}
    adapter, inner = _make_adapter(gap_indices=gap)
    try:
        keys = [_object_key(i) for i in range(N_KEYS)]
        _store_all(adapter, keys)
        # Lookup is never faulted (lookup says present).
        lookup = _lookup_bitmap(adapter, keys)
        assert lookup.popcount() == N_KEYS
        load = _load_bitmap(adapter, keys)
        for i in range(N_KEYS):
            assert load.test(i) == (i not in gap)
    finally:
        adapter.close()


# =============================================================================
# gap_tail_ratios: distance-from-tail / load-length; workload-agnostic.
# =============================================================================


def test_gap_tail_ratio_position_and_scaling():
    """A tail-ratio drops the chunk at round((1-ratio)*(n-1)); the absolute
    index is computed from the load batch, so it scales with length -- no
    content or position is known in advance."""
    keys = [_object_key(i) for i in range(N_KEYS)]
    adapter, inner = _make_adapter(gap_tail_ratios=(0.5,))
    try:
        _store_all(adapter, keys)
        _lookup_bitmap(adapter, keys)
        mid = round(0.5 * (N_KEYS - 1))
        load = _load_bitmap(adapter, keys)
        assert not load.test(mid)
        assert all(load.test(i) for i in range(N_KEYS) if i != mid)
        # Same ratio on a SHORTER load drops a proportionally-different index
        # (relative to that batch's tail) -- self-scaling, workload-agnostic.
        short = keys[:4]
        load2 = _load_bitmap(adapter, short)
        assert not load2.test(round(0.5 * (len(short) - 1)))
    finally:
        adapter.close()


def test_gap_tail_ratio_endpoints():
    """ratio=0.0 drops the last chunk; ratio=1.0 drops the first."""
    keys = [_object_key(i) for i in range(N_KEYS)]
    adapter, inner = _make_adapter(gap_tail_ratios=(0.0, 1.0))
    try:
        _store_all(adapter, keys)
        _lookup_bitmap(adapter, keys)
        load = _load_bitmap(adapter, keys)
        assert not load.test(0) and not load.test(N_KEYS - 1)
        assert all(load.test(i) for i in range(1, N_KEYS - 1))
    finally:
        adapter.close()


# =============================================================================
# rate-based drops: deterministic across instances with the same seed.
# =============================================================================


def test_rate_drop_is_deterministic_across_instances():
    """Same seed drops the same keys; a different seed (very likely) differs."""
    keys = [_object_key(i) for i in range(64)]

    def dropped_positions(seed):
        adapter, inner = _make_adapter(rate=0.3, seed=seed)
        try:
            _store_all(adapter, keys)
            _lookup_bitmap(adapter, keys)
            load = _load_bitmap(adapter, keys)
            return {i for i in range(len(keys)) if not load.test(i)}
        finally:
            adapter.close()

    a = dropped_positions(seed=42)
    b = dropped_positions(seed=42)
    c = dropped_positions(seed=1234)
    assert a, "rate=0.3 should drop at least one of 64 keys"
    assert a == b, "same seed must drop the same keys"
    # A different seed should (with overwhelming probability) differ.
    assert a != c


def test_rate_drop_within_tolerance():
    """Rate-based bucketing drops roughly ``rate * N`` keys (wide tolerance)."""
    keys = [_object_key(i) for i in range(200)]
    adapter, inner = _make_adapter(rate=0.25, seed=7)
    try:
        _store_all(adapter, keys)
        _lookup_bitmap(adapter, keys)
        load = _load_bitmap(adapter, keys)
        dropped = sum(1 for i in range(len(keys)) if not load.test(i))
        # Deterministic hash bucketing -> roughly rate * N, allow a wide band.
        assert 25 <= dropped <= 75
    finally:
        adapter.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
