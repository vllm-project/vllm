# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for HybridOffloadingManager.
"""
import time

import numpy as np
import pytest

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.hybrid_manager import HybridOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec


def to_hashes(int_hashes: list[int]) -> list[BlockHash]:
    return [BlockHash(str(i).encode()) for i in int_hashes]


def test_basic_store_and_lookup():
    """Store blocks and verify lookup works."""
    block_size = 256
    backend = CPUBackend(block_size=block_size, num_blocks=4)
    manager = HybridOffloadingManager(backend, enable_events=True)

    output = manager.prepare_store(to_hashes([1, 2]))
    assert output is not None
    assert output.block_hashes_to_store == to_hashes([1, 2])

    # Not ready
    assert manager.lookup(to_hashes([1, 2])) == 0

    manager.complete_store(to_hashes([1, 2]))

    assert manager.lookup(to_hashes([1])) == 1
    assert manager.lookup(to_hashes([1, 2])) == 2
    assert manager.lookup(to_hashes([1, 2, 3])) == 2


def test_weight_validation():
    """Weights must sum to 1.0."""
    backend = CPUBackend(block_size=256, num_blocks=4)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        HybridOffloadingManager(backend, alpha=0.5, beta=0.5, gamma=0.5)


def test_attention_dominant_eviction():
    """With high alpha, attention scores dominate eviction decisions."""
    backend = CPUBackend(block_size=256, num_blocks=4)
    manager = HybridOffloadingManager(
        backend,
        alpha=0.9,
        beta=0.05,
        gamma=0.05,
        score_decay=1.0,
    )

    manager.prepare_store(to_hashes([1, 2, 3, 4]))
    manager.complete_store(to_hashes([1, 2, 3, 4]))

    # Give block 3 highest attention, block 1 lowest
    manager.update_attention_scores({
        to_hashes([1])[0]: 0.1,
        to_hashes([2])[0]: 5.0,
        to_hashes([3])[0]: 10.0,
        to_hashes([4])[0]: 0.2,
    })

    # Store block 5 -> evict block 1 (lowest attention)
    output = manager.prepare_store(to_hashes([5]))
    assert output is not None
    assert output.block_hashes_evicted == to_hashes([1])


def test_recency_dominant_eviction():
    """With high beta, recency dominates eviction decisions."""
    backend = CPUBackend(block_size=256, num_blocks=4)
    manager = HybridOffloadingManager(
        backend,
        alpha=0.05,
        beta=0.9,
        gamma=0.05,
        score_decay=1.0,
    )

    manager.prepare_store(to_hashes([1, 2, 3, 4]))
    manager.complete_store(to_hashes([1, 2, 3, 4]))

    # Touch blocks 2, 3, 4 (making 1 the least recent)
    manager.touch(to_hashes([2]))
    time.sleep(0.01)
    manager.touch(to_hashes([3]))
    time.sleep(0.01)
    manager.touch(to_hashes([4]))

    # Give block 1 highest attention (should be overridden by recency)
    manager.update_attention_scores({
        to_hashes([1])[0]: 100.0,
    })

    # Store block 5 -> evict block 1 (least recent, despite high attention)
    output = manager.prepare_store(to_hashes([5]))
    assert output is not None
    assert output.block_hashes_evicted == to_hashes([1])


def test_frequency_dominant_eviction():
    """With high gamma, frequency dominates eviction decisions."""
    backend = CPUBackend(block_size=256, num_blocks=4)
    manager = HybridOffloadingManager(
        backend,
        alpha=0.05,
        beta=0.05,
        gamma=0.9,
        score_decay=1.0,
    )

    manager.prepare_store(to_hashes([1, 2, 3, 4]))
    manager.complete_store(to_hashes([1, 2, 3, 4]))

    # Touch blocks 2, 3, 4 many times (block 1 stays at 0 accesses)
    for _ in range(10):
        manager.touch(to_hashes([2, 3, 4]))

    # Store block 5 -> evict block 1 (lowest frequency)
    output = manager.prepare_store(to_hashes([5]))
    assert output is not None
    assert output.block_hashes_evicted == to_hashes([1])


def test_eviction_respects_ref_cnt():
    """Blocks being loaded cannot be evicted."""
    backend = CPUBackend(block_size=256, num_blocks=4)
    manager = HybridOffloadingManager(backend)

    manager.prepare_store(to_hashes([1, 2, 3, 4]))
    manager.complete_store(to_hashes([1, 2, 3, 4]))

    # Load blocks [1, 2]
    manager.prepare_load(to_hashes([1, 2]))

    # Try to store 3 more -> fail (only 2 evictable)
    assert manager.prepare_store(to_hashes([5, 6, 7])) is None

    manager.complete_load(to_hashes([1, 2]))

    output = manager.prepare_store(to_hashes([5, 6, 7]))
    assert output is not None
    assert len(output.block_hashes_evicted) == 3


def test_failed_store_cleanup():
    """Failed store removes blocks correctly."""
    backend = CPUBackend(block_size=256, num_blocks=4)
    manager = HybridOffloadingManager(backend)

    manager.prepare_store(to_hashes([1, 2]))
    manager.complete_store(to_hashes([1, 2]))

    manager.prepare_store(to_hashes([3]))
    manager.complete_store(to_hashes([3]), success=False)

    assert manager.lookup(to_hashes([3])) == 0
    assert manager.lookup(to_hashes([1, 2])) == 2


def test_events_tracking():
    """Events are correctly emitted."""
    backend = CPUBackend(block_size=256, num_blocks=2)
    manager = HybridOffloadingManager(backend, enable_events=True)

    manager.prepare_store(to_hashes([1, 2]))
    manager.complete_store(to_hashes([1, 2]))

    events = list(manager.take_events())
    assert len(events) == 1
    assert not events[0].removed

    # Trigger eviction
    manager.prepare_store(to_hashes([3]))
    manager.complete_store(to_hashes([3]))

    events = list(manager.take_events())
    evictions = [e for e in events if e.removed]
    stores = [e for e in events if not e.removed]
    assert len(evictions) == 1
    assert len(stores) == 1


def test_prepare_load_returns_correct_spec():
    """prepare_load returns correct block IDs."""
    backend = CPUBackend(block_size=256, num_blocks=4)
    manager = HybridOffloadingManager(backend)

    manager.prepare_store(to_hashes([1, 2]))
    manager.complete_store(to_hashes([1, 2]))

    spec = manager.prepare_load(to_hashes([1, 2]))
    assert isinstance(spec, CPULoadStoreSpec)
    expected = np.array([0, 1], dtype=np.int64)
    assert np.array_equal(spec.block_ids, expected)


def test_get_stats():
    """get_stats returns correct statistics."""
    backend = CPUBackend(block_size=256, num_blocks=4)
    manager = HybridOffloadingManager(
        backend, alpha=0.5, beta=0.3, gamma=0.2
    )

    manager.prepare_store(to_hashes([1, 2]))
    manager.complete_store(to_hashes([1, 2]))

    stats = manager.get_stats()
    assert stats["total_blocks"] == 2
    assert stats["ready_blocks"] == 2
    assert stats["weights"]["alpha"] == 0.5
    assert stats["weights"]["beta"] == 0.3
    assert stats["weights"]["gamma"] == 0.2


def test_duplicate_store_skipped():
    """Already-stored blocks are not re-stored."""
    backend = CPUBackend(block_size=256, num_blocks=4)
    manager = HybridOffloadingManager(backend)

    manager.prepare_store(to_hashes([1, 2]))
    manager.complete_store(to_hashes([1, 2]))

    output = manager.prepare_store(to_hashes([1, 2, 3]))
    assert output is not None
    assert output.block_hashes_to_store == to_hashes([3])


def test_empty_store():
    """Storing all-duplicate blocks returns empty output."""
    backend = CPUBackend(block_size=256, num_blocks=4)
    manager = HybridOffloadingManager(backend)

    manager.prepare_store(to_hashes([1, 2]))
    manager.complete_store(to_hashes([1, 2]))

    output = manager.prepare_store(to_hashes([1, 2]))
    assert output is not None
    assert output.block_hashes_to_store == []
    assert output.block_hashes_evicted == []
