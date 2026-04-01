# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for AttentionWeightedOffloadingManager.
"""
import numpy as np

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.attention_manager import (
    AttentionWeightedOffloadingManager,
)
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec


def to_hashes(int_hashes: list[int]) -> list[BlockHash]:
    return [BlockHash(str(i).encode()) for i in int_hashes]


def test_basic_store_and_lookup():
    """Store blocks and verify lookup works."""
    block_size = 256
    backend = CPUBackend(block_size=block_size, num_blocks=4)
    manager = AttentionWeightedOffloadingManager(backend, enable_events=True)

    # Store blocks [1, 2]
    output = manager.prepare_store(to_hashes([1, 2]))
    assert output is not None
    assert output.block_hashes_to_store == to_hashes([1, 2])
    assert output.block_hashes_evicted == []

    # Not ready yet
    assert manager.lookup(to_hashes([1, 2])) == 0

    # Complete store
    manager.complete_store(to_hashes([1, 2]))

    # Now ready
    assert manager.lookup(to_hashes([1])) == 1
    assert manager.lookup(to_hashes([1, 2])) == 2
    assert manager.lookup(to_hashes([1, 2, 3])) == 2


def test_eviction_by_attention_score():
    """Blocks with lower attention scores are evicted first."""
    block_size = 256
    backend = CPUBackend(block_size=block_size, num_blocks=4)
    manager = AttentionWeightedOffloadingManager(
        backend, enable_events=True, score_decay=1.0  # no decay for test
    )

    # Fill cache with blocks [1, 2, 3, 4]
    manager.prepare_store(to_hashes([1, 2, 3, 4]))
    manager.complete_store(to_hashes([1, 2, 3, 4]))

    # Give block 3 a high attention score, block 1 the lowest
    manager.update_attention_scores({
        to_hashes([1])[0]: 0.1,
        to_hashes([2])[0]: 0.5,
        to_hashes([3])[0]: 10.0,
        to_hashes([4])[0]: 0.3,
    })

    # Store block 5 -> should evict block 1 (lowest score: 0.1)
    output = manager.prepare_store(to_hashes([5]))
    assert output is not None
    assert output.block_hashes_evicted == to_hashes([1])
    manager.complete_store(to_hashes([5]))

    # Block 3 (highest score) should still be present
    assert manager.lookup(to_hashes([3])) == 1
    # Block 1 should be gone
    assert manager.lookup(to_hashes([1])) == 0


def test_eviction_respects_ref_cnt():
    """Blocks being loaded (ref_cnt > 0) cannot be evicted."""
    block_size = 256
    backend = CPUBackend(block_size=block_size, num_blocks=4)
    manager = AttentionWeightedOffloadingManager(backend)

    # Fill cache
    manager.prepare_store(to_hashes([1, 2, 3, 4]))
    manager.complete_store(to_hashes([1, 2, 3, 4]))

    # Load blocks [1, 2] (increases ref_cnt)
    manager.prepare_load(to_hashes([1, 2]))

    # Try to store 3 more blocks - should fail (only 2 evictable)
    assert manager.prepare_store(to_hashes([5, 6, 7])) is None

    # Complete load
    manager.complete_load(to_hashes([1, 2]))

    # Now it should succeed
    output = manager.prepare_store(to_hashes([5, 6, 7]))
    assert output is not None
    assert len(output.block_hashes_evicted) == 3


def test_touch_updates_access_count():
    """Touch increments access count and updates recency."""
    block_size = 256
    backend = CPUBackend(block_size=block_size, num_blocks=4)
    manager = AttentionWeightedOffloadingManager(
        backend, score_decay=1.0
    )

    # Store and complete blocks
    manager.prepare_store(to_hashes([1, 2, 3, 4]))
    manager.complete_store(to_hashes([1, 2, 3, 4]))

    # Touch block 1 multiple times
    manager.touch(to_hashes([1]))
    manager.touch(to_hashes([1]))
    manager.touch(to_hashes([1]))

    meta_1 = manager.blocks[to_hashes([1])[0]]
    meta_2 = manager.blocks[to_hashes([2])[0]]
    assert meta_1.access_count == 3
    assert meta_2.access_count == 0


def test_score_decay():
    """Exponential decay reduces scores over eviction rounds."""
    block_size = 256
    backend = CPUBackend(block_size=block_size, num_blocks=4)
    manager = AttentionWeightedOffloadingManager(
        backend, score_decay=0.5
    )

    manager.prepare_store(to_hashes([1, 2]))
    manager.complete_store(to_hashes([1, 2]))

    # Set score for block 1
    manager.update_attention_scores({to_hashes([1])[0]: 10.0})
    assert manager.blocks[to_hashes([1])[0]].cumulative_attention_score == 10.0

    # Apply decay
    manager.apply_score_decay()
    assert manager.blocks[to_hashes([1])[0]].cumulative_attention_score == 5.0

    # Apply decay again
    manager.apply_score_decay()
    assert manager.blocks[to_hashes([1])[0]].cumulative_attention_score == 2.5


def test_failed_store_cleanup():
    """Failed store removes blocks that weren't ready."""
    block_size = 256
    backend = CPUBackend(block_size=block_size, num_blocks=4)
    manager = AttentionWeightedOffloadingManager(backend, enable_events=True)

    manager.prepare_store(to_hashes([1, 2]))
    manager.complete_store(to_hashes([1, 2]))

    # Prepare store for block 3
    manager.prepare_store(to_hashes([3]))
    # Fail the store
    manager.complete_store(to_hashes([3]), success=False)

    # Block 3 should not be in cache
    assert manager.lookup(to_hashes([3])) == 0
    # Block 1 and 2 should still be present
    assert manager.lookup(to_hashes([1, 2])) == 2


def test_events_tracking():
    """Events are emitted for stores and evictions."""
    block_size = 256
    backend = CPUBackend(block_size=block_size, num_blocks=2)
    manager = AttentionWeightedOffloadingManager(backend, enable_events=True)

    # Store [1, 2]
    manager.prepare_store(to_hashes([1, 2]))
    manager.complete_store(to_hashes([1, 2]))

    events = list(manager.take_events())
    assert len(events) == 1  # one store event
    assert not events[0].removed
    assert set(events[0].block_hashes) == set(to_hashes([1, 2]))

    # Store [3] -> evicts lowest score block
    manager.prepare_store(to_hashes([3]))
    manager.complete_store(to_hashes([3]))

    events = list(manager.take_events())
    evictions = [e for e in events if e.removed]
    stores = [e for e in events if not e.removed]
    assert len(evictions) == 1
    assert len(stores) == 1


def test_prepare_load_returns_correct_spec():
    """prepare_load returns a LoadStoreSpec with correct block IDs."""
    block_size = 256
    backend = CPUBackend(block_size=block_size, num_blocks=4)
    manager = AttentionWeightedOffloadingManager(backend)

    manager.prepare_store(to_hashes([1, 2]))
    manager.complete_store(to_hashes([1, 2]))

    spec = manager.prepare_load(to_hashes([1, 2]))
    assert isinstance(spec, CPULoadStoreSpec)
    expected = np.array([0, 1], dtype=np.int64)
    assert np.array_equal(spec.block_ids, expected)


def test_get_stats():
    """get_stats returns correct statistics."""
    block_size = 256
    backend = CPUBackend(block_size=block_size, num_blocks=4)
    manager = AttentionWeightedOffloadingManager(backend)

    manager.prepare_store(to_hashes([1, 2]))
    manager.complete_store(to_hashes([1, 2]))

    manager.update_attention_scores({
        to_hashes([1])[0]: 5.0,
        to_hashes([2])[0]: 3.0,
    })

    stats = manager.get_stats()
    assert stats["total_blocks"] == 2
    assert stats["ready_blocks"] == 2
    assert stats["avg_attention_score"] == 4.0
    assert stats["free_backend_blocks"] == 2


def test_duplicate_store_skipped():
    """Storing already-stored blocks is a no-op."""
    block_size = 256
    backend = CPUBackend(block_size=block_size, num_blocks=4)
    manager = AttentionWeightedOffloadingManager(backend)

    manager.prepare_store(to_hashes([1, 2]))
    manager.complete_store(to_hashes([1, 2]))

    # Store again with overlap
    output = manager.prepare_store(to_hashes([1, 2, 3]))
    assert output is not None
    # Only block 3 should be newly stored
    assert output.block_hashes_to_store == to_hashes([3])
