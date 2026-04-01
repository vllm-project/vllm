# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for SequentialPrefetcher and FrequencyPrefetcher.
"""
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.prefetcher import (
    FrequencyPrefetcher,
    SequentialPrefetcher,
)


def to_hash(i: int) -> BlockHash:
    return BlockHash(str(i).encode())


def to_hashes(ints: list[int]) -> list[BlockHash]:
    return [to_hash(i) for i in ints]


# --- SequentialPrefetcher tests ---


def test_sequential_basic_prediction():
    """Predict next blocks after sequential access."""
    prefetcher = SequentialPrefetcher(
        lookahead=2, max_pending=8, cooldown_seconds=0.0
    )

    # Build sequence: blocks 1, 2, 3, 4, 5
    prefetcher.record_access("req1", to_hashes([1, 2, 3, 4, 5]))

    # Access block 2, predict blocks 3, 4
    offloaded = set(to_hashes([3, 4, 5]))
    requests = prefetcher.predict(
        "req1",
        current_blocks=[to_hash(2)],
        offloaded_blocks=offloaded,
    )

    predicted_hashes = {r.block_hash for r in requests}
    assert to_hash(3) in predicted_hashes
    assert to_hash(4) in predicted_hashes


def test_sequential_only_prefetch_offloaded():
    """Only blocks on CPU (offloaded) are prefetched."""
    prefetcher = SequentialPrefetcher(
        lookahead=3, max_pending=8, cooldown_seconds=0.0
    )

    prefetcher.record_access("req1", to_hashes([1, 2, 3, 4, 5]))

    # Only block 4 is offloaded (3 and 5 are on GPU)
    offloaded = {to_hash(4)}
    requests = prefetcher.predict(
        "req1",
        current_blocks=[to_hash(2)],
        offloaded_blocks=offloaded,
    )

    assert len(requests) == 1
    assert requests[0].block_hash == to_hash(4)


def test_sequential_no_duplicate_pending():
    """Don't prefetch blocks that are already pending."""
    prefetcher = SequentialPrefetcher(
        lookahead=2, max_pending=8, cooldown_seconds=0.0
    )

    prefetcher.record_access("req1", to_hashes([1, 2, 3, 4]))
    offloaded = set(to_hashes([3, 4]))

    # First prediction
    requests1 = prefetcher.predict(
        "req1", current_blocks=[to_hash(2)], offloaded_blocks=offloaded
    )
    assert len(requests1) == 2

    # Second prediction for same position: already pending
    requests2 = prefetcher.predict(
        "req1", current_blocks=[to_hash(2)], offloaded_blocks=offloaded
    )
    assert len(requests2) == 0


def test_sequential_max_pending():
    """Respect max_pending limit."""
    prefetcher = SequentialPrefetcher(
        lookahead=10, max_pending=2, cooldown_seconds=0.0
    )

    prefetcher.record_access("req1", to_hashes(list(range(20))))
    offloaded = set(to_hashes(list(range(5, 15))))

    requests = prefetcher.predict(
        "req1", current_blocks=[to_hash(4)], offloaded_blocks=offloaded
    )
    assert len(requests) <= 2


def test_sequential_useful_prefetch_tracking():
    """Track when prefetched blocks are actually used."""
    prefetcher = SequentialPrefetcher(
        lookahead=2, max_pending=8, cooldown_seconds=0.0
    )

    prefetcher.record_access("req1", to_hashes([1, 2, 3, 4]))
    offloaded = set(to_hashes([3, 4]))

    prefetcher.predict(
        "req1", current_blocks=[to_hash(2)], offloaded_blocks=offloaded
    )

    assert prefetcher.stats.total_prefetches == 2
    assert prefetcher.stats.useful_prefetches == 0

    # Access prefetched block 3
    prefetcher.record_access("req1", to_hashes([3]))
    assert prefetcher.stats.useful_prefetches == 1


def test_sequential_cancel_prefetch():
    """Cancel tracks wasted prefetches."""
    prefetcher = SequentialPrefetcher(
        lookahead=2, max_pending=8, cooldown_seconds=0.0
    )

    prefetcher.record_access("req1", to_hashes([1, 2, 3, 4]))
    offloaded = set(to_hashes([3, 4]))

    requests = prefetcher.predict(
        "req1", current_blocks=[to_hash(2)], offloaded_blocks=offloaded
    )

    for req in requests:
        prefetcher.cancel_prefetch(req.block_hash)

    assert prefetcher.stats.wasted_prefetches == 2


def test_sequential_remove_request():
    """Clean up when request completes."""
    prefetcher = SequentialPrefetcher(lookahead=2, cooldown_seconds=0.0)

    prefetcher.record_access("req1", to_hashes([1, 2, 3]))
    assert "req1" in prefetcher._request_sequences

    prefetcher.remove_request("req1")
    assert "req1" not in prefetcher._request_sequences


def test_sequential_get_stats():
    """get_stats returns correct info."""
    prefetcher = SequentialPrefetcher(
        lookahead=2, max_pending=8, cooldown_seconds=0.0
    )

    prefetcher.record_access("req1", to_hashes([1, 2, 3]))
    offloaded = set(to_hashes([2, 3]))
    prefetcher.predict(
        "req1", current_blocks=[to_hash(1)], offloaded_blocks=offloaded
    )

    stats = prefetcher.get_stats()
    assert stats["total_prefetches"] == 2
    assert stats["tracked_requests"] == 1
    assert stats["pending_count"] == 2


def test_sequential_priority_ordering():
    """Closer blocks have higher priority."""
    prefetcher = SequentialPrefetcher(
        lookahead=3, max_pending=8, cooldown_seconds=0.0
    )

    prefetcher.record_access("req1", to_hashes([1, 2, 3, 4, 5]))
    offloaded = set(to_hashes([3, 4, 5]))

    requests = prefetcher.predict(
        "req1", current_blocks=[to_hash(2)], offloaded_blocks=offloaded
    )

    # First request should have highest priority
    assert requests[0].priority > requests[-1].priority


# --- FrequencyPrefetcher tests ---


def test_frequency_basic_prediction():
    """Predict high-frequency offloaded blocks."""
    prefetcher = FrequencyPrefetcher(
        top_k=2, min_frequency=3, max_pending=8
    )

    # Build frequency: block 1 accessed 5 times, block 2 accessed 4 times
    for _ in range(5):
        prefetcher.record_access(to_hash(1))
    for _ in range(4):
        prefetcher.record_access(to_hash(2))
    for _ in range(1):
        prefetcher.record_access(to_hash(3))

    offloaded = set(to_hashes([1, 2, 3]))
    requests = prefetcher.predict(offloaded)

    # Block 3 has freq=1, below min_frequency=3
    predicted_hashes = {r.block_hash for r in requests}
    assert to_hash(1) in predicted_hashes
    assert to_hash(2) in predicted_hashes
    assert to_hash(3) not in predicted_hashes


def test_frequency_min_threshold():
    """Blocks below min_frequency are not prefetched."""
    prefetcher = FrequencyPrefetcher(
        top_k=4, min_frequency=5, max_pending=8
    )

    for _ in range(4):
        prefetcher.record_access(to_hash(1))  # freq=4, below threshold

    offloaded = {to_hash(1)}
    requests = prefetcher.predict(offloaded)
    assert len(requests) == 0


def test_frequency_useful_tracking():
    """Track useful prefetches."""
    prefetcher = FrequencyPrefetcher(
        top_k=2, min_frequency=2, max_pending=8
    )

    for _ in range(3):
        prefetcher.record_access(to_hash(1))

    offloaded = {to_hash(1)}
    prefetcher.predict(offloaded)

    # Access the prefetched block
    prefetcher.record_access(to_hash(1))
    assert prefetcher.stats.useful_prefetches == 1


def test_frequency_top_k_limit():
    """Only top_k blocks are returned."""
    prefetcher = FrequencyPrefetcher(
        top_k=2, min_frequency=1, max_pending=8
    )

    for i in range(5):
        for _ in range(5 - i):
            prefetcher.record_access(to_hash(i))

    offloaded = set(to_hashes(list(range(5))))
    requests = prefetcher.predict(offloaded)
    assert len(requests) == 2

    # Highest frequency blocks should be returned
    predicted_hashes = {r.block_hash for r in requests}
    assert to_hash(0) in predicted_hashes  # freq=5
    assert to_hash(1) in predicted_hashes  # freq=4
