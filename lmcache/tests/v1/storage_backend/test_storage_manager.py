# SPDX-License-Identifier: Apache-2.0
"""
Test cases for StorageManager.

This module tests the critical logic in prefetch_all_done_callback that handles:
1. Calculating the actual number of retrieved chunks based on batched_get_non_blocking
   results (not batched_async_contains results)
2. Handling chunk eviction between contains check and actual retrieval
3. Ensuring prefix-based continuity: if a tier retrieves fewer chunks than expected,
   all subsequent tiers are ignored
4. Properly cleaning up (ref_count_down) memory objects that won't be used due to
   discontinuity

Key scenarios tested:
- All chunks retrieved successfully from all tiers
- Middle tier partial retrieval (subsequent tiers ignored)
- First tier partial retrieval (all subsequent tiers ignored)
- Last chunk not being full size
- Single tier partial retrieval
"""

# Standard
import asyncio

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventType
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.storage_manager import StorageManager


class MockMemoryObj:
    """Mock MemoryObj for testing."""

    def __init__(self, obj_id: int):
        self.obj_id = obj_id
        self.ref_count = 1
        self.ref_count_down_called = False

    def ref_count_down(self):
        self.ref_count -= 1
        self.ref_count_down_called = True

    def __repr__(self):
        return f"MockMemoryObj(id={self.obj_id}, ref_count={self.ref_count})"


class MockAsyncLookupServer:
    """Mock async lookup server for testing."""

    def __init__(self):
        self.responses = []

    def send_response_to_scheduler(self, lookup_id: str, retrieved_length: int):
        self.responses.append((lookup_id, retrieved_length))


@pytest.fixture
def event_manager():
    """Create an EventManager for testing."""
    return EventManager()


@pytest.fixture
def storage_manager_config():
    """Create a test configuration for StorageManager."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=False,
        lmcache_instance_id="test_instance",
    )
    return config


@pytest.fixture
def storage_manager_metadata():
    """Create test metadata for StorageManager."""
    metadata = LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(28, 2, 256, 8, 128),
        role="scheduler",
    )
    return metadata


@pytest.fixture
def storage_manager(storage_manager_config, storage_manager_metadata, event_manager):
    """Create a StorageManager for testing."""
    manager = StorageManager(
        config=storage_manager_config,
        metadata=storage_manager_metadata,
        event_manager=event_manager,
    )
    # Mock the async lookup server
    manager.async_lookup_server = MockAsyncLookupServer()
    yield manager
    manager.close()


class TestStorageManagerPrefetchCallback:
    """Test cases for StorageManager prefetch_all_done_callback."""

    def test_all_chunks_retrieved_successfully(self, storage_manager):
        """Test Case 1: All chunks retrieved successfully from all tiers."""
        # Setup: 5 chunks total (1280 tokens), distributed across 2 tiers
        # Tier 0: 3 chunks, Tier 1: 2 chunks
        cum_chunk_lengths_total = [0, 256, 512, 768, 1024, 1280]
        tier_expected_chunks = [3, 2]

        # Create mock memory objects for all chunks. At runtime,
        # gather_with_keys() in async_lookup_and_prefetch produces
        # (key, mem_obj) tuples per chunk, so res mirrors that shape.
        tier0_objs = [MockMemoryObj(i) for i in range(3)]
        tier1_objs = [MockMemoryObj(i + 3) for i in range(2)]
        res = [
            [(f"k{i}", obj) for i, obj in enumerate(tier0_objs)],
            [(f"k{i + 3}", obj) for i, obj in enumerate(tier1_objs)],
        ]

        # Create a mock future that returns the result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = loop.create_future()
        future.set_result(res)

        # Register the event before calling callback
        storage_manager.event_manager.add_event(
            EventType.LOADING, "test_lookup_1", future
        )

        # Call the callback
        storage_manager.prefetch_all_done_callback(
            future, "test_lookup_1", cum_chunk_lengths_total, tier_expected_chunks
        )
        loop.close()

        # Verify: All 5 chunks should be counted, total 1280 tokens
        assert len(storage_manager.async_lookup_server.responses) == 1
        lookup_id, retrieved_length = storage_manager.async_lookup_server.responses[0]
        assert lookup_id == "test_lookup_1"
        assert retrieved_length == 1280

        # Verify: No memory objects should have ref_count_down called
        for obj in tier0_objs + tier1_objs:
            assert not obj.ref_count_down_called

    def test_middle_tier_partial_retrieval(self, storage_manager):
        """Test Case 2: Middle tier only got partial chunks, subsequent tier ignored."""
        # Setup: 7 chunks total (1792 tokens), distributed across 3 tiers
        # Tier 0: 3 chunks, Tier 1: 2 chunks, Tier 2: 2 chunks
        cum_chunk_lengths_total = [0, 256, 512, 768, 1024, 1280, 1536, 1792]
        tier_expected_chunks = [3, 2, 2]

        # Tier 0 got all 3, Tier 1 only got 1 (eviction), Tier 2 got all 2
        tier0_objs = [MockMemoryObj(i) for i in range(3)]
        tier1_objs = [MockMemoryObj(i + 3) for i in range(1)]  # Only 1 instead of 2
        tier2_objs = [MockMemoryObj(i + 5) for i in range(2)]  # Got all 2
        res = [
            [(f"k{i}", obj) for i, obj in enumerate(tier0_objs)],
            [(f"k{i + 3}", obj) for i, obj in enumerate(tier1_objs)],
            [(f"k{i + 5}", obj) for i, obj in enumerate(tier2_objs)],
        ]

        # Create a mock future that returns the result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = loop.create_future()
        future.set_result(res)

        # Register the event before calling callback
        storage_manager.event_manager.add_event(
            EventType.LOADING, "test_lookup_2", future
        )

        # Call the callback
        storage_manager.prefetch_all_done_callback(
            future, "test_lookup_2", cum_chunk_lengths_total, tier_expected_chunks
        )
        loop.close()

        # Verify: Only 4 chunks counted (3 from tier0 + 1 from tier1)
        # Total: 1024 tokens
        assert len(storage_manager.async_lookup_server.responses) == 1
        lookup_id, retrieved_length = storage_manager.async_lookup_server.responses[0]
        assert lookup_id == "test_lookup_2"
        assert retrieved_length == 1024

        # Verify: Tier 0 and Tier 1 objects should NOT have ref_count_down called
        for obj in tier0_objs + tier1_objs:
            assert not obj.ref_count_down_called

        # Verify: All Tier 2 objects should have ref_count_down called
        for obj in tier2_objs:
            assert obj.ref_count_down_called

    def test_first_tier_partial_retrieval(self, storage_manager):
        """
        Test Case 3: First tier only got partial chunks,
        all subsequent tiers ignored.
        """
        # Setup: 7 chunks total (1792 tokens), distributed across 3 tiers
        # Tier 0: 3 chunks, Tier 1: 2 chunks, Tier 2: 2 chunks
        cum_chunk_lengths_total = [0, 256, 512, 768, 1024, 1280, 1536, 1792]
        tier_expected_chunks = [3, 2, 2]

        # Tier 0 only got 2 (eviction), Tier 1 got all 2, Tier 2 got all 2
        tier0_objs = [MockMemoryObj(i) for i in range(2)]  # Only 2 instead of 3
        tier1_objs = [MockMemoryObj(i + 3) for i in range(2)]  # Got all 2
        tier2_objs = [MockMemoryObj(i + 5) for i in range(2)]  # Got all 2
        res = [
            [(f"k{i}", obj) for i, obj in enumerate(tier0_objs)],
            [(f"k{i + 3}", obj) for i, obj in enumerate(tier1_objs)],
            [(f"k{i + 5}", obj) for i, obj in enumerate(tier2_objs)],
        ]

        # Create a mock future that returns the result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = loop.create_future()
        future.set_result(res)

        # Register the event before calling callback
        storage_manager.event_manager.add_event(
            EventType.LOADING, "test_lookup_3", future
        )

        # Call the callback
        storage_manager.prefetch_all_done_callback(
            future, "test_lookup_3", cum_chunk_lengths_total, tier_expected_chunks
        )
        loop.close()

        # Verify: Only 2 chunks counted (2 from tier0)
        # Total: 512 tokens
        assert len(storage_manager.async_lookup_server.responses) == 1
        lookup_id, retrieved_length = storage_manager.async_lookup_server.responses[0]
        assert lookup_id == "test_lookup_3"
        assert retrieved_length == 512

        # Verify: Tier 0 objects should NOT have ref_count_down called
        for obj in tier0_objs:
            assert not obj.ref_count_down_called

        # Verify: All Tier 1 and Tier 2 objects should have ref_count_down called
        for obj in tier1_objs + tier2_objs:
            assert obj.ref_count_down_called

    def test_last_chunk_not_full(self, storage_manager):
        """Test with last chunk not being full size."""
        # Setup: 3 chunks with last chunk only 128 tokens (640 tokens total)
        # Tier 0: 2 chunks, Tier 1: 1 chunk
        cum_chunk_lengths_total = [0, 256, 512, 640]  # Last chunk is 128 tokens
        tier_expected_chunks = [2, 1]

        # All chunks retrieved successfully
        tier0_objs = [MockMemoryObj(i) for i in range(2)]
        tier1_objs = [MockMemoryObj(i + 2) for i in range(1)]
        res = [
            [(f"k{i}", obj) for i, obj in enumerate(tier0_objs)],
            [(f"k{i + 2}", obj) for i, obj in enumerate(tier1_objs)],
        ]

        # Create a mock future that returns the result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = loop.create_future()
        future.set_result(res)

        # Register the event before calling callback
        storage_manager.event_manager.add_event(
            EventType.LOADING, "test_lookup_4", future
        )

        # Call the callback
        storage_manager.prefetch_all_done_callback(
            future, "test_lookup_4", cum_chunk_lengths_total, tier_expected_chunks
        )
        loop.close()

        # Verify: All 3 chunks counted, total 640 tokens
        assert len(storage_manager.async_lookup_server.responses) == 1
        lookup_id, retrieved_length = storage_manager.async_lookup_server.responses[0]
        assert lookup_id == "test_lookup_4"
        assert retrieved_length == 640

        # Verify: No memory objects should have ref_count_down called
        for obj in tier0_objs + tier1_objs:
            assert not obj.ref_count_down_called

    def test_single_tier_partial_retrieval(self, storage_manager):
        """Test with single tier that only got partial chunks."""
        # Setup: 5 chunks total (1280 tokens), single tier
        cum_chunk_lengths_total = [0, 256, 512, 768, 1024, 1280]
        tier_expected_chunks = [5]

        # Only got 3 chunks instead of 5
        tier0_objs = [MockMemoryObj(i) for i in range(3)]
        res = [[(f"k{i}", obj) for i, obj in enumerate(tier0_objs)]]

        # Create a mock future that returns the result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = loop.create_future()
        future.set_result(res)

        # Register the event before calling callback
        storage_manager.event_manager.add_event(
            EventType.LOADING, "test_lookup_5", future
        )

        # Call the callback
        storage_manager.prefetch_all_done_callback(
            future, "test_lookup_5", cum_chunk_lengths_total, tier_expected_chunks
        )
        loop.close()

        # Verify: Only 3 chunks counted, total 768 tokens
        assert len(storage_manager.async_lookup_server.responses) == 1
        lookup_id, retrieved_length = storage_manager.async_lookup_server.responses[0]
        assert lookup_id == "test_lookup_5"
        assert retrieved_length == 768

        # Verify: No memory objects should have ref_count_down called
        # (no remaining chunks in current tier, no subsequent tiers)
        for obj in tier0_objs:
            assert not obj.ref_count_down_called

    def test_layerwise_partial_chunk_tail_released(self, storage_manager):
        """keys_per_chunk=4, backend returns 7 mem_objs for 2 chunks (8 keys);
        the 3 mem_objs in the rounded-off partial-chunk tail must be released."""
        keys_per_chunk = 4
        cum_chunk_lengths_total = [0, 256, 512]
        tier_expected_chunks = [2]

        tier0_objs = [MockMemoryObj(i) for i in range(7)]
        res = [[(f"k{i}", obj) for i, obj in enumerate(tier0_objs)]]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = loop.create_future()
        future.set_result(res)
        storage_manager.event_manager.add_event(
            EventType.LOADING, "test_layerwise_tail", future
        )

        storage_manager.prefetch_all_done_callback(
            future,
            "test_layerwise_tail",
            cum_chunk_lengths_total,
            tier_expected_chunks,
            keys_per_chunk=keys_per_chunk,
        )
        loop.close()

        assert storage_manager.async_lookup_server.responses == [
            ("test_layerwise_tail", 256)
        ]
        for obj in tier0_objs[:4]:
            assert not obj.ref_count_down_called
        for obj in tier0_objs[4:]:
            assert obj.ref_count_down_called
