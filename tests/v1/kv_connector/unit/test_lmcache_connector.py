# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_events import BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1,
    LMCacheKVEvents,
)
from vllm.v1.outputs import KVConnectorOutput


@pytest.fixture
def mock_lmcache_engine_event():
    """Create a mock event object that mimics what the lmcache engine returns."""

    class MockEvent:
        def __init__(
            self,
            block_hashes,
            parent_block_hash,
            token_ids,
            lora_id,
            block_size,
            medium,
            lora_name,
        ):
            self.block_hashes = block_hashes
            self.parent_block_hash = parent_block_hash
            self.token_ids = token_ids
            self.lora_id = lora_id
            self.block_size = block_size
            self.medium = medium
            self.lora_name = lora_name

    return MockEvent(
        block_hashes=["hash1", "hash2"],
        parent_block_hash="parent_hash",
        token_ids=[1, 2, 3, 4],
        lora_id=None,
        block_size=16,
        medium="GPU",
        lora_name=None,
    )


@pytest.fixture
def mock_connector():
    """Create a mock LMCacheConnectorV1 instance with mocked dependencies."""
    connector = MagicMock(spec=LMCacheConnectorV1)
    connector._kv_cache_events = None
    connector._lmcache_engine = MagicMock()

    # Make the methods use the real implementation
    connector.get_kv_connector_kv_cache_events = (
        LMCacheConnectorV1.get_kv_connector_kv_cache_events.__get__(
            connector, LMCacheConnectorV1
        )
    )
    connector.update_connector_output = (
        LMCacheConnectorV1.update_connector_output.__get__(
            connector, LMCacheConnectorV1
        )
    )
    connector.take_events = LMCacheConnectorV1.take_events.__get__(
        connector, LMCacheConnectorV1
    )

    return connector


class TestGetKVConnectorKVCacheEvents:
    """Test get_kv_connector_kv_cache_events method."""

    def test_returns_none_when_no_events(self, mock_connector):
        """Test that None is returned when lmcache engine has no events."""
        mock_connector._lmcache_engine.get_kv_events.return_value = None

        result = mock_connector.get_kv_connector_kv_cache_events()

        assert result is None
        mock_connector._lmcache_engine.get_kv_events.assert_called_once()

    def test_returns_none_when_empty_list(self, mock_connector):
        """Test that None is returned when lmcache engine returns empty list."""
        mock_connector._lmcache_engine.get_kv_events.return_value = []

        result = mock_connector.get_kv_connector_kv_cache_events()

        assert result is None

    def test_converts_single_event(self, mock_connector, mock_lmcache_engine_event):
        """Test conversion of a single event from lmcache engine format."""
        mock_connector._lmcache_engine.get_kv_events.return_value = [
            mock_lmcache_engine_event
        ]

        result = mock_connector.get_kv_connector_kv_cache_events()

        assert result is not None
        assert isinstance(result, LMCacheKVEvents)
        assert result.get_number_of_workers() == 1

        events = result.get_all_events()
        assert len(events) == 1
        assert isinstance(events[0], BlockStored)
        assert events[0].block_hashes == ["hash1", "hash2"]
        assert events[0].parent_block_hash == "parent_hash"
        assert events[0].token_ids == [1, 2, 3, 4]
        assert events[0].lora_id is None
        assert events[0].block_size == 16
        assert events[0].medium == "GPU"
        assert events[0].lora_name is None

    def test_converts_multiple_events(self, mock_connector):
        """Test conversion of multiple events from lmcache engine format."""

        class MockEvent:
            def __init__(self, i):
                self.block_hashes = [f"hash{i}"]
                self.parent_block_hash = f"parent{i}"
                self.token_ids = [i]
                self.lora_id = None
                self.block_size = 16
                self.medium = "GPU"
                self.lora_name = None

        events = [MockEvent(i) for i in range(5)]
        mock_connector._lmcache_engine.get_kv_events.return_value = events

        result = mock_connector.get_kv_connector_kv_cache_events()

        assert result is not None
        assert isinstance(result, LMCacheKVEvents)

        converted_events = result.get_all_events()
        assert len(converted_events) == 5

        for i, event in enumerate(converted_events):
            assert isinstance(event, BlockStored)
            assert event.block_hashes == [f"hash{i}"]
            assert event.parent_block_hash == f"parent{i}"
            assert event.token_ids == [i]

    def test_preserves_event_attributes(self, mock_connector):
        """Test that all event attributes are correctly preserved."""

        class MockEventWithLora:
            def __init__(self):
                self.block_hashes = ["hash_a", "hash_b", "hash_c"]
                self.parent_block_hash = "parent_xyz"
                self.token_ids = [100, 200, 300]
                self.lora_id = 42
                self.block_size = 32
                self.medium = "DISK"
                self.lora_name = "lora_example"

        mock_connector._lmcache_engine.get_kv_events.return_value = [
            MockEventWithLora()
        ]

        result = mock_connector.get_kv_connector_kv_cache_events()

        events = result.get_all_events()
        event = events[0]

        assert event.block_hashes == ["hash_a", "hash_b", "hash_c"]
        assert event.parent_block_hash == "parent_xyz"
        assert event.token_ids == [100, 200, 300]
        assert event.lora_id == 42
        assert event.block_size == 32
        assert event.medium == "DISK"
        assert event.lora_name == "lora_example"

    def test_handles_none_parent_block_hash(self, mock_connector):
        """Test handling of events with None parent_block_hash."""

        class MockEventNoParent:
            def __init__(self):
                self.block_hashes = ["hash1"]
                self.parent_block_hash = None
                self.token_ids = [1, 2]
                self.lora_id = None
                self.block_size = 16
                self.medium = "GPU"
                self.lora_name = None

        mock_connector._lmcache_engine.get_kv_events.return_value = [
            MockEventNoParent()
        ]

        result = mock_connector.get_kv_connector_kv_cache_events()

        events = result.get_all_events()
        assert events[0].parent_block_hash is None


class TestUpdateConnectorOutput:
    """Test update_connector_output method."""

    def test_does_nothing_when_kv_cache_events_is_none(self, mock_connector):
        """Test that method returns early when kv_cache_events is None."""
        connector_output = KVConnectorOutput(kv_cache_events=None)

        mock_connector.update_connector_output(connector_output)

        assert mock_connector._kv_cache_events is None

    def test_does_nothing_when_kv_cache_events_is_not_lmcache_kv_events(
        self, mock_connector
    ):
        """Test that method returns early when kv_cache_events is not
        LMCacheKVEvents."""
        # Create a mock object that is not LMCacheKVEvents
        fake_events = MagicMock()
        connector_output = KVConnectorOutput(kv_cache_events=fake_events)

        mock_connector.update_connector_output(connector_output)

        assert mock_connector._kv_cache_events is None

    def test_sets_kv_cache_events_when_none(self, mock_connector):
        """Test that _kv_cache_events is set when it was None."""
        kv_events = LMCacheKVEvents(num_workers=1)
        event = BlockStored(
            block_hashes=["hash1"],
            parent_block_hash=None,
            token_ids=[1, 2],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )
        kv_events.add_events([event])

        connector_output = KVConnectorOutput(kv_cache_events=kv_events)

        mock_connector.update_connector_output(connector_output)

        assert mock_connector._kv_cache_events is kv_events

    def test_adds_events_when_kv_cache_events_already_exists(self, mock_connector):
        """Test that events are added when _kv_cache_events already exists."""
        # Set up existing events
        existing_events = LMCacheKVEvents(num_workers=2)
        event1 = BlockStored(
            block_hashes=["hash1"],
            parent_block_hash=None,
            token_ids=[1],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )
        existing_events.add_events([event1])
        existing_events.add_events([event1])  # Simulate 2 workers reporting

        mock_connector._kv_cache_events = existing_events

        # Create new events to add
        new_events = LMCacheKVEvents(num_workers=1)
        event2 = BlockStored(
            block_hashes=["hash2"],
            parent_block_hash=None,
            token_ids=[2],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )
        new_events.add_events([event2])

        connector_output = KVConnectorOutput(kv_cache_events=new_events)

        mock_connector.update_connector_output(connector_output)

        # Check that events were added
        all_events = mock_connector._kv_cache_events.get_all_events()
        assert len(all_events) == 3  # 2 from existing + 1 from new
        assert event1 in all_events
        assert event2 in all_events

    def test_increments_workers_when_kv_cache_events_already_exists(
        self, mock_connector
    ):
        """Test that worker count is incremented correctly."""
        # Set up existing events with 2 workers
        existing_events = LMCacheKVEvents(num_workers=2)
        mock_connector._kv_cache_events = existing_events

        # Create new events from 3 workers
        new_events = LMCacheKVEvents(num_workers=3)
        event = BlockStored(
            block_hashes=["hash1"],
            parent_block_hash=None,
            token_ids=[1],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )
        new_events.add_events([event])

        connector_output = KVConnectorOutput(kv_cache_events=new_events)

        mock_connector.update_connector_output(connector_output)

        # Worker count should be 2 + 3 = 5
        assert mock_connector._kv_cache_events.get_number_of_workers() == 5

    def test_multiple_updates(self, mock_connector):
        """Test multiple consecutive updates."""
        # First update
        events1 = LMCacheKVEvents(num_workers=1)
        event1 = BlockStored(
            block_hashes=["hash1"],
            parent_block_hash=None,
            token_ids=[1],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )
        events1.add_events([event1])
        output1 = KVConnectorOutput(kv_cache_events=events1)
        mock_connector.update_connector_output(output1)

        # Second update
        events2 = LMCacheKVEvents(num_workers=2)
        event2 = BlockStored(
            block_hashes=["hash2"],
            parent_block_hash=None,
            token_ids=[2],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )
        events2.add_events([event2])
        output2 = KVConnectorOutput(kv_cache_events=events2)
        mock_connector.update_connector_output(output2)

        # Third update
        events3 = LMCacheKVEvents(num_workers=1)
        event3 = BlockStored(
            block_hashes=["hash3"],
            parent_block_hash=None,
            token_ids=[3],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )
        events3.add_events([event3])
        output3 = KVConnectorOutput(kv_cache_events=events3)
        mock_connector.update_connector_output(output3)

        # Check final state
        all_events = mock_connector._kv_cache_events.get_all_events()
        assert len(all_events) == 3
        assert mock_connector._kv_cache_events.get_number_of_workers() == 4  # 1+2+1

    def test_updates_with_empty_events(self, mock_connector):
        """Test updating with empty event lists."""
        # First update with actual events
        events1 = LMCacheKVEvents(num_workers=1)
        event1 = BlockStored(
            block_hashes=["hash1"],
            parent_block_hash=None,
            token_ids=[1],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )
        events1.add_events([event1])
        output1 = KVConnectorOutput(kv_cache_events=events1)
        mock_connector.update_connector_output(output1)

        # Second update with empty events
        events2 = LMCacheKVEvents(num_workers=2)
        # No events added
        output2 = KVConnectorOutput(kv_cache_events=events2)
        mock_connector.update_connector_output(output2)

        # Should still have the original event
        all_events = mock_connector._kv_cache_events.get_all_events()
        assert len(all_events) == 1
        assert mock_connector._kv_cache_events.get_number_of_workers() == 3


class TestTakeEvents:
    """Test take_events method."""

    def test_yields_nothing_when_kv_cache_events_is_none(self, mock_connector):
        """Test that nothing is yielded when _kv_cache_events is None."""
        mock_connector._kv_cache_events = None

        events = list(mock_connector.take_events())

        assert events == []

    def test_yields_events_and_clears(self, mock_connector):
        """Test that events are yielded and then cleared."""
        # Set up events
        kv_events = LMCacheKVEvents(num_workers=1)
        event1 = BlockStored(
            block_hashes=["hash1"],
            parent_block_hash=None,
            token_ids=[1],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )
        event2 = BlockStored(
            block_hashes=["hash2"],
            parent_block_hash=None,
            token_ids=[2],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )
        kv_events.add_events([event1, event2])
        mock_connector._kv_cache_events = kv_events

        # Take events
        events = list(mock_connector.take_events())

        # Check that events were yielded
        assert len(events) == 2
        assert event1 in events
        assert event2 in events

        # Check that _kv_cache_events was cleared
        assert mock_connector._kv_cache_events is None

    def test_aggregates_before_yielding(self, mock_connector):
        """Test that events are aggregated before yielding."""
        # Set up events from multiple workers
        kv_events = LMCacheKVEvents(num_workers=3)
        common_event = BlockStored(
            block_hashes=["hash_common"],
            parent_block_hash=None,
            token_ids=[1],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )
        uncommon_event = BlockStored(
            block_hashes=["hash_uncommon"],
            parent_block_hash=None,
            token_ids=[2],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )

        # All 3 workers report common_event
        kv_events.add_events([common_event])
        kv_events.add_events([common_event])
        kv_events.add_events([common_event])

        # Only 1 worker reports uncommon_event
        kv_events.add_events([uncommon_event])

        mock_connector._kv_cache_events = kv_events

        # Take events
        events = list(mock_connector.take_events())

        # Only the common event should be yielded
        assert len(events) == 1
        assert events[0] == common_event

    def test_multiple_take_events_calls(self, mock_connector):
        """Test calling take_events multiple times."""
        # First call with events
        kv_events1 = LMCacheKVEvents(num_workers=1)
        event1 = BlockStored(
            block_hashes=["hash1"],
            parent_block_hash=None,
            token_ids=[1],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )
        kv_events1.add_events([event1])
        mock_connector._kv_cache_events = kv_events1

        events1 = list(mock_connector.take_events())
        assert len(events1) == 1
        assert events1[0] == event1
        assert mock_connector._kv_cache_events is None

        # Second call with no events
        events2 = list(mock_connector.take_events())
        assert events2 == []

        # Third call after adding new events
        kv_events2 = LMCacheKVEvents(num_workers=1)
        event2 = BlockStored(
            block_hashes=["hash2"],
            parent_block_hash=None,
            token_ids=[2],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )
        kv_events2.add_events([event2])
        mock_connector._kv_cache_events = kv_events2

        events3 = list(mock_connector.take_events())
        assert len(events3) == 1
        assert events3[0] == event2

    def test_yields_empty_after_aggregation_removes_all(self, mock_connector):
        """Test that nothing is yielded if aggregation removes all events."""
        # Set up events from 2 workers with no common events
        kv_events = LMCacheKVEvents(num_workers=2)
        event1 = BlockStored(
            block_hashes=["hash1"],
            parent_block_hash=None,
            token_ids=[1],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )
        event2 = BlockStored(
            block_hashes=["hash2"],
            parent_block_hash=None,
            token_ids=[2],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )

        # Worker 1 reports event1
        kv_events.add_events([event1])
        # Worker 2 reports event2
        kv_events.add_events([event2])

        mock_connector._kv_cache_events = kv_events

        # Take events
        events = list(mock_connector.take_events())

        # No common events, so nothing should be yielded
        assert events == []
        assert mock_connector._kv_cache_events is None


class TestIntegrationScenarios:
    """Test integration scenarios."""

    def test_full_workflow(self, mock_connector, mock_lmcache_engine_event):
        """Test a complete workflow from getting events to taking them."""
        # Step 1: Get events from lmcache engine
        mock_connector._lmcache_engine.get_kv_events.return_value = [
            mock_lmcache_engine_event
        ]
        kv_events = mock_connector.get_kv_connector_kv_cache_events()

        assert kv_events is not None
        assert len(kv_events.get_all_events()) == 1

        # Step 2: Update connector output (simulate receiving from worker)
        output1 = KVConnectorOutput(kv_cache_events=kv_events)
        mock_connector.update_connector_output(output1)

        assert mock_connector._kv_cache_events is not None

        # Step 3: Take events
        taken_events = list(mock_connector.take_events())

        assert len(taken_events) == 1
        assert mock_connector._kv_cache_events is None

    def test_multiple_workers_workflow(self, mock_connector):
        """Test workflow with multiple workers."""

        class MockEvent:
            def __init__(self, hash_val):
                self.block_hashes = [hash_val]
                self.parent_block_hash = None
                self.token_ids = [1]
                self.lora_id = None
                self.block_size = 16
                self.medium = "GPU"
                self.lora_name = None

        # Worker 1
        mock_connector._lmcache_engine.get_kv_events.return_value = [
            MockEvent("hash_common"),
            MockEvent("hash_worker1"),
        ]
        kv_events1 = mock_connector.get_kv_connector_kv_cache_events()
        output1 = KVConnectorOutput(kv_cache_events=kv_events1)
        mock_connector.update_connector_output(output1)

        # Worker 2
        mock_connector._lmcache_engine.get_kv_events.return_value = [
            MockEvent("hash_common"),
            MockEvent("hash_worker2"),
        ]
        kv_events2 = mock_connector.get_kv_connector_kv_cache_events()
        output2 = KVConnectorOutput(kv_cache_events=kv_events2)
        mock_connector.update_connector_output(output2)

        # Take events (should only get common events)
        taken_events = list(mock_connector.take_events())

        # With aggregation, only events reported by both workers should be present
        # In this case, hash_common was reported by both
        event_hashes = [e.block_hashes[0] for e in taken_events]
        assert "hash_common" in event_hashes

    def test_empty_workflow(self, mock_connector):
        """Test workflow when there are no events at any stage."""
        # Get events returns None
        mock_connector._lmcache_engine.get_kv_events.return_value = None
        kv_events = mock_connector.get_kv_connector_kv_cache_events()

        assert kv_events is None

        # Update with None
        output = KVConnectorOutput(kv_cache_events=None)
        mock_connector.update_connector_output(output)

        # Take events
        taken_events = list(mock_connector.take_events())

        assert taken_events == []
        assert mock_connector._kv_cache_events is None

    def test_repeated_cycles(self, mock_connector):
        """Test multiple cycles of the complete workflow."""

        class MockEvent:
            def __init__(self, cycle_num):
                self.block_hashes = [f"hash_cycle_{cycle_num}"]
                self.parent_block_hash = None
                self.token_ids = [cycle_num]
                self.lora_id = None
                self.block_size = 16
                self.medium = "GPU"
                self.lora_name = None

        for cycle in range(3):
            # Get events
            mock_connector._lmcache_engine.get_kv_events.return_value = [
                MockEvent(cycle)
            ]
            kv_events = mock_connector.get_kv_connector_kv_cache_events()

            # Update
            output = KVConnectorOutput(kv_cache_events=kv_events)
            mock_connector.update_connector_output(output)

            # Take
            taken_events = list(mock_connector.take_events())

            # Verify
            assert len(taken_events) == 1
            assert taken_events[0].block_hashes[0] == f"hash_cycle_{cycle}"
            assert mock_connector._kv_cache_events is None

    def test_lmcache_kv_events_aggregation(self):
        """
        Test LMCacheKVEvents aggregation across TP ranks using
        KVOutputAggregator (used by MultiprocExecutor).
        """
        from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
        from vllm.v1.outputs import ModelRunnerOutput

        # Create KVOutputAggregator for 3 workers (simulating TP=3)
        aggregator = KVOutputAggregator(expected_finished_count=3)

        # Define common and unique events
        common_event = BlockStored(
            block_hashes=["hash_common"],
            parent_block_hash="parent_common",
            token_ids=[1, 2, 3],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )

        worker1_unique_event = BlockStored(
            block_hashes=["hash_worker1"],
            parent_block_hash="parent_w1",
            token_ids=[4, 5],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )

        worker2_unique_event = BlockStored(
            block_hashes=["hash_worker2"],
            parent_block_hash="parent_w2",
            token_ids=[6, 7],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )

        worker3_unique_event = BlockStored(
            block_hashes=["hash_worker3"],
            parent_block_hash="parent_w3",
            token_ids=[8, 9],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )

        # Create events for each worker
        # Worker 0: reports common event and its unique event
        worker0_events = LMCacheKVEvents(num_workers=1)
        worker0_events.add_events([common_event, worker1_unique_event])

        # Worker 1: reports common event and its unique event
        worker1_events = LMCacheKVEvents(num_workers=1)
        worker1_events.add_events([common_event, worker2_unique_event])

        # Worker 2: reports common event and its unique event
        worker2_events = LMCacheKVEvents(num_workers=1)
        worker2_events.add_events([common_event, worker3_unique_event])

        # Create ModelRunnerOutput instances for each worker
        worker_outputs = []
        for i, worker_events in enumerate(
            [worker0_events, worker1_events, worker2_events]
        ):
            output = ModelRunnerOutput(
                req_ids=[f"req_{i}"],
                req_id_to_index={f"req_{i}": 0},
                sampled_token_ids=[[123]],  # dummy token
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[None],
                kv_connector_output=KVConnectorOutput(
                    finished_sending=set([f"req_{i}_send"])
                    if i < 2
                    else None,  # Workers 0,1 finished sending
                    finished_recving=set([f"req_{i}_recv"])
                    if i > 0
                    else None,  # Workers 1,2 finished receiving
                    kv_cache_events=worker_events,
                ),
            )
            worker_outputs.append(output)

        # Use the real aggregation mechanism (like MultiprocExecutor.execute_model)
        aggregated_output = aggregator.aggregate(worker_outputs, output_rank=0)
        kv_cache_events = aggregated_output.kv_connector_output.kv_cache_events

        assert isinstance(kv_cache_events, LMCacheKVEvents)

        # After aggregation, events should be combined from all workers
        # The aggregator doesn't automatically aggregate events, so we need to call
        # aggregate() to get only common events
        kv_cache_events.aggregate()
        aggregated_events = kv_cache_events.get_all_events()

        # Only the common event should remain after aggregation
        # because it's the only event reported by all 3 workers
        assert len(aggregated_events) == 1
        assert aggregated_events[0] == common_event

        # Verify the common event properties
        assert aggregated_events[0].block_hashes == ["hash_common"]
        assert aggregated_events[0].parent_block_hash == "parent_common"
        assert aggregated_events[0].token_ids == [1, 2, 3]
