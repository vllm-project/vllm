# SPDX-License-Identifier: Apache-2.0
"""Unit tests for KVController."""

# Standard
from unittest.mock import Mock, patch

# Third Party
import pytest

# First Party
from lmcache.v1.cache_controller.controllers.kv_controller import KVController
from lmcache.v1.cache_controller.message import (
    BatchedKVOperationMsg,
    BatchedP2PLookupMsg,
    CheckFinishMsg,
    ClearMsg,
    CompressMsg,
    DecompressMsg,
    KVOpEvent,
    LookupMsg,
    MoveMsg,
    OpType,
    PinMsg,
)


@pytest.fixture
def kv_controller(mock_reg_controller, mock_cluster_executor):
    """Create a KVController instance for testing."""
    controller = KVController(registry=mock_reg_controller.registry)
    controller.post_init(
        reg_controller=mock_reg_controller,
        cluster_executor=mock_cluster_executor,
    )
    return controller


class TestKVControllerAdmit:
    """Test KVController admit operations."""

    @pytest.mark.asyncio
    async def test_admit_new_key(self, kv_controller):
        """Test admitting a new KV chunk to the controller."""
        msg = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.ADMIT, key=12345, seq_num=1)],
        )

        await kv_controller.handle_batched_kv_operations(msg)

        # Access kv_pool through the registry mock
        report_id = ("test_instance", 0)
        assert report_id in kv_controller.registry.kv_pool
        assert "LocalCPUBackend" in kv_controller.registry.kv_pool[report_id]
        assert 12345 in kv_controller.registry.kv_pool[report_id]["LocalCPUBackend"]

    @pytest.mark.asyncio
    async def test_admit_duplicate_key(self, kv_controller):
        """Test admitting same key to different instances."""
        msg1 = BatchedKVOperationMsg(
            instance_id="instance1",
            worker_id=0,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.ADMIT, key=12345, seq_num=1)],
        )
        msg2 = BatchedKVOperationMsg(
            instance_id="instance2",
            worker_id=1,
            location="LocalDiskBackend",
            operations=[KVOpEvent(op_type=OpType.ADMIT, key=12345, seq_num=1)],
        )

        await kv_controller.handle_batched_kv_operations(msg1)
        await kv_controller.handle_batched_kv_operations(msg2)

        report_id1 = ("instance1", 0)
        report_id2 = ("instance2", 1)
        assert 12345 in kv_controller.registry.kv_pool[report_id1]["LocalCPUBackend"]
        assert 12345 in kv_controller.registry.kv_pool[report_id2]["LocalDiskBackend"]

    @pytest.mark.asyncio
    async def test_admit_multiple_keys(self, kv_controller):
        """Test admitting multiple different keys."""
        operations = [
            KVOpEvent(op_type=OpType.ADMIT, key=1000 + i, seq_num=i + 1)
            for i in range(5)
        ]
        msg = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=operations,
        )
        await kv_controller.handle_batched_kv_operations(msg)

        report_id = ("test_instance", 0)
        assert report_id in kv_controller.registry.kv_pool
        assert "LocalCPUBackend" in kv_controller.registry.kv_pool[report_id]
        for i in range(5):
            assert (1000 + i) in kv_controller.registry.kv_pool[report_id][
                "LocalCPUBackend"
            ]


class TestKVControllerEvict:
    """Test KVController evict operations."""

    @pytest.mark.asyncio
    async def test_evict_existing_key(self, kv_controller):
        """Test evicting an existing KV chunk."""
        # First admit
        admit_msg = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.ADMIT, key=12345, seq_num=1)],
        )
        await kv_controller.handle_batched_kv_operations(admit_msg)

        # Then evict
        evict_msg = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.EVICT, key=12345, seq_num=2)],
        )
        await kv_controller.handle_batched_kv_operations(evict_msg)

        report_id = ("test_instance", 0)
        assert (
            report_id not in kv_controller.registry.kv_pool
            or 12345
            not in kv_controller.registry.kv_pool.get(report_id, {}).get(
                "LocalCPUBackend", set()
            )
        )

    @pytest.mark.asyncio
    async def test_evict_nonexistent_key(self, kv_controller):
        """Test evicting a non-existent key does nothing."""
        evict_msg = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.EVICT, key=99999, seq_num=1)],
        )

        # Should not raise an error
        await kv_controller.handle_batched_kv_operations(evict_msg)
        # Verify key doesn't exist in registry
        result = kv_controller.registry.find_kv(99999)
        assert result is None

    @pytest.mark.asyncio
    async def test_evict_partial_metadata(self, kv_controller):
        """Test evicting from one instance while another instance still has it."""
        # Admit two entries for the same key from different instances
        msg1 = BatchedKVOperationMsg(
            instance_id="instance1",
            worker_id=0,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.ADMIT, key=12345, seq_num=1)],
        )
        msg2 = BatchedKVOperationMsg(
            instance_id="instance2",
            worker_id=1,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.ADMIT, key=12345, seq_num=1)],
        )
        await kv_controller.handle_batched_kv_operations(msg1)
        await kv_controller.handle_batched_kv_operations(msg2)

        # Evict only from the first instance
        evict_msg = BatchedKVOperationMsg(
            instance_id="instance1",
            worker_id=0,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.EVICT, key=12345, seq_num=2)],
        )
        await kv_controller.handle_batched_kv_operations(evict_msg)

        # Key should still exist in instance2
        report_id1 = ("instance1", 0)
        report_id2 = ("instance2", 1)
        assert (
            report_id1 not in kv_controller.registry.kv_pool
            or 12345
            not in kv_controller.registry.kv_pool[report_id1].get(
                "LocalCPUBackend", set()
            )
        )
        assert 12345 in kv_controller.registry.kv_pool[report_id2]["LocalCPUBackend"]

    @pytest.mark.asyncio
    async def test_evict_by_location(self, kv_controller):
        """Test evicting only matches specific location."""
        # Admit same key to different locations
        msg1 = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.ADMIT, key=12345, seq_num=1)],
        )
        msg2 = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalDiskBackend",
            operations=[KVOpEvent(op_type=OpType.ADMIT, key=12345, seq_num=2)],
        )
        await kv_controller.handle_batched_kv_operations(msg1)
        await kv_controller.handle_batched_kv_operations(msg2)

        # Evict only from CPU backend
        evict_msg = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.EVICT, key=12345, seq_num=3)],
        )
        await kv_controller.handle_batched_kv_operations(evict_msg)

        # Key should still exist in disk backend
        report_id = ("test_instance", 0)
        assert report_id in kv_controller.registry.kv_pool
        assert "LocalDiskBackend" in kv_controller.registry.kv_pool[report_id]
        assert 12345 in kv_controller.registry.kv_pool[report_id]["LocalDiskBackend"]
        assert (
            "LocalCPUBackend" not in kv_controller.registry.kv_pool[report_id]
            or 12345 not in kv_controller.registry.kv_pool[report_id]["LocalCPUBackend"]
        )


class TestKVControllerSequenceTracking:
    """Test KVController sequence number tracking."""

    @pytest.mark.asyncio
    async def test_sequence_number_first_message(self, kv_controller):
        """Test first message from a source initializes sequence tracker."""
        msg = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.ADMIT, key=12345, seq_num=5)],
        )

        await kv_controller.handle_batched_kv_operations(msg)

        seq_num = kv_controller.reg_controller.registry.get_seq_num(
            "test_instance", 0, "LocalCPUBackend"
        )
        assert seq_num == 5

    @pytest.mark.asyncio
    async def test_sequence_number_continuous(self, kv_controller):
        """Test continuous sequence numbers don't trigger warnings."""
        operations = [
            KVOpEvent(op_type=OpType.ADMIT, key=12345 + i, seq_num=i)
            for i in range(1, 6)
        ]
        msg = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=operations,
        )
        await kv_controller.handle_batched_kv_operations(msg)

        seq_num = kv_controller.reg_controller.registry.get_seq_num(
            "test_instance", 0, "LocalCPUBackend"
        )
        assert seq_num == 5
        # Sequence discontinuity is now tracked in the registry
        assert kv_controller.registry.get_seq_discontinuity_count() == 0

    @pytest.mark.asyncio
    async def test_sequence_discontinuity_detection(self, kv_controller):
        """Test detection of gaps in sequence numbers."""
        # First message
        msg1 = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.ADMIT, key=12345, seq_num=1)],
        )
        await kv_controller.handle_batched_kv_operations(msg1)

        # Skip to sequence 5 (gap of 3)
        msg2 = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.ADMIT, key=12346, seq_num=5)],
        )
        await kv_controller.handle_batched_kv_operations(msg2)

        # Sequence discontinuity is now tracked in the registry
        assert kv_controller.registry.get_seq_discontinuity_count() == 1
        seq_num = kv_controller.reg_controller.registry.get_seq_num(
            "test_instance", 0, "LocalCPUBackend"
        )
        assert seq_num == 5

    @pytest.mark.asyncio
    async def test_sequence_tracking_per_source(self, kv_controller):
        """Test sequence tracking is independent per source."""
        # Different sources should have independent tracking
        msg1 = BatchedKVOperationMsg(
            instance_id="instance1",
            worker_id=0,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.ADMIT, key=100, seq_num=1)],
        )
        msg2 = BatchedKVOperationMsg(
            instance_id="instance2",
            worker_id=0,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.ADMIT, key=200, seq_num=1)],
        )

        await kv_controller.handle_batched_kv_operations(msg1)
        await kv_controller.handle_batched_kv_operations(msg2)

        seq_num1 = kv_controller.reg_controller.registry.get_seq_num(
            "instance1", 0, "LocalCPUBackend"
        )
        seq_num2 = kv_controller.reg_controller.registry.get_seq_num(
            "instance2", 0, "LocalCPUBackend"
        )

        assert seq_num1 == 1
        assert seq_num2 == 1


class TestKVControllerLookup:
    """Test KVController lookup operations."""

    @pytest.mark.asyncio
    async def test_lookup_hit(self, kv_controller):
        """Test successful lookup with prefix match."""
        # Admit some chunks
        operations = [
            KVOpEvent(op_type=OpType.ADMIT, key=1000 + i, seq_num=i + 1)
            for i in range(3)
        ]
        msg = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=operations,
        )
        await kv_controller.handle_batched_kv_operations(msg)

        # Mock token database to return our keys
        with patch.object(
            kv_controller.token_database, "process_tokens"
        ) as mock_process:
            mock_process.return_value = [
                (0, 256, 1000),
                (256, 512, 1001),
                (512, 768, 1002),
            ]

            lookup_msg = LookupMsg(
                event_id="event_123",
                tokens=list(range(768)),
            )

            result = await kv_controller.lookup(lookup_msg)

            assert result.event_id == "event_123"
            assert "test_instance" in result.layout_info
            assert result.layout_info["test_instance"] == ("LocalCPUBackend", 768)

    @pytest.mark.asyncio
    async def test_lookup_miss(self, kv_controller):
        """Test lookup with no matches."""
        with patch.object(
            kv_controller.token_database, "process_tokens"
        ) as mock_process:
            mock_process.return_value = [(0, 256, 9999)]

            lookup_msg = LookupMsg(
                event_id="event_123",
                tokens=list(range(256)),
            )

            result = await kv_controller.lookup(lookup_msg)

            assert result.event_id == "event_123"
            assert len(result.layout_info) == 0

    @pytest.mark.asyncio
    async def test_lookup_partial_match(self, kv_controller):
        """Test lookup with partial prefix match."""
        # Admit only first chunk
        msg = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.ADMIT, key=1000, seq_num=1)],
        )
        await kv_controller.handle_batched_kv_operations(msg)

        with patch.object(
            kv_controller.token_database, "process_tokens"
        ) as mock_process:
            # Return two chunks but only first exists
            mock_process.return_value = [
                (0, 256, 1000),
                (256, 512, 1001),  # This doesn't exist
            ]

            lookup_msg = LookupMsg(
                event_id="event_123",
                tokens=list(range(512)),
            )

            result = await kv_controller.lookup(lookup_msg)

            # Should only match up to first chunk
            assert "test_instance" in result.layout_info
            assert result.layout_info["test_instance"] == ("LocalCPUBackend", 256)


class TestKVControllerBatchedP2PLookup:
    """Test KVController batched P2P lookup operations."""

    @pytest.mark.asyncio
    async def test_batched_p2p_lookup_success(self, kv_controller):
        """Test successful batched P2P lookup."""
        # Admit chunks from different instance
        operations = [
            KVOpEvent(op_type=OpType.ADMIT, key=1000 + i, seq_num=i + 1)
            for i in range(3)
        ]
        msg = BatchedKVOperationMsg(
            instance_id="remote_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=operations,
        )
        await kv_controller.handle_batched_kv_operations(msg)

        # Query from different instance
        lookup_msg = BatchedP2PLookupMsg(
            hashes=[1000, 1001, 1002],
            instance_id="query_instance",
            worker_id=0,
        )

        result = await kv_controller.batched_p2p_lookup(lookup_msg)

        assert len(result.layout_info) == 1
        instance_id, location, num_hits, peer_url = result.layout_info[0]
        assert instance_id == "remote_instance"
        assert location == "LocalCPUBackend"
        assert num_hits == 3
        assert peer_url == "tcp://localhost:5000"

    @pytest.mark.asyncio
    async def test_batched_p2p_lookup_no_match(self, kv_controller):
        """Test batched P2P lookup with no matches."""
        lookup_msg = BatchedP2PLookupMsg(
            hashes=[9999, 9998, 9997],
            instance_id="query_instance",
            worker_id=0,
        )

        result = await kv_controller.batched_p2p_lookup(lookup_msg)

        assert len(result.layout_info) == 1
        instance_id, location, num_hits, peer_url = result.layout_info[0]
        assert instance_id == ""
        assert location == ""
        assert num_hits == 0
        assert peer_url == ""

    @pytest.mark.asyncio
    async def test_batched_p2p_lookup_same_instance(self, kv_controller):
        """Test batched P2P lookup filters out same instance."""
        # Admit chunks from same instance as query
        msg = BatchedKVOperationMsg(
            instance_id="query_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.ADMIT, key=1000, seq_num=1)],
        )
        await kv_controller.handle_batched_kv_operations(msg)

        lookup_msg = BatchedP2PLookupMsg(
            hashes=[1000],
            instance_id="query_instance",
            worker_id=0,
        )

        result = await kv_controller.batched_p2p_lookup(lookup_msg)

        # Should not match same instance
        instance_id, location, num_hits, peer_url = result.layout_info[0]
        assert instance_id == ""
        assert location == ""
        assert num_hits == 0
        assert peer_url == ""

    @pytest.mark.asyncio
    async def test_batched_p2p_lookup_partial_match(self, kv_controller):
        """Test batched P2P lookup with partial matches."""
        # Admit only first two chunks
        operations = [
            KVOpEvent(op_type=OpType.ADMIT, key=1000 + i, seq_num=i + 1)
            for i in range(2)
        ]
        msg = BatchedKVOperationMsg(
            instance_id="remote_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=operations,
        )
        await kv_controller.handle_batched_kv_operations(msg)

        # Query for three chunks
        lookup_msg = BatchedP2PLookupMsg(
            hashes=[1000, 1001, 1002],
            instance_id="query_instance",
            worker_id=0,
        )

        result = await kv_controller.batched_p2p_lookup(lookup_msg)

        # Should only match first two
        instance_id, location, num_hits, peer_url = result.layout_info[0]
        assert num_hits == 2


class TestKVControllerDeregister:
    """Test KVController deregister operations."""

    @pytest.mark.asyncio
    async def test_deregister_cleanup(self, kv_controller):
        """Test deregister cleans up KV pool and sequence tracker."""
        # Admit some chunks
        operations = [
            KVOpEvent(op_type=OpType.ADMIT, key=1000 + i, seq_num=i + 1)
            for i in range(3)
        ]
        msg = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=operations,
        )
        await kv_controller.handle_batched_kv_operations(msg)

        # Verify data exists
        report_id = ("test_instance", 0)
        assert report_id in kv_controller.registry.kv_pool
        assert len(kv_controller.registry.kv_pool[report_id]["LocalCPUBackend"]) == 3
        seq_num = kv_controller.reg_controller.registry.get_seq_num(
            "test_instance", 0, "LocalCPUBackend"
        )
        assert seq_num is not None

        # Deregister directly through registry
        kv_controller.registry.deregister_worker("test_instance", 0)

        # Verify cleanup
        assert len(kv_controller.registry.kv_pool) == 0
        # Note: seq_tracker is managed by registry, cleaned up by deregister_worker

    @pytest.mark.asyncio
    async def test_deregister_partial_cleanup(self, kv_controller):
        """Test deregister only removes specific instance-worker."""
        # Admit chunks from different workers
        msg1 = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.ADMIT, key=1000, seq_num=1)],
        )
        msg2 = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=1,
            location="LocalCPUBackend",
            operations=[KVOpEvent(op_type=OpType.ADMIT, key=1000, seq_num=1)],
        )

        await kv_controller.handle_batched_kv_operations(msg1)
        await kv_controller.handle_batched_kv_operations(msg2)

        # Deregister only worker 0 directly through registry
        kv_controller.registry.deregister_worker("test_instance", 0)

        # Worker 1's data should remain
        report_id1 = ("test_instance", 1)
        assert report_id1 in kv_controller.registry.kv_pool
        assert 1000 in kv_controller.registry.kv_pool[report_id1]["LocalCPUBackend"]


class TestKVControllerDelegation:
    """Test KVController delegation to cluster executor."""

    @pytest.mark.asyncio
    async def test_clear_delegates_to_executor(self, kv_controller):
        """Test clear operation delegates to cluster executor."""
        msg = ClearMsg(
            event_id="event_123",
            instance_id="test_instance",
            location="LocalCPUBackend",
        )

        await kv_controller.clear(msg)

        kv_controller.cluster_executor.execute.assert_called_once_with("clear", msg)

    @pytest.mark.asyncio
    async def test_pin_delegates_to_executor(self, kv_controller):
        """Test pin operation delegates to cluster executor."""
        msg = PinMsg(
            event_id="event_123",
            instance_id="test_instance",
            location="LocalCPUBackend",
            tokens=[1, 2, 3],
        )

        await kv_controller.pin(msg)

        kv_controller.cluster_executor.execute.assert_called_once_with("pin", msg)

    @pytest.mark.asyncio
    async def test_compress_delegates_to_executor(self, kv_controller):
        """Test compress operation delegates to cluster executor."""
        msg = CompressMsg(
            event_id="event_123",
            instance_id="test_instance",
            method="gzip",
            location="LocalCPUBackend",
            tokens=[1, 2, 3],
        )

        await kv_controller.compress(msg)

        kv_controller.cluster_executor.execute.assert_called_once_with("compress", msg)

    @pytest.mark.asyncio
    async def test_decompress_delegates_to_executor(self, kv_controller):
        """Test decompress operation delegates to cluster executor."""
        msg = DecompressMsg(
            event_id="event_123",
            instance_id="test_instance",
            method="gzip",
            location="LocalCPUBackend",
            tokens=[1, 2, 3],
        )

        await kv_controller.decompress(msg)

        kv_controller.cluster_executor.execute.assert_called_once_with(
            "decompress", msg
        )

    @pytest.mark.asyncio
    async def test_move_delegates_to_executor(self, kv_controller):
        """Test move operation delegates to cluster executor."""
        msg = MoveMsg(
            event_id="event_123",
            old_position=("instance1", "LocalCPUBackend"),
            new_position=("instance2", "LocalCPUBackend"),
            tokens=[1, 2, 3],
            copy=False,
        )

        await kv_controller.move(msg)

        kv_controller.cluster_executor.execute.assert_called_once_with("move", msg)

    @pytest.mark.asyncio
    async def test_check_finish_delegates_to_executor(self, kv_controller):
        """Test check_finish operation delegates to cluster executor."""
        msg = CheckFinishMsg(event_id="event_123")

        await kv_controller.check_finish(msg)

        kv_controller.cluster_executor.execute.assert_called_once_with(
            "check_finish", msg
        )


class TestKVControllerMetrics:
    """Test KVController metrics setup."""

    @patch("lmcache.v1.cache_controller.controllers.kv_controller.PrometheusLogger")
    def test_metrics_setup(
        self, mock_prometheus_logger, mock_reg_controller, mock_cluster_executor
    ):
        """Test metrics are properly set up."""
        mock_logger_instance = Mock()
        mock_prometheus_logger.GetInstanceOrNone.return_value = mock_logger_instance

        _controller = KVController(registry=mock_reg_controller.registry)
        _controller.post_init(
            reg_controller=mock_reg_controller,
            cluster_executor=mock_cluster_executor,
        )

        # Verify metrics functions were set
        mock_logger_instance.kv_pool_keys_count.set_function.assert_called_once()
        mock_logger_instance.kv_op_seq_discontinuity_count.set_function.assert_called_once()

    @patch("lmcache.v1.cache_controller.controllers.kv_controller.PrometheusLogger")
    def test_metrics_setup_no_logger(
        self, mock_prometheus_logger, mock_reg_controller, mock_cluster_executor
    ):
        """Test metrics setup when no logger is available."""
        mock_prometheus_logger.GetInstanceOrNone.return_value = None

        # Should not raise an error
        controller = KVController(registry=mock_reg_controller.registry)

        assert controller is not None
