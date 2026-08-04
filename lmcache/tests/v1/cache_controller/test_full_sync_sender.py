# SPDX-License-Identifier: Apache-2.0
"""Unit tests for FullSyncSender."""

# Third Party
import pytest

# First Party
from lmcache.v1.cache_controller.full_sync_sender import FullSyncSender
from lmcache.v1.cache_controller.message import (
    FullSyncBatchMsg,
    FullSyncEndMsg,
)

# Local
# Test utilities from conftest
from .conftest import (
    H,
    MockLMCacheEngine,
    MockLocalCPUBackend,
    MockWorker,
    create_test_config,
    create_test_key,
)


@pytest.fixture
def full_sync_sender():
    """Create a FullSyncSender instance for testing."""
    config = create_test_config()
    worker = MockWorker()
    engine = MockLMCacheEngine()
    backend = MockLocalCPUBackend()
    sender = FullSyncSender(
        config=config,
        worker=worker,
        lmcache_engine=engine,
        local_cpu_backend=backend,
    )
    return sender, worker, engine, backend


class TestFullSyncSenderInit:
    """Test cases for FullSyncSender initialization."""

    @pytest.mark.parametrize(
        "batch_size,batch_interval_ms,startup_delay,max_retry",
        [(100, 0, 0.0, 3), (500, 10, 2.0, 5)],
    )
    def test_init_config(self, batch_size, batch_interval_ms, startup_delay, max_retry):
        """Test initialization with various configs."""
        config = create_test_config(
            batch_size=batch_size,
            batch_interval_ms=batch_interval_ms,
            startup_delay_s=startup_delay,
            max_retry_count=max_retry,
        )
        sender = FullSyncSender(
            config=config,
            worker=MockWorker(),
            lmcache_engine=MockLMCacheEngine(),
            local_cpu_backend=MockLocalCPUBackend(),
        )
        assert sender.batch_size == batch_size
        assert sender.batch_interval_ms == batch_interval_ms
        assert sender.startup_delay_range_s == startup_delay
        assert sender.max_retry_count == max_retry
        assert sender._is_syncing is False


class TestFullSyncSenderProperties:
    """Test cases for FullSyncSender properties."""

    @pytest.mark.parametrize(
        "prop,expected",
        [
            ("instance_id", "test_instance"),
            ("worker_id", 0),
            ("location", "LocalCPUBackend"),
            ("is_syncing", False),
        ],
    )
    def test_properties(self, full_sync_sender, prop, expected):
        """Test all properties."""
        sender, _, _, _ = full_sync_sender
        assert getattr(sender, prop) == expected


class TestFullSyncSenderSyncId:
    """Test cases for sync ID generation."""

    def test_generate_sync_id_format_and_uniqueness(self, full_sync_sender):
        """Test sync ID format and uniqueness."""
        sender, _, _, _ = full_sync_sender
        sync_ids = [sender._generate_sync_id() for _ in range(100)]

        # Check format
        assert all(s.startswith("test_instance_0_") for s in sync_ids)
        # Check uniqueness
        assert len(set(sync_ids)) == 100


class TestFullSyncSenderGetKeys:
    """Test cases for getting hot cache keys."""

    @pytest.mark.parametrize("key_count", [0, 10])
    def test_get_all_hot_cache_keys(self, key_count):
        """Test getting keys from cache."""
        test_keys = [create_test_key(i) for i in range(key_count)]
        sender = FullSyncSender(
            config=create_test_config(),
            worker=MockWorker(),
            lmcache_engine=MockLMCacheEngine(),
            local_cpu_backend=MockLocalCPUBackend(keys=test_keys),
        )
        keys = sender._get_all_hot_cache_keys()
        assert len(keys) == key_count
        assert all(isinstance(k, int) for k in keys)


class TestFullSyncSenderSendBatch:
    """Test cases for sending sync batch messages."""

    def test_send_sync_batch(self, full_sync_sender):
        """Test sending batch messages."""
        sender, worker, _, _ = full_sync_sender

        for i in range(3):
            sender._send_sync_batch("sync_123", i, [i * 3, i * 3 + 1, i * 3 + 2])

        assert len(worker.messages) == 3
        for i, msg in enumerate(worker.messages):
            assert isinstance(msg, FullSyncBatchMsg)
            assert msg.sync_id == "sync_123"
            assert msg.batch_id == i


class TestFullSyncSenderSendEnd:
    """Test cases for sending sync end message."""

    def test_send_sync_end(self, full_sync_sender):
        """Test sending end message."""
        sender, worker, _, _ = full_sync_sender
        sender._send_sync_end("sync_123", 1000)

        assert len(worker.messages) == 1
        msg = worker.messages[0]
        assert isinstance(msg, FullSyncEndMsg)
        assert msg.sync_id == "sync_123"
        assert msg.actual_total_keys == 1000


class TestFullSyncSenderStartSync:
    """Test cases for the full sync process."""

    @pytest.mark.asyncio
    async def test_start_full_sync_empty_cache(self, full_sync_sender):
        """Test full sync with empty cache."""
        sender, worker, engine, _ = full_sync_sender
        worker.set_responses([H.start_ret("test"), H.status_ret("test")])

        success = await sender.start_full_sync("test_reason")

        assert success is True
        assert sender.is_syncing is False
        assert engine._freeze is False

    @pytest.mark.asyncio
    async def test_start_full_sync_with_keys(self):
        """Test full sync with keys in cache."""
        config = create_test_config(batch_size=5)
        worker = MockWorker()
        engine = MockLMCacheEngine()
        test_keys = [create_test_key(i) for i in range(12)]  # 3 batches

        sender = FullSyncSender(
            config=config,
            worker=worker,
            lmcache_engine=engine,
            local_cpu_backend=MockLocalCPUBackend(keys=test_keys),
        )
        worker.set_responses([H.start_ret("test"), H.status_ret("test")])

        success = await sender.start_full_sync("test_reason")

        assert success is True
        batch_msgs = [m for m in worker.messages if isinstance(m, FullSyncBatchMsg)]
        end_msgs = [m for m in worker.messages if isinstance(m, FullSyncEndMsg)]
        assert len(batch_msgs) == 3
        assert len(end_msgs) == 1

    @pytest.mark.asyncio
    async def test_start_full_sync_already_syncing(self, full_sync_sender):
        """Test that concurrent sync is prevented."""
        sender, _, _, _ = full_sync_sender
        sender._is_syncing = True

        assert await sender.start_full_sync("test_reason") is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "responses,expected",
        [
            # All retries rejected
            ([H.start_ret("t", False, "Busy")] * 3, False),
            # First fails, second succeeds
            (
                [H.start_ret("t", False, "Busy"), H.start_ret("t"), H.status_ret("t")],
                True,
            ),
        ],
    )
    async def test_start_full_sync_retry(self, full_sync_sender, responses, expected):
        """Test sync retry scenarios."""
        sender, worker, engine, _ = full_sync_sender
        worker.set_responses(responses)

        success = await sender.start_full_sync("test_reason")

        assert success is expected
        assert engine._freeze is False

    @pytest.mark.asyncio
    async def test_start_full_sync_freeze_mode(self, full_sync_sender):
        """Test that freeze mode is properly managed."""
        sender, worker, engine, _ = full_sync_sender
        worker.set_responses([H.start_ret("test"), H.status_ret("test")])

        await sender.start_full_sync("test_reason")

        assert True in engine.freeze_calls
        assert False in engine.freeze_calls
        assert engine.freeze_calls[-1] is False


class TestFullSyncSenderMessages:
    """Test cases for sync message methods."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "accepted,error_msg",
        [(True, None), (False, "Already syncing")],
    )
    async def test_send_sync_start(self, full_sync_sender, accepted, error_msg):
        """Test sync start responses."""
        sender, worker, _, _ = full_sync_sender
        worker.set_responses([H.start_ret("sync_123", accepted, error_msg)])

        ret = await sender._send_sync_start("sync_123", 100, 5)

        assert ret.accepted is accepted
        if error_msg:
            assert ret.error_msg == error_msg

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "is_complete,progress,can_exit",
        [(True, 0.85, True), (False, 0.3, False)],
    )
    async def test_query_sync_status(
        self, full_sync_sender, is_complete, progress, can_exit
    ):
        """Test sync status queries."""
        sender, worker, _, _ = full_sync_sender
        worker.set_responses(
            [H.status_ret("sync_123", is_complete, progress, can_exit)]
        )

        ret = await sender._query_sync_status("sync_123")

        assert ret.is_complete is is_complete
        assert ret.global_progress == progress
        assert ret.can_exit_freeze is can_exit
