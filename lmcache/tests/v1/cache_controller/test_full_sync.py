# SPDX-License-Identifier: Apache-2.0
"""Tests for full sync functionality (Tracker and KVController)."""

# Standard
import asyncio
import time

# Third Party
import pytest

# First Party
from lmcache.v1.cache_controller.commands.full_sync import FullSyncCommand
from lmcache.v1.cache_controller.controllers.registration_controller import (
    RegistrationController,
)
from lmcache.v1.cache_controller.message import (
    FullSyncStatusMsg,
    HeartbeatMsg,
    OpType,
)
from lmcache.v1.cache_controller.utils import FullSyncState, RegistryTree

# Local
# Test utilities from conftest
from .conftest import LOCATION, H

# ============= FullSyncTracker Tests =============


class TestFullSyncTracker:
    """Tests for FullSyncTracker."""

    @pytest.mark.parametrize(
        "state,reason",
        [
            ("none", "controller_restart"),
            ("syncing", None),
            ("completed", None),
            ("failed", "sync_failed_retry"),
        ],
    )
    def test_should_request_full_sync(self, state, reason):
        """Test sync request logic."""
        tracker, reg = H.create_tracker()
        H.reg_worker(reg, "i1", 0)

        if state == "syncing":
            tracker.start_sync("i1", 0, "s1", 100, 5)
        elif state == "completed":
            tracker.start_sync("i1", 0, "s1", 100, 5)
            tracker.complete_sync("i1", 0, "s1", 100)
        elif state == "failed":
            tracker.set_need_full_sync_all(False)
            tracker.start_sync("i1", 0, "s1", 100, 5)
            tracker.mark_failed("i1", 0, "timeout")

        need, r = tracker.should_request_full_sync("i1", 0)
        assert (need, r) == (reason is not None, reason)

    @pytest.mark.parametrize(
        "action,syncing,state",
        [
            ("none", False, None),
            ("start", True, FullSyncState.SYNCING),
            ("complete", False, FullSyncState.COMPLETED),
            ("fail", False, FullSyncState.FAILED),
        ],
    )
    def test_is_worker_syncing(self, action, syncing, state):
        """Test worker syncing status."""
        tracker, reg = H.create_tracker()
        H.reg_worker(reg, "i1", 0)

        if action == "start":
            tracker.start_sync("i1", 0, "s1", 100, 5)
        elif action == "complete":
            tracker.start_sync("i1", 0, "s1", 100, 5)
            tracker.complete_sync("i1", 0, "s1", 100)
        elif action == "fail":
            tracker.start_sync("i1", 0, "s1", 100, 5)
            tracker.mark_failed("i1", 0, "reason")

        assert tracker.is_worker_syncing("i1", 0) == syncing
        if action != "none":
            assert reg.get_worker("i1", 0).sync_info.state == state

    def test_start_sync_conflict(self):
        """Test sync start conflict handling."""
        tracker, reg = H.create_tracker()
        H.reg_worker(reg, "i1", 0)

        tracker.start_sync("i1", 0, "s1", 100, 5)
        assert tracker.start_sync("i1", 0, "s2", 200, 10) is False  # Different ID
        assert tracker.start_sync("i1", 0, "s1", 100, 5) is True  # Same ID (retry)

    @pytest.mark.parametrize("batches,keys_per", [(5, 20), (2, 50), (1, 100)])
    def test_receive_batches(self, batches, keys_per):
        """Test batch receiving."""
        tracker, reg = H.create_tracker()
        H.reg_worker(reg, "i1", 0)
        tracker.start_sync("i1", 0, "s1", 100, batches)

        for i in range(batches):
            assert tracker.receive_batch("i1", 0, "s1", i, keys_per) is True

        info = reg.get_worker("i1", 0).sync_info
        assert info.received_batches == set(range(batches))
        assert info.received_keys_count == batches * keys_per

    def test_receive_batch_wrong_sync_id(self):
        """Test batch with wrong sync ID rejected."""
        tracker, reg = H.create_tracker()
        H.reg_worker(reg, "i1", 0)
        tracker.start_sync("i1", 0, "correct", 100, 5)

        assert tracker.receive_batch("i1", 0, "wrong", 0, 20) is False
        assert reg.get_worker("i1", 0).sync_info.received_keys_count == 0

    def test_complete_sync(self):
        """Test sync completion."""
        tracker, reg = H.create_tracker()
        H.reg_worker(reg, "i1", 0)
        tracker.start_sync("i1", 0, "s1", 100, 5)

        for i in range(5):
            tracker.receive_batch("i1", 0, "s1", i, 20)

        assert tracker.complete_sync("i1", 0, "s1", 100) is True
        assert reg.get_worker("i1", 0).sync_info.state == FullSyncState.COMPLETED
        assert tracker.complete_sync("i1", 0, "wrong", 100) is False

    def test_check_sync_timeout(self):
        """Test timeout detection."""
        tracker, reg = H.create_tracker(timeout=0.1)
        tracker.set_need_full_sync_all(False)
        H.reg_worker(reg, "i1", 0)
        tracker.start_sync("i1", 0, "s1", 100, 5)

        time.sleep(0.15)
        tracker.check_sync_timeout()

        assert reg.get_worker("i1", 0).sync_info.state == FullSyncState.FAILED
        need, reason = tracker.should_request_full_sync("i1", 0)
        assert need and reason == "sync_failed_retry"

    @pytest.mark.parametrize(
        "total,completed,expected",
        [(4, 0, 0.0), (4, 1, 0.25), (4, 2, 0.5), (4, 4, 1.0)],
    )
    def test_get_global_progress(self, total, completed, expected):
        """Test progress calculation."""
        tracker, reg = H.create_tracker()
        tracker.set_need_full_sync_all(False)

        for i in range(total):
            H.reg_worker(reg, "i1", i)
            tracker.start_sync("i1", i, f"s{i}", 100, 5)
        for i in range(completed):
            tracker.complete_sync("i1", i, f"s{i}", 100)

        assert abs(tracker.get_global_progress() - expected) < 0.001

    @pytest.mark.parametrize(
        "total,completed,threshold,can_exit",
        [(4, 2, 0.5, True), (4, 1, 0.5, False), (4, 3, 0.8, False), (4, 4, 0.8, True)],
    )
    def test_can_exit_freeze(self, total, completed, threshold, can_exit):
        """Test freeze exit check."""
        tracker, reg = H.create_tracker(threshold)
        tracker.set_need_full_sync_all(True)

        for i in range(total):
            H.reg_worker(reg, "i1", i)
            tracker.start_sync("i1", i, f"s{i}", 100, 5)
        for i in range(completed):
            tracker.complete_sync("i1", i, f"s{i}", 100)

        assert tracker.can_exit_freeze() == can_exit

    @pytest.mark.parametrize(
        "received,missing",
        [([], [0, 1, 2, 3, 4]), ([0, 2, 4], [1, 3]), ([0, 1, 2, 3, 4], [])],
    )
    def test_get_missing_batches(self, received, missing):
        """Test missing batches detection."""
        tracker, reg = H.create_tracker()
        H.reg_worker(reg, "i1", 0)
        tracker.start_sync("i1", 0, "s1", 100, 5)

        for b in received:
            tracker.receive_batch("i1", 0, "s1", b, 20)

        assert tracker.get_missing_batches("i1", 0, "s1") == missing

    @pytest.mark.parametrize(
        "setup,expected",
        [
            ("no_sync", []),  # No sync started
            ("wrong_id", []),  # Wrong sync ID
            ("completed", []),  # After completion
        ],
    )
    def test_get_missing_batches_edge_cases(self, setup, expected):
        """Test missing batches edge cases."""
        tracker, reg = H.create_tracker()
        H.reg_worker(reg, "i1", 0)

        if setup == "wrong_id":
            tracker.start_sync("i1", 0, "correct", 100, 5)
            assert tracker.get_missing_batches("i1", 0, "wrong") == expected
        elif setup == "completed":
            tracker.start_sync("i1", 0, "correct", 100, 5)
            tracker.complete_sync("i1", 0, "correct", 100)
            assert tracker.get_missing_batches("i1", 0, "correct") == expected
        else:
            assert tracker.get_missing_batches("i1", 0, "s1") == expected


# ============= KVController Tests =============


class TestKVController:
    """Tests for KVController full sync handling."""

    @pytest.fixture
    def ctrl(self):
        """Create KVController."""
        return H.create_controller()

    @pytest.mark.asyncio
    async def test_full_sync_start_clears_keys(self, ctrl):
        """Test sync start clears existing keys."""
        H.reg_worker(ctrl.registry, "i1", 0)
        for seq, key in enumerate([1, 2, 3]):
            ctrl.registry.handle_batched_kv_operations(
                H.batched_op("i1", 0, OpType.ADMIT, key, seq)
            )

        ret = await ctrl.handle_full_sync_start(H.start_msg("i1", 0, "s1", 100, 5))

        assert ret.accepted is True
        assert ctrl.full_sync_tracker.is_worker_syncing("i1", 0)
        assert len(ctrl.registry.get_worker_kv_keys("i1", 0, LOCATION)) == 0

    @pytest.mark.asyncio
    async def test_sync_start_conflict(self, ctrl):
        """Test sync start conflict."""
        H.reg_worker(ctrl.registry, "i1", 0)

        r1 = await ctrl.handle_full_sync_start(H.start_msg("i1", 0, "s1", 100, 5))
        r2 = await ctrl.handle_full_sync_start(H.start_msg("i1", 0, "s2", 200, 10))
        r3 = await ctrl.handle_full_sync_start(H.start_msg("i1", 0, "s1", 100, 5))

        assert r1.accepted and not r2.accepted and r3.accepted

    @pytest.mark.asyncio
    async def test_batch_handling(self, ctrl):
        """Test batch adds keys."""
        H.reg_worker(ctrl.registry, "i1", 0)
        await ctrl.handle_full_sync_start(H.start_msg("i1", 0, "s1", 15, 3))

        for i in range(3):
            await ctrl.handle_full_sync_batch(
                H.batch_msg("i1", 0, "s1", i, list(range(i * 5, (i + 1) * 5)))
            )

        keys = ctrl.registry.get_worker_kv_keys("i1", 0, LOCATION)
        assert len(keys) == 15 and all(i in keys for i in range(15))

    @pytest.mark.asyncio
    async def test_batch_wrong_sync_id_rejected(self, ctrl):
        """Test batch with wrong sync ID rejected."""
        H.reg_worker(ctrl.registry, "i1", 0)
        await ctrl.handle_full_sync_start(H.start_msg("i1", 0, "correct", 10, 1))

        keys_before = ctrl.registry.get_worker_kv_keys("i1", 0, LOCATION).copy()
        await ctrl.handle_full_sync_batch(H.batch_msg("i1", 0, "wrong", 0, [1, 2, 3]))

        assert ctrl.registry.get_worker_kv_keys("i1", 0, LOCATION) == keys_before

    @pytest.mark.asyncio
    async def test_sync_end_completes(self, ctrl):
        """Test sync end marks completion."""
        H.reg_worker(ctrl.registry, "i1", 0)
        await ctrl.handle_full_sync_start(H.start_msg("i1", 0, "s1", 5, 1))
        await ctrl.handle_full_sync_batch(
            H.batch_msg("i1", 0, "s1", 0, [1, 2, 3, 4, 5])
        )
        await ctrl.handle_full_sync_end(H.end_msg("i1", 0, "s1", 5))

        assert not ctrl.full_sync_tracker.is_worker_syncing("i1", 0)
        assert (
            ctrl.registry.get_worker("i1", 0).sync_info.state == FullSyncState.COMPLETED
        )

    @pytest.mark.asyncio
    async def test_sync_status(self, ctrl):
        """Test status query."""
        ctrl.full_sync_tracker.completion_threshold = 0.5
        for i in range(4):
            H.reg_worker(ctrl.registry, "i1", i)
            ctrl.full_sync_tracker.start_sync("i1", i, f"s{i}", 100, 5)
        for i in range(2):
            ctrl.full_sync_tracker.complete_sync("i1", i, f"s{i}", 100)

        ret = await ctrl.handle_full_sync_status(FullSyncStatusMsg("i1", 0, "s0"))

        assert (
            ret.is_complete
            and abs(ret.global_progress - 0.5) < 0.001
            and ret.can_exit_freeze
        )

    @pytest.mark.asyncio
    async def test_sync_status_missing_batches(self, ctrl):
        """Test status returns missing batches."""
        H.reg_worker(ctrl.registry, "i1", 0)
        await ctrl.handle_full_sync_start(H.start_msg("i1", 0, "s1", 100, 5))
        await ctrl.handle_full_sync_batch(H.batch_msg("i1", 0, "s1", 0, [1, 2, 3]))
        await ctrl.handle_full_sync_batch(H.batch_msg("i1", 0, "s1", 2, [7, 8, 9]))

        ret = await ctrl.handle_full_sync_status(FullSyncStatusMsg("i1", 0, "s1"))

        assert not ret.is_complete and ret.missing_batches == [1, 3, 4]

    @pytest.mark.asyncio
    async def test_incremental_discarded_during_sync(self, ctrl):
        """Test incremental messages discarded during sync."""
        H.reg_worker(ctrl.registry, "i1", 0)
        await ctrl.handle_full_sync_start(H.start_msg("i1", 0, "s1", 5, 1))

        await ctrl.handle_batched_kv_operations(
            H.batched_op("i1", 0, OpType.ADMIT, 888)
        )
        assert 888 not in ctrl.registry.get_worker_kv_keys("i1", 0, LOCATION)

        await ctrl.handle_full_sync_batch(H.batch_msg("i1", 0, "s1", 0, [999]))
        await ctrl.handle_batched_kv_operations(
            H.batched_op("i1", 0, OpType.EVICT, 999)
        )
        assert 999 in ctrl.registry.get_worker_kv_keys("i1", 0, LOCATION)

    @pytest.mark.asyncio
    async def test_incremental_allowed_after_sync(self, ctrl):
        """Test incremental works after sync completion."""
        H.reg_worker(ctrl.registry, "i1", 0)
        await ctrl.handle_full_sync_start(H.start_msg("i1", 0, "s1", 0, 1))
        await ctrl.handle_full_sync_end(H.end_msg("i1", 0, "s1", 0))

        await ctrl.handle_batched_kv_operations(
            H.batched_op("i1", 0, OpType.ADMIT, 1000, 1)
        )

        assert 1000 in ctrl.registry.get_worker_kv_keys("i1", 0, LOCATION)

    @pytest.mark.asyncio
    async def test_sync_timeout(self):
        """Test sync timeout marks failed."""
        ctrl = H.create_controller(timeout=0.1)
        ctrl.full_sync_tracker.set_need_full_sync_all(False)
        H.reg_worker(ctrl.registry, "i1", 0)

        await ctrl.handle_full_sync_start(H.start_msg("i1", 0, "s1", 100, 5))
        await asyncio.sleep(0.15)
        ctrl.full_sync_tracker.check_sync_timeout()

        assert ctrl.registry.get_worker("i1", 0).sync_info.state == FullSyncState.FAILED
        need, reason = ctrl.full_sync_tracker.should_request_full_sync("i1", 0)
        assert need and reason == "sync_failed_retry"


# ============= Integration Tests =============


class TestIntegration:
    """Integration tests for full sync flow."""

    @pytest.fixture
    def setup(self):
        """Setup controllers with shared registry."""
        # Standard
        from unittest.mock import MagicMock

        # First Party
        from lmcache.v1.cache_controller.controllers.kv_controller import KVController

        reg = RegistryTree()
        kv = KVController(reg, 0.5, 300.0)
        kv.cluster_executor = MagicMock()
        rc = RegistrationController()
        rc.registry = reg
        rc.kv_controller = kv
        rc.cluster_executor = MagicMock()
        return kv, rc

    @pytest.mark.asyncio
    async def test_complete_sync_flow(self, setup):
        """Test complete sync flow for single worker."""
        kv, rc = setup

        await H.reg_worker_async(rc, "i1", 0)
        for seq, key in enumerate([100, 200, 300]):
            kv.registry.handle_batched_kv_operations(
                H.batched_op("i1", 0, OpType.ADMIT, key, seq)
            )

        hb = await rc.heartbeat(HeartbeatMsg("i1", 0, "127.0.0.1", 8000, None))
        assert len(hb.commands) == 1 and isinstance(hb.commands[0], FullSyncCommand)

        await kv.handle_full_sync_start(H.start_msg("i1", 0, "s1", 10, 2))
        assert len(kv.registry.get_worker_kv_keys("i1", 0, LOCATION)) == 0

        await kv.handle_full_sync_batch(H.batch_msg("i1", 0, "s1", 0, [1, 2, 3, 4, 5]))
        await kv.handle_full_sync_batch(H.batch_msg("i1", 0, "s1", 1, [6, 7, 8, 9, 10]))
        await kv.handle_full_sync_end(H.end_msg("i1", 0, "s1", 10))

        keys = kv.registry.get_worker_kv_keys("i1", 0, LOCATION)
        assert keys == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

        hb2 = await rc.heartbeat(HeartbeatMsg("i1", 0, "127.0.0.1", 8000, None))
        assert len(hb2.commands) == 0

        status = await kv.handle_full_sync_status(FullSyncStatusMsg("i1", 0, "s1"))
        assert status.is_complete and status.global_progress == 1.0

    @pytest.mark.asyncio
    async def test_sync_progress_multiple_workers(self, setup):
        """Test sync progress with multiple workers."""
        kv, rc = setup
        workers = [(f"i{i}", i, f"192.168.1.{i + 1}") for i in range(4)]

        for i in range(len(workers)):
            inst, wid, ip = workers[i]
            sid = f"s{i}"
            await H.reg_worker_async(rc, inst, wid, ip)
            await kv.handle_full_sync_start(H.start_msg(inst, wid, sid, 5, 1))

        for i in range(2):
            inst, wid, _ = workers[i]
            sid = f"s{i}"
            await kv.handle_full_sync_batch(
                H.batch_msg(inst, wid, sid, 0, list(range(5)))
            )
            await kv.handle_full_sync_end(H.end_msg(inst, wid, sid, 5))

        assert kv.full_sync_tracker.get_global_progress() == 0.5
        assert kv.full_sync_tracker.can_exit_freeze() is True
