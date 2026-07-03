# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for LMCacheMPConnector.update_state_after_alloc state transition logic.

Verifies the fix for the non-chosen connector scenario in MultiConnector:
When LMCacheMPConnector is a non-chosen sub-connector (num_external_tokens=0),
the state machine should NOT transition to WAITING_FOR_LOAD, but block tracking
(store path) should still execute, and lookup locks must be released.

Ref: PR #47294 review discussion, PR #46865
"""

import pytest

# Import conftest first to set up lmcache mocks (must be before any lmcache imports)
import tests.v1.kv_connector.unit.conftest  # noqa: F401
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (
    LMCacheMPRequestState,
)


class TestUpdateStateAfterAllocNonChosen:
    """
    Non-chosen connector scenario:
    - LMCache has cached data (num_lmcache_hit_blocks > 0)
    - But another connector was chosen (num_external_tokens == 0)
    """

    @pytest.fixture
    def mock_request(self, mock_request_factory):
        return mock_request_factory("test-request-123")

    # ---- Tests ----

    def test_state_should_be_ready_not_waiting_for_load(
        self, connector, mock_request, mock_blocks, setup_tracker_with_hits
    ):
        """
        Core bug: non-chosen connector must NOT enter WAITING_FOR_LOAD.
        It was never asked to load, so it should go directly to READY.
        """
        setup_tracker_with_hits(connector, mock_request)

        connector.update_state_after_alloc(mock_request, mock_blocks, 0)

        assert (
            connector.request_trackers[mock_request.request_id].state
            == LMCacheMPRequestState.READY
        ), "Non-chosen connector should be READY, not WAITING_FOR_LOAD."

    def test_block_tracking_must_still_execute(
        self, connector, mock_request, mock_blocks, setup_tracker_with_hits
    ):
        """
        Store path depends on allocated_block_ids.
        Even non-chosen connectors must track blocks for future store ops.
        (This is what PR #47294 got wrong — it guarded block tracking.)
        """
        setup_tracker_with_hits(connector, mock_request)

        connector.update_state_after_alloc(mock_request, mock_blocks, 0)

        tracker = connector.request_trackers[mock_request.request_id]
        assert len(tracker.allocated_block_ids) > 0, (
            "Block tracking must execute even for non-chosen connector. "
            "Store path needs allocated_block_ids."
        )

    def test_cleanup_lookup_result_must_be_called(
        self, connector, mock_request, mock_blocks, setup_tracker_with_hits
    ):
        """Lookup future must be cleaned up to prevent memory leak."""
        setup_tracker_with_hits(connector, mock_request)

        connector.update_state_after_alloc(mock_request, mock_blocks, 0)

        connector._mock_adapter.cleanup_lookup_result.assert_called_once_with(
            mock_request.request_id
        )

    def test_free_lookup_locks_must_be_called(
        self, connector, mock_request, mock_blocks, setup_tracker_with_hits
    ):
        """
        Lookup acquires read locks in LMCache.
        end_session does NOT release them.
        Non-chosen connector must call free_lookup_locks to prevent
        locks from being held until TTL expiry (10 minutes).
        """
        setup_tracker_with_hits(connector, mock_request)

        connector.update_state_after_alloc(mock_request, mock_blocks, 0)

        connector._mock_adapter.free_lookup_locks.assert_called_once()
        call_kwargs = connector._mock_adapter.free_lookup_locks.call_args.kwargs
        assert call_kwargs["request_id"] == mock_request.request_id
        assert call_kwargs["start"] == 0
        assert call_kwargs["end"] == 10 * 16  # 10 blocks * 16 tokens/block


class TestChosenConnector:
    """Chosen connector (num_external_tokens > 0) should work as before."""

    @pytest.fixture
    def mock_request(self, mock_request_factory):
        return mock_request_factory("test-chosen")

    def test_with_lmcache_hit_enters_waiting_for_load(
        self, connector, mock_request, mock_blocks, setup_tracker_with_hits
    ):
        """Chosen connector with LMCache hits should enter WAITING_FOR_LOAD."""
        setup_tracker_with_hits(connector, mock_request)

        connector.update_state_after_alloc(mock_request, mock_blocks, 10)  # chosen

        assert (
            connector.request_trackers[mock_request.request_id].state
            == LMCacheMPRequestState.WAITING_FOR_LOAD
        )

    def test_without_lmcache_hit_enters_ready(
        self, connector, mock_request, mock_blocks
    ):
        """Chosen connector without LMCache hits should enter READY."""
        tracker = connector._get_or_create_request_tracker(mock_request)
        tracker.num_lmcache_hit_blocks = 0
        tracker.num_vllm_hit_blocks = 0

        connector.update_state_after_alloc(mock_request, mock_blocks, 0)

        assert (
            connector.request_trackers[mock_request.request_id].state
            == LMCacheMPRequestState.READY
        )


class TestColdRequest:
    """Cold request: no LMCache hit at all."""

    @pytest.fixture
    def mock_request(self, mock_request_factory):
        return mock_request_factory("test-cold")

    def test_no_free_locks_when_no_hits(self, connector, mock_request, mock_blocks):
        """No LMCache hits means no locks to release."""
        tracker = connector._get_or_create_request_tracker(mock_request)
        tracker.num_lmcache_hit_blocks = 0
        tracker.num_vllm_hit_blocks = 0

        connector.update_state_after_alloc(mock_request, mock_blocks, 0)

        assert (
            connector.request_trackers[mock_request.request_id].state
            == LMCacheMPRequestState.READY
        )
        connector._mock_adapter.free_lookup_locks.assert_not_called()
