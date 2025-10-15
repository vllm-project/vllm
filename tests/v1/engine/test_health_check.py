# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for stats-based health check functionality."""
import time

import pytest

from vllm.v1.engine.async_llm import HealthStateTracker
from vllm.v1.engine.exceptions import EngineDeadError


class TestHealthStateTracker:
    """Test the HealthStateTracker class."""

    def test_initial_state_is_healthy(self):
        """Test that tracker starts in healthy state (no active requests = idle)."""
        tracker = HealthStateTracker(stall_timeout_seconds=1.0)
        # Initial state: 0 waiting, 0 running = idle = healthy
        assert tracker.num_waiting_reqs == 0
        assert tracker.num_running_reqs == 0
        assert tracker.is_healthy()

    def test_remains_healthy_after_first_update(self):
        """Test that tracker remains healthy after first update with active requests."""
        tracker = HealthStateTracker(stall_timeout_seconds=1.0)
        tracker.update(step_counter=1, current_wave=0, num_waiting_reqs=1, num_running_reqs=0)
        # Has active requests and just made progress = healthy
        assert tracker.is_healthy()

    def test_remains_healthy_with_progress(self):
        """Test that tracker remains healthy when step_counter advances."""
        tracker = HealthStateTracker(stall_timeout_seconds=1.0)
        tracker.update(step_counter=1, current_wave=0, num_waiting_reqs=1, num_running_reqs=0)
        time.sleep(0.5)
        tracker.update(step_counter=2, current_wave=0, num_waiting_reqs=1, num_running_reqs=0)
        assert tracker.is_healthy()

    def test_becomes_unhealthy_after_timeout_with_active_requests(self):
        """Test that tracker becomes unhealthy after stall timeout with active requests."""
        tracker = HealthStateTracker(stall_timeout_seconds=0.5)
        tracker.update(step_counter=1, current_wave=0, num_waiting_reqs=1, num_running_reqs=0)
        time.sleep(0.6)
        # Still has active requests - should be unhealthy
        tracker.update(step_counter=1, current_wave=0, num_waiting_reqs=1, num_running_reqs=0)
        assert not tracker.is_healthy()

    def test_idle_engine_remains_healthy(self):
        """Test that an idle engine (no active requests) remains healthy."""
        tracker = HealthStateTracker(stall_timeout_seconds=0.5)
        tracker.update(step_counter=1, current_wave=0, num_waiting_reqs=1, num_running_reqs=0)
        time.sleep(0.6)
        # No active requests - should be healthy even with no progress
        tracker.update(step_counter=1, current_wave=0, num_waiting_reqs=0, num_running_reqs=0)
        assert tracker.is_healthy()

    def test_ignores_non_advancing_counter(self):
        """Test that tracker doesn't reset timer for non-advancing counter."""
        tracker = HealthStateTracker(stall_timeout_seconds=0.5)
        tracker.update(step_counter=5, current_wave=0, num_waiting_reqs=1, num_running_reqs=0)
        initial_time = tracker.last_update_time

        time.sleep(0.1)
        tracker.update(
            step_counter=5, current_wave=0, num_waiting_reqs=1, num_running_reqs=0
        )  # Same counter, should not reset
        assert tracker.last_update_time == initial_time

        time.sleep(0.1)
        tracker.update(
            step_counter=4, current_wave=0, num_waiting_reqs=1, num_running_reqs=0
        )  # Going backwards, should not reset
        assert tracker.last_update_time == initial_time

    def test_updates_on_advancing_counter(self):
        """Test that tracker updates timestamp when counter advances."""
        tracker = HealthStateTracker(stall_timeout_seconds=1.0)
        tracker.update(step_counter=1, current_wave=0, num_waiting_reqs=1, num_running_reqs=0)
        initial_time = tracker.last_update_time

        time.sleep(0.1)
        tracker.update(
            step_counter=2, current_wave=0, num_waiting_reqs=1, num_running_reqs=0
        )  # Advances, should reset
        assert tracker.last_update_time > initial_time

    def test_custom_timeout(self):
        """Test that custom timeout values work correctly."""
        tracker = HealthStateTracker(stall_timeout_seconds=2.0)
        tracker.update(step_counter=1, current_wave=0, num_waiting_reqs=1, num_running_reqs=0)

        # Should still be healthy after 1.5 seconds (with active requests)
        time.sleep(1.5)
        tracker.update(step_counter=1, current_wave=0, num_waiting_reqs=1, num_running_reqs=0)
        assert tracker.is_healthy()

        # Should be unhealthy after 2.1 seconds (with active requests)
        time.sleep(0.6)
        tracker.update(step_counter=1, current_wave=0, num_waiting_reqs=1, num_running_reqs=0)
        assert not tracker.is_healthy()


@pytest.mark.asyncio
async def test_async_llm_health_check_integration():
    """Integration test for AsyncLLM health check with HealthStateTracker.

    Note: This is a basic structure. Full integration tests would require
    mocking the engine core and scheduler stats.
    """
    # This test would require setting up a full AsyncLLM instance
    # with mocked components, which is beyond the scope of this unit test.
    # The unit tests above verify the core HealthStateTracker logic.
    pass


def test_health_tracker_with_zero_timeout():
    """Test edge case where timeout is set to 0 (always unhealthy after init with active requests)."""
    tracker = HealthStateTracker(stall_timeout_seconds=0.0)
    tracker.update(step_counter=1, current_wave=0, num_waiting_reqs=1, num_running_reqs=0)
    # Should immediately be unhealthy since timeout is 0 and there are active requests
    assert not tracker.is_healthy()


def test_health_tracker_with_large_timeout():
    """Test that very large timeout values work correctly."""
    tracker = HealthStateTracker(stall_timeout_seconds=10000.0)
    tracker.update(step_counter=1, current_wave=0, num_waiting_reqs=1, num_running_reqs=0)
    time.sleep(0.1)
    # Should still be healthy with a very large timeout
    assert tracker.is_healthy()


def test_transition_from_active_to_idle():
    """Test that transitioning from active to idle keeps engine healthy."""
    tracker = HealthStateTracker(stall_timeout_seconds=0.5)

    # Start with active requests
    tracker.update(step_counter=1, current_wave=0, num_waiting_reqs=2, num_running_reqs=1)
    assert tracker.is_healthy()

    # Wait longer than timeout with no progress
    time.sleep(0.6)
    tracker.update(step_counter=1, current_wave=0, num_waiting_reqs=2, num_running_reqs=1)
    # Should be unhealthy (stalled with active requests)
    assert not tracker.is_healthy()

    # Requests complete - transition to idle
    tracker.update(step_counter=1, current_wave=0, num_waiting_reqs=0, num_running_reqs=0)
    # Should now be healthy (idle)
    assert tracker.is_healthy()


def test_running_and_waiting_requests():
    """Test that both running and waiting requests are considered."""
    tracker = HealthStateTracker(stall_timeout_seconds=0.5)

    # Test with only waiting requests
    tracker.update(step_counter=1, current_wave=0, num_waiting_reqs=5, num_running_reqs=0)
    time.sleep(0.6)
    tracker.update(step_counter=1, current_wave=0, num_waiting_reqs=5, num_running_reqs=0)
    assert not tracker.is_healthy()

    # Reset with progress
    tracker.update(step_counter=2, current_wave=0, num_waiting_reqs=0, num_running_reqs=0)
    assert tracker.is_healthy()

    # Test with only running requests
    tracker.update(step_counter=2, current_wave=0, num_waiting_reqs=0, num_running_reqs=3)
    time.sleep(0.6)
    tracker.update(step_counter=2, current_wave=0, num_waiting_reqs=0, num_running_reqs=3)
    assert not tracker.is_healthy()

    # Reset with progress
    tracker.update(step_counter=3, current_wave=0, num_waiting_reqs=0, num_running_reqs=0)
    assert tracker.is_healthy()

    # Test with both
    tracker.update(step_counter=3, current_wave=0, num_waiting_reqs=2, num_running_reqs=3)
    time.sleep(0.6)
    tracker.update(step_counter=3, current_wave=0, num_waiting_reqs=2, num_running_reqs=3)
    assert not tracker.is_healthy()


def test_wave_advancement_is_progress():
    """Test that wave advancement is treated as progress even if step_counter resets."""
    tracker = HealthStateTracker(stall_timeout_seconds=0.5)

    # Start at wave 0, step 100
    tracker.update(step_counter=100, current_wave=0, num_waiting_reqs=1, num_running_reqs=0)
    initial_time = tracker.last_update_time

    time.sleep(0.1)

    # Wave advances to 1, step resets to 0 - this is progress!
    tracker.update(step_counter=0, current_wave=1, num_waiting_reqs=1, num_running_reqs=0)
    assert tracker.last_update_time > initial_time  # Timer should reset
    assert tracker.is_healthy()

    # Even after timeout, since wave advanced, we're still healthy
    time.sleep(0.6)
    assert tracker.is_healthy()


def test_step_counter_reset_without_wave_advancement():
    """Test that step_counter reset without wave advancement is NOT progress."""
    tracker = HealthStateTracker(stall_timeout_seconds=0.5)

    # Start at wave 0, step 100
    tracker.update(step_counter=100, current_wave=0, num_waiting_reqs=1, num_running_reqs=0)
    initial_time = tracker.last_update_time

    time.sleep(0.1)

    # Step counter goes backwards but wave doesn't advance - NOT progress
    tracker.update(step_counter=0, current_wave=0, num_waiting_reqs=1, num_running_reqs=0)
    assert tracker.last_update_time == initial_time  # Timer should NOT reset

    # Should become unhealthy after timeout
    time.sleep(0.5)
    tracker.update(step_counter=0, current_wave=0, num_waiting_reqs=1, num_running_reqs=0)
    assert not tracker.is_healthy()


def test_multiple_wave_advancements():
    """Test that multiple wave advancements are correctly tracked."""
    tracker = HealthStateTracker(stall_timeout_seconds=1.0)

    # Wave 0
    tracker.update(step_counter=10, current_wave=0, num_waiting_reqs=1, num_running_reqs=0)
    assert tracker.is_healthy()

    time.sleep(0.5)

    # Wave 1
    tracker.update(step_counter=0, current_wave=1, num_waiting_reqs=1, num_running_reqs=0)
    assert tracker.is_healthy()

    time.sleep(0.5)

    # Wave 2
    tracker.update(step_counter=0, current_wave=2, num_waiting_reqs=1, num_running_reqs=0)
    assert tracker.is_healthy()

    # Now stall at wave 2 - should become unhealthy
    time.sleep(1.1)
    tracker.update(step_counter=0, current_wave=2, num_waiting_reqs=1, num_running_reqs=0)
    assert not tracker.is_healthy()
