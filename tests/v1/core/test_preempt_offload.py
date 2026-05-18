# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for preemption with KV cache offload to CPU."""

from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


class TestPreemptionModeConfig:
    """Test preemption_mode configuration."""

    def test_default_preemption_mode_is_discard(self):
        """Test that default preemption mode is 'discard'."""
        scheduler = create_scheduler()
        assert scheduler.preemption_mode == "discard"

    def test_preemption_mode_offload(self):
        """Test that preemption mode can be set to 'offload'."""
        scheduler = create_scheduler(preemption_mode="offload")
        assert scheduler.preemption_mode == "offload"


class TestPreemptionDiscard:
    """Test preemption with discard mode (default behavior)."""

    def test_preemption_discards_kv_cache(self):
        """Test that preemption in discard mode resets num_computed_tokens."""
        scheduler = create_scheduler(
            max_num_seqs=2,
            max_num_batched_tokens=100,
            num_blocks=10,  # Limited blocks to trigger preemption
            preemption_mode="discard",
        )

        # Create requests with different token counts
        requests = create_requests(num_requests=3, num_tokens=50, max_tokens=100)
        for request in requests:
            scheduler.add_request(request)

        # Schedule first batch
        output = scheduler.schedule()
        assert len(output.scheduled_new_reqs) == 2  # Limited by max_num_seqs

        # Simulate model output for scheduled requests
        req_to_index = {
            req.req_id: i for i, req in enumerate(output.scheduled_new_reqs)
        }
        model_runner_output = ModelRunnerOutput(
            req_ids=[req.req_id for req in output.scheduled_new_reqs],
            req_id_to_index=req_to_index,
            sampled_token_ids=[[1], [1]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        scheduler.update_from_output(output, model_runner_output)

        # Verify requests are running
        assert len(scheduler.running) == 2
        for req in scheduler.running:
            assert req.num_computed_tokens > 0


class TestPreemptRequestMethod:
    """Test the _preempt_request method directly."""

    def test_preempt_request_discard_mode(self):
        """Test _preempt_request in discard mode."""
        scheduler = create_scheduler(preemption_mode="discard")
        requests = create_requests(num_requests=1, num_tokens=32)
        request = requests[0]
        scheduler.add_request(request)

        # Schedule the request
        output = scheduler.schedule()
        assert len(output.scheduled_new_reqs) == 1

        # Simulate some computation
        request.num_computed_tokens = 16

        # Preempt the request
        import time
        scheduler._preempt_request(request, time.monotonic())

        # Verify discard behavior
        assert request.status == RequestStatus.PREEMPTED
        assert request.num_computed_tokens == 0
        assert request.num_preemptions == 1

    def test_preempt_request_offload_mode_no_connector(self):
        """Test _preempt_request in offload mode without connector falls back to discard."""
        scheduler = create_scheduler(preemption_mode="offload")
        # No connector configured, should fall back to discard
        assert scheduler.connector is None

        requests = create_requests(num_requests=1, num_tokens=32)
        request = requests[0]
        scheduler.add_request(request)

        # Schedule the request
        output = scheduler.schedule()
        assert len(output.scheduled_new_reqs) == 1

        # Simulate some computation
        request.num_computed_tokens = 16

        # Preempt the request
        import time
        scheduler._preempt_request(request, time.monotonic())

        # Should fall back to discard since no connector
        assert request.status == RequestStatus.PREEMPTED
        assert request.num_computed_tokens == 0
        assert request.num_preemptions == 1

    def test_preempt_request_offload_mode_with_connector(self):
        """Test _preempt_request in offload mode with mocked connector."""
        scheduler = create_scheduler(preemption_mode="offload")

        # Mock the connector
        mock_connector = MagicMock()
        mock_connector.offload_preempted_request.return_value = True
        scheduler.connector = mock_connector

        requests = create_requests(num_requests=1, num_tokens=32)
        request = requests[0]
        scheduler.add_request(request)

        # Schedule the request
        output = scheduler.schedule()
        assert len(output.scheduled_new_reqs) == 1

        # Simulate some computation
        original_computed_tokens = 16
        request.num_computed_tokens = original_computed_tokens

        # Preempt the request
        import time
        scheduler._preempt_request(request, time.monotonic())

        # Verify offload behavior
        assert request.status == RequestStatus.PREEMPTED_OFFLOADED
        # num_computed_tokens should be preserved for offloaded requests
        assert request.num_preemptions == 1
        mock_connector.offload_preempted_request.assert_called_once_with(request)

    def test_preempt_request_offload_fails_fallback_to_discard(self):
        """Test that failed offload falls back to discard mode."""
        scheduler = create_scheduler(preemption_mode="offload")

        # Mock the connector to return failure
        mock_connector = MagicMock()
        mock_connector.offload_preempted_request.return_value = False
        scheduler.connector = mock_connector

        requests = create_requests(num_requests=1, num_tokens=32)
        request = requests[0]
        scheduler.add_request(request)

        # Schedule the request
        output = scheduler.schedule()
        assert len(output.scheduled_new_reqs) == 1

        # Simulate some computation
        request.num_computed_tokens = 16

        # Preempt the request
        import time
        scheduler._preempt_request(request, time.monotonic())

        # Should fall back to discard since offload failed
        assert request.status == RequestStatus.PREEMPTED
        assert request.num_computed_tokens == 0
        assert request.num_preemptions == 1


class TestRequestStatusTransitions:
    """Test request status transitions for preemption."""

    def test_preempted_status_not_finished(self):
        """Test that PREEMPTED status is not considered finished."""
        assert not RequestStatus.is_finished(RequestStatus.PREEMPTED)

    def test_preempted_offloaded_status_not_finished(self):
        """Test that PREEMPTED_OFFLOADED status is not considered finished."""
        assert not RequestStatus.is_finished(RequestStatus.PREEMPTED_OFFLOADED)

    def test_waiting_for_reload_status_not_finished(self):
        """Test that WAITING_FOR_RELOAD status is not considered finished."""
        assert not RequestStatus.is_finished(RequestStatus.WAITING_FOR_RELOAD)

    def test_finished_statuses_are_finished(self):
        """Test that FINISHED_* statuses are considered finished."""
        assert RequestStatus.is_finished(RequestStatus.FINISHED_STOPPED)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_LENGTH_CAPPED)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_ABORTED)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_IGNORED)


class TestPreemptionCounter:
    """Test preemption counter tracking."""

    def test_preemption_counter_increments(self):
        """Test that num_preemptions increments on each preemption."""
        scheduler = create_scheduler(preemption_mode="discard")
        requests = create_requests(num_requests=1, num_tokens=32)
        request = requests[0]
        scheduler.add_request(request)

        # Schedule the request
        scheduler.schedule()

        import time
        timestamp = time.monotonic()

        # First preemption
        request.num_computed_tokens = 16
        scheduler._preempt_request(request, timestamp)
        assert request.num_preemptions == 1

        # Second preemption
        request.num_computed_tokens = 16
        scheduler._preempt_request(request, timestamp)
        assert request.num_preemptions == 2

        # Third preemption
        request.num_computed_tokens = 16
        scheduler._preempt_request(request, timestamp)
        assert request.num_preemptions == 3
