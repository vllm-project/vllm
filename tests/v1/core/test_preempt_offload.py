# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for preemption with KV cache offload to CPU.

Behavior under test:
- `preemption_mode="discard"` (default): preemption frees GPU blocks and
  resets num_computed_tokens. Resume recomputes from scratch.
- `preemption_mode="offload"`: identical control flow (PREEMPTED status,
  num_computed_tokens reset), but the connector is asked to flush the
  request's KV cache to CPU first. The standard resume path
  (get_num_new_matched_tokens -> update_state_after_alloc) then hits the
  CPU cache and loads the prior tokens instead of recomputing.
"""

import time
from unittest.mock import MagicMock

import pytest

from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


class TestPreemptionModeConfig:
    """Test preemption_mode configuration."""

    def test_default_preemption_mode_is_discard(self):
        scheduler = create_scheduler()
        assert scheduler.preemption_mode == "discard"

    def test_preemption_mode_offload(self):
        scheduler = create_scheduler(preemption_mode="offload")
        assert scheduler.preemption_mode == "offload"


class TestPreemptRequestMethod:
    """Test the _preempt_request method directly."""

    def _scheduled_request(self, scheduler):
        requests = create_requests(num_requests=1, num_tokens=32)
        request = requests[0]
        scheduler.add_request(request)
        output = scheduler.schedule()
        assert len(output.scheduled_new_reqs) == 1
        request.num_computed_tokens = 16
        return request

    def test_discard_mode(self):
        scheduler = create_scheduler(preemption_mode="discard")
        request = self._scheduled_request(scheduler)

        scheduler._preempt_request(request, time.monotonic())

        assert request.status == RequestStatus.PREEMPTED
        assert request.num_computed_tokens == 0
        assert request.num_preemptions == 1

    def test_offload_mode_no_connector_is_discard_equivalent(self):
        scheduler = create_scheduler(preemption_mode="offload")
        # No connector configured: nothing to call, but the control flow
        # still ends in PREEMPTED + tokens reset.
        assert scheduler.connector is None

        request = self._scheduled_request(scheduler)
        scheduler._preempt_request(request, time.monotonic())

        assert request.status == RequestStatus.PREEMPTED
        assert request.num_computed_tokens == 0
        assert request.num_preemptions == 1

    def test_offload_mode_invokes_connector(self):
        scheduler = create_scheduler(preemption_mode="offload")

        mock_connector = MagicMock()
        mock_connector.offload_preempted_request.return_value = True
        scheduler.connector = mock_connector

        request = self._scheduled_request(scheduler)
        scheduler._preempt_request(request, time.monotonic())

        # Connector was given the chance to flush the request's KV cache.
        mock_connector.offload_preempted_request.assert_called_once_with(request)
        # Control flow converges with discard: PREEMPTED + tokens reset.
        # Reload happens via the standard resume path (lookup + load), not
        # via a separate preempt-offloaded status machine.
        assert request.status == RequestStatus.PREEMPTED
        assert request.num_computed_tokens == 0
        assert request.num_preemptions == 1

    def test_offload_mode_connector_miss_still_preempts(self):
        """Connector returning False is non-fatal; request still preempts."""
        scheduler = create_scheduler(preemption_mode="offload")

        mock_connector = MagicMock()
        mock_connector.offload_preempted_request.return_value = False
        scheduler.connector = mock_connector

        request = self._scheduled_request(scheduler)
        scheduler._preempt_request(request, time.monotonic())

        mock_connector.offload_preempted_request.assert_called_once_with(request)
        assert request.status == RequestStatus.PREEMPTED
        assert request.num_computed_tokens == 0
        assert request.num_preemptions == 1


class TestPreemptionCounter:
    def test_preemption_counter_increments(self):
        scheduler = create_scheduler(preemption_mode="discard")
        requests = create_requests(num_requests=1, num_tokens=32)
        request = requests[0]
        scheduler.add_request(request)
        scheduler.schedule()

        timestamp = time.monotonic()
        for expected in (1, 2, 3):
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = 16
            scheduler._preempt_request(request, timestamp)
            assert request.num_preemptions == expected
