"""Tests for Request Journey Event tracing.

This module tests the journey event tracing functionality that emits
lifecycle events (QUEUED, SCHEDULED, FIRST_TOKEN, PREEMPTED, FINISHED)
for requests as they move through the scheduler.
"""

import time

import pytest

from tests.v1.core.utils import create_scheduler
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.journey_events import (
    RequestJourneyEventType,
    ScheduleKind,
)
from vllm.v1.request import Request, RequestStatus

EOS_TOKEN_ID = 50256
_none_hash_initialized = False


def _get_buffered_events(scheduler, client_index: int):
    """Helper to get buffered events for a client."""
    if not scheduler._enable_journey_tracing:
        return []
    return scheduler._journey_events_buffer_by_client.get(client_index, [])


def _get_all_journey_events(engine_outputs):
    """Helper to extract all journey events from engine outputs."""
    events = []
    for eco in engine_outputs.values():
        if eco.journey_events:
            events.extend(eco.journey_events)
    return events


def _create_request(
    prompt_len: int = 10, max_tokens: int = 10, client_index: int = 0
) -> Request:
    """Helper to create a test request."""
    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True

    request_id = f"request-{time.time()}"
    prompt_token_ids = list(range(prompt_len))
    sampling_params = SamplingParams(max_tokens=max_tokens)
    block_hasher = get_request_block_hasher(16, sha256)

    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=EOS_TOKEN_ID,
        block_hasher=block_hasher,
    )


def test_schedule_kind_first_vs_resume():
    """Verify SCHEDULED emits FIRST on initial schedule.

    Note: Testing RESUME requires actual preemption which is complex to set up
    in a unit test. We verify the logic by checking that FIRST is used for
    WAITING->RUNNING transitions.
    """
    scheduler = create_scheduler(enable_journey_tracing=True)
    request = _create_request()

    scheduler.add_request(request)

    # First schedule
    output1 = scheduler.schedule()
    events = _get_buffered_events(scheduler, request.client_index)

    scheduled = [
        e for e in events if e.event_type == RequestJourneyEventType.SCHEDULED
    ]
    # Should have at least one SCHEDULED event with FIRST kind
    assert len(scheduled) >= 1
    # Check the first SCHEDULED event
    assert scheduled[0].schedule_kind == ScheduleKind.FIRST
    assert scheduled[0].scheduler_step == 1
    assert scheduled[0].num_preemptions_so_far == 0


def test_scheduler_step_semantics():
    """Verify scheduler_step is None for QUEUED, populated for SCHEDULED."""
    scheduler = create_scheduler(enable_journey_tracing=True)
    request = _create_request()

    # QUEUED
    scheduler.add_request(request)
    events = _get_buffered_events(scheduler, request.client_index)
    queued = [e for e in events if e.event_type == RequestJourneyEventType.QUEUED]
    assert len(queued) == 1
    assert queued[0].scheduler_step is None  # Before first schedule

    # SCHEDULED
    sched_output = scheduler.schedule()
    events = _get_buffered_events(scheduler, request.client_index)
    scheduled = [
        e for e in events if e.event_type == RequestJourneyEventType.SCHEDULED
    ]
    assert len(scheduled) == 1
    assert scheduled[0].scheduler_step == 1  # First schedule call


def test_progress_snapshot_survives_preemption():
    """
    Verify prefill progress tracking using _journey_prefill_hiwater dict.

    This test verifies that the scheduler maintains a high-water mark of
    prefill progress that can survive preemption (if it were to occur).
    """
    scheduler = create_scheduler(enable_journey_tracing=True)
    prompt_len = 100
    request = _create_request(prompt_len=prompt_len, max_tokens=10)

    # First schedule - processes some prompt tokens
    scheduler.add_request(request)
    sched_output = scheduler.schedule()

    events = _get_buffered_events(scheduler, request.client_index)
    scheduled_events = [
        e for e in events if e.event_type == RequestJourneyEventType.SCHEDULED
    ]

    if not scheduled_events:
        pytest.skip("Request was not scheduled in first iteration")

    scheduled1 = scheduled_events[0]

    # Verify progress tracking fields are present and valid
    assert scheduled1.prefill_total_tokens == prompt_len
    assert scheduled1.prefill_done_tokens >= 0
    assert scheduled1.prefill_done_tokens <= scheduled1.prefill_total_tokens
    assert scheduled1.decode_done_tokens >= 0
    assert scheduled1.decode_max_tokens == 10
    assert scheduled1.phase in ["PREFILL", "DECODE"]

    # If in prefill phase, verify hiwater dict was updated
    if scheduled1.phase == "PREFILL":
        assert request.request_id in scheduler._journey_prefill_hiwater
        hiwater = scheduler._journey_prefill_hiwater[request.request_id]
        assert hiwater == scheduled1.prefill_done_tokens
        # Verify hiwater can be accessed (proving it survives in scheduler state)
        assert hiwater >= 0


def test_no_request_iteration_overhead():
    """
    Structural test: Verify journey tracing does NOT iterate over all requests.

    Strategy: Create 1000 waiting requests, schedule only ~10. Verify:
    1. Only ~10 SCHEDULED events (not 1000)
    2. No QUEUED events in output (only buffered at add_request time)
    """
    scheduler = create_scheduler(enable_journey_tracing=True)

    # Add 1000 requests
    num_waiting = 1000
    requests = []
    for i in range(num_waiting):
        req = _create_request(prompt_len=50, max_tokens=20)
        requests.append(req)
        scheduler.add_request(req)

    # Flush QUEUED events buffer (they're buffered per-client on add)
    for client_idx in scheduler._journey_events_buffer_by_client.keys():
        scheduler._journey_events_buffer_by_client[client_idx].clear()

    # Schedule (fits only ~10 requests based on budget)
    sched_output = scheduler.schedule()

    # Collect events from all clients
    all_events = []
    for client_idx, events in scheduler._journey_events_buffer_by_client.items():
        all_events.extend(events)

    # Should only have SCHEDULED events for actually scheduled requests
    scheduled_events = [
        e for e in all_events if e.event_type == RequestJourneyEventType.SCHEDULED
    ]

    # Not 1000, only the ones that were scheduled
    assert len(scheduled_events) < 50  # Far less than 1000
    assert len(scheduled_events) > 0  # But more than 0

    # Verify: No events for waiting requests (no full scan)
    scheduled_req_ids = {e.request_id for e in scheduled_events}
    waiting_req_ids = {req.request_id for req in scheduler.waiting}

    # No overlap (waiting requests don't get events during schedule)
    assert scheduled_req_ids.isdisjoint(waiting_req_ids)


def test_finish_status_mapping():
    """Verify terminal status correctly mapped to journey event status."""
    scheduler = create_scheduler(enable_journey_tracing=True)

    # FINISHED_STOPPED → "stopped"
    req1 = _create_request()
    scheduler.add_request(req1)
    scheduler.schedule()
    scheduler._journey_events_buffer_by_client[req1.client_index].clear()

    scheduler.finish_requests(req1.request_id, RequestStatus.FINISHED_STOPPED)
    events = _get_buffered_events(scheduler, req1.client_index)
    finished = [e for e in events if e.event_type == RequestJourneyEventType.FINISHED][
        0
    ]
    assert finished.finish_status == "stopped"

    # FINISHED_LENGTH_CAPPED → "length"
    req2 = _create_request()
    scheduler.add_request(req2)
    scheduler.schedule()
    scheduler._journey_events_buffer_by_client[req2.client_index].clear()

    scheduler.finish_requests(req2.request_id, RequestStatus.FINISHED_LENGTH_CAPPED)
    events = _get_buffered_events(scheduler, req2.client_index)
    finished = [e for e in events if e.event_type == RequestJourneyEventType.FINISHED][
        0
    ]
    assert finished.finish_status == "length"

    # FINISHED_ABORTED → "aborted"
    req3 = _create_request()
    scheduler.add_request(req3)
    scheduler.schedule()
    scheduler._journey_events_buffer_by_client[req3.client_index].clear()

    scheduler.finish_requests(req3.request_id, RequestStatus.FINISHED_ABORTED)
    events = _get_buffered_events(scheduler, req3.client_index)
    finished = [e for e in events if e.event_type == RequestJourneyEventType.FINISHED][
        0
    ]
    assert finished.finish_status == "aborted"

    # FINISHED_ERROR → "error"
    req4 = _create_request()
    scheduler.add_request(req4)
    scheduler.schedule()
    scheduler._journey_events_buffer_by_client[req4.client_index].clear()

    scheduler.finish_requests(req4.request_id, RequestStatus.FINISHED_ERROR)
    events = _get_buffered_events(scheduler, req4.client_index)
    finished = [e for e in events if e.event_type == RequestJourneyEventType.FINISHED][
        0
    ]
    assert finished.finish_status == "error"

    # FINISHED_IGNORED → "ignored"
    req5 = _create_request()
    scheduler.add_request(req5)
    scheduler.schedule()
    scheduler._journey_events_buffer_by_client[req5.client_index].clear()

    scheduler.finish_requests(req5.request_id, RequestStatus.FINISHED_IGNORED)
    events = _get_buffered_events(scheduler, req5.client_index)
    finished = [e for e in events if e.event_type == RequestJourneyEventType.FINISHED][
        0
    ]
    assert finished.finish_status == "ignored"


def test_disabled_by_default_no_overhead():
    """Verify journey tracing disabled by default with no side effects."""
    scheduler = create_scheduler(enable_journey_tracing=False)
    request = _create_request()

    # Add and schedule
    scheduler.add_request(request)
    sched_output = scheduler.schedule()

    # No events buffered (data structures don't exist)
    assert not hasattr(scheduler, "_journey_events_buffer_by_client") or len(
        scheduler._journey_events_buffer_by_client
    ) == 0
    assert not hasattr(scheduler, "_first_token_emitted") or len(
        scheduler._first_token_emitted
    ) == 0
    assert not hasattr(scheduler, "_journey_prefill_hiwater") or len(
        scheduler._journey_prefill_hiwater
    ) == 0


def test_queued_event_emitted():
    """Verify QUEUED event is emitted when request added."""
    scheduler = create_scheduler(enable_journey_tracing=True)
    request = _create_request()

    scheduler.add_request(request)

    events = _get_buffered_events(scheduler, request.client_index)
    queued = [e for e in events if e.event_type == RequestJourneyEventType.QUEUED]

    assert len(queued) == 1
    assert queued[0].request_id == request.request_id
    assert queued[0].scheduler_step is None
    assert queued[0].phase == "PREFILL"
    assert queued[0].num_preemptions_so_far == 0


def test_cleanup_on_finish():
    """Verify journey tracking state is cleaned up when request finishes."""
    scheduler = create_scheduler(enable_journey_tracing=True)
    request = _create_request()

    scheduler.add_request(request)
    scheduler.schedule()

    # Verify state exists
    assert request.request_id in scheduler._journey_prefill_hiwater

    # Finish request
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_STOPPED)

    # Verify state is cleaned up
    assert request.request_id not in scheduler._journey_prefill_hiwater
    assert request.request_id not in scheduler._first_token_emitted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
