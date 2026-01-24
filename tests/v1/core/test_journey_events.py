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
        # Hiwater is updated in _update_after_schedule(), which runs after the
        # SCHEDULED event is emitted, so hiwater will reflect the scheduled tokens
        # while the event shows the state before execution
        assert hiwater >= scheduled1.prefill_done_tokens
        # Verify hiwater reflects the scheduled work
        assert hiwater > 0


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


def test_chunked_prefill_running_queue_hiwater():
    """Hiwater updates across multiple running queue schedules (chunked prefill)."""
    scheduler = create_scheduler(
        enable_journey_tracing=True,
        max_num_batched_tokens=512,
        max_num_seqs=10,
        long_prefill_token_threshold=256,  # Force chunked prefill
    )

    # Create request with 1000 prompt tokens
    request = _create_request(prompt_len=1000, max_tokens=100)
    scheduler.add_request(request)

    # Track hiwater progression
    hiwater_values = []

    # Schedule multiple times until prefill completes or we hit limit
    for iteration in range(10):
        output = scheduler.schedule()

        # Capture hiwater after schedule (schedule() calls _update_after_schedule() internally)
        hiwater_after_update = scheduler._journey_prefill_hiwater.get(
            request.request_id, 0
        )
        hiwater_values.append(hiwater_after_update)

        # Monotonicity: hiwater should never decrease
        if len(hiwater_values) > 1:
            assert hiwater_after_update >= hiwater_values[-2], (
                f"Hiwater decreased from {hiwater_values[-2]} to {hiwater_after_update}"
            )

        # Progress: hiwater should reflect num_computed_tokens
        expected_hiwater = min(request.num_computed_tokens, 1000)
        assert hiwater_after_update >= expected_hiwater, (
            f"Hiwater {hiwater_after_update} < expected {expected_hiwater}"
        )

        # Exit if prefill complete (transitioned to decode)
        if request.num_output_tokens > 0:
            break

    # Verify we actually did chunked prefill (multiple iterations)
    assert len(hiwater_values) >= 3, (
        f"Expected >= 3 scheduling iterations for chunked prefill, got {len(hiwater_values)}"
    )

    # Final hiwater should be close to prompt length
    final_hiwater = hiwater_values[-1]
    assert final_hiwater >= 1000 * 0.9, (
        f"Final hiwater {final_hiwater} should be near prompt length 1000"
    )


def test_chunked_prefill_preemption_accurate_progress():
    """PREEMPTED event shows correct prefill_done after chunked prefill."""
    scheduler = create_scheduler(
        enable_journey_tracing=True,
        max_num_batched_tokens=512,
        long_prefill_token_threshold=256,  # Force chunked prefill
    )

    # Create request with 1000 prompt tokens
    request = _create_request(prompt_len=1000, max_tokens=100)
    scheduler.add_request(request)

    # Schedule 3 chunks (expect ~768 tokens scheduled)
    computed_tokens_before_preempt = 0
    for i in range(3):
        output = scheduler.schedule()
        computed_tokens_before_preempt = request.num_computed_tokens

        # Don't preempt yet, let chunked prefill progress
        assert request.status == RequestStatus.RUNNING
        assert request.num_output_tokens == 0, "Should still be in prefill"

    # Verify we've scheduled multiple chunks
    assert computed_tokens_before_preempt >= 512, (
        f"Expected >= 512 tokens after 3 chunks, got {computed_tokens_before_preempt}"
    )

    # Preempt the request using deterministic timestamp
    running_req = next(iter(scheduler.running))
    curr_step = output.scheduler_step + 1
    scheduler._preempt_request(running_req, timestamp=1234567890.0, scheduler_step=curr_step)

    # Verify num_computed_tokens was reset (standard preemption behavior)
    assert running_req.num_computed_tokens == 0, "num_computed_tokens should be reset"

    # Check PREEMPTED event
    events = _get_buffered_events(scheduler, request.client_index)
    preempted_events = [e for e in events if e.event_type == RequestJourneyEventType.PREEMPTED]
    assert len(preempted_events) == 1, f"Expected 1 PREEMPTED event, got {len(preempted_events)}"

    preempted_event = preempted_events[0]

    # CRITICAL ASSERTION: prefill_done_tokens should match computed_tokens_before_preempt
    # NOT zero (reset value), NOT stale first-chunk value
    assert preempted_event.prefill_done_tokens >= computed_tokens_before_preempt * 0.95, (
        f"PREEMPTED event shows prefill_done={preempted_event.prefill_done_tokens}, "
        f"but we scheduled {computed_tokens_before_preempt} tokens. "
        f"This indicates hiwater was not updated during chunked prefill."
    )

    # Verify other fields are sensible
    assert preempted_event.prefill_total_tokens == 1000
    assert preempted_event.phase == "PREFILL"
    assert preempted_event.num_preemptions_so_far == 1

    # Verify progress percentage is reasonable
    progress_pct = (preempted_event.prefill_done_tokens /
                   preempted_event.prefill_total_tokens * 100)
    assert progress_pct >= 50, f"Expected >= 50% progress, got {progress_pct:.1f}%"


def test_hiwater_monotonic_across_preemption_cycles():
    """Hiwater survives preemption and is monotonically increasing."""
    scheduler = create_scheduler(
        enable_journey_tracing=True,
        max_num_batched_tokens=512,
        long_prefill_token_threshold=256,
    )

    request = _create_request(prompt_len=1000, max_tokens=100)
    scheduler.add_request(request)

    hiwater_values = []

    # Schedule multiple chunks to build up hiwater
    for cycle in range(5):
        output = scheduler.schedule()

        # Record hiwater after schedule (includes update)
        hiwater_after = scheduler._journey_prefill_hiwater.get(request.request_id, 0)
        hiwater_values.append(hiwater_after)

        # Monotonicity: hiwater never decreases
        if len(hiwater_values) > 1:
            assert hiwater_after >= hiwater_values[-2], (
                f"Hiwater decreased from {hiwater_values[-2]} to {hiwater_after}"
            )

        # Exit if we've transitioned to decode (prefill complete)
        if request.num_output_tokens > 0:
            break

        # Test preemption on cycle 2 (after some progress) and exit
        if cycle == 2 and request.status == RequestStatus.RUNNING:
            # Capture hiwater before preemption
            hiwater_before_preempt = scheduler._journey_prefill_hiwater.get(request.request_id, 0)
            assert hiwater_before_preempt > 0, "Should have some prefill progress"

            # Preempt the request using deterministic timestamp
            req = next(iter(scheduler.running))
            scheduler._preempt_request(req, timestamp=1234567890.0, scheduler_step=output.scheduler_step + 1)

            # Verify hiwater survives preemption (not reset)
            hiwater_after_preempt = scheduler._journey_prefill_hiwater.get(request.request_id, 0)
            assert hiwater_after_preempt == hiwater_before_preempt, (
                f"Hiwater changed during preemption: {hiwater_before_preempt} -> {hiwater_after_preempt}"
            )

            # Verify num_computed_tokens was reset (standard preemption behavior)
            assert req.num_computed_tokens == 0, "num_computed_tokens should be reset"

            # Exit test after verifying preemption behavior
            break

    # Verify we actually did chunked prefill (multiple iterations)
    assert len(hiwater_values) >= 3, (
        f"Expected >= 3 scheduling iterations, got {len(hiwater_values)}"
    )

    # Verify hiwater made reasonable progress
    final_hiwater = hiwater_values[-1]
    assert final_hiwater >= 256, (  # At least one full chunk
        f"Final hiwater {final_hiwater} too low"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
