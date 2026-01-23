# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for journey event integration with OTEL tracing."""

import time
from unittest.mock import Mock

import pytest

from vllm.sampling_params import SamplingParams
from vllm.tracing import is_otel_available
from vllm.v1.core.sched.journey_events import (
    RequestJourneyEvent,
    RequestJourneyEventType,
    ScheduleKind,
)
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.metrics.stats import IterationStats


def test_journey_events_accumulation():
    """Test that journey events are correctly accumulated in RequestState."""
    output_processor = OutputProcessor(tokenizer=None, log_stats=False)

    # Create a request
    request = EngineCoreRequest(
        request_id="request-0-int",
        external_req_id="request-0",
        prompt_token_ids=[1, 2, 3, 4, 5],
        mm_features=None,
        eos_token_id=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
    )

    # Add request to output processor
    output_processor.add_request(request, None)

    # Create journey events
    ts = time.monotonic()
    journey_events = [
        RequestJourneyEvent(
            request_id=request.request_id,
            event_type=RequestJourneyEventType.QUEUED,
            ts_monotonic=ts,
            scheduler_step=None,
            prefill_done_tokens=0,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=0,
            schedule_kind=None,
            finish_status=None,
        ),
        RequestJourneyEvent(
            request_id=request.request_id,
            event_type=RequestJourneyEventType.SCHEDULED,
            ts_monotonic=ts + 0.1,
            scheduler_step=1,
            prefill_done_tokens=0,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=0,
            schedule_kind=ScheduleKind.FIRST,
            finish_status=None,
        ),
    ]

    # Create mock engine core output
    engine_output = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[42],
    )

    # Process outputs with journey events
    output_processor.process_outputs([engine_output], journey_events=journey_events)

    # Verify events were accumulated
    req_state = output_processor.request_states.get("request-0-int")
    assert req_state is not None
    assert len(req_state.journey_events) == 2
    assert req_state.journey_events[0].event_type == RequestJourneyEventType.QUEUED
    assert (
        req_state.journey_events[1].event_type == RequestJourneyEventType.SCHEDULED
    )


def test_journey_events_accumulation_across_iterations():
    """Test that journey events accumulate across multiple process_outputs calls."""
    output_processor = OutputProcessor(tokenizer=None, log_stats=False)

    request = EngineCoreRequest(
        request_id="request-0-int",
        external_req_id="request-0",
        prompt_token_ids=[1, 2, 3, 4, 5],
        mm_features=None,
        eos_token_id=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
    )

    output_processor.add_request(request, None)

    ts = time.monotonic()

    # First iteration: QUEUED event
    journey_events_1 = [
        RequestJourneyEvent(
            request_id=request.request_id,
            event_type=RequestJourneyEventType.QUEUED,
            ts_monotonic=ts,
            scheduler_step=None,
            prefill_done_tokens=0,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=0,
            schedule_kind=None,
            finish_status=None,
        ),
    ]

    engine_output_1 = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[42],
    )
    output_processor.process_outputs([engine_output_1], journey_events=journey_events_1)

    # Second iteration: SCHEDULED event
    journey_events_2 = [
        RequestJourneyEvent(
            request_id=request.request_id,
            event_type=RequestJourneyEventType.SCHEDULED,
            ts_monotonic=ts + 0.1,
            scheduler_step=1,
            prefill_done_tokens=0,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=0,
            schedule_kind=ScheduleKind.FIRST,
            finish_status=None,
        ),
    ]

    engine_output_2 = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[43],
    )
    output_processor.process_outputs([engine_output_2], journey_events=journey_events_2)

    # Verify both events were accumulated
    req_state = output_processor.request_states.get("request-0-int")
    assert req_state is not None
    assert len(req_state.journey_events) == 2
    assert req_state.journey_events[0].event_type == RequestJourneyEventType.QUEUED
    assert (
        req_state.journey_events[1].event_type == RequestJourneyEventType.SCHEDULED
    )


def test_journey_events_ignored_for_unknown_requests():
    """Test that journey events for unknown request IDs are ignored."""
    output_processor = OutputProcessor(tokenizer=None, log_stats=False)

    request = EngineCoreRequest(
        request_id="request-0-int",
        external_req_id="request-0",
        prompt_token_ids=[1, 2, 3, 4, 5],
        mm_features=None,
        eos_token_id=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
    )

    output_processor.add_request(request, None)

    ts = time.monotonic()

    # Create journey events for known and unknown requests
    journey_events = [
        RequestJourneyEvent(
            request_id="request-0-int",  # Known
            event_type=RequestJourneyEventType.QUEUED,
            ts_monotonic=ts,
            scheduler_step=None,
            prefill_done_tokens=0,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=0,
            schedule_kind=None,
            finish_status=None,
        ),
        RequestJourneyEvent(
            request_id="unknown-request",  # Unknown
            event_type=RequestJourneyEventType.QUEUED,
            ts_monotonic=ts,
            scheduler_step=None,
            prefill_done_tokens=0,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=0,
            schedule_kind=None,
            finish_status=None,
        ),
    ]

    engine_output = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[42],
    )

    # Should not raise an error
    output_processor.process_outputs([engine_output], journey_events=journey_events)

    # Verify only the known request's event was accumulated
    req_state = output_processor.request_states.get("request-0-int")
    assert req_state is not None
    assert len(req_state.journey_events) == 1
    assert req_state.journey_events[0].request_id == "request-0-int"


@pytest.mark.skipif(not is_otel_available(), reason="Requires OpenTelemetry")
def test_otel_journey_events_span_events():
    """Test that journey events are added as OTEL span events."""
    output_processor = OutputProcessor(tokenizer=None, log_stats=True)

    # Mock the tracer
    mock_tracer = Mock()
    mock_span = Mock()
    mock_cm = Mock()
    mock_cm.__enter__ = Mock(return_value=mock_span)
    mock_cm.__exit__ = Mock(return_value=None)
    mock_tracer.start_as_current_span.return_value = mock_cm
    output_processor.tracer = mock_tracer

    request = EngineCoreRequest(
        request_id="request-0-int",
        external_req_id="request-0",
        prompt_token_ids=[1, 2, 3, 4, 5],
        mm_features=None,
        eos_token_id=None,
        arrival_time=time.monotonic(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
    )

    output_processor.add_request(request, None)

    ts = time.monotonic()

    # Create comprehensive journey events
    journey_events = [
        RequestJourneyEvent(
            request_id=request.request_id,
            event_type=RequestJourneyEventType.QUEUED,
            ts_monotonic=ts,
            scheduler_step=None,
            prefill_done_tokens=0,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=0,
            schedule_kind=None,
            finish_status=None,
        ),
        RequestJourneyEvent(
            request_id=request.request_id,
            event_type=RequestJourneyEventType.SCHEDULED,
            ts_monotonic=ts + 0.1,
            scheduler_step=1,
            prefill_done_tokens=0,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=0,
            schedule_kind=ScheduleKind.FIRST,
            finish_status=None,
        ),
        RequestJourneyEvent(
            request_id=request.request_id,
            event_type=RequestJourneyEventType.FIRST_TOKEN,
            ts_monotonic=ts + 0.2,
            scheduler_step=2,
            prefill_done_tokens=10,
            prefill_total_tokens=10,
            decode_done_tokens=1,
            decode_max_tokens=50,
            phase="DECODE",
            num_preemptions_so_far=0,
            schedule_kind=None,
            finish_status=None,
        ),
    ]

    # Create finished output
    engine_output = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[42, 43, 44],
        finish_reason=FinishReason.LENGTH,
    )

    output_processor.process_outputs(
        [engine_output],
        engine_core_timestamp=time.monotonic(),
        iteration_stats=IterationStats(),
        journey_events=journey_events,
    )

    # Verify tracer was called
    assert mock_tracer.start_as_current_span.called

    # Verify span.add_event was called for each journey event
    add_event_calls = mock_span.add_event.call_args_list
    assert len(add_event_calls) == 3

    # Verify QUEUED event
    queued_call = add_event_calls[0]
    assert queued_call[1]["name"] == "journey.QUEUED"
    queued_attrs = queued_call[1]["attributes"]
    assert queued_attrs["event.type"] == "QUEUED"
    assert queued_attrs["ts.monotonic"] == ts
    assert "scheduler.step" not in queued_attrs  # None values excluded
    assert queued_attrs["phase"] == "PREFILL"
    assert queued_attrs["prefill.done_tokens"] == 0
    assert queued_attrs["prefill.total_tokens"] == 10

    # Verify SCHEDULED event
    scheduled_call = add_event_calls[1]
    assert scheduled_call[1]["name"] == "journey.SCHEDULED"
    scheduled_attrs = scheduled_call[1]["attributes"]
    assert scheduled_attrs["event.type"] == "SCHEDULED"
    assert scheduled_attrs["scheduler.step"] == 1
    assert scheduled_attrs["schedule.kind"] == "FIRST"

    # Verify FIRST_TOKEN event
    first_token_call = add_event_calls[2]
    assert first_token_call[1]["name"] == "journey.FIRST_TOKEN"
    first_token_attrs = first_token_call[1]["attributes"]
    assert first_token_attrs["event.type"] == "FIRST_TOKEN"
    assert first_token_attrs["phase"] == "DECODE"
    assert first_token_attrs["decode.done_tokens"] == 1

    # Verify request was removed from request_states (finished requests are cleaned up)
    req_state = output_processor.request_states.get("request-0-int")
    assert req_state is None  # Request state removed after completion


@pytest.mark.skipif(not is_otel_available(), reason="Requires OpenTelemetry")
def test_otel_journey_events_with_preemption():
    """Test that journey events with preemption info are correctly exported."""
    output_processor = OutputProcessor(tokenizer=None, log_stats=True)

    # Mock the tracer
    mock_tracer = Mock()
    mock_span = Mock()
    mock_cm = Mock()
    mock_cm.__enter__ = Mock(return_value=mock_span)
    mock_cm.__exit__ = Mock(return_value=None)
    mock_tracer.start_as_current_span.return_value = mock_cm
    output_processor.tracer = mock_tracer

    request = EngineCoreRequest(
        request_id="request-0-int",
        external_req_id="request-0",
        prompt_token_ids=[1, 2, 3, 4, 5],
        mm_features=None,
        eos_token_id=None,
        arrival_time=time.monotonic(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
    )

    output_processor.add_request(request, None)

    ts = time.monotonic()

    # Create journey events with preemption
    journey_events = [
        RequestJourneyEvent(
            request_id=request.request_id,
            event_type=RequestJourneyEventType.PREEMPTED,
            ts_monotonic=ts,
            scheduler_step=5,
            prefill_done_tokens=5,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=1,
            schedule_kind=None,
            finish_status=None,
        ),
        RequestJourneyEvent(
            request_id=request.request_id,
            event_type=RequestJourneyEventType.SCHEDULED,
            ts_monotonic=ts + 0.1,
            scheduler_step=10,
            prefill_done_tokens=5,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=1,
            schedule_kind=ScheduleKind.RESUME,
            finish_status=None,
        ),
        RequestJourneyEvent(
            request_id=request.request_id,
            event_type=RequestJourneyEventType.FINISHED,
            ts_monotonic=ts + 0.2,
            scheduler_step=15,
            prefill_done_tokens=10,
            prefill_total_tokens=10,
            decode_done_tokens=50,
            decode_max_tokens=50,
            phase="DECODE",
            num_preemptions_so_far=1,
            schedule_kind=None,
            finish_status="length",
        ),
    ]

    # Create finished output
    engine_output = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[42, 43, 44],
        finish_reason=FinishReason.LENGTH,
    )

    output_processor.process_outputs(
        [engine_output],
        engine_core_timestamp=time.monotonic(),
        iteration_stats=IterationStats(),
        journey_events=journey_events,
    )

    # Verify span events were added
    add_event_calls = mock_span.add_event.call_args_list
    assert len(add_event_calls) == 3

    # Verify PREEMPTED event
    preempted_call = add_event_calls[0]
    assert preempted_call[1]["name"] == "journey.PREEMPTED"
    preempted_attrs = preempted_call[1]["attributes"]
    assert preempted_attrs["num_preemptions"] == 1

    # Verify SCHEDULED with RESUME
    scheduled_call = add_event_calls[1]
    scheduled_attrs = scheduled_call[1]["attributes"]
    assert scheduled_attrs["schedule.kind"] == "RESUME"
    assert scheduled_attrs["num_preemptions"] == 1

    # Verify FINISHED event
    finished_call = add_event_calls[2]
    assert finished_call[1]["name"] == "journey.FINISHED"
    finished_attrs = finished_call[1]["attributes"]
    assert finished_attrs["finish.status"] == "length"

    # Verify request was removed from request_states (finished requests are cleaned up)
    req_state = output_processor.request_states.get("request-0-int")
    assert req_state is None  # Request state removed after completion


def test_otel_journey_events_without_tracer():
    """Test that journey events are accumulated but not exported without tracer."""
    output_processor = OutputProcessor(tokenizer=None, log_stats=False)
    # No tracer set
    assert output_processor.tracer is None

    request = EngineCoreRequest(
        request_id="request-0-int",
        external_req_id="request-0",
        prompt_token_ids=[1, 2, 3, 4, 5],
        mm_features=None,
        eos_token_id=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
    )

    output_processor.add_request(request, None)

    ts = time.monotonic()
    journey_events = [
        RequestJourneyEvent(
            request_id=request.request_id,
            event_type=RequestJourneyEventType.QUEUED,
            ts_monotonic=ts,
            scheduler_step=None,
            prefill_done_tokens=0,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=0,
            schedule_kind=None,
            finish_status=None,
        ),
    ]

    engine_output = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[42],
    )

    # Process outputs - should not raise an error
    output_processor.process_outputs([engine_output], journey_events=journey_events)

    # Verify events were accumulated (request not finished yet, so not cleared)
    req_state = output_processor.request_states.get("request-0-int")
    assert req_state is not None
    assert len(req_state.journey_events) == 1


@pytest.mark.skipif(not is_otel_available(), reason="Requires OpenTelemetry")
def test_otel_journey_events_not_exported_when_span_not_recording():
    """Test that journey events are not exported when span is not recording."""
    output_processor = OutputProcessor(tokenizer=None, log_stats=True)

    # Mock the tracer with a span that is NOT recording
    mock_tracer = Mock()
    mock_span = Mock()
    mock_span.is_recording.return_value = False  # Span not recording
    mock_cm = Mock()
    mock_cm.__enter__ = Mock(return_value=mock_span)
    mock_cm.__exit__ = Mock(return_value=None)
    mock_tracer.start_as_current_span.return_value = mock_cm
    output_processor.tracer = mock_tracer

    request = EngineCoreRequest(
        request_id="request-0-int",
        external_req_id="request-0",
        prompt_token_ids=[1, 2, 3, 4, 5],
        mm_features=None,
        eos_token_id=None,
        arrival_time=time.monotonic(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
    )

    output_processor.add_request(request, None)

    ts = time.monotonic()
    journey_events = [
        RequestJourneyEvent(
            request_id=request.request_id,
            event_type=RequestJourneyEventType.QUEUED,
            ts_monotonic=ts,
            scheduler_step=None,
            prefill_done_tokens=0,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=0,
            schedule_kind=None,
            finish_status=None,
        ),
    ]

    engine_output = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[42, 43],
        finish_reason=FinishReason.LENGTH,
    )

    output_processor.process_outputs(
        [engine_output],
        engine_core_timestamp=time.monotonic(),
        iteration_stats=IterationStats(),
        journey_events=journey_events,
    )

    # Verify span.is_recording() was checked
    assert mock_span.is_recording.called

    # Verify span.add_event was NOT called (because span not recording)
    assert not mock_span.add_event.called

    # Verify request was removed from request_states (finished requests are cleaned up)
    req_state = output_processor.request_states.get("request-0-int")
    assert req_state is None  # Request state removed after completion


@pytest.mark.skipif(not is_otel_available(), reason="Requires OpenTelemetry")
def test_otel_journey_events_no_duplication_across_iterations():
    """Test that journey events are not duplicated when exported across multiple iterations."""
    output_processor = OutputProcessor(tokenizer=None, log_stats=True)

    # Mock the tracer
    mock_tracer = Mock()
    mock_span = Mock()
    mock_cm = Mock()
    mock_cm.__enter__ = Mock(return_value=mock_span)
    mock_cm.__exit__ = Mock(return_value=None)
    mock_tracer.start_as_current_span.return_value = mock_cm
    output_processor.tracer = mock_tracer

    request = EngineCoreRequest(
        request_id="request-0-int",
        external_req_id="request-0",
        prompt_token_ids=[1, 2, 3, 4, 5],
        mm_features=None,
        eos_token_id=None,
        arrival_time=time.monotonic(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
    )

    output_processor.add_request(request, None)

    ts = time.monotonic()

    # Iteration 1: QUEUED event
    journey_events_1 = [
        RequestJourneyEvent(
            request_id=request.request_id,
            event_type=RequestJourneyEventType.QUEUED,
            ts_monotonic=ts,
            scheduler_step=None,
            prefill_done_tokens=0,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=0,
            schedule_kind=None,
            finish_status=None,
        ),
    ]

    engine_output_1 = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[42],
    )
    output_processor.process_outputs(
        [engine_output_1],
        engine_core_timestamp=time.monotonic(),
        iteration_stats=IterationStats(),
        journey_events=journey_events_1,
    )

    # Request not finished yet, so no tracing call
    assert not mock_tracer.start_as_current_span.called

    # Iteration 2: SCHEDULED event
    journey_events_2 = [
        RequestJourneyEvent(
            request_id=request.request_id,
            event_type=RequestJourneyEventType.SCHEDULED,
            ts_monotonic=ts + 0.1,
            scheduler_step=1,
            prefill_done_tokens=0,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=0,
            schedule_kind=ScheduleKind.FIRST,
            finish_status=None,
        ),
    ]

    engine_output_2 = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[43],
    )
    output_processor.process_outputs(
        [engine_output_2],
        engine_core_timestamp=time.monotonic(),
        iteration_stats=IterationStats(),
        journey_events=journey_events_2,
    )

    # Still not finished
    assert not mock_tracer.start_as_current_span.called

    # Iteration 3: FIRST_TOKEN event + FINISH
    journey_events_3 = [
        RequestJourneyEvent(
            request_id=request.request_id,
            event_type=RequestJourneyEventType.FIRST_TOKEN,
            ts_monotonic=ts + 0.2,
            scheduler_step=2,
            prefill_done_tokens=10,
            prefill_total_tokens=10,
            decode_done_tokens=1,
            decode_max_tokens=50,
            phase="DECODE",
            num_preemptions_so_far=0,
            schedule_kind=None,
            finish_status=None,
        ),
    ]

    engine_output_3 = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[44],
        finish_reason=FinishReason.LENGTH,
    )
    output_processor.process_outputs(
        [engine_output_3],
        engine_core_timestamp=time.monotonic(),
        iteration_stats=IterationStats(),
        journey_events=journey_events_3,
    )

    # Now finished, tracer should be called ONCE
    assert mock_tracer.start_as_current_span.call_count == 1

    # Verify span.add_event was called for ALL THREE events (no duplication)
    add_event_calls = mock_span.add_event.call_args_list
    assert len(add_event_calls) == 3, f"Expected 3 events, got {len(add_event_calls)}"

    # Verify the events are the right ones (one of each type)
    event_names = [call[1]["name"] for call in add_event_calls]
    assert "journey.QUEUED" in event_names
    assert "journey.SCHEDULED" in event_names
    assert "journey.FIRST_TOKEN" in event_names

    # Verify no duplicates
    assert len(event_names) == len(set(event_names)), "Duplicate events detected!"


@pytest.mark.skipif(not is_otel_available(), reason="Requires OpenTelemetry")
def test_otel_journey_events_cleared_after_each_do_tracing_call():
    """Test that journey events are cleared after each do_tracing call to prevent duplication."""
    output_processor = OutputProcessor(tokenizer=None, log_stats=True)

    # Mock the tracer
    mock_tracer = Mock()
    output_processor.tracer = mock_tracer

    # We'll simulate multiple finish calls with do_tracing to verify clearing

    request = EngineCoreRequest(
        request_id="request-0-int",
        external_req_id="request-0",
        prompt_token_ids=[1, 2, 3, 4, 5],
        mm_features=None,
        eos_token_id=None,
        arrival_time=time.monotonic(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
    )

    output_processor.add_request(request, None)
    req_state = output_processor.request_states["request-0-int"]

    ts = time.monotonic()

    # Add some journey events
    req_state.journey_events.extend(
        [
            RequestJourneyEvent(
                request_id=request.request_id,
                event_type=RequestJourneyEventType.QUEUED,
                ts_monotonic=ts,
                scheduler_step=None,
                prefill_done_tokens=0,
                prefill_total_tokens=10,
                decode_done_tokens=0,
                decode_max_tokens=50,
                phase="PREFILL",
                num_preemptions_so_far=0,
                schedule_kind=None,
                finish_status=None,
            ),
        ]
    )

    # Mock span
    mock_span = Mock()
    mock_cm = Mock()
    mock_cm.__enter__ = Mock(return_value=mock_span)
    mock_cm.__exit__ = Mock(return_value=None)
    mock_tracer.start_as_current_span.return_value = mock_cm

    engine_output = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[42],
        finish_reason=FinishReason.LENGTH,
    )

    # Call do_tracing directly
    output_processor.do_tracing(
        engine_output, req_state, IterationStats()
    )

    # Verify events were exported
    assert mock_span.add_event.call_count == 1

    # CRITICAL: Verify events were cleared after export
    assert len(req_state.journey_events) == 0, "Events not cleared after do_tracing!"

    # Add more events
    req_state.journey_events.append(
        RequestJourneyEvent(
            request_id=request.request_id,
            event_type=RequestJourneyEventType.SCHEDULED,
            ts_monotonic=ts + 0.1,
            scheduler_step=1,
            prefill_done_tokens=0,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=0,
            schedule_kind=ScheduleKind.FIRST,
            finish_status=None,
        )
    )

    # Call do_tracing again
    output_processor.do_tracing(
        engine_output, req_state, IterationStats()
    )

    # Verify only the NEW event was exported (total should be 2, not 3)
    assert mock_span.add_event.call_count == 2

    # Verify events cleared again
    assert len(req_state.journey_events) == 0


def test_journey_events_cleared_on_finish_without_tracer():
    """Test that journey events are cleared when request finishes even without tracer."""
    output_processor = OutputProcessor(tokenizer=None, log_stats=False)
    # No tracer set
    assert output_processor.tracer is None

    request = EngineCoreRequest(
        request_id="request-0-int",
        external_req_id="request-0",
        prompt_token_ids=[1, 2, 3, 4, 5],
        mm_features=None,
        eos_token_id=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
    )

    output_processor.add_request(request, None)

    ts = time.monotonic()

    # Add events across multiple iterations
    journey_events_1 = [
        RequestJourneyEvent(
            request_id=request.request_id,
            event_type=RequestJourneyEventType.QUEUED,
            ts_monotonic=ts,
            scheduler_step=None,
            prefill_done_tokens=0,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=0,
            schedule_kind=None,
            finish_status=None,
        ),
    ]

    engine_output_1 = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[42],
    )
    output_processor.process_outputs([engine_output_1], journey_events=journey_events_1)

    # Events should be accumulated (not finished yet)
    req_state = output_processor.request_states.get("request-0-int")
    assert req_state is not None
    assert len(req_state.journey_events) == 1

    # Add more events
    journey_events_2 = [
        RequestJourneyEvent(
            request_id=request.request_id,
            event_type=RequestJourneyEventType.SCHEDULED,
            ts_monotonic=ts + 0.1,
            scheduler_step=1,
            prefill_done_tokens=0,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=0,
            schedule_kind=ScheduleKind.FIRST,
            finish_status=None,
        ),
    ]

    engine_output_2 = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[43],
    )
    output_processor.process_outputs([engine_output_2], journey_events=journey_events_2)

    # Both events should be accumulated
    req_state = output_processor.request_states.get("request-0-int")
    assert req_state is not None
    assert len(req_state.journey_events) == 2

    # Finish the request
    engine_output_3 = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[44],
        finish_reason=FinishReason.LENGTH,
    )
    output_processor.process_outputs([engine_output_3], journey_events=[])

    # Request should be removed (finished and cleaned up)
    req_state = output_processor.request_states.get("request-0-int")
    assert req_state is None  # Request state removed, events cleared


def test_journey_events_without_outputs_are_accumulated():
    """Test that events for requests without outputs are still accumulated (QUEUED events)."""
    output_processor = OutputProcessor(tokenizer=None, log_stats=False)

    # Add two requests
    request1 = EngineCoreRequest(
        request_id="request-1-int",
        external_req_id="request-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        eos_token_id=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
    )

    request2 = EngineCoreRequest(
        request_id="request-2-int",
        external_req_id="request-2",
        prompt_token_ids=[4, 5, 6],
        mm_features=None,
        eos_token_id=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
    )

    output_processor.add_request(request1, None)
    output_processor.add_request(request2, None)

    ts = time.monotonic()

    # Create events for both requests, but only request-1 has an output
    journey_events = [
        RequestJourneyEvent(
            request_id="request-1-int",
            event_type=RequestJourneyEventType.QUEUED,
            ts_monotonic=ts,
            scheduler_step=None,
            prefill_done_tokens=0,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=0,
            schedule_kind=None,
            finish_status=None,
        ),
        RequestJourneyEvent(
            request_id="request-2-int",  # No output for this request
            event_type=RequestJourneyEventType.QUEUED,
            ts_monotonic=ts,
            scheduler_step=None,
            prefill_done_tokens=0,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=0,
            schedule_kind=None,
            finish_status=None,
        ),
    ]

    # Only request-1 has an output
    engine_outputs = [
        EngineCoreOutput(
            request_id="request-1-int",
            new_token_ids=[42],
        )
    ]

    output_processor.process_outputs(engine_outputs, journey_events=journey_events)

    # CRITICAL: Verify both requests have their events accumulated
    req_state1 = output_processor.request_states.get("request-1-int")
    assert req_state1 is not None
    assert len(req_state1.journey_events) == 1
    assert req_state1.journey_events[0].event_type == RequestJourneyEventType.QUEUED

    req_state2 = output_processor.request_states.get("request-2-int")
    assert req_state2 is not None
    assert len(req_state2.journey_events) == 1, "QUEUED event for request without output was lost!"
    assert req_state2.journey_events[0].event_type == RequestJourneyEventType.QUEUED


def test_journey_events_with_async_chunking():
    """Test that journey events are correctly handled with AsyncLLM-style chunking.
    
    This simulates the AsyncLLM pattern where outputs are split into chunks
    but all events must be preserved.
    """
    output_processor = OutputProcessor(tokenizer=None, log_stats=False)

    # Create 3 requests
    requests = []
    for i in range(3):
        request = EngineCoreRequest(
            request_id=f"request-{i}-int",
            external_req_id=f"request-{i}",
            prompt_token_ids=[1, 2, 3],
            mm_features=None,
            eos_token_id=None,
            arrival_time=0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            sampling_params=SamplingParams(),
            pooling_params=None,
        )
        output_processor.add_request(request, None)
        requests.append(request)

    ts = time.monotonic()

    # Create events for all 3 requests
    journey_events = [
        RequestJourneyEvent(
            request_id=f"request-{i}-int",
            event_type=RequestJourneyEventType.QUEUED,
            ts_monotonic=ts,
            scheduler_step=None,
            prefill_done_tokens=0,
            prefill_total_tokens=10,
            decode_done_tokens=0,
            decode_max_tokens=50,
            phase="PREFILL",
            num_preemptions_so_far=0,
            schedule_kind=None,
            finish_status=None,
        )
        for i in range(3)
    ]

    # Simulate AsyncLLM: distribute events ONCE before chunking
    for event in journey_events:
        if req_state := output_processor.request_states.get(event.request_id):
            req_state.journey_events.append(event)

    # Create outputs for only 2 of the 3 requests
    engine_outputs = [
        EngineCoreOutput(request_id="request-0-int", new_token_ids=[42]),
        EngineCoreOutput(request_id="request-1-int", new_token_ids=[43]),
        # request-2 has NO output
    ]

    # Simulate chunking: process in 2 chunks (chunk_size=1)
    chunk1 = engine_outputs[0:1]
    chunk2 = engine_outputs[1:2]

    # Process chunks WITHOUT passing journey_events (already distributed)
    output_processor.process_outputs(chunk1)
    output_processor.process_outputs(chunk2)

    # Verify ALL THREE requests have their events
    for i in range(3):
        req_state = output_processor.request_states.get(f"request-{i}-int")
        assert req_state is not None, f"Request {i} state missing"
        assert len(req_state.journey_events) == 1, \
            f"Request {i} lost its event (chunking or no-output bug)"
        assert req_state.journey_events[0].event_type == RequestJourneyEventType.QUEUED
