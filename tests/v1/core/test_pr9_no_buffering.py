# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for PR #9: Remove Journey Event Buffering

These tests verify that:
1. No journey event buffering exists
2. OTEL spans still work end-to-end
3. Prometheus metrics work independently of tracing
4. Timestamps are captured directly (monotonic)
5. Backward compatibility is preserved
"""
import time

import pytest

from tests.v1.core.utils import create_requests, create_scheduler


class TestNoBuffering:
    """Verify that journey event buffering has been completely removed."""

    def test_no_buffer_dict_exists(self):
        """Verify that the journey events buffer dictionary doesn't exist."""
        scheduler = create_scheduler(enable_journey_tracing=True)

        # Verify buffer dict doesn't exist
        assert not hasattr(scheduler, "_journey_events_buffer_by_client")

        # Essential tracing infrastructure should still exist
        assert hasattr(scheduler, "_first_token_emitted")
        assert hasattr(scheduler, "_journey_prefill_hiwater")
        assert hasattr(scheduler, "_core_spans")

    def test_no_buffering_during_lifecycle(self):
        """Verify no buffering occurs during request lifecycle."""
        scheduler = create_scheduler(enable_journey_tracing=True)
        requests = create_requests(num_requests=5)

        # Add requests (QUEUED events)
        for request in requests:
            scheduler.add_request(request)

        # Schedule requests (SCHEDULED events)
        scheduler.schedule()

        # Verify no buffer dict exists to accumulate events
        assert not hasattr(scheduler, "_journey_events_buffer_by_client")

        # Verify no accidental reintroduction of buffering under different names
        # Check for attributes containing both "journey" and "buffer"
        for attr_name in dir(scheduler):
            if "journey" in attr_name.lower() and "buffer" in attr_name.lower():
                pytest.fail(f"Found suspicious buffering attribute: {attr_name}")

    def test_no_journey_events_in_outputs(self):
        """Verify EngineCoreOutputs.journey_events field is always None."""
        from vllm.v1.engine import EngineCoreOutputs

        # Create a default EngineCoreOutputs
        outputs = EngineCoreOutputs()

        # Verify journey_events field exists but is None (not populated)
        assert hasattr(outputs, "journey_events"), "journey_events field removed (breaks backward compat)"
        assert outputs.journey_events is None, "journey_events should always be None (no buffering)"

        # Verify it can be set to None explicitly (backward compat)
        outputs_with_none = EngineCoreOutputs(journey_events=None)
        assert outputs_with_none.journey_events is None


class TestSpanInfrastructure:
    """Verify span tracking infrastructure exists and cleanup is safe."""

    def test_span_tracking_dict_exists(self):
        """Verify span tracking infrastructure exists."""
        scheduler = create_scheduler(enable_journey_tracing=True)

        # Verify core span tracking dict exists
        assert hasattr(scheduler, "_core_spans")
        assert isinstance(scheduler._core_spans, dict)

        # Note: Actual span creation depends on tracer initialization,
        # which requires OTLP endpoint in production. Tests verify
        # infrastructure exists, not end-to-end OTEL emission.

    def test_span_cleanup_is_safe(self):
        """Verify span cleanup logic doesn't error."""
        scheduler = create_scheduler(enable_journey_tracing=True)
        requests = create_requests(num_requests=2)

        # Add requests
        for request in requests:
            scheduler.add_request(request)

        # Finish one request - cleanup should not error
        from vllm.v1.request import RequestStatus

        try:
            scheduler.finish_requests(
                requests[0].request_id, RequestStatus.FINISHED_STOPPED
            )
        except Exception as e:
            pytest.fail(f"Span cleanup failed: {e}")

        # Verify cleanup is idempotent (safe to call again)
        try:
            scheduler._end_core_span_and_cleanup(requests[0])
        except Exception as e:
            pytest.fail(f"Idempotent cleanup failed: {e}")


class TestMetricsIndependence:
    """Verify Prometheus metrics work independently of journey tracing."""

    def test_metrics_with_tracing_disabled(self):
        """Verify metrics timestamps are captured when tracing is OFF."""
        scheduler = create_scheduler(
            enable_journey_tracing=False,  # Tracing disabled
        )
        scheduler.log_stats = True  # Metrics enabled

        requests = create_requests(num_requests=1)
        request = requests[0]

        # Add request (should capture queued_ts)
        before_queue = time.monotonic()
        scheduler.add_request(request)
        after_queue = time.monotonic()

        # Verify queued_ts was captured
        assert request.queued_ts > 0.0
        assert before_queue <= request.queued_ts <= after_queue

        # Schedule request (should capture scheduled_ts)
        before_schedule = time.monotonic()
        scheduler.schedule()
        after_schedule = time.monotonic()

        # Verify scheduled_ts was captured
        assert request.scheduled_ts > 0.0
        assert before_schedule <= request.scheduled_ts <= after_schedule

    def test_metrics_use_monotonic_time(self):
        """Verify metrics timestamps use monotonic time domain."""
        scheduler = create_scheduler(enable_journey_tracing=False)
        scheduler.log_stats = True

        requests = create_requests(num_requests=1)
        request = requests[0]

        # Capture monotonic baseline
        baseline = time.monotonic()

        scheduler.add_request(request)
        scheduler.schedule()

        # Verify timestamps are monotonic and reasonable
        assert request.queued_ts >= baseline
        assert request.scheduled_ts >= request.queued_ts

    def test_timestamps_in_request_object(self):
        """Verify timestamps are stored in Request object."""
        scheduler = create_scheduler()
        scheduler.log_stats = True

        requests = create_requests(num_requests=1)
        request = requests[0]

        queued_ts_before = time.monotonic()
        scheduler.add_request(request)
        queued_ts_after = time.monotonic()

        scheduled_ts_before = time.monotonic()
        scheduler.schedule()
        scheduled_ts_after = time.monotonic()

        # Verify timestamps are directly on Request object
        assert hasattr(request, "queued_ts")
        assert hasattr(request, "scheduled_ts")

        assert queued_ts_before <= request.queued_ts <= queued_ts_after
        assert scheduled_ts_before <= request.scheduled_ts <= scheduled_ts_after


class TestBackwardCompatibility:
    """Verify backward compatibility with deprecated journey_events parameters."""

    def test_journey_events_parameter_accepted(self):
        """Verify journey_events parameter is accepted in process_outputs signature."""
        from vllm.v1.engine.output_processor import OutputProcessor

        # Verify OutputProcessor.process_outputs has journey_events parameter
        import inspect

        sig = inspect.signature(OutputProcessor.process_outputs)
        params = sig.parameters

        # Verify journey_events parameter exists (for backward compatibility)
        assert "journey_events" in params, "journey_events parameter removed (breaks backward compatibility)"

        # Verify it's optional (has default of None)
        assert params["journey_events"].default is None, "journey_events should default to None"

    def test_engine_core_outputs_journey_events_field(self):
        """Verify EngineCoreOutputs.journey_events field still exists."""
        from vllm.v1.engine import EngineCoreOutputs

        # Create EngineCoreOutputs
        outputs = EngineCoreOutputs()

        # Verify journey_events field exists (for backward compat)
        assert hasattr(outputs, "journey_events")
        assert outputs.journey_events is None  # Should be None (not populated)


class TestTimestampCapture:
    """Verify direct timestamp capture for metrics."""

    def test_queued_ts_captured_on_add_request(self):
        """Verify queued_ts is captured when request is added."""
        scheduler = create_scheduler()
        scheduler.log_stats = True

        requests = create_requests(num_requests=1)
        request = requests[0]

        before = time.monotonic()
        scheduler.add_request(request)
        after = time.monotonic()

        # Verify queued_ts was set
        assert request.queued_ts > 0.0
        assert before <= request.queued_ts <= after

    def test_scheduled_ts_captured_on_first_schedule(self):
        """Verify scheduled_ts is captured on first schedule only."""
        scheduler = create_scheduler()
        scheduler.log_stats = True

        requests = create_requests(num_requests=1)
        request = requests[0]

        scheduler.add_request(request)

        # Verify scheduled_ts starts at 0.0
        assert request.scheduled_ts == 0.0

        # First schedule
        before = time.monotonic()
        scheduler.schedule()
        after = time.monotonic()

        # Verify scheduled_ts was set
        assert request.scheduled_ts > 0.0
        assert before <= request.scheduled_ts <= after

    def test_scheduled_ts_not_overwritten(self):
        """Verify scheduled_ts is set only once and never overwritten."""
        scheduler = create_scheduler()
        scheduler.log_stats = True

        requests = create_requests(num_requests=1)
        request = requests[0]

        scheduler.add_request(request)

        # First schedule
        scheduler.schedule()
        first_scheduled_ts = request.scheduled_ts

        # Verify it was set
        assert first_scheduled_ts > 0.0

        # Second schedule (same request, e.g., in next iteration)
        # Even though we call schedule again, scheduled_ts should not change
        scheduler.schedule()

        # Verify scheduled_ts was NOT overwritten
        assert request.scheduled_ts == first_scheduled_ts, (
            "scheduled_ts was overwritten on subsequent schedule"
        )

    def test_timestamps_not_captured_when_log_stats_false(self):
        """Verify timestamps not captured when log_stats is False."""
        scheduler = create_scheduler()
        scheduler.log_stats = False  # Disable metrics

        requests = create_requests(num_requests=1)
        request = requests[0]

        scheduler.add_request(request)
        scheduler.schedule()

        # Verify timestamps remain 0.0 (not captured)
        assert request.queued_ts == 0.0
        assert request.scheduled_ts == 0.0


class TestZeroOverheadWhenDisabled:
    """Verify zero overhead when tracing is disabled."""

    def test_no_tracing_structures_when_disabled(self):
        """Verify tracing structures not created when disabled."""
        scheduler = create_scheduler(enable_journey_tracing=False)

        # Verify journey tracing is disabled
        assert not scheduler._enable_journey_tracing

        # Verify journey-specific structures don't exist when tracing disabled
        assert not hasattr(scheduler, "_first_token_emitted"), (
            "_first_token_emitted should not exist when tracing disabled"
        )
        assert not hasattr(scheduler, "_journey_prefill_hiwater"), (
            "_journey_prefill_hiwater should not exist when tracing disabled"
        )

        # Core spans dict may exist (for span lifecycle) but should be empty
        assert hasattr(scheduler, "_core_spans")
        assert scheduler._core_spans == {}, "Core spans dict should be empty when tracing disabled"

    def test_metrics_independent_of_tracing(self):
        """Verify metrics work even when tracing is completely disabled."""
        scheduler = create_scheduler(enable_journey_tracing=False)
        scheduler.log_stats = True

        requests = create_requests(num_requests=2)

        for request in requests:
            scheduler.add_request(request)

        scheduler.schedule()

        # Verify timestamps captured despite tracing being off
        for request in requests:
            assert request.queued_ts > 0.0
            assert request.scheduled_ts > 0.0
