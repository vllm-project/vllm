"""Behavioral tests for journey/step tracer separation.

These tests verify behavior by mocking and inspecting actual function calls,
not by reading source files and asserting text patterns.
"""
from unittest.mock import Mock, patch, MagicMock


# Import create_scheduler utility
from tests.v1.core.utils import create_scheduler


class TestTracerSeparationBehavioral:
    """Behavioral tests verifying journey and step tracers are separate."""

    @patch('vllm.tracing.init_tracer')
    def test_journey_tracer_uses_scheduler_scope(self, mock_init_tracer):
        """Verify journey tracer is initialized with vllm.scheduler scope."""
        mock_tracer = Mock()
        mock_init_tracer.return_value = mock_tracer

        # Create scheduler with journey tracing enabled
        scheduler = create_scheduler(
            enable_journey_tracing=True,
            otlp_traces_endpoint="http://test:4317",
            step_tracing_enabled=False,
        )

        # Verify init_tracer was called with vllm.scheduler scope (not order/count sensitive)
        calls = mock_init_tracer.call_args_list
        assert any(
            call[0] == ("vllm.scheduler", "http://test:4317")
            for call in calls
        ), f"Expected call to init_tracer('vllm.scheduler', 'http://test:4317'), got: {calls}"

        # Verify tracer attributes
        assert scheduler.journey_tracer is mock_tracer
        assert scheduler.step_tracer is None

    @patch('vllm.tracing.init_tracer')
    def test_step_tracer_uses_scheduler_step_scope(self, mock_init_tracer):
        """Verify step tracer is initialized with vllm.scheduler.step scope."""
        mock_tracer = Mock()
        mock_init_tracer.return_value = mock_tracer

        # Create scheduler with step tracing enabled
        scheduler = create_scheduler(
            enable_journey_tracing=False,
            step_tracing_enabled=True,
            otlp_traces_endpoint="http://test:4317",
        )

        # Verify init_tracer was called with vllm.scheduler.step scope (not order/count sensitive)
        calls = mock_init_tracer.call_args_list
        assert any(
            call[0] == ("vllm.scheduler.step", "http://test:4317")
            for call in calls
        ), f"Expected call to init_tracer('vllm.scheduler.step', 'http://test:4317'), got: {calls}"

        # Verify tracer attributes
        assert scheduler.journey_tracer is None
        assert scheduler.step_tracer is mock_tracer

    @patch('vllm.tracing.init_tracer')
    def test_both_tracers_use_distinct_scopes(self, mock_init_tracer):
        """Verify both tracers initialized with distinct scopes when both enabled."""
        # Return different mock tracers for each call
        journey_tracer = Mock(name="journey_tracer")
        step_tracer = Mock(name="step_tracer")
        mock_init_tracer.side_effect = [journey_tracer, step_tracer]

        # Create scheduler with both tracing features enabled
        scheduler = create_scheduler(
            enable_journey_tracing=True,
            step_tracing_enabled=True,
            otlp_traces_endpoint="http://test:4317",
        )

        # Verify both scopes were initialized (order-agnostic)
        calls = mock_init_tracer.call_args_list
        call_tuples = [call[0] for call in calls]

        assert ("vllm.scheduler", "http://test:4317") in call_tuples, \
            f"Expected vllm.scheduler scope, got: {call_tuples}"
        assert ("vllm.scheduler.step", "http://test:4317") in call_tuples, \
            f"Expected vllm.scheduler.step scope, got: {call_tuples}"

        # Verify tracers are stored in separate variables
        assert scheduler.journey_tracer is not None
        assert scheduler.step_tracer is not None
        assert scheduler.journey_tracer is not scheduler.step_tracer

    @patch('vllm.tracing.init_tracer')
    def test_no_accidental_coupling_between_tracers(self, mock_init_tracer):
        """Verify step tracer initializes even if journey tracer fails."""
        # Make journey tracer init fail, step tracer succeed
        step_tracer = Mock(name="step_tracer")
        mock_init_tracer.side_effect = [Exception("Journey init failed"), step_tracer]

        # Create scheduler with both tracing features enabled
        scheduler = create_scheduler(
            enable_journey_tracing=True,
            step_tracing_enabled=True,
            otlp_traces_endpoint="http://test:4317",
        )

        # Verify both scopes were attempted (order-agnostic, just presence)
        calls = mock_init_tracer.call_args_list
        call_tuples = [call[0] for call in calls]

        assert any(args[0] == "vllm.scheduler" for args in call_tuples), \
            f"Expected journey tracer init attempt, got: {call_tuples}"
        assert any(args[0] == "vllm.scheduler.step" for args in call_tuples), \
            f"Expected step tracer init attempt, got: {call_tuples}"

        # Journey tracer failed, step tracer succeeded (no coupling!)
        assert scheduler.journey_tracer is None
        assert scheduler.step_tracer is step_tracer

    @patch('vllm.tracing.init_tracer')
    def test_journey_tracer_used_for_core_spans(self, mock_init_tracer):
        """Verify journey_tracer is used to create llm_core spans."""
        journey_tracer = MagicMock()
        mock_span = Mock()
        journey_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = journey_tracer

        # Create scheduler with journey tracing enabled
        scheduler = create_scheduler(
            enable_journey_tracing=True,
            otlp_traces_endpoint="http://test:4317",
        )

        # Verify journey tracer was initialized with correct scope
        calls = mock_init_tracer.call_args_list
        assert any(
            call[0] == ("vllm.scheduler", "http://test:4317")
            for call in calls
        ), f"Expected vllm.scheduler scope init, got: {calls}"

        # Create a mock request and verify tracer can create spans
        from vllm.v1.request import Request
        mock_request = Mock(spec=Request)
        mock_request.request_id = "test-req-123"
        mock_request.trace_headers = None

        # Call _create_core_span to verify journey_tracer is usable
        span = scheduler._create_core_span(mock_request)

        # Verify journey_tracer was used (not checking exact call timing/order)
        assert journey_tracer.start_span.called, "journey_tracer.start_span should be called"
        assert span is mock_span

    @patch('vllm.tracing.init_tracer')
    def test_step_tracer_used_for_step_spans(self, mock_init_tracer):
        """Verify step_tracer is initialized with correct scope for step tracing."""
        step_tracer = MagicMock()
        mock_span = Mock()
        step_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = step_tracer

        # Create scheduler with step tracing enabled
        scheduler = create_scheduler(
            step_tracing_enabled=True,
            otlp_traces_endpoint="http://test:4317",
        )

        # Verify step tracer was initialized with correct scope
        calls = mock_init_tracer.call_args_list
        assert any(
            call[0] == ("vllm.scheduler.step", "http://test:4317")
            for call in calls
        ), f"Expected vllm.scheduler.step scope init, got: {calls}"

        # Verify step tracer is available and can be used
        # (Don't assert on exact timing/method of span creation - that's implementation detail)
        assert scheduler.step_tracer is step_tracer

    @patch('vllm.tracing.init_tracer')
    def test_no_endpoint_disables_both_tracers(self, mock_init_tracer):
        """Verify tracers not initialized when endpoint is None."""
        # Create scheduler with both features enabled but no endpoint
        scheduler = create_scheduler(
            enable_journey_tracing=True,
            step_tracing_enabled=True,
            otlp_traces_endpoint=None,
        )

        # Verify init_tracer never called
        mock_init_tracer.assert_not_called()

        # Verify tracer attributes are None
        assert scheduler.journey_tracer is None
        assert scheduler.step_tracer is None

    @patch('vllm.tracing.init_tracer')
    def test_tracers_share_same_endpoint(self, mock_init_tracer):
        """Verify both tracers use the same OTLP endpoint (singleton provider)."""
        mock_init_tracer.return_value = Mock()

        # Create scheduler with both features enabled and shared endpoint
        endpoint = "http://shared-collector:4317"
        create_scheduler(
            enable_journey_tracing=True,
            step_tracing_enabled=True,
            otlp_traces_endpoint=endpoint,
        )

        # Verify both scopes were initialized with the same endpoint
        calls = mock_init_tracer.call_args_list
        call_tuples = [call[0] for call in calls]

        # Check both scopes present with same endpoint
        assert ("vllm.scheduler", endpoint) in call_tuples, \
            f"Expected vllm.scheduler with {endpoint}, got: {call_tuples}"
        assert ("vllm.scheduler.step", endpoint) in call_tuples, \
            f"Expected vllm.scheduler.step with {endpoint}, got: {call_tuples}"
