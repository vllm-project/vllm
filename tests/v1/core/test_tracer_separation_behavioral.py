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

        # Verify init_tracer called with vllm.scheduler scope
        mock_init_tracer.assert_called_once_with("vllm.scheduler", "http://test:4317")

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

        # Verify init_tracer called with vllm.scheduler.step scope
        mock_init_tracer.assert_called_once_with("vllm.scheduler.step", "http://test:4317")

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

        # Verify init_tracer called twice with distinct scopes
        assert mock_init_tracer.call_count == 2
        calls = mock_init_tracer.call_args_list
        assert calls[0][0] == ("vllm.scheduler", "http://test:4317")
        assert calls[1][0] == ("vllm.scheduler.step", "http://test:4317")

        # Verify tracers are stored in separate variables
        assert scheduler.journey_tracer is journey_tracer
        assert scheduler.step_tracer is step_tracer
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

        # Verify both init calls were attempted
        assert mock_init_tracer.call_count == 2

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

        # Verify init_tracer was called
        mock_init_tracer.assert_called_with("vllm.scheduler", "http://test:4317")

        # Create a mock request
        from vllm.v1.request import Request
        mock_request = Mock(spec=Request)
        mock_request.request_id = "test-req-123"
        mock_request.trace_headers = None

        # Call _create_core_span (internal method, but testing behavior)
        span = scheduler._create_core_span(mock_request)

        # Verify journey_tracer.start_span was called
        assert journey_tracer.start_span.called
        # Check first positional arg or 'name' kwarg
        call_args, call_kwargs = journey_tracer.start_span.call_args
        span_name = call_args[0] if call_args else call_kwargs.get('name')
        assert span_name == "llm_core"
        assert span is mock_span

    @patch('vllm.tracing.init_tracer')
    def test_step_tracer_used_for_step_spans(self, mock_init_tracer):
        """Verify step_tracer is used to create scheduler_steps span."""
        step_tracer = MagicMock()
        mock_span = Mock()
        step_tracer.start_span.return_value = mock_span
        mock_init_tracer.return_value = step_tracer

        # Create scheduler with step tracing enabled
        scheduler = create_scheduler(
            step_tracing_enabled=True,
            otlp_traces_endpoint="http://test:4317",
        )

        # Verify init_tracer was called
        mock_init_tracer.assert_called_with("vllm.scheduler.step", "http://test:4317")

        # Verify step_tracer.start_span was called during __init__
        assert step_tracer.start_span.called
        # Check first positional arg or 'name' kwarg
        call_args, call_kwargs = step_tracer.start_span.call_args
        span_name = call_args[0] if call_args else call_kwargs.get('name')
        assert span_name == "scheduler_steps"
        assert scheduler._step_span is mock_span

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

        # Verify both calls used the same endpoint
        assert mock_init_tracer.call_count == 2
        for call in mock_init_tracer.call_args_list:
            assert call[0][1] == endpoint  # Second arg is endpoint
