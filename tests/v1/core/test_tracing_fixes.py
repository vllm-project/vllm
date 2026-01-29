"""Unit tests for journey tracing coordination fixes.

These tests verify the global provider pattern and integration behavior:
1. TracerProvider singleton pattern prevents overwrites
2. API span creation with events
3. Trace context injection/extraction
4. End-to-end global provider flow

Note: Behavioral tests for tracer separation and vllm.llm_engine removal
are in test_tracer_separation_behavioral.py and test_vllm_llm_engine_removed_behavioral.py
"""

import pytest
from unittest.mock import Mock, patch


@pytest.fixture
def reset_global_tracer_provider():
    """Reset global tracer provider and endpoint before and after test."""
    import vllm.tracing as tracing_module
    original_provider = tracing_module._global_tracer_provider
    original_endpoint = tracing_module._global_tracer_endpoint
    tracing_module._global_tracer_provider = None
    tracing_module._global_tracer_endpoint = None
    yield
    # Restore original provider and endpoint after test
    tracing_module._global_tracer_provider = original_provider
    tracing_module._global_tracer_endpoint = original_endpoint


class TestTracerProviderSingleton:
    """Test that TracerProvider is properly shared across scopes."""

    def test_singleton_prevents_overwrites(self, reset_global_tracer_provider):
        """Verify multiple init_tracer calls don't overwrite the provider."""

        # Mock components
        mock_provider = Mock()
        mock_tracer_api = Mock()
        mock_tracer_scheduler = Mock()

        def mock_get_tracer(scope_name):
            if scope_name == "vllm.api":
                return mock_tracer_api
            elif scope_name == "vllm.scheduler":
                return mock_tracer_scheduler
            return Mock()

        mock_provider.get_tracer = mock_get_tracer

        with patch('vllm.tracing.TracerProvider', return_value=mock_provider):
            with patch('vllm.tracing.get_span_exporter', return_value=Mock()):
                with patch('vllm.tracing.BatchSpanProcessor', return_value=Mock()):
                    with patch('vllm.tracing.set_tracer_provider') as mock_set_provider:
                        from vllm.tracing import init_tracer

                        # First call should set provider
                        tracer1 = init_tracer("vllm.api", "http://localhost:4317")
                        first_set_count = mock_set_provider.call_count

                        # Second call should NOT set provider again
                        tracer2 = init_tracer("vllm.scheduler", "http://localhost:4317")
                        second_set_count = mock_set_provider.call_count

                        assert tracer1 is mock_tracer_api
                        assert tracer2 is mock_tracer_scheduler
                        assert first_set_count == 1
                        assert second_set_count == 1  # Should NOT increase
                        assert second_set_count == first_set_count  # Provider not overwritten


class TestAPISpanCreation:
    """Test API span creation and event emission."""

    @pytest.mark.asyncio
    async def test_api_span_with_arrived_event(self):
        """Verify API span is created with api.ARRIVED event using global provider."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_span.add_event = Mock()
        mock_span.set_attribute = Mock()

        mock_tracer = Mock()
        mock_tracer.start_span.return_value = mock_span

        mock_provider = Mock()
        mock_provider.get_tracer.return_value = mock_tracer

        # Patch where the functions are imported FROM (not where they're used)
        with patch('opentelemetry.trace.get_tracer_provider', return_value=mock_provider):
            with patch('opentelemetry.trace.SpanKind'):
                with patch('vllm.tracing.extract_trace_context', return_value=None):
                    from vllm.entrypoints.openai.engine.serving import OpenAIServing

                    # Create instance with necessary attributes
                    serving = OpenAIServing.__new__(OpenAIServing)
                    serving._cached_is_tracing_enabled = True  # Mock tracing enabled

                    span = await serving._create_api_span("test-request-123", None)

                    # Verify span was created with global provider
                    assert span is mock_span
                    assert mock_tracer.start_span.called
                    # Verify get_tracer was called with "vllm.api" scope
                    mock_provider.get_tracer.assert_called_with("vllm.api")

                    # Verify api.ARRIVED event was emitted
                    assert mock_span.add_event.called
                    event_call = mock_span.add_event.call_args
                    event_name = event_call[1].get('name', event_call[0][0] if event_call[0] else None)
                    assert event_name == "api.ARRIVED"


class TestTraceContextPropagation:
    """Test trace context injection and extraction."""

    def test_inject_trace_context(self):
        """Verify trace context is properly injected into carrier."""
        mock_span = Mock()
        mock_context = Mock()

        with patch('vllm.tracing.TraceContextTextMapPropagator') as mock_propagator_class:
            mock_propagator = Mock()
            mock_propagator.inject = Mock(
                side_effect=lambda carrier, context: carrier.update({"traceparent": "00-trace123-span456-01"})
            )
            mock_propagator_class.return_value = mock_propagator

            with patch('opentelemetry.trace.set_span_in_context', return_value=mock_context):
                from vllm.tracing import inject_trace_context

                carrier = {}
                result = inject_trace_context(mock_span, carrier)

                assert "traceparent" in carrier
                assert carrier["traceparent"] == "00-trace123-span456-01"
                assert result is carrier

    def test_extract_trace_context(self):
        """Verify trace context is properly extracted from headers."""
        mock_context = Mock()

        with patch('vllm.tracing.TraceContextTextMapPropagator') as mock_propagator_class:
            mock_propagator = Mock()
            mock_propagator.extract = Mock(return_value=mock_context)
            mock_propagator_class.return_value = mock_propagator

            from vllm.tracing import extract_trace_context

            headers = {"traceparent": "00-trace123-span456-01"}
            context = extract_trace_context(headers)

            assert context is mock_context
            assert mock_propagator.extract.called


class TestGlobalProviderIntegration:
    """Test that global provider pattern works end-to-end."""

    @pytest.mark.asyncio
    async def test_global_provider_flow(self, reset_global_tracer_provider):
        """Verify init_tracer() sets global provider and _create_api_span() uses it."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_span.add_event = Mock()
        mock_span.set_attribute = Mock()

        mock_tracer = Mock()
        mock_tracer.start_span.return_value = mock_span

        mock_provider = Mock()
        mock_provider.get_tracer.return_value = mock_tracer

        # Mock the OTEL components (patch where they're imported TO, not FROM)
        with patch('vllm.tracing.TracerProvider', return_value=mock_provider):
            with patch('vllm.tracing.set_tracer_provider') as mock_set_provider:
                with patch('opentelemetry.trace.get_tracer_provider', return_value=mock_provider):
                    with patch('vllm.tracing.get_span_exporter'):
                        with patch('opentelemetry.trace.SpanKind'):
                            with patch('vllm.tracing.extract_trace_context', return_value=None):
                                # Step 1: Initialize tracer (simulating api_server.py)
                                from vllm.tracing import init_tracer
                                init_tracer("vllm.api", "http://localhost:4317")

                            # Verify set_tracer_provider was called to set global
                            assert mock_set_provider.called
                            mock_set_provider.assert_called_with(mock_provider)

                            # Verify get_tracer was called with correct scope
                            mock_provider.get_tracer.assert_called_with("vllm.api")

                            # Step 2: Create API span (simulating _create_api_span)
                            from vllm.entrypoints.openai.engine.serving import OpenAIServing

                            serving = OpenAIServing.__new__(OpenAIServing)
                            serving._cached_is_tracing_enabled = True

                            span = await serving._create_api_span("test-req-456", None)

                            # Verify span was created using the global provider
                            assert span is mock_span
                            # get_tracer should be called at least twice:
                            # once in init_tracer, once in _create_api_span
                            assert mock_provider.get_tracer.call_count >= 2
                            # Both should use "vllm.api" scope
                            for call in mock_provider.get_tracer.call_args_list:
                                assert call[0][0] == "vllm.api"
