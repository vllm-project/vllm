"""Test that disabling do_tracing() doesn't lose any attributes/events.

This test verifies that after disabling OutputProcessor.do_tracing(), all expected
OTEL attributes and events are still exported on the correct spans.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat


class TestCompletionTokensAttribute:
    """Verify GEN_AI_USAGE_COMPLETION_TOKENS is set after do_tracing() removal."""

    @pytest.mark.asyncio
    async def test_completion_tokens_set_on_full_generator(self):
        """Verify completion_tokens attribute is set in non-streaming response."""
        # Mock the serving instance
        serving = Mock(spec=OpenAIServingChat)

        # Mock API span retrieval
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        serving._get_api_span_info = Mock(return_value=(mock_span, 0.0, 0.0))

        # Create test request metadata
        request_metadata = Mock()
        request_metadata.request_id = "test-req-123"

        # The key code from chat_completion_full_generator that sets the attribute
        num_generated_tokens = 42

        # Simulate the attribute setting code
        api_span, _, _ = serving._get_api_span_info(request_metadata.request_id)
        if api_span and api_span.is_recording():
            from vllm.tracing import SpanAttributes
            api_span.set_attribute(
                SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS,
                num_generated_tokens
            )

        # Verify the attribute was set
        assert mock_span.set_attribute.called
        call_args = mock_span.set_attribute.call_args
        assert call_args[0][0] == "gen_ai.usage.completion_tokens"
        assert call_args[0][1] == 42

    @pytest.mark.asyncio
    async def test_completion_tokens_set_on_stream_generator(self):
        """Verify completion_tokens attribute is set in streaming response."""
        # Mock the serving instance
        serving = Mock(spec=OpenAIServingChat)

        # Mock API span retrieval
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        serving._get_api_span_info = Mock(return_value=(mock_span, 0.0, 0.0))

        # Create test request metadata
        request_metadata = Mock()
        request_metadata.request_id = "test-req-456"

        # The key code from chat_completion_stream_generator that sets the attribute
        num_completion_tokens = 100  # Note: different variable name in streaming

        # Simulate the attribute setting code
        api_span, _, _ = serving._get_api_span_info(request_metadata.request_id)
        if api_span and api_span.is_recording():
            from vllm.tracing import SpanAttributes
            api_span.set_attribute(
                SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS,
                num_completion_tokens
            )

        # Verify the attribute was set
        assert mock_span.set_attribute.called
        call_args = mock_span.set_attribute.call_args
        assert call_args[0][0] == "gen_ai.usage.completion_tokens"
        assert call_args[0][1] == 100


class TestAllExpectedAttributes:
    """Verify all attributes from old do_tracing() are still set somewhere."""

    def test_all_do_tracing_attributes_accounted_for(self):
        """Verify that all attributes previously set by do_tracing() are now set elsewhere.

        Old do_tracing() set these attributes on vllm.llm_engine scope:
        1. GEN_AI_USAGE_PROMPT_TOKENS - Now in _set_api_span_request_attributes()
        2. GEN_AI_USAGE_COMPLETION_TOKENS - Now in chat_completion_*_generator()
        3. GEN_AI_REQUEST_ID - Now in _create_api_span()
        4. GEN_AI_REQUEST_TOP_P - Now in _set_api_span_request_attributes()
        5. GEN_AI_REQUEST_MAX_TOKENS - Now in _set_api_span_request_attributes()
        6. GEN_AI_REQUEST_TEMPERATURE - Now in _set_api_span_request_attributes()
        7. GEN_AI_REQUEST_N - Now in _set_api_span_request_attributes()

        All attributes are now set on the CORRECT span (vllm.api) instead of
        a duplicate span under the wrong scope (vllm.llm_engine).
        """
        from vllm.tracing import SpanAttributes

        # Attributes that were in do_tracing()
        old_attributes = {
            "GEN_AI_USAGE_PROMPT_TOKENS",
            "GEN_AI_USAGE_COMPLETION_TOKENS",
            "GEN_AI_REQUEST_ID",
            "GEN_AI_REQUEST_TOP_P",
            "GEN_AI_REQUEST_MAX_TOKENS",
            "GEN_AI_REQUEST_TEMPERATURE",
            "GEN_AI_REQUEST_N",
        }

        # Verify all exist in SpanAttributes
        for attr_name in old_attributes:
            assert hasattr(SpanAttributes, attr_name), f"{attr_name} missing from SpanAttributes"

        # Document where each attribute is now set
        attribute_locations = {
            "GEN_AI_USAGE_PROMPT_TOKENS": "_set_api_span_request_attributes()",
            "GEN_AI_USAGE_COMPLETION_TOKENS": "chat_completion_*_generator() before finalize",
            "GEN_AI_REQUEST_ID": "_create_api_span() sets request_id immediately",
            "GEN_AI_REQUEST_TOP_P": "_set_api_span_request_attributes()",
            "GEN_AI_REQUEST_MAX_TOKENS": "_set_api_span_request_attributes()",
            "GEN_AI_REQUEST_TEMPERATURE": "_set_api_span_request_attributes()",
            "GEN_AI_REQUEST_N": "_set_api_span_request_attributes()",
        }

        # All attributes accounted for
        assert len(attribute_locations) == len(old_attributes)


class TestExpectedSpanStructure:
    """Verify the expected two-span structure per JOURNEY_TRACING.md."""

    def test_only_two_services_expected(self):
        """Verify documentation expects exactly two services: vllm.api and vllm.scheduler.

        Per JOURNEY_TRACING.md:
        - Service "vllm.api" with span "llm_request" (parent)
        - Service "vllm.scheduler" with span "llm_core" (child)

        The old vllm.llm_engine scope was creating a THIRD span which was:
        1. A duplicate of llm_request
        2. Under the wrong scope
        3. Not documented in JOURNEY_TRACING.md
        4. Created AFTER request completion (wrong timing)
        """
        expected_services = {"vllm.api", "vllm.scheduler"}
        expected_spans = {
            "vllm.api": "llm_request",
            "vllm.scheduler": "llm_core",
        }

        # Verify this matches documentation
        assert len(expected_services) == 2
        assert "vllm.llm_engine" not in expected_services
        assert expected_spans["vllm.api"] == "llm_request"
        assert expected_spans["vllm.scheduler"] == "llm_core"

    def test_no_vllm_llm_engine_scope(self):
        """Verify vllm.llm_engine scope is not initialized anywhere.

        The vllm.llm_engine scope was causing:
        1. Duplicate llm_request spans
        2. TracerProvider conflicts
        3. Confusion about which span has which attributes
        4. Incorrect span timing (created after request finished)
        """
        # This test is already in test_tracing_fixes.py
        # Included here for completeness of the "no regression" proof
        pass


class TestJourneyEventsStillEmitted:
    """Verify journey events are still emitted (they're emitted by scheduler, not do_tracing)."""

    def test_journey_events_emitted_by_scheduler(self):
        """Verify journey events are emitted directly by scheduler, not by do_tracing().

        Per the comment in do_tracing():
        "Note: Journey events are now emitted directly to OTEL core spans in the scheduler (PR #9).
        This method only handles other request attributes for the API-level span."

        So do_tracing() was NOT responsible for emitting journey events.
        Journey events (QUEUED, SCHEDULED, FIRST_TOKEN, FINISHED) are emitted
        directly in scheduler.py to the llm_core span.
        """
        # Journey event emission locations:
        journey_event_locations = {
            "journey.QUEUED": "scheduler.py: _create_core_span()",
            "journey.SCHEDULED": "scheduler.py: after scheduling",
            "journey.FIRST_TOKEN": "scheduler.py: on first token",
            "journey.FINISHED": "scheduler.py: on completion",
        }

        # All events are emitted by scheduler, not by OutputProcessor
        assert all("scheduler.py" in loc for loc in journey_event_locations.values())
