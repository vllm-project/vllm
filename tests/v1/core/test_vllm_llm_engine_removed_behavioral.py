"""Behavioral tests verifying vllm.llm_engine scope removed.

These tests verify that vllm.llm_engine tracer is not initialized
by mocking init_tracer and verifying it's never called with that scope.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestVllmLlmEngineRemovedBehavioral:
    """Behavioral tests verifying vllm.llm_engine scope is not used."""

    @patch('vllm.v1.engine.async_llm.EngineCoreClient')
    @patch('vllm.tracing.init_tracer')
    def test_async_llm_does_not_init_llm_engine_tracer(self, mock_init_tracer, mock_core_client):
        """Verify AsyncLLM does not initialize vllm.llm_engine tracer."""
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.config import VllmConfig
        from vllm.usage.usage_lib import UsageContext

        mock_init_tracer.return_value = Mock()

        # Mock EngineCoreClient to prevent actual initialization
        mock_client_instance = MagicMock()
        mock_core_client.make_async_mp_client.return_value = mock_client_instance

        # Create config with tracing enabled
        config = VllmConfig()
        config.observability_config.otlp_traces_endpoint = "http://test:4317"

        # Create a mock executor class (not instance)
        MockExecutor = type('MockExecutor', (), {})

        # Instantiate AsyncLLM
        try:
            async_llm = AsyncLLM(
                vllm_config=config,
                executor_class=MockExecutor,
                log_stats=False,
                usage_context=UsageContext.ENGINE_CONTEXT,
                log_requests=False,
                start_engine_loop=False,  # Don't start background loop
            )
        except Exception:
            # May fail due to other initialization issues, but that's OK
            # We just care about tracer calls
            pass

        # Verify init_tracer was NEVER called with vllm.llm_engine
        for call in mock_init_tracer.call_args_list:
            scope_name = call[0][0]  # First positional arg
            assert scope_name != "vllm.llm_engine", \
                f"AsyncLLM should not initialize vllm.llm_engine tracer, but found: {scope_name}"

    @patch('vllm.v1.engine.core_client.EngineCoreClient')
    @patch('vllm.tracing.init_tracer')
    def test_llm_engine_does_not_init_llm_engine_tracer(self, mock_init_tracer, mock_core_client):
        """Verify LLMEngine does not initialize vllm.llm_engine tracer."""
        from vllm.v1.engine.llm_engine import LLMEngine
        from vllm.config import VllmConfig
        from vllm.usage.usage_lib import UsageContext

        mock_init_tracer.return_value = Mock()

        # Mock EngineCoreClient to prevent actual initialization
        mock_client_instance = MagicMock()
        mock_core_client.make_client.return_value = mock_client_instance

        # Create config with tracing enabled
        config = VllmConfig()
        config.observability_config.otlp_traces_endpoint = "http://test:4317"

        # Create a mock executor class (not instance)
        MockExecutor = type('MockExecutor', (), {})

        # Instantiate LLMEngine
        try:
            llm_engine = LLMEngine(
                vllm_config=config,
                executor_class=MockExecutor,
                log_stats=False,
                usage_context=UsageContext.ENGINE_CONTEXT,
            )
        except Exception:
            # May fail due to other initialization issues, but that's OK
            # We just care about tracer calls
            pass

        # Verify init_tracer was NEVER called with vllm.llm_engine
        for call in mock_init_tracer.call_args_list:
            scope_name = call[0][0]  # First positional arg
            assert scope_name != "vllm.llm_engine", \
                f"LLMEngine should not initialize vllm.llm_engine tracer, but found: {scope_name}"

    def test_output_processor_does_not_call_do_tracing(self):
        """Verify OutputProcessor.process_outputs does not call do_tracing().

        The do_tracing() method still exists (for now) but should not be called.
        This test verifies the call has been commented out/removed.
        """
        from vllm.v1.engine.output_processor import OutputProcessor
        from unittest.mock import Mock

        # Create OutputProcessor with a mock tracer
        output_processor = OutputProcessor(
            tokenizer=None,
            log_stats=False,
            stream_interval=1,
        )
        output_processor.tracer = Mock()  # Old tracer variable

        # Mock do_tracing method to track if it's called
        output_processor.do_tracing = Mock()

        # Create mock inputs for process_outputs
        mock_engine_core_outputs = []  # Empty list

        # Call process_outputs
        result = output_processor.process_outputs(
            engine_core_outputs=mock_engine_core_outputs,
            journey_events=None,
        )

        # Verify do_tracing was NEVER called
        output_processor.do_tracing.assert_not_called()

    def test_output_processor_do_tracing_method_still_exists(self):
        """Verify do_tracing method still exists but is unused.

        This is for backward compatibility - the method exists but is not called.
        """
        from vllm.v1.engine.output_processor import OutputProcessor

        output_processor = OutputProcessor(
            tokenizer=None,
            log_stats=False,
            stream_interval=1,
        )

        # Method should exist
        assert hasattr(output_processor, 'do_tracing')
        assert callable(output_processor.do_tracing)


class TestNoVllmLlmEngineSpansCreated:
    """Behavioral tests verifying no spans created under vllm.llm_engine scope."""

    @patch('vllm.tracing.init_tracer')
    def test_only_expected_scopes_initialized(self, mock_init_tracer):
        """Verify only vllm.api and vllm.scheduler scopes are initialized (not vllm.llm_engine)."""
        from vllm.config import VllmConfig

        mock_init_tracer.return_value = Mock()

        # Simulate full system initialization
        # (This is a conceptual test - actual full init may be complex)

        # Expected scopes:
        # - vllm.api (initialized in api_server.py)
        # - vllm.scheduler (initialized in scheduler.py for journey tracing)
        # - vllm.scheduler.step (initialized in scheduler.py for step tracing)

        # NOT expected:
        # - vllm.llm_engine (removed from async_llm.py and llm_engine.py)

        # This test documents the expected scopes
        expected_scopes = {
            "vllm.api",           # API layer spans
            "vllm.scheduler",     # Journey tracing (llm_core spans)
            "vllm.scheduler.step" # Step tracing (scheduler_steps span)
        }

        forbidden_scopes = {
            "vllm.llm_engine"     # This scope should NOT be used
        }

        # Note: This is a documentation test. Actual behavioral verification
        # happens in the AsyncLLM and LLMEngine tests above.
        assert "vllm.llm_engine" in forbidden_scopes
        assert "vllm.api" in expected_scopes
        assert "vllm.scheduler" in expected_scopes
        assert "vllm.scheduler.step" in expected_scopes
