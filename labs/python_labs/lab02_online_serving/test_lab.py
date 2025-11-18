"""
Lab 02: Online Serving with AsyncEngine - Tests

pytest tests to verify your async implementation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from solution import (
    create_async_engine,
    generate_request_id,
    generate_single_request,
    generate_concurrent_requests
)


class TestAsyncEngineCreation:
    """Tests for AsyncLLMEngine initialization."""

    @pytest.mark.asyncio
    @patch('solution.AsyncLLMEngine')
    @patch('solution.AsyncEngineArgs')
    async def test_create_async_engine_defaults(self, mock_args_class, mock_engine_class):
        """Test async engine creation with default parameters."""
        mock_args_instance = Mock()
        mock_args_class.return_value = mock_args_instance

        mock_engine_instance = AsyncMock()
        mock_engine_class.from_engine_args.return_value = mock_engine_instance

        result = await create_async_engine()

        mock_args_class.assert_called_once_with(
            model="facebook/opt-125m",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )
        mock_engine_class.from_engine_args.assert_called_once_with(mock_args_instance)
        assert result == mock_engine_instance

    @pytest.mark.asyncio
    @patch('solution.AsyncLLMEngine')
    @patch('solution.AsyncEngineArgs')
    async def test_create_async_engine_custom(self, mock_args_class, mock_engine_class):
        """Test async engine creation with custom parameters."""
        mock_args_instance = Mock()
        mock_args_class.return_value = mock_args_instance

        mock_engine_instance = AsyncMock()
        mock_engine_class.from_engine_args.return_value = mock_engine_instance

        result = await create_async_engine(
            model_name="gpt2",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8
        )

        mock_args_class.assert_called_once_with(
            model="gpt2",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
        )


class TestRequestID:
    """Tests for request ID generation."""

    def test_generate_request_id_format(self):
        """Test that request ID has correct format."""
        request_id = generate_request_id()
        assert request_id.startswith("req-")
        assert len(request_id) > 4

    def test_generate_request_id_uniqueness(self):
        """Test that generated IDs are unique."""
        ids = [generate_request_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_generate_request_id_custom_prefix(self):
        """Test custom prefix."""
        request_id = generate_request_id(prefix="test")
        assert request_id.startswith("test-")


class TestSingleRequest:
    """Tests for single request generation."""

    @pytest.mark.asyncio
    async def test_generate_single_request(self):
        """Test single request generation."""
        mock_engine = AsyncMock()

        # Create mock output
        mock_output = Mock()
        mock_output.outputs = [Mock(text="Generated response")]

        # Create async generator
        async def mock_generator():
            yield mock_output

        mock_engine.generate.return_value = mock_generator()

        mock_sampling_params = Mock()

        result = await generate_single_request(
            mock_engine,
            "Test prompt",
            mock_sampling_params,
            "test-id"
        )

        assert result == "Generated response"
        mock_engine.generate.assert_called_once_with(
            "Test prompt",
            mock_sampling_params,
            "test-id"
        )

    @pytest.mark.asyncio
    async def test_generate_single_request_empty_output(self):
        """Test handling of empty output."""
        mock_engine = AsyncMock()

        async def mock_generator():
            yield Mock(outputs=[])

        mock_engine.generate.return_value = mock_generator()

        result = await generate_single_request(
            mock_engine, "Test", Mock(), "id"
        )

        assert result == ""


class TestConcurrentRequests:
    """Tests for concurrent request processing."""

    @pytest.mark.asyncio
    async def test_generate_concurrent_requests(self):
        """Test concurrent request processing."""
        mock_engine = AsyncMock()

        # Mock outputs for different prompts
        outputs = ["Output 1", "Output 2", "Output 3"]

        call_count = 0

        async def mock_generator():
            nonlocal call_count
            mock_output = Mock()
            mock_output.outputs = [Mock(text=outputs[call_count])]
            call_count += 1
            yield mock_output

        mock_engine.generate.side_effect = lambda *args: mock_generator()

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        mock_sampling_params = Mock()

        results = await generate_concurrent_requests(
            mock_engine, prompts, mock_sampling_params
        )

        assert len(results) == 3
        # Check that all outputs are present (order may vary due to concurrency)
        result_texts = [text for _, text in results]
        assert set(result_texts) == set(outputs)

        # Check that request IDs are unique
        request_ids = [req_id for req_id, _ in results]
        assert len(set(request_ids)) == 3


class TestIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    @patch('solution.AsyncLLMEngine')
    @patch('solution.AsyncEngineArgs')
    async def test_full_async_pipeline(self, mock_args_class, mock_engine_class):
        """Test the full async inference pipeline."""
        # Setup engine mock
        mock_engine_instance = AsyncMock()

        mock_output = Mock()
        mock_output.outputs = [Mock(text="Test output")]

        async def mock_generator():
            yield mock_output

        mock_engine_instance.generate.return_value = mock_generator()
        mock_engine_class.from_engine_args.return_value = mock_engine_instance

        # Create engine
        engine = await create_async_engine()

        # Generate request
        result = await generate_single_request(
            engine,
            "Test prompt",
            Mock(),
            generate_request_id()
        )

        assert result == "Test output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
