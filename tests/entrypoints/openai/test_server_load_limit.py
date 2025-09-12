# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for server load limit functionality."""

from unittest.mock import MagicMock

import pytest
from fastapi.responses import JSONResponse

from vllm.entrypoints.utils import load_aware_call


class TestServerLoadLimit:
    """Test suite for server load limiting functionality."""

    @pytest.mark.asyncio
    async def test_load_aware_call_max_load_exceeded(self):
        """Test that requests are rejected when max load is exceeded."""

        @load_aware_call
        async def dummy_handler(raw_request):
            return {"message": "success"}

        # Mock request with load exceeding limit
        mock_request = MagicMock()
        mock_request.app.state.enable_server_load_tracking = True
        mock_request.app.state.max_server_load = 10
        mock_request.app.state.server_load_metrics = 15  # Exceeds limit

        response = await dummy_handler(raw_request=mock_request)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 503

        # Verify error content
        import json
        content = json.loads(response.body.decode('utf-8'))
        assert content["error"]["type"] == "server_overloaded"
        assert "Server is currently overloaded" in content["error"]["message"]
        assert "Current load: 15" in content["error"]["message"]
        assert "Max load: 10" in content["error"]["message"]

    @pytest.mark.asyncio
    async def test_load_aware_call_max_load_at_limit(self):
        """Test that requests are rejected when load equals limit."""

        @load_aware_call
        async def dummy_handler(raw_request):
            return {"message": "success"}

        # Mock request with load exactly at limit
        mock_request = MagicMock()
        mock_request.app.state.enable_server_load_tracking = True
        mock_request.app.state.max_server_load = 10
        mock_request.app.state.server_load_metrics = 10  # At limit

        response = await dummy_handler(raw_request=mock_request)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_load_aware_call_max_load_under_limit(self):
        """Test that requests proceed normally when under limit."""

        @load_aware_call
        async def dummy_handler(raw_request):
            return {"message": "success"}

        # Mock request with load under limit
        mock_request = MagicMock()
        mock_request.app.state.enable_server_load_tracking = True
        mock_request.app.state.max_server_load = 10
        mock_request.app.state.server_load_metrics = 5  # Under limit

        response = await dummy_handler(raw_request=mock_request)

        # Should proceed normally
        assert response == {"message": "success"}

    @pytest.mark.asyncio
    async def test_load_aware_call_max_load_not_set(self):
        """Test that requests proceed normally when max_server_load is None."""

        @load_aware_call
        async def dummy_handler(raw_request):
            return {"message": "success"}

        # Mock request with no max load set
        mock_request = MagicMock()
        mock_request.app.state.enable_server_load_tracking = True
        mock_request.app.state.max_server_load = None  # No limit
        mock_request.app.state.server_load_metrics = 100  # High load

        response = await dummy_handler(raw_request=mock_request)

        # Should proceed normally despite high load
        assert response == {"message": "success"}

    @pytest.mark.asyncio
    async def test_load_aware_call_tracking_disabled(self):
        """Test that load limiting is bypassed when tracking is disabled."""

        @load_aware_call
        async def dummy_handler(raw_request):
            return {"message": "success"}

        # Mock request with tracking disabled
        mock_request = MagicMock()
        mock_request.app.state.enable_server_load_tracking = False
        mock_request.app.state.max_server_load = 5
        mock_request.app.state.server_load_metrics = 100  # High load

        response = await dummy_handler(raw_request=mock_request)

        # Should proceed normally when tracking is disabled
        assert response == {"message": "success"}

    @pytest.mark.asyncio
    async def test_load_aware_call_with_exception(self):
        """Test that load counter is properly decremented on exception."""

        @load_aware_call
        async def failing_handler(raw_request):
            raise ValueError("Test exception")

        # Mock request under limit
        mock_request = MagicMock()
        mock_request.app.state.enable_server_load_tracking = True
        mock_request.app.state.max_server_load = 10
        mock_request.app.state.server_load_metrics = 5

        # Should raise the original exception
        with pytest.raises(ValueError, match="Test exception"):
            await failing_handler(raw_request=mock_request)

        # Load counter should be decremented back to 5
        assert mock_request.app.state.server_load_metrics == 5

    @pytest.mark.asyncio
    async def test_load_aware_call_increments_counter(self):
        """Test that load counter is properly incremented."""

        @load_aware_call
        async def dummy_handler(raw_request):
            # Verify counter was incremented
            assert raw_request.app.state.server_load_metrics == 6
            return {"message": "success"}

        # Mock request under limit
        mock_request = MagicMock()
        mock_request.app.state.enable_server_load_tracking = True
        mock_request.app.state.max_server_load = 10
        mock_request.app.state.server_load_metrics = 5

        response = await dummy_handler(raw_request=mock_request)

        assert response == {"message": "success"}

    @pytest.mark.asyncio
    async def test_load_aware_call_zero_max_load(self):
        """Test behavior when max_server_load is set to 0."""

        @load_aware_call
        async def dummy_handler(raw_request):
            return {"message": "success"}

        # Mock request with zero max load
        mock_request = MagicMock()
        mock_request.app.state.enable_server_load_tracking = True
        mock_request.app.state.max_server_load = 0
        mock_request.app.state.server_load_metrics = 0

        response = await dummy_handler(raw_request=mock_request)

        # Should be rejected since 0 >= 0
        assert isinstance(response, JSONResponse)
        assert response.status_code == 503

    def test_max_server_load_parameter_exists(self):
        """Test that max_server_load parameter is properly defined."""
        from vllm.entrypoints.openai.cli_args import FrontendArgs

        # Check that the parameter exists in FrontendArgs
        frontend_args = FrontendArgs()
        assert hasattr(frontend_args, 'max_server_load')
        assert frontend_args.max_server_load is None  # Default value

    def test_frontend_args_annotation(self):
        """Test that max_server_load has proper type annotation."""
        from vllm.entrypoints.openai.cli_args import FrontendArgs

        # Get type hints
        annotations = FrontendArgs.__annotations__
        assert 'max_server_load' in annotations

        # Should be Optional[int]
        import typing
        expected_type = typing.Optional[int]
        assert annotations['max_server_load'] == expected_type
