# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for VLLMUnprocessableEntityError and media fetch error handling."""

from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import requests

from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.exceptions import VLLMUnprocessableEntityError
from vllm.multimodal.media import MediaConnector


class TestVLLMUnprocessableEntityError:
    """Tests for VLLMUnprocessableEntityError exception."""

    def test_creation(self):
        exc = VLLMUnprocessableEntityError("Test error")
        assert str(exc) == "Test error"
        assert exc.parameter is None

    def test_creation_with_parameter_and_value(self):
        exc = VLLMUnprocessableEntityError(
            "Test error",
            parameter="image_url",
            value="https://example.com/image.jpg",
        )
        assert "parameter=image_url" in str(exc)
        assert "value=https://example.com/image.jpg" in str(exc)

    def test_is_value_error_subclass(self):
        exc = VLLMUnprocessableEntityError("Test")
        assert isinstance(exc, ValueError)


class TestMediaConnectorErrorHandling:
    """Tests for MediaConnector error handling."""

    @pytest.mark.asyncio
    async def test_fetch_image_async_404(self):
        connector = MediaConnector()

        with patch.object(
            connector.connection, "async_get_bytes", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=404,
                message="Not Found",
            )

            with pytest.raises(VLLMUnprocessableEntityError) as exc_info:
                await connector.fetch_image_async("https://example.com/missing.jpg")

            assert exc_info.value.parameter == "image_url"

    @pytest.mark.asyncio
    async def test_fetch_image_async_dns_error(self):
        """DNS errors should remain retryable transport failures."""
        connector = MediaConnector()

        with patch.object(
            connector.connection, "async_get_bytes", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientConnectorDNSError(
                connection_key=MagicMock(),
                os_error=MagicMock(),
            )

            with pytest.raises(aiohttp.ClientConnectorDNSError):
                await connector.fetch_image_async(
                    "https://nonexistent.example/image.jpg"
                )

    @pytest.mark.asyncio
    async def test_fetch_image_async_429(self):
        """HTTP 429 should remain a retryable upstream response."""
        connector = MediaConnector()

        with patch.object(
            connector.connection, "async_get_bytes", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=429,
                message="Too Many Requests",
            )

            with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                await connector.fetch_image_async("https://example.com/image.jpg")

            assert exc_info.value.status == 429

    @pytest.mark.asyncio
    async def test_fetch_image_async_invalid_url(self):
        connector = MediaConnector()

        with patch.object(
            connector.connection, "async_get_bytes", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = aiohttp.InvalidURL(
                "http://internal.example:8080/image.jpg"
            )

            with pytest.raises(VLLMUnprocessableEntityError) as exc_info:
                await connector.fetch_image_async("http:// bad")

            assert exc_info.value.parameter == "image_url"
            assert str(exc_info.value).startswith(
                "Failed to fetch media from URL: Invalid URL"
            )
            assert "internal.example" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_image_async_500_preserved(self):
        """5xx errors should remain as server errors."""
        connector = MediaConnector()

        with patch.object(
            connector.connection, "async_get_bytes", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=500,
                message="Internal Server Error",
            )

            with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                await connector.fetch_image_async("https://example.com/image.jpg")

            assert exc_info.value.status == 500

    def test_fetch_image_404(self):
        connector = MediaConnector()

        with patch.object(
            connector.connection, "get_bytes", new_callable=MagicMock
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=404,
                message="Not Found",
            )

            with pytest.raises(VLLMUnprocessableEntityError) as exc_info:
                connector.fetch_image("https://example.com/missing.jpg")

            assert exc_info.value.parameter == "image_url"

    def test_fetch_image_connection_error(self):
        """Connection errors should remain retryable transport failures."""
        connector = MediaConnector()

        with patch.object(
            connector.connection, "get_bytes", new_callable=MagicMock
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientConnectionError("Connection refused")

            with pytest.raises(aiohttp.ClientConnectionError):
                connector.fetch_image("https://example.com/image.jpg")

    def test_fetch_image_requests_connection_error(self):
        """Requests connection errors should remain retryable failures."""
        connector = MediaConnector()

        with patch.object(
            connector.connection, "get_bytes", new_callable=MagicMock
        ) as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError(
                "Connection refused"
            )

            with pytest.raises(requests.exceptions.ConnectionError):
                connector.fetch_image("https://example.com/image.jpg")

    def test_fetch_image_requests_invalid_url(self):
        connector = MediaConnector()

        with patch.object(
            connector.connection, "get_bytes", new_callable=MagicMock
        ) as mock_get:
            mock_get.side_effect = requests.exceptions.InvalidURL(
                "No host supplied for internal.example:8080"
            )

            with pytest.raises(VLLMUnprocessableEntityError) as exc_info:
                connector.fetch_image("http:// bad")

            assert exc_info.value.parameter == "image_url"
            assert str(exc_info.value).startswith(
                "Failed to fetch media from URL: Invalid URL"
            )
            assert "internal.example" not in str(exc_info.value)


class TestErrorResponse:
    """Tests for error response creation."""

    def test_unprocessable_entity_returns_422(self):
        exc = VLLMUnprocessableEntityError(
            "Failed to fetch media from URL: Cannot connect",
            parameter="image_url",
            value="https://example.com/image.jpg",
        )

        response = create_error_response(exc)

        assert response.error.code == HTTPStatus.UNPROCESSABLE_ENTITY.value
        assert response.error.type == "UnprocessableEntityError"
        assert response.error.param == "image_url"

    def test_unprocessable_entity_message(self):
        exc = VLLMUnprocessableEntityError("Test error message")
        response = create_error_response(exc)

        assert response.error.message == "Test error message"
        assert response.error.code == 422
