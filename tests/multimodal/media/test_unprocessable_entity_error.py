# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for VLLMUnprocessableEntityError and media fetch error handling.

Verifies that unprocessable image URLs (404, 403, DNS failures, etc.) return
HTTP 422 instead of 500.
"""

from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.exceptions import VLLMUnprocessableEntityError
from vllm.multimodal.media import MediaConnector
from vllm.multimodal.media.connector import _wrap_media_fetch_error


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
        """DNS errors are transient and should remain as-is for retry."""
        connector = MediaConnector()

        with patch.object(
            connector.connection, "async_get_bytes", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientConnectorDNSError(
                connection_key=MagicMock(),
                os_error=MagicMock(),
            )

            with pytest.raises(aiohttp.ClientConnectorDNSError) as exc_info:
                await connector.fetch_image_async(
                    "https://nonexistent.example/image.jpg"
                )

            assert isinstance(exc_info.value, aiohttp.ClientConnectorDNSError)

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
        """Connection errors are transient and should remain as-is for retry."""
        connector = MediaConnector()

        with patch.object(
            connector.connection, "get_bytes", new_callable=MagicMock
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientConnectionError("Connection refused")

            with pytest.raises(aiohttp.ClientConnectionError) as exc_info:
                connector.fetch_image("https://example.com/image.jpg")

            assert isinstance(exc_info.value, aiohttp.ClientConnectionError)


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


class TestIntegration:
    """Integration tests for full error flow."""

    def test_404_to_http_422(self):
        """Verify 404 error results in HTTP 422 response."""

        url = "https://example.com/missing.jpg"
        exc = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=404,
            message="Not Found",
        )

        wrapped = _wrap_media_fetch_error(url, exc)
        response = create_error_response(wrapped)

        assert response.error.code == 422
        assert response.error.type == "UnprocessableEntityError"

    def test_dns_remains_retryable(self):
        """Verify DNS error remains as-is for retry (not converted to 422)."""

        url = "https://nonexistent.example/image.jpg"
        exc = aiohttp.ClientConnectorDNSError(
            connection_key=MagicMock(),
            os_error=MagicMock(),
        )

        wrapped = _wrap_media_fetch_error(url, exc)
        response = create_error_response(wrapped)

        assert response.error.code == 500
        assert response.error.type == "InternalServerError"

    def test_500_remains_500(self):
        """Verify 500 error remains HTTP 500."""

        url = "https://example.com/image.jpg"
        exc = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=500,
            message="Internal Server Error",
        )

        wrapped = _wrap_media_fetch_error(url, exc)
        response = create_error_response(wrapped)

        assert response.error.code == 500
        assert response.error.type == "InternalServerError"
