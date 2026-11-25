# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for VLLMUnprocessableEntityError and media fetch error handling.

Verifies that unprocessable image URLs (404, 403, DNS failures, etc.) return
HTTP 422 instead of 500.
"""

import errno
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import requests

from vllm.connections import (
    HTTPConnection,
    MediaDownloadSizeExceededError,
    MediaRequestDownloadSizeExceededError,
)
from vllm.entrypoints.serve import create_error_response
from vllm.exceptions import VLLMClientError, VLLMUnprocessableEntityError
from vllm.multimodal.media import MediaConnector

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


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

    def test_is_client_error_subclass(self):
        exc = VLLMUnprocessableEntityError("Test")
        assert isinstance(exc, VLLMClientError)


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
        """DNS errors that survive the fetch retries are reported as 422."""
        connector = MediaConnector()

        with patch.object(
            connector.connection, "async_get_bytes", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientConnectorDNSError(
                connection_key=MagicMock(),
                os_error=OSError(-2, "Name or service not known"),
            )

            with pytest.raises(VLLMUnprocessableEntityError) as exc_info:
                await connector.fetch_image_async(
                    "https://nonexistent.example/image.jpg"
                )

            assert exc_info.value.parameter == "image_url"
            assert "Could not connect to host" in str(exc_info.value)
            assert isinstance(exc_info.value.__cause__, aiohttp.ClientConnectorDNSError)

    @pytest.mark.asyncio
    async def test_fetch_image_async_500_after_retries(self):
        """5xx from the media host, after retries, is reported as 422."""
        connector = MediaConnector()

        with patch.object(
            connector.connection, "async_get_bytes", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=503,
                message="Service Unavailable",
            )

            with pytest.raises(VLLMUnprocessableEntityError) as exc_info:
                await connector.fetch_image_async("https://example.com/image.jpg")

            assert "HTTP 503 error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_image_async_timeout(self):
        connector = MediaConnector()

        with patch.object(
            connector.connection, "async_get_bytes", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = aiohttp.ServerTimeoutError("Timeout on reading")

            with pytest.raises(VLLMUnprocessableEntityError) as exc_info:
                await connector.fetch_image_async("https://example.com/image.jpg")

            assert "Request timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_image_async_local_fd_exhaustion_preserved(self):
        """EMFILE on this server is a server fault, not a bad URL."""
        connector = MediaConnector()

        with patch.object(
            connector.connection, "async_get_bytes", new_callable=AsyncMock
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientConnectorError(
                connection_key=MagicMock(),
                os_error=OSError(errno.EMFILE, "Too many open files"),
            )

            with pytest.raises(aiohttp.ClientConnectorError):
                await connector.fetch_image_async("https://example.com/image.jpg")

    @pytest.mark.asyncio
    async def test_fetch_image_async_429_preserved(self):
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
        """Connection errors that survive the fetch retries are reported as 422."""
        connector = MediaConnector()

        with patch.object(
            connector.connection, "get_bytes", new_callable=MagicMock
        ) as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError(
                "HTTPConnectionPool(host='10.0.0.5', port=80): Max retries exceeded"
            )

            with pytest.raises(VLLMUnprocessableEntityError) as exc_info:
                connector.fetch_image("https://example.com/image.jpg")

            # The message must not echo the internal host or address.
            assert "10.0.0.5" not in str(exc_info.value)
            assert "Could not connect to host" in str(exc_info.value)

    def test_fetch_image_dns_error_retried_then_422(self, monkeypatch):
        """The retry loop in HTTPConnection runs before the error is wrapped."""
        monkeypatch.setenv("VLLM_MEDIA_FETCH_MAX_RETRIES", "3")
        monkeypatch.setattr("vllm.connections.time.sleep", lambda _: None)
        connector = MediaConnector(connection=HTTPConnection())

        dns_error = requests.exceptions.ConnectionError("Name or service not known")
        with (
            patch.object(
                connector.connection, "get_response", side_effect=dns_error
            ) as mock_get,
            pytest.raises(VLLMUnprocessableEntityError),
        ):
            connector.fetch_image("https://nonexistent.example/image.jpg")

        assert mock_get.call_count == 3

    def test_fetch_image_download_size_limit_error(self):
        connector = MediaConnector()

        with patch.object(
            connector.connection, "get_bytes", new_callable=MagicMock
        ) as mock_get:
            mock_get.side_effect = MediaDownloadSizeExceededError(256 * 1024 * 1024)

            with pytest.raises(VLLMUnprocessableEntityError) as exc_info:
                connector.fetch_image("https://example.com/image.jpg")

            assert "VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB=256" in str(exc_info.value)

    def test_fetch_image_request_download_size_limit_error(self):
        connector = MediaConnector()

        with patch.object(
            connector.connection, "get_bytes", new_callable=MagicMock
        ) as mock_get:
            mock_get.side_effect = MediaRequestDownloadSizeExceededError(
                "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB=256"
            )

            with pytest.raises(VLLMUnprocessableEntityError) as exc_info:
                connector.fetch_image("https://example.com/image.jpg")

            assert "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB=256" in str(exc_info.value)


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
