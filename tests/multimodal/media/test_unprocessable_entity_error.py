# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for VLLMUnprocessableEntityError and media fetch error handling.

Verifies that unprocessable image URLs (404, 403, DNS failures, etc.) return
HTTP 422 instead of 500.
"""

from http import HTTPStatus
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import numpy as np
import pytest

from vllm.connections import MediaDownloadSizeExceededError
from vllm.entrypoints.serve import create_error_response
from vllm.exceptions import VLLMClientError, VLLMUnprocessableEntityError
from vllm.multimodal.media import AudioMediaIO, ImageMediaIO, MediaConnector, MediaRef
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems
from vllm.multimodal.processing.processor import BaseMultiModalProcessor


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
                await connector.fetch_image_async(
                    "https://example.com/missing.jpg", ImageMediaIO()
                )

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
                    "https://nonexistent.example/image.jpg", ImageMediaIO()
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
                await connector.fetch_image_async(
                    "https://example.com/image.jpg", ImageMediaIO()
                )

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
                connector.fetch_image("https://example.com/missing.jpg", ImageMediaIO())

            assert exc_info.value.parameter == "image_url"

    def test_fetch_image_connection_error(self):
        """Connection errors are transient and should remain as-is for retry."""
        connector = MediaConnector()

        with patch.object(
            connector.connection, "get_bytes", new_callable=MagicMock
        ) as mock_get:
            mock_get.side_effect = aiohttp.ClientConnectionError("Connection refused")

            with pytest.raises(aiohttp.ClientConnectionError) as exc_info:
                connector.fetch_image("https://example.com/image.jpg", ImageMediaIO())

            assert isinstance(exc_info.value, aiohttp.ClientConnectionError)

    def test_fetch_image_download_size_limit_error(self):
        connector = MediaConnector()

        with patch.object(
            connector.connection, "get_bytes", new_callable=MagicMock
        ) as mock_get:
            mock_get.side_effect = MediaDownloadSizeExceededError(256 * 1024 * 1024)

            with pytest.raises(VLLMUnprocessableEntityError) as exc_info:
                connector.fetch_image("https://example.com/image.jpg", ImageMediaIO())

            assert "VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB=256" in str(exc_info.value)


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

    def test_processor_decode_error_returns_422(self):
        """Corrupt media bytes decoded inside the mm processor surface as 422.

        Intentional behavior change (400 -> 422): with eager decoding a
        corrupt payload raised a plain decode error at fetch time, which
        fell through to the 400 BadRequest fallback; the processor's
        lazy-decode phase now wraps it as VLLMUnprocessableEntityError.
        """
        corrupt = b"corrupt-not-an-audio"
        lazy = AudioMediaIO().load_bytes_ref(corrupt)
        mm_items = cast(
            MultiModalDataItems,
            {"audio": AudioProcessorItems([lazy])},  # type: ignore[list-item]
        )

        with pytest.raises(VLLMUnprocessableEntityError) as exc_info:
            # `_decode_ref_items` uses no instance state; call it unbound to
            # exercise the real decode-and-wrap path without a full processor.
            BaseMultiModalProcessor._decode_ref_items(
                cast("BaseMultiModalProcessor", None), mm_items
            )

        assert exc_info.value.parameter is None

        response = create_error_response(exc_info.value)

        assert response.error.code == HTTPStatus.UNPROCESSABLE_ENTITY.value
        assert response.error.type == "UnprocessableEntityError"
        assert response.error.param is None
        assert "Failed to decode audio media at index 0" in response.error.message

    def test_processor_decode_error_names_the_item_index(self):
        """The index is the item's position within its modality, so a client
        sending several items can tell which one is corrupt."""
        decoded = MediaRef(lambda: (np.zeros(4, dtype=np.float32), 16000.0), b"ok")
        decoded.decode()
        corrupt = AudioMediaIO().load_bytes_ref(b"corrupt-not-an-audio")
        mm_items = cast(
            MultiModalDataItems,
            # type: ignore[list-item]
            {"audio": AudioProcessorItems([decoded, corrupt])},
        )

        with pytest.raises(VLLMUnprocessableEntityError) as exc_info:
            BaseMultiModalProcessor._decode_ref_items(
                cast("BaseMultiModalProcessor", None), mm_items
            )

        assert "audio media at index 1" in str(exc_info.value)
