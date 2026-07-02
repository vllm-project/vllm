# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import aiohttp
import requests

from vllm.connections import _is_retryable


class TestIsRetryable:
    def test_timeouterror(self):
        assert _is_retryable(TimeoutError()) is True

    def test_asyncio_timeouterror(self):
        assert _is_retryable(asyncio.TimeoutError()) is True

    def test_requests_timeout(self):
        exc = requests.exceptions.Timeout()
        assert _is_retryable(exc) is True

    def test_aiohttp_server_timeout(self):
        exc = aiohttp.ServerTimeoutError()
        assert _is_retryable(exc) is True

    def test_connectionerror(self):
        assert _is_retryable(ConnectionError()) is True

    def test_requests_connection_error(self):
        exc = requests.exceptions.ConnectionError()
        assert _is_retryable(exc) is True

    def test_aiohttp_client_connection_error(self):
        exc = aiohttp.ClientConnectionError()
        assert _is_retryable(exc) is True

    def test_aiohttp_server_disconnected(self):
        exc = aiohttp.ServerDisconnectedError()
        assert _is_retryable(exc) is True

    def test_requests_http_5xx(self):
        resp = requests.Response()
        resp.status_code = 503
        exc = requests.exceptions.HTTPError(response=resp)
        assert _is_retryable(exc) is True

    def test_requests_http_500(self):
        resp = requests.Response()
        resp.status_code = 500
        exc = requests.exceptions.HTTPError(response=resp)
        assert _is_retryable(exc) is True

    def test_requests_http_4xx_not_retryable(self):
        resp = requests.Response()
        resp.status_code = 404
        exc = requests.exceptions.HTTPError(response=resp)
        assert _is_retryable(exc) is False

    def test_aiohttp_client_response_5xx(self):
        exc = aiohttp.ClientResponseError(
            request_info=None,  # type: ignore[arg-type]
            history=(),
            status=503,
        )
        assert _is_retryable(exc) is True

    def test_aiohttp_client_response_4xx_not_retryable(self):
        exc = aiohttp.ClientResponseError(
            request_info=None,  # type: ignore[arg-type]
            history=(),
            status=404,
        )
        assert _is_retryable(exc) is False

    def test_valueerror_not_retryable(self):
        assert _is_retryable(ValueError()) is False

    def test_typeerror_not_retryable(self):
        assert _is_retryable(TypeError()) is False

    def test_keyerror_not_retryable(self):
        assert _is_retryable(KeyError()) is False
