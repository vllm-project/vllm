# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import functools
import time
from collections.abc import Callable, Coroutine, Mapping, MutableMapping
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import aiohttp
import requests
from urllib3.util import parse_url

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

_P = ParamSpec("_P")
_T = TypeVar("_T")

# Multiplier applied to timeout and sleep on each retry attempt.
# Attempt N uses: base_timeout * (_RETRY_BACKOFF_FACTOR ** N) for the
# per-attempt timeout and sleeps _RETRY_BACKOFF_FACTOR ** N seconds.
_RETRY_BACKOFF_FACTOR = 4


def _is_retryable(exc: Exception) -> bool:
    """Return True for transient errors that are worth retrying.

    Retryable:
      - Timeouts (aiohttp, requests, stdlib)
      - Connection-level failures (refused, reset, DNS)
      - Server errors (5xx) -- includes S3 503 SlowDown
    Not retryable:
      - Client errors (4xx) -- bad URL, auth, not-found
      - Programming errors (ValueError, TypeError, ...)
    """
    # Timeouts
    if isinstance(
        exc,
        (
            TimeoutError,
            asyncio.TimeoutError,
            requests.exceptions.Timeout,
            aiohttp.ServerTimeoutError,
        ),
    ):
        return True
    # Connection-level failures
    if isinstance(
        exc,
        (
            ConnectionError,
            aiohttp.ClientConnectionError,
            requests.exceptions.ConnectionError,
        ),
    ):
        return True
    # aiohttp server-side disconnects
    if isinstance(exc, aiohttp.ServerDisconnectedError):
        return True
    # requests 5xx -- raise_for_status() throws HTTPError
    if (
        isinstance(exc, requests.exceptions.HTTPError)
        and exc.response is not None
        and exc.response.status_code >= 500
    ):
        return True
    # aiohttp 5xx -- raise_for_status() throws ClientResponseError
    return isinstance(exc, aiohttp.ClientResponseError) and exc.status >= 500


def _log_retry(
    args: tuple,
    kwargs: dict,
    attempt: int,
    max_retries: int,
    attempt_timeout: float | None,
    exc: Exception,
    backoff: float,
    base_timeout: float | None,
) -> None:
    # args[0] is `self` (bound method), args[1] is the URL
    url = args[1] if len(args) > 1 else kwargs.get("url")
    timeout_info = (
        f"timeout={attempt_timeout:.3f}s" if base_timeout is not None else "no timeout"
    )
    next_timeout = (
        f" with timeout={base_timeout * (_RETRY_BACKOFF_FACTOR ** (attempt + 1)):.3f}s"
        if base_timeout is not None
        else ""
    )
    logger.warning(
        "HTTP fetch failed for %s (attempt %d/%d, %s): %s -- retrying in %.3fs%s",
        url,
        attempt + 1,
        max_retries,
        timeout_info,
        exc,
        backoff,
        next_timeout,
    )


def _sync_retry(
    fn: Callable[_P, _T],
) -> Callable[_P, _T]:
    """Add retry logic with exponential backoff to a sync method.

    The decorated method must accept ``timeout`` as a keyword argument.
    The decorator replaces it with a per-attempt timeout that grows by
    ``_RETRY_BACKOFF_FACTOR`` on each retry so transient slowness on busy
    hosts is absorbed.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> _T:
        base_timeout: float | None = kwargs.get("timeout")
        max_retries = max(envs.VLLM_MEDIA_FETCH_MAX_RETRIES, 1)

        for attempt in range(max_retries):
            attempt_timeout = (
                base_timeout * (_RETRY_BACKOFF_FACTOR**attempt)
                if base_timeout is not None
                else None
            )
            kwargs["timeout"] = attempt_timeout
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if not _is_retryable(e) or attempt + 1 >= max_retries:
                    raise
                backoff = _RETRY_BACKOFF_FACTOR**attempt
                _log_retry(
                    args,
                    kwargs,
                    attempt,
                    max_retries,
                    attempt_timeout,
                    e,
                    backoff,
                    base_timeout,
                )
                time.sleep(backoff)

        raise AssertionError("unreachable")

    return wrapper  # type: ignore[return-value]


def _async_retry(
    fn: Callable[_P, Coroutine[Any, Any, _T]],
) -> Callable[_P, Coroutine[Any, Any, _T]]:
    """Add retry logic with exponential backoff to an async method.

    The decorated method must accept ``timeout`` as a keyword argument.
    The decorator replaces it with a per-attempt timeout that grows by
    ``_RETRY_BACKOFF_FACTOR`` on each retry so transient slowness on busy
    hosts is absorbed.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> _T:
        base_timeout: float | None = kwargs.get("timeout")
        max_retries = max(envs.VLLM_MEDIA_FETCH_MAX_RETRIES, 1)

        for attempt in range(max_retries):
            attempt_timeout = (
                base_timeout * (_RETRY_BACKOFF_FACTOR**attempt)
                if base_timeout is not None
                else None
            )
            kwargs["timeout"] = attempt_timeout
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                if not _is_retryable(e) or attempt + 1 >= max_retries:
                    raise
                backoff = _RETRY_BACKOFF_FACTOR**attempt
                _log_retry(
                    args,
                    kwargs,
                    attempt,
                    max_retries,
                    attempt_timeout,
                    e,
                    backoff,
                    base_timeout,
                )
                await asyncio.sleep(backoff)

        raise AssertionError("unreachable")

    return wrapper  # type: ignore[return-value]


class HTTPConnection:
    """Helper class to send HTTP requests."""

    def __init__(self, *, reuse_client: bool = True) -> None:
        super().__init__()

        self.reuse_client = reuse_client

        self._sync_client: requests.Session | None = None
        self._async_client: aiohttp.ClientSession | None = None

    def get_sync_client(self) -> requests.Session:
        if self._sync_client is None or not self.reuse_client:
            self._sync_client = requests.Session()

        return self._sync_client

    # NOTE: We intentionally use an async function even though it is not
    # required, so that the client is only accessible inside async event loop
    async def get_async_client(self) -> aiohttp.ClientSession:
        if self._async_client is None or not self.reuse_client:
            self._async_client = aiohttp.ClientSession(trust_env=True)

        return self._async_client

    def _validate_http_url(self, url: str):
        parsed_url = parse_url(url)

        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(
                "Invalid HTTP URL: A valid HTTP URL must have scheme 'http' or 'https'."
            )

    def _headers(self, **extras: str) -> MutableMapping[str, str]:
        return {"User-Agent": f"vLLM/{VLLM_VERSION}", **extras}

    def get_response(
        self,
        url: str,
        *,
        stream: bool = False,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
        allow_redirects: bool = True,
    ):
        self._validate_http_url(url)

        client = self.get_sync_client()
        extra_headers = extra_headers or {}

        return client.get(
            url,
            headers=self._headers(**extra_headers),
            stream=stream,
            timeout=timeout,
            allow_redirects=allow_redirects,
        )

    async def get_async_response(
        self,
        url: str,
        *,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
        allow_redirects: bool = True,
    ):
        self._validate_http_url(url)

        client = await self.get_async_client()
        extra_headers = extra_headers or {}

        return client.get(
            url,
            headers=self._headers(**extra_headers),
            timeout=timeout,
            allow_redirects=allow_redirects,
        )

    @_sync_retry
    def get_bytes(
        self, url: str, *, timeout: float | None = None, allow_redirects: bool = True
    ) -> bytes:
        with self.get_response(
            url, timeout=timeout, allow_redirects=allow_redirects
        ) as r:
            r.raise_for_status()

            return r.content

    @_async_retry
    async def async_get_bytes(
        self,
        url: str,
        *,
        timeout: float | None = None,
        allow_redirects: bool = True,
    ) -> bytes:
        async with await self.get_async_response(
            url, timeout=timeout, allow_redirects=allow_redirects
        ) as r:
            r.raise_for_status()

            return await r.read()

    def get_text(self, url: str, *, timeout: float | None = None) -> str:
        with self.get_response(url, timeout=timeout) as r:
            r.raise_for_status()

            return r.text

    async def async_get_text(
        self,
        url: str,
        *,
        timeout: float | None = None,
    ) -> str:
        async with await self.get_async_response(url, timeout=timeout) as r:
            r.raise_for_status()

            return await r.text()

    def get_json(self, url: str, *, timeout: float | None = None) -> str:
        with self.get_response(url, timeout=timeout) as r:
            r.raise_for_status()

            return r.json()

    async def async_get_json(
        self,
        url: str,
        *,
        timeout: float | None = None,
    ) -> str:
        async with await self.get_async_response(url, timeout=timeout) as r:
            r.raise_for_status()

            return await r.json()

    @_sync_retry
    def download_file(
        self,
        url: str,
        save_path: Path,
        *,
        timeout: float | None = None,
        chunk_size: int = 128,
    ) -> Path:
        try:
            with self.get_response(url, timeout=timeout) as r:
                r.raise_for_status()

                with save_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size):
                        f.write(chunk)

            return save_path
        except Exception:
            # Clean up partial downloads before retrying or propagating
            if save_path.exists():
                save_path.unlink()
            raise

    @_async_retry
    async def async_download_file(
        self,
        url: str,
        save_path: Path,
        *,
        timeout: float | None = None,
        chunk_size: int = 128,
    ) -> Path:
        try:
            async with await self.get_async_response(
                url,
                timeout=timeout,
            ) as r:
                r.raise_for_status()

                with save_path.open("wb") as f:
                    async for chunk in r.content.iter_chunked(chunk_size):
                        f.write(chunk)

            return save_path
        except Exception:
            # Clean up partial downloads before retrying or propagating
            if save_path.exists():
                save_path.unlink()
            raise


global_http_connection = HTTPConnection()
"""
The global [`HTTPConnection`][vllm.connections.HTTPConnection] instance used
by vLLM.
"""
