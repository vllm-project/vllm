# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import functools
import ssl
import time
from collections.abc import Callable, Coroutine, Mapping, MutableMapping
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
from urllib3.util import Url, parse_url

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


class SSRFSafeClient(requests.Session):
    """Wrapper class of the request.Session client that sends requests
    using the pre-validated IP obtained for the original domain used in the URL.
    This client is used when the --forbid-media-private-networks-access flag is
    enforced to prevent TOCTOU DNS rebinding SSRF bypass.
    """

    def __init__(self, url: str, pre_validated_ip: str, *args, **kwargs):
        self.pre_validated_ip = pre_validated_ip
        # Extract hostname and scheme to be injected into adapter
        self.parsed_url = parse_url(url)
        self.scheme = self.parsed_url.scheme
        self.original_hostname = self.parsed_url.hostname or self.parsed_url.host or ""

        # Initialize requests.session
        super().__init__(*args, **kwargs)

        # Enforce original host headers on that session
        self._update_headers()
        # Mount original-host-aware custom adapter for that session session
        self._setup_adapter()

    # Monky-patch the client.request of the request.Session object. This ensure
    # that all the helpers methods of the client (GET, OPTIONS, etc ...) will
    # use the pre-validated-ip instead of the original domain.
    def request(self, method: str, url: str, *args, **kwargs):  # type: ignore
        # Reconstruct the url, replacing the hostname with the pre-validated IP address
        ip_url = self._get_ip_url(self.parsed_url, self.pre_validated_ip)
        return super().request(method, ip_url, *args, **kwargs)

    def _setup_adapter(self):
        is_ssl = self.scheme == "https"
        adapter = SSRFSafeAdapter(self.original_hostname, is_ssl)
        # Mount a custom adapter to the schema matching the url.
        # For https requests, the adapter will ensure https certificate is
        # checked against cn=original_hostname and not the ip present in url.
        self.mount(f"{self.scheme}://", adapter)

    def _update_headers(self):
        # Send request with proper Host header (original hostname)
        self.headers.update(
            {
                "Host": self.original_hostname,
            }
        )

    @staticmethod
    def _get_ip_url(parsed_url, ip_to_enforce):
        return Url(
            scheme=parsed_url.scheme,
            auth=parsed_url.auth,
            host=ip_to_enforce,
            port=parsed_url.port,
            path=parsed_url.path,
            query=parsed_url.query,
            fragment=parsed_url.fragment,
        )


class SSRFSafeAdapter(HTTPAdapter):
    """Adapter that connects to an IP address
    but validates TLS for the original host.
    """

    def __init__(self, original_hostname, is_ssl, *args, **kwargs):
        self.original_hostname = original_hostname
        self.is_ssl = is_ssl
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["server_hostname"] = self.original_hostname
        if self.is_ssl:
            context = ssl.create_default_context()
            kwargs["ssl_context"] = context
        self.poolmanager = PoolManager(*args, **kwargs)


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

    def get_sync_ssrf_safe_client(
        self, url: str, pre_validated_ip: str
    ) -> SSRFSafeClient:
        return SSRFSafeClient(url, pre_validated_ip)

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
        pre_validated_ip: str | None = None,
    ):
        self._validate_http_url(url)

        client = self.get_sync_client()
        if pre_validated_ip:
            # If a pre-validated IP has been provided, we use a custom client
            # to enforce that IP and prevent DNS rebinding
            client = self.get_sync_ssrf_safe_client(url, pre_validated_ip)
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
        pre_validated_ip: str | None = None,
    ):
        self._validate_http_url(url)

        # TODO: if a pre-validated IP is provided, build a custom
        # aiohttp.TCPConnector with custom resolver that returns the pre-validated IP

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
        self,
        url: str,
        *,
        timeout: float | None = None,
        allow_redirects: bool = True,
        pre_validated_ip: str | None = None,
    ) -> bytes:
        with self.get_response(
            url,
            timeout=timeout,
            allow_redirects=allow_redirects,
            pre_validated_ip=pre_validated_ip,
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
        pre_validated_ip: str | None = None,
    ) -> bytes:
        async with await self.get_async_response(
            url,
            timeout=timeout,
            allow_redirects=allow_redirects,
            pre_validated_ip=pre_validated_ip,
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
