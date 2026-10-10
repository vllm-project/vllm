# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import functools
import ipaddress
import socket
import threading
import time
from collections.abc import Callable, Collection, Coroutine, Mapping, MutableMapping
from pathlib import Path
from typing import Any, ParamSpec, TypeVar
from urllib.parse import urljoin, urlsplit

import aiohttp
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import InvalidURL, TooManyRedirects
from requests.utils import select_proxy
from urllib3 import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.connection import HTTPConnection as Urllib3HTTPConnection
from urllib3.connection import HTTPSConnection as Urllib3HTTPSConnection
from urllib3.util import parse_url

import vllm.envs as envs
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.utils.mem_constants import KiB_bytes, MiB_bytes
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

_P = ParamSpec("_P")
_T = TypeVar("_T")

# Multiplier applied to timeout and sleep on each retry attempt.
# Attempt N uses: base_timeout * (_RETRY_BACKOFF_FACTOR ** N) for the
# per-attempt timeout and sleeps _RETRY_BACKOFF_FACTOR ** N seconds.
_RETRY_BACKOFF_FACTOR = 4
_RESPONSE_READ_CHUNK_SIZE = 64 * KiB_bytes


class HTTPResponseSizeExceededError(VLLMValidationError):
    """Raised when an HTTP response exceeds a caller-supplied byte ceiling."""

    def __init__(self, max_bytes: int, received_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.received_bytes = received_bytes
        super().__init__(
            "Maximum file size exceeded: HTTP response exceeded maximum size "
            f"of {max_bytes} bytes (received at least {received_bytes} bytes)",
            parameter="audio_filesize_mb",
            value=received_bytes / MiB_bytes,
        )


class MediaDownloadSizeExceededError(ValueError):
    """Raised when a remote media download exceeds the configured byte limit."""

    def __init__(self, max_bytes: int, received_bytes: int | None = None) -> None:
        self.max_bytes = max_bytes
        self.received_bytes = received_bytes
        max_size_mb = max_bytes / MiB_bytes
        message = (
            "Remote media download exceeds "
            f"VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB={max_size_mb:g}"
        )
        if received_bytes is not None:
            message += f" (received at least {received_bytes} bytes)"
        super().__init__(message)


def _get_media_download_limit_bytes() -> int | None:
    max_size_mb = envs.VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB
    if max_size_mb <= 0:
        return None
    return max_size_mb * MiB_bytes


def _get_response_size_limit(max_bytes: int | None) -> tuple[int | None, bool]:
    media_max_bytes = _get_media_download_limit_bytes()
    if media_max_bytes is not None and (
        max_bytes is None or media_max_bytes <= max_bytes
    ):
        return media_max_bytes, True
    return max_bytes, False


def _raise_size_exceeded(
    max_bytes: int, received_bytes: int, media_download_limit: bool
) -> None:
    if media_download_limit:
        raise MediaDownloadSizeExceededError(max_bytes, received_bytes)
    raise HTTPResponseSizeExceededError(max_bytes, received_bytes)


def _check_content_length(
    content_length: str | int | None,
    max_bytes: int | None,
    media_download_limit: bool,
) -> None:
    if content_length is None or max_bytes is None:
        return
    try:
        parsed_length = int(content_length)
    except (TypeError, ValueError):
        return
    if parsed_length > max_bytes:
        _raise_size_exceeded(max_bytes, parsed_length, media_download_limit)


# Maximum redirect hops followed by the media fetch helpers below. The
# libraries' own redirect following is disabled on this path so that every
# hop is re-validated (see _check_allowed_domains / _resolve_validated_ip);
# 10 matches aiohttp's default limit.
_MEDIA_MAX_REDIRECTS = 10


def _resolve_host_ips(hostname: str) -> list[str]:
    """Resolve a hostname once, returning every unique IP in answer order.

    A single resolution per hop (instead of one at check time and another at
    connect time) closes the DNS-rebinding TOCTOU: the addresses validated
    here are the addresses dialed below.
    """
    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"Cannot resolve media URL host {hostname!r}: {e}") from e
    ips = list(dict.fromkeys(info[4][0] for info in infos))
    if not ips:
        raise ValueError(f"Cannot resolve media URL host {hostname!r}: no addresses")
    return ips


def _resolve_validated_ip(hostname: str) -> str:
    """Resolve `hostname` once and refuse non-public destinations.

    Every resolved IP must be globally reachable (this rejects loopback,
    RFC1918, link-local, multicast, reserved, and unspecified addresses);
    a hostname with mixed answers is refused outright so a rebinding
    rotation cannot smuggle one bad record past the check. Returns the
    first resolved IP, which the caller must dial directly instead of
    resolving again. `VLLM_MEDIA_URL_ALLOW_PRIVATE_IPS=1` opts out (tests,
    trusted networks) but still pins the resolved IP.
    """
    if not hostname:
        raise ValueError("Cannot fetch media URL with a missing hostname")
    ips = _resolve_host_ips(hostname)
    if envs.VLLM_MEDIA_URL_ALLOW_PRIVATE_IPS:
        return ips[0]
    for ip in ips:
        try:
            public = ipaddress.ip_address(ip).is_global
        except ValueError:
            public = False
        if not public:
            raise ValueError(
                f"Refusing to fetch media URL with non-public IP {ip} "
                f"(host {hostname!r}). Loopback, private, link-local, "
                "multicast, and reserved destinations are blocked by "
                "default; set VLLM_MEDIA_URL_ALLOW_PRIVATE_IPS=1 to allow "
                "them."
            )
    return ips[0]


def _check_allowed_domains(url: str, allowed_domains: Collection[str] | None) -> None:
    """Re-check one redirect hop against the domain allowlist.

    Mirrors MediaConnector._assert_url_in_allowed_media_domains, which only
    sees the initial URL; every subsequent hop repeats the same check here.
    An empty/None allowlist keeps the historical allow-all behavior.
    """
    if not allowed_domains:
        return
    hostname = urlsplit(url).hostname
    if hostname not in allowed_domains:
        raise ValueError(
            f"The URL must be from one of the allowed domains: "
            f"{list(allowed_domains)}. Input URL domain: {hostname}"
        )


def _append_chunk(
    chunks: list[bytes],
    chunk: bytes,
    received_bytes: int,
    max_bytes: int | None,
    media_download_limit: bool,
) -> int:
    if not chunk:
        return received_bytes
    received_bytes += len(chunk)
    if max_bytes is not None and received_bytes > max_bytes:
        _raise_size_exceeded(max_bytes, received_bytes, media_download_limit)
    chunks.append(chunk)
    return received_bytes


def _read_response_bytes(response: requests.Response, max_bytes: int | None) -> bytes:
    max_bytes, media_download_limit = _get_response_size_limit(max_bytes)
    if max_bytes is None:
        return response.content

    _check_content_length(
        response.headers.get("Content-Length"),
        max_bytes,
        media_download_limit,
    )
    chunks: list[bytes] = []
    received_bytes = 0
    for chunk in response.iter_content(_RESPONSE_READ_CHUNK_SIZE):
        received_bytes = _append_chunk(
            chunks, chunk, received_bytes, max_bytes, media_download_limit
        )
    return b"".join(chunks)


async def _async_read_response_bytes(
    response: aiohttp.ClientResponse, max_bytes: int | None
) -> bytes:
    max_bytes, media_download_limit = _get_response_size_limit(max_bytes)
    if max_bytes is None:
        return await response.read()

    _check_content_length(response.content_length, max_bytes, media_download_limit)
    chunks: list[bytes] = []
    received_bytes = 0
    async for chunk in response.content.iter_chunked(_RESPONSE_READ_CHUNK_SIZE):
        received_bytes = _append_chunk(
            chunks, chunk, received_bytes, max_bytes, media_download_limit
        )
    return b"".join(chunks)


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


class _PinnedHTTPConnection(Urllib3HTTPConnection):
    """urllib3 connection that dials a pre-validated IP.

    Only `_dns_host` (the TCP destination) is overridden; the `host`
    property keeps returning the original hostname, so the `Host` header,
    SNI, and certificate checks are unchanged. Used proxy-free only.
    """

    @property  # type: ignore[override]
    def host(self) -> str:
        return self._orig_host

    @host.setter
    def host(self, value: str) -> None:
        self._orig_host = value
        self._dns_host = value


class _PinnedHTTPSConnection(Urllib3HTTPSConnection):
    """TLS sibling of _PinnedHTTPConnection (same split)."""

    @property  # type: ignore[override]
    def host(self) -> str:
        return self._orig_host

    @host.setter
    def host(self, value: str) -> None:
        self._orig_host = value
        self._dns_host = value


# Pool kwargs that plain-HTTP pools accept. The TLS keys requests attaches
# (ssl_context, cert_reqs, ca_certs, ...) are only valid on HTTPS pools and
# would break HTTPConnection.__init__.
_HTTP_POOL_KWARGS = frozenset({"timeout", "maxsize", "block", "headers", "retries"})


class _PinnedHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _PinnedHTTPConnection

    def __init__(self, *args: Any, dial_host: str, **kwargs: Any) -> None:
        self._dial_host = dial_host
        super().__init__(*args, **kwargs)

    def _new_conn(self) -> _PinnedHTTPConnection:
        conn = super()._new_conn()
        conn._dns_host = self._dial_host
        return conn


class _PinnedHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _PinnedHTTPSConnection

    def __init__(self, *args: Any, dial_host: str, **kwargs: Any) -> None:
        self._dial_host = dial_host
        super().__init__(*args, **kwargs)

    def _new_conn(self) -> _PinnedHTTPSConnection:
        conn = super()._new_conn()
        conn._dns_host = self._dial_host
        return conn


class _SSRFProtectiveAdapter(HTTPAdapter):
    """requests adapter that resolves once, validates, and dials the IP.

    For every proxy-free request the hostname is resolved a single time,
    every answer is IP-validated (_resolve_validated_ip), and the returned
    pool dials that IP while keeping the original hostname for the `Host`
    header, SNI, and certificate checks. Because requests resolves the pool
    per redirect hop, each hop is re-validated automatically. Proxied
    requests keep default behavior (the proxy dials) after the same
    hostname validation.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pinned_lock = threading.Lock()
        self._pinned_pools: dict[
            tuple[str, str, int, str], HTTPConnectionPool | HTTPSConnectionPool
        ] = {}

    def get_connection_with_tls_context(
        self,
        request: requests.PreparedRequest,
        verify: bool | str,
        proxies: dict[str, str] | None = None,
        cert: str | tuple[str, str] | None = None,
    ) -> HTTPConnectionPool | HTTPSConnectionPool:
        proxy = select_proxy(request.url, proxies)
        if proxy:
            _resolve_validated_ip(urlsplit(request.url).hostname or "")
            return super().get_connection_with_tls_context(
                request, verify, proxies, cert
            )
        try:
            host_params, pool_kwargs = self.build_connection_pool_key_attributes(
                request, verify, cert
            )
        except ValueError as e:
            raise InvalidURL(e, request=request) from e
        scheme = host_params["scheme"]
        host = host_params["host"]
        port = host_params["port"]
        ip = _resolve_validated_ip(host)
        key = (scheme, host, port, ip)
        with self._pinned_lock:
            pool = self._pinned_pools.get(key)
            if pool is None:
                if scheme == "https":
                    pool = _PinnedHTTPSConnectionPool(
                        host, port, dial_host=ip, **pool_kwargs
                    )
                else:
                    safe_kwargs = {
                        k: v for k, v in pool_kwargs.items() if k in _HTTP_POOL_KWARGS
                    }
                    pool = _PinnedHTTPConnectionPool(
                        host, port, dial_host=ip, **safe_kwargs
                    )
                self._pinned_pools[key] = pool
            return pool


class _ValidatingTCPConnector(aiohttp.TCPConnector):
    """aiohttp connector that only dials validated IPs.

    Hostname, SNI, and certificate verification keep using the original
    URL host; DNS answers are restricted to the IPs validated here, so a
    rebinding rotation cannot smuggle a private address past the check.
    Every redirect hop re-resolves through this path.
    """

    async def _resolve_host(
        self,
        host: str,
        port: int,
        traces: Any = None,
    ) -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(
                host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
            )
        except socket.gaierror as e:
            raise ValueError(f"Cannot resolve media URL host {host!r}: {e}") from e
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for family, _type, proto, _canon, sockaddr in infos:
            ip = sockaddr[0]
            if ip in seen:
                continue
            seen.add(ip)
            if not envs.VLLM_MEDIA_URL_ALLOW_PRIVATE_IPS:
                try:
                    public = ipaddress.ip_address(ip).is_global
                except ValueError:
                    public = False
                if not public:
                    raise ValueError(
                        f"Refusing to fetch media URL with non-public IP "
                        f"{ip} (host {host!r}). Loopback, private, "
                        "link-local, multicast, and reserved destinations "
                        "are blocked by default; set "
                        "VLLM_MEDIA_URL_ALLOW_PRIVATE_IPS=1 to allow them."
                    )
            results.append(
                {
                    "hostname": host,
                    "host": ip,
                    "port": port,
                    "family": family,
                    "proto": proto,
                    "flags": 0,
                }
            )
        if not results:
            raise ValueError(f"Cannot resolve media URL host {host!r}: no addresses")
        return results


class HTTPConnection:
    """Helper class to send HTTP requests."""

    def __init__(self, *, reuse_client: bool = True) -> None:
        super().__init__()

        self.reuse_client = reuse_client

        self._sync_client: requests.Session | None = None
        self._async_client: aiohttp.ClientSession | None = None
        self._async_connector: _ValidatingTCPConnector | None = None

    def get_sync_client(self) -> requests.Session:
        if self._sync_client is None or not self.reuse_client:
            session = requests.Session()
            protective_adapter = _SSRFProtectiveAdapter()
            session.mount("http://", protective_adapter)
            session.mount("https://", protective_adapter)
            self._sync_client = session

        return self._sync_client

    # NOTE: We intentionally use an async function even though it is not
    # required, so that the client is only accessible inside async event loop
    async def get_async_client(self) -> aiohttp.ClientSession:
        if self._async_client is None or not self.reuse_client:
            if self.reuse_client:
                if self._async_connector is None:
                    self._async_connector = _ValidatingTCPConnector()
                connector = self._async_connector
            else:
                # Owned by the session below, so it is released together.
                connector = _ValidatingTCPConnector()
            self._async_client = aiohttp.ClientSession(
                trust_env=True, connector=connector
            )

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
        self,
        url: str,
        *,
        timeout: float | None = None,
        allow_redirects: bool = True,
        max_bytes: int | None = None,
        allowed_domains: Collection[str] | None = None,
    ) -> bytes:
        self._validate_http_url(url)

        client = self.get_sync_client()
        current_url = url
        for _ in range(_MEDIA_MAX_REDIRECTS + 1):
            # Every hop — including the first — repeats the domain check the
            # caller performed, because earlier hops are invisible upstream.
            _check_allowed_domains(current_url, allowed_domains)
            # Resolve once and IP-validate before connecting; the transport
            # dials exactly this address (see _SSRFProtectiveAdapter), so a
            # rebinding rotation between check and connect cannot smuggle a
            # private destination past the check.
            _resolve_validated_ip(urlsplit(current_url).hostname or "")
            with client.get(
                current_url,
                headers=self._headers(),
                stream=True,
                timeout=timeout,
                allow_redirects=False,
            ) as r:
                if allow_redirects and r.is_redirect and r.headers.get("location"):
                    current_url = urljoin(current_url, r.headers["location"])
                    continue
                r.raise_for_status()

                return _read_response_bytes(r, max_bytes)

        raise TooManyRedirects(
            f"Exceeded {_MEDIA_MAX_REDIRECTS} redirects when fetching media URL {url!r}"
        )

    @_async_retry
    async def async_get_bytes(
        self,
        url: str,
        *,
        timeout: float | None = None,
        allow_redirects: bool = True,
        max_bytes: int | None = None,
        allowed_domains: Collection[str] | None = None,
    ) -> bytes:
        self._validate_http_url(url)

        client = await self.get_async_client()
        current_url = url
        for _ in range(_MEDIA_MAX_REDIRECTS + 1):
            _check_allowed_domains(current_url, allowed_domains)
            _resolve_validated_ip(urlsplit(current_url).hostname or "")
            async with await client.get(
                current_url,
                headers=self._headers(),
                timeout=timeout,
                allow_redirects=False,
            ) as r:
                location = r.headers.get("Location")
                if (
                    allow_redirects
                    and r.status in (301, 302, 303, 307, 308)
                    and location
                ):
                    current_url = urljoin(current_url, location)
                    continue
                r.raise_for_status()

                return await _async_read_response_bytes(r, max_bytes)

        raise aiohttp.TooManyRedirects(
            None,
            (),
            message=(
                f"Exceeded {_MEDIA_MAX_REDIRECTS} redirects when "
                f"fetching media URL {url!r}"
            ),
        )

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
