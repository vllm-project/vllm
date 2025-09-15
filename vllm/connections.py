# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import aiohttp
import requests

from vllm.version import __version__ as VLLM_VERSION


class HTTPConnection:
    """Helper class to send HTTP requests."""

    def __init__(self, *, reuse_client: bool = True) -> None:
        super().__init__()

        self.reuse_client = reuse_client

        self._sync_client: Optional[requests.Session] = None
        self._async_client: Optional[aiohttp.ClientSession] = None

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
        parsed_url = urlparse(url)

        if parsed_url.scheme not in ("http", "https"):
            raise ValueError("Invalid HTTP URL: A valid HTTP URL "
                             "must have scheme 'http' or 'https'.")

    def _headers(self, **extras: str) -> MutableMapping[str, str]:
        return {"User-Agent": f"vLLM/{VLLM_VERSION}", **extras}

    def get_response(
        self,
        url: str,
        *,
        stream: bool = False,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ):
        self._validate_http_url(url)

        client = self.get_sync_client()
        extra_headers = extra_headers or {}

        return client.get(url,
                          headers=self._headers(**extra_headers),
                          stream=stream,
                          timeout=timeout)

    async def get_async_response(
        self,
        url: str,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ):
        self._validate_http_url(url)

        client = await self.get_async_client()
        extra_headers = extra_headers or {}

        return client.get(url,
                          headers=self._headers(**extra_headers),
                          timeout=timeout)

    def get_bytes(self, url: str, *, timeout: Optional[float] = None) -> bytes:
        with self.get_response(url, timeout=timeout) as r:
            r.raise_for_status()

            return r.content

    async def async_get_bytes(
        self,
        url: str,
        *,
        timeout: Optional[float] = None,
    ) -> bytes:
        async with await self.get_async_response(url, timeout=timeout) as r:
            r.raise_for_status()

            return await r.read()

    def get_text(self, url: str, *, timeout: Optional[float] = None) -> str:
        with self.get_response(url, timeout=timeout) as r:
            r.raise_for_status()

            return r.text

    async def async_get_text(
        self,
        url: str,
        *,
        timeout: Optional[float] = None,
    ) -> str:
        async with await self.get_async_response(url, timeout=timeout) as r:
            r.raise_for_status()

            return await r.text()

    def get_json(self, url: str, *, timeout: Optional[float] = None) -> str:
        with self.get_response(url, timeout=timeout) as r:
            r.raise_for_status()

            return r.json()

    async def async_get_json(
        self,
        url: str,
        *,
        timeout: Optional[float] = None,
    ) -> str:
        async with await self.get_async_response(url, timeout=timeout) as r:
            r.raise_for_status()

            return await r.json()

    def download_file(
        self,
        url: str,
        save_path: Path,
        *,
        timeout: Optional[float] = None,
        chunk_size: int = 128,
    ) -> Path:
        with self.get_response(url, timeout=timeout) as r:
            r.raise_for_status()

            with save_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size):
                    f.write(chunk)

        return save_path

    async def async_download_file(
        self,
        url: str,
        save_path: Path,
        *,
        timeout: Optional[float] = None,
        chunk_size: int = 128,
    ) -> Path:
        async with await self.get_async_response(url, timeout=timeout) as r:
            r.raise_for_status()

            with save_path.open("wb") as f:
                async for chunk in r.content.iter_chunked(chunk_size):
                    f.write(chunk)

        return save_path


global_http_connection = HTTPConnection()
"""The global :class:`HTTPConnection` instance used by vLLM."""
