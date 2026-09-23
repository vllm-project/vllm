# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import atexit
import contextlib
import hashlib
import os
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar
from urllib.request import url2pathname

import aiohttp
import numpy as np
import numpy.typing as npt
import requests
import torch
from PIL import Image
from urllib3.util import Url, parse_url

import vllm.envs as envs
from vllm.connections import (
    HTTPConnection,
    MediaDownloadSizeExceededError,
    global_http_connection,
)
from vllm.exceptions import VLLMUnprocessableEntityError, VLLMValidationError
from vllm.logger import init_logger
from vllm.utils.registry import ExtensionManager

from .audio import AudioEmbeddingMediaIO, AudioMediaIO
from .base import MediaIO, MediaRef
from .image import ImageEmbeddingMediaIO, ImageMediaIO
from .video import VideoEmbeddingMediaIO, VideoMediaIO

logger = init_logger(__name__)

_M = TypeVar("_M")
_V = TypeVar("_V")
_A = TypeVar("_A")

global_thread_pool = ThreadPoolExecutor(
    max_workers=envs.VLLM_MEDIA_LOADING_THREAD_COUNT
)
atexit.register(global_thread_pool.shutdown)

MEDIA_CONNECTOR_REGISTRY = ExtensionManager()

MODALITY_IO_MAP: dict[str, type[MediaIO]] = {
    "audio": AudioMediaIO,
    "image": ImageMediaIO,
    "video": VideoMediaIO,
}


def _wrap_media_fetch_error(
    url: str, exc: Exception
) -> VLLMUnprocessableEntityError | Exception:
    """Convert media fetch exceptions to VLLMUnprocessableEntityError.

    This handles HTTP errors that indicate the media resource is invalid
    (4xx responses except 408/429, malformed URLs) and converts them to a
    422 Unprocessable Entity error instead of 500.

    Transient errors (5xx, 408, 429, DNS failures, connection errors,
    timeouts) are returned as-is to allow retry logic to handle them
    appropriately.

    Returns:
        VLLMUnprocessableEntityError for permanent client errors (4xx except
            408/429, invalid URL)
        Original exception for transient errors (5xx, 408, 429, network blips)
            or other exceptions

    """
    if isinstance(exc, VLLMValidationError):
        return exc

    if isinstance(exc, aiohttp.ClientResponseError):
        if exc.status in (408, 429):
            return exc
        if exc.status < 500:
            return VLLMUnprocessableEntityError(
                f"Failed to fetch media from URL: HTTP {exc.status} error",
                parameter="image_url",
                value=url,
            )
        return exc

    if isinstance(exc, requests.exceptions.HTTPError):
        if exc.response is not None:
            status_code = exc.response.status_code
            if status_code in (408, 429):
                return exc
            if status_code < 500:
                return VLLMUnprocessableEntityError(
                    f"Failed to fetch media from URL: HTTP {status_code} error",
                    parameter="image_url",
                    value=url,
                )
        return exc

    if isinstance(exc, requests.exceptions.InvalidURL):
        return VLLMUnprocessableEntityError(
            "Failed to fetch media from URL: Invalid URL format",
            parameter="image_url",
            value=url,
        )

    if isinstance(exc, MediaDownloadSizeExceededError):
        return VLLMUnprocessableEntityError(
            f"Failed to fetch media from URL: {exc}",
            parameter="image_url",
            value=url,
        )

    if isinstance(exc, ValueError):
        return VLLMUnprocessableEntityError(
            "Failed to fetch media from URL: Invalid URL",
            parameter="image_url",
            value=url,
        )
    return exc


def merge_media_io_kwargs(
    defaults: dict[str, dict[str, Any]] | None,
    overrides: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    """Merge config-level and per-request media_io_kwargs per modality.

    Each modality key is merged using the corresponding MediaIO subclass's
    ``merge_kwargs``, which may apply modality-specific logic (e.g.
    VideoMediaIO clears cross-dependent fps/num_frames fields).
    """
    if not defaults and not overrides:
        return None
    all_keys = set(defaults or {}) | set(overrides or {})
    merged = {}
    for key in all_keys:
        io_cls = MODALITY_IO_MAP.get(key, MediaIO)
        merged[key] = io_cls.merge_kwargs(
            (defaults or {}).get(key),
            (overrides or {}).get(key),
        )
    return merged or None


def _is_data_url(url: str) -> bool:
    return url[:5].lower() == "data:"


def _parse_data_url(url: str) -> tuple[str, str]:
    """Split a `data:` URL into its media type and base64 payload."""
    # Format per RFC 2397:
    # data:[<mediatype>][;<param>=<value>]*[;base64],<data>
    data_spec, sep, data = url[5:].partition(",")
    if not sep:
        msg = f"Invalid data URL {url[:32]!r}: missing ',' separator."
        raise ValueError(msg)

    media_type, sep, encoding = data_spec.rpartition(";")
    if not sep or encoding != "base64":
        msg = "Only base64 data URLs are supported for now."
        raise NotImplementedError(msg)

    return media_type.partition(";")[0], data


def _strictest_max_bytes(media_ios: Sequence[MediaIO[Any]]) -> int | None:
    """The encoded-size cap of one download feeding several decoders.

    Each decoder would have enforced its own cap on a download of its own, so
    a shared download keeps the strictest of them.
    """
    caps = [
        cap
        for cap in (media_io.get_max_bytes() for media_io in media_ios)
        if cap is not None
    ]
    return min(caps) if caps else None


def _pair_bytes(
    media_ios: Sequence[MediaIO[Any]],
    data: bytes,
) -> list[MediaRef[Any]]:
    return [media_io.load_bytes_ref(data) for media_io in media_ios]


@MEDIA_CONNECTOR_REGISTRY.register("http")
class MediaConnector:
    """Fetches media URLs down to their encoded bytes.

    Transport only: scheme dispatch (`data:` / `http(s):` / `file:`), the
    domain allow-list, the encoded-size cap, the on-disk download cache, the
    redirect policy and error normalization. How bytes decode -- and therefore
    what identifies the resulting item in the processor cache -- is decided by
    the caller-supplied [`MediaIO`][vllm.multimodal.media.base.MediaIO], which
    the multi-modal processor's `info` resolves from the model config.
    """

    def __init__(
        self,
        connection: HTTPConnection = global_http_connection,
        *,
        allowed_local_media_path: str = "",
        allowed_media_domains: list[str] | None = None,
    ) -> None:
        """Args:
        connection: HTTP connection client to download media contents.
        allowed_local_media_path: A local directory to load media files from.
        allowed_media_domains: If set, only media URLs that belong to this
                               domain can be used for multi-modal inputs.

        """
        super().__init__()

        self.connection = connection

        if allowed_local_media_path:
            allowed_local_media_path_ = Path(allowed_local_media_path).resolve()

            if not allowed_local_media_path_.exists():
                raise ValueError(
                    "Invalid `--allowed-local-media-path`: The path "
                    f"{allowed_local_media_path_} does not exist."
                )
            if not allowed_local_media_path_.is_dir():
                raise ValueError(
                    "Invalid `--allowed-local-media-path`: The path "
                    f"{allowed_local_media_path_} must be a directory."
                )
        else:
            allowed_local_media_path_ = None

        self.allowed_local_media_path = allowed_local_media_path_
        if allowed_media_domains is None:
            allowed_media_domains = []
        self.allowed_media_domains = allowed_media_domains

        # Media download cache (opt-in via VLLM_MEDIA_CACHE)
        self._media_cache_dir: str | None = None
        self._media_cache_max_bytes: int = 0
        self._media_cache_ttl_secs: float = 0
        media_cache = envs.VLLM_MEDIA_CACHE
        if media_cache:
            try:
                os.makedirs(media_cache, exist_ok=True)
                # Verify the directory is writable before enabling caching
                with tempfile.NamedTemporaryFile(dir=media_cache, delete=True):
                    pass
                self._media_cache_dir = media_cache
                self._media_cache_max_bytes = (
                    envs.VLLM_MEDIA_CACHE_MAX_SIZE_MB * 1024 * 1024
                )
                self._media_cache_ttl_secs = envs.VLLM_MEDIA_CACHE_TTL_HOURS * 3600
                logger.info(
                    "Media cache enabled at %s (max %d MB, TTL %s hours)",
                    media_cache,
                    envs.VLLM_MEDIA_CACHE_MAX_SIZE_MB,
                    envs.VLLM_MEDIA_CACHE_TTL_HOURS,
                )
            except OSError:
                logger.warning(
                    "VLLM_MEDIA_CACHE path %s is not writable, media caching disabled",
                    media_cache,
                )

    def _get_cached_bytes(self, url: str) -> bytes | None:
        """Return cached bytes for a URL, or None if not cached/expired."""
        if not self._media_cache_dir:
            return None
        cache_path = self._media_cache_path(url)
        # Check TTL
        try:
            age = time.time() - cache_path.stat().st_mtime
        except OSError:
            return None
        if age > self._media_cache_ttl_secs:
            cache_path.unlink(missing_ok=True)
            return None
        # Touch mtime for LRU ordering
        try:
            cache_path.touch()
            return cache_path.read_bytes()
        except OSError:
            return None

    def _put_cached_bytes(self, url: str, data: bytes) -> None:
        """Store downloaded bytes and evict if over budget."""
        if not self._media_cache_dir:
            return
        cache_path = self._media_cache_path(url)
        # Atomic write via temp file + rename
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self._media_cache_dir, delete=False
            ) as tmp_file:
                tmp_file.write(data)
                tmp_path = tmp_file.name
            os.rename(tmp_path, str(cache_path))
        except OSError:
            # Another process beat us or disk issue
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)
            return
        self._maybe_evict(exclude=cache_path)

    def _maybe_evict(self, exclude: Path | None = None) -> None:
        """Evict expired entries first, then LRU until under size limit."""
        cache_dir = Path(self._media_cache_dir)  # type: ignore[arg-type]
        entries = []
        expired = []
        total_size = 0
        now = time.time()
        for f in cache_dir.iterdir():
            if f.name.startswith("."):
                continue
            try:
                stat = f.stat()
            except OSError:
                continue
            age = now - stat.st_mtime
            if age > self._media_cache_ttl_secs:
                expired.append(f)
                continue
            total_size += stat.st_size
            # Never evict the file we just wrote
            if exclude is not None and f.name == exclude.name:
                continue
            entries.append((stat.st_mtime, stat.st_size, f))

        # Evict items according to LRU policy
        entries.sort(key=lambda e: e[0], reverse=True)
        while total_size > self._media_cache_max_bytes and entries:
            mtime, size, f = entries.pop()
            expired.append(f)
            total_size -= size

        for f in expired:
            f.unlink(missing_ok=True)

    def _media_cache_path(self, url: str) -> Path:
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:20]
        ext = Path(url.split("?", 1)[0]).suffix or ""
        return Path(self._media_cache_dir) / f"{url_hash}{ext}"  # type: ignore[arg-type]

    def _resolve_allowed_file(self, url_spec: Url) -> Path:
        """Resolve a `file:` URL, checking it against the allowed media path."""
        allowed_local_media_path = self.allowed_local_media_path
        if allowed_local_media_path is None:
            raise RuntimeError(
                "Cannot load local files without `--allowed-local-media-path`."
            )

        url_spec_path = url_spec.path or ""
        url_spec_netloc = url_spec.netloc or ""
        filepath = Path(url2pathname(url_spec_netloc + url_spec_path))
        if allowed_local_media_path not in filepath.resolve().parents:
            raise ValueError(
                f"The file path {filepath} must be a subpath "
                f"of `--allowed-local-media-path {allowed_local_media_path}`."
            )

        return filepath

    def _assert_url_in_allowed_media_domains(self, url_spec: Url) -> None:
        if (
            self.allowed_media_domains
            and url_spec.hostname not in self.allowed_media_domains
        ):
            raise ValueError(
                f"The URL must be from one of the allowed domains: "
                f"{self.allowed_media_domains}. Input URL domain: "
                f"{url_spec.hostname}"
            )

    def _fetch_url_bytes(
        self,
        url: str,
        url_spec: Url,
        *,
        max_bytes: int | None = None,
        fetch_timeout: int | None = None,
    ) -> bytes:
        """Download an HTTP(S) URL down to its encoded bytes."""
        self._assert_url_in_allowed_media_domains(url_spec)

        cached = self._get_cached_bytes(url)
        if cached is not None:
            return cached

        try:
            data = self.connection.get_bytes(
                url_spec.url,
                timeout=fetch_timeout,
                allow_redirects=envs.VLLM_MEDIA_URL_ALLOW_REDIRECTS,
                max_bytes=max_bytes,
            )
        except Exception as e:
            wrapped = _wrap_media_fetch_error(url, e)
            if isinstance(wrapped, VLLMUnprocessableEntityError):
                raise wrapped from e
            raise

        self._put_cached_bytes(url, data)
        return data

    async def _fetch_url_bytes_async(
        self,
        url: str,
        url_spec: Url,
        *,
        max_bytes: int | None = None,
        fetch_timeout: int | None = None,
    ) -> bytes:
        """Asynchronous `_fetch_url_bytes`."""
        loop = asyncio.get_running_loop()

        self._assert_url_in_allowed_media_domains(url_spec)

        cached = await loop.run_in_executor(
            global_thread_pool, self._get_cached_bytes, url
        )
        if cached is not None:
            return cached

        try:
            data = await self.connection.async_get_bytes(
                url_spec.url,
                timeout=fetch_timeout,
                allow_redirects=envs.VLLM_MEDIA_URL_ALLOW_REDIRECTS,
                max_bytes=max_bytes,
            )
        except Exception as e:
            wrapped = _wrap_media_fetch_error(url, e)
            if isinstance(wrapped, VLLMUnprocessableEntityError):
                raise wrapped from e
            raise

        await loop.run_in_executor(
            global_thread_pool, self._put_cached_bytes, url, data
        )
        return data

    def _load_refs(
        self,
        url: str,
        media_ios: Sequence[MediaIO[Any]],
        *,
        fetch_timeout: int | None = None,
    ) -> list[MediaRef[Any]]:
        """Fetch `url` once and pair the payload with each of `media_ios`.

        One payload can legitimately feed several decoders -- reading the audio
        track out of a video file needs the same bytes as its frames -- and
        fetching once per decoder would repeat the download.
        """
        if _is_data_url(url):
            media_type, encoded = _parse_data_url(url)
            return [
                media_io.load_base64_ref(media_type, encoded) for media_io in media_ios
            ]

        url_spec = parse_url(url)

        if url_spec.scheme == "file":
            filepath = self._resolve_allowed_file(url_spec)
            return [media_io.load_file_ref(filepath) for media_io in media_ios]

        if url_spec.scheme and url_spec.scheme.startswith("http"):
            data = self._fetch_url_bytes(
                url,
                url_spec,
                max_bytes=_strictest_max_bytes(media_ios),
                fetch_timeout=fetch_timeout,
            )
            return _pair_bytes(media_ios, data)

        msg = "The URL must be either a HTTP, data or file URL."
        raise ValueError(msg)

    async def _load_refs_async(
        self,
        url: str,
        media_ios: Sequence[MediaIO[Any]],
        *,
        fetch_timeout: int | None = None,
    ) -> list[MediaRef[Any]]:
        """Asynchronous `_load_refs`."""
        loop = asyncio.get_running_loop()
        url_spec = None if _is_data_url(url) else parse_url(url)

        if (
            url_spec is not None
            and url_spec.scheme
            and url_spec.scheme.startswith("http")
        ):
            data = await self._fetch_url_bytes_async(
                url,
                url_spec,
                max_bytes=_strictest_max_bytes(media_ios),
                fetch_timeout=fetch_timeout,
            )
            # Building the refs digests the payload for their cache keys (and
            # parses an image header), so keep it off the event loop.
            return await loop.run_in_executor(
                global_thread_pool, _pair_bytes, media_ios, data
            )

        # `data:` and `file:` URLs never touch the network, but splitting the
        # payload, reading the file and building the refs are all linear in it.
        return await loop.run_in_executor(
            global_thread_pool, self._load_refs, url, media_ios
        )

    def load_from_url(
        self,
        url: str,
        media_io: MediaIO[_M],
        *,
        fetch_timeout: int | None = None,
    ) -> MediaRef[_M]:
        """Fetch `url` and pair it with a caller-supplied decoder.

        Returns a lazy handle: decoding happens on first access (inside the
        multi-modal processor), so decode errors surface there rather than at
        the fetch site.
        """
        (ref,) = self._load_refs(url, [media_io], fetch_timeout=fetch_timeout)
        return ref

    async def load_from_url_async(
        self,
        url: str,
        media_io: MediaIO[_M],
        *,
        fetch_timeout: int | None = None,
    ) -> MediaRef[_M]:
        """Asynchronously fetch `url` and pair it with a caller-supplied decoder."""
        (ref,) = await self._load_refs_async(
            url, [media_io], fetch_timeout=fetch_timeout
        )
        return ref

    def fetch_audio(
        self,
        audio_url: str,
        media_io: MediaIO[tuple[np.ndarray, int | float]],
    ) -> MediaRef[tuple[np.ndarray, int | float]]:
        """Fetch audio and pair it with a caller-supplied decoder.

        Returns a lazy handle: decoding happens on first access (inside the
        multi-modal processor), so decode errors surface there rather than at
        the fetch site.
        """
        return self.load_from_url(
            audio_url,
            media_io,
            fetch_timeout=envs.VLLM_AUDIO_FETCH_TIMEOUT,
        )

    async def fetch_audio_async(
        self,
        audio_url: str,
        media_io: MediaIO[tuple[np.ndarray, int | float]],
    ) -> MediaRef[tuple[np.ndarray, int | float]]:
        """Asynchronously fetch audio and pair it with a caller-supplied decoder."""
        return await self.load_from_url_async(
            audio_url,
            media_io,
            fetch_timeout=envs.VLLM_AUDIO_FETCH_TIMEOUT,
        )

    def fetch_image(
        self,
        image_url: str,
        media_io: MediaIO[Image.Image],
    ) -> MediaRef[Image.Image]:
        """Fetch an image and pair it with a caller-supplied decoder.

        Returns a lazy handle: decoding happens on first access (inside the
        multi-modal processor), so decode errors surface there rather than at
        the fetch site.
        """
        return self.load_from_url(
            image_url,
            media_io,
            fetch_timeout=envs.VLLM_IMAGE_FETCH_TIMEOUT,
        )

    async def fetch_image_async(
        self,
        image_url: str,
        media_io: MediaIO[Image.Image],
    ) -> MediaRef[Image.Image]:
        """Asynchronously fetch an image and pair it with a supplied decoder."""
        return await self.load_from_url_async(
            image_url,
            media_io,
            fetch_timeout=envs.VLLM_IMAGE_FETCH_TIMEOUT,
        )

    def fetch_video(
        self,
        video_url: str,
        media_io: MediaIO[tuple[npt.NDArray, dict[str, Any]]],
    ) -> MediaRef[tuple[npt.NDArray, dict[str, Any]]]:
        """Fetch a video and pair it with a caller-supplied decoder.

        Returns a lazy handle: decoding happens on first access (inside the
        multi-modal processor), so decode errors surface there rather than at
        the fetch site.
        """
        return self.load_from_url(
            video_url,
            media_io,
            fetch_timeout=envs.VLLM_VIDEO_FETCH_TIMEOUT,
        )

    async def fetch_video_async(
        self,
        video_url: str,
        media_io: MediaIO[tuple[npt.NDArray, dict[str, Any]]],
    ) -> MediaRef[tuple[npt.NDArray, dict[str, Any]]]:
        """Asynchronously fetch a video and pair it with a supplied decoder."""
        return await self.load_from_url_async(
            video_url,
            media_io,
            fetch_timeout=envs.VLLM_VIDEO_FETCH_TIMEOUT,
        )

    def fetch_video_and_audio(
        self,
        video_url: str,
        video_io: MediaIO[_V],
        audio_io: MediaIO[_A],
    ) -> tuple[MediaRef[_V], MediaRef[_A]]:
        """Fetch `video_url` once and decode it as both video and audio.

        `use_audio_in_video` reads the audio track out of the video payload, so
        both refs are built from one download instead of two.
        """
        video, audio = self._load_refs(
            video_url,
            [video_io, audio_io],
            fetch_timeout=envs.VLLM_VIDEO_FETCH_TIMEOUT,
        )
        return video, audio

    async def fetch_video_and_audio_async(
        self,
        video_url: str,
        video_io: MediaIO[_V],
        audio_io: MediaIO[_A],
    ) -> tuple[MediaRef[_V], MediaRef[_A]]:
        """Asynchronous `fetch_video_and_audio`."""
        video, audio = await self._load_refs_async(
            video_url,
            [video_io, audio_io],
            fetch_timeout=envs.VLLM_VIDEO_FETCH_TIMEOUT,
        )
        return video, audio

    def fetch_image_embedding(
        self,
        data: str,
    ) -> torch.Tensor:
        """Load image embedding from a URL."""
        image_embedding_io = ImageEmbeddingMediaIO()

        return image_embedding_io.load_base64("", data)

    async def fetch_image_embedding_async(
        self,
        data: str,
    ) -> torch.Tensor:
        """Asynchronously load image embedding from a URL."""
        image_embedding_io = ImageEmbeddingMediaIO()
        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(
            global_thread_pool, image_embedding_io.load_base64, "", data
        )

    def fetch_audio_embedding(
        self,
        data: str,
    ) -> torch.Tensor:
        """Load audio embedding from a URL."""
        audio_embedding_io = AudioEmbeddingMediaIO()

        return audio_embedding_io.load_base64("", data)

    async def fetch_audio_embedding_async(
        self,
        data: str,
    ) -> torch.Tensor:
        """Asynchronously load audio embedding from a URL."""
        audio_embedding_io = AudioEmbeddingMediaIO()
        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(
            global_thread_pool, audio_embedding_io.load_base64, "", data
        )

    def fetch_video_embedding(
        self,
        data: str,
    ) -> torch.Tensor:
        """Load video embedding from a URL."""
        video_embedding_io = VideoEmbeddingMediaIO()

        return video_embedding_io.load_base64("", data)

    async def fetch_video_embedding_async(
        self,
        data: str,
    ) -> torch.Tensor:
        """Asynchronously load video embedding from a URL."""
        video_embedding_io = VideoEmbeddingMediaIO()
        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(
            global_thread_pool, video_embedding_io.load_base64, "", data
        )
