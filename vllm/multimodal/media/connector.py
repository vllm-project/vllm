# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import atexit
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar
from urllib.request import url2pathname

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image, UnidentifiedImageError
from urllib3.util import Url, parse_url

import vllm.envs as envs
from vllm.connections import HTTPConnection, global_http_connection
from vllm.utils.registry import ExtensionManager

from .audio import AudioEmbeddingMediaIO, AudioMediaIO
from .base import MediaIO
from .image import ImageEmbeddingMediaIO, ImageMediaIO
from .video import VideoMediaIO

_M = TypeVar("_M")

global_thread_pool = ThreadPoolExecutor(
    max_workers=envs.VLLM_MEDIA_LOADING_THREAD_COUNT
)
atexit.register(global_thread_pool.shutdown)

MEDIA_CONNECTOR_REGISTRY = ExtensionManager()


@MEDIA_CONNECTOR_REGISTRY.register("http")
class MediaConnector:
    def __init__(
        self,
        media_io_kwargs: dict[str, dict[str, Any]] | None = None,
        connection: HTTPConnection = global_http_connection,
        *,
        allowed_local_media_path: str = "",
        allowed_media_domains: list[str] | None = None,
    ) -> None:
        """
        Args:
            media_io_kwargs: Additional args passed to process media
                             inputs, keyed by modalities. For example,
                             to set num_frames for video, set
                             `--media-io-kwargs '{"video":{"num_frames":40}}'`
            connection: HTTP connection client to download media contents.
            allowed_local_media_path: A local directory to load media files from.
            allowed_media_domains: If set, only media URLs that belong to this
                                   domain can be used for multi-modal inputs.
        """
        super().__init__()

        self.media_io_kwargs: dict[str, dict[str, Any]] = (
            media_io_kwargs if media_io_kwargs else {}
        )
        self.connection = connection

        if allowed_local_media_path:
            allowed_local_media_path_ = Path(allowed_local_media_path)

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

    def _load_data_url(
        self,
        url_spec: Url,
        media_io: MediaIO[_M],
    ) -> _M:  # type: ignore[type-var]
        url_spec_path = url_spec.path or ""
        data_spec, data = url_spec_path.split(",", 1)
        media_type, data_type = data_spec.split(";", 1)
        # media_type starts with a leading "/" (e.g., "/video/jpeg")
        media_type = media_type.lstrip("/")

        if data_type != "base64":
            msg = "Only base64 data URLs are supported for now."
            raise NotImplementedError(msg)

        return media_io.load_base64(media_type, data)

    def _load_file_url(
        self,
        url_spec: Url,
        media_io: MediaIO[_M],
    ) -> _M:  # type: ignore[type-var]
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

        return media_io.load_file(filepath)

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

    def load_from_url(
        self,
        url: str,
        media_io: MediaIO[_M],
        *,
        fetch_timeout: int | None = None,
    ) -> _M:  # type: ignore[type-var]
        url_spec = parse_url(url)

        if url_spec.scheme and url_spec.scheme.startswith("http"):
            self._assert_url_in_allowed_media_domains(url_spec)

            connection = self.connection
            data = connection.get_bytes(
                url,
                timeout=fetch_timeout,
                allow_redirects=envs.VLLM_MEDIA_URL_ALLOW_REDIRECTS,
            )

            return media_io.load_bytes(data)

        if url_spec.scheme == "data":
            return self._load_data_url(url_spec, media_io)

        if url_spec.scheme == "file":
            return self._load_file_url(url_spec, media_io)

        msg = "The URL must be either a HTTP, data or file URL."
        raise ValueError(msg)

    async def load_from_url_async(
        self,
        url: str,
        media_io: MediaIO[_M],
        *,
        fetch_timeout: int | None = None,
    ) -> _M:
        url_spec = parse_url(url)
        loop = asyncio.get_running_loop()

        if url_spec.scheme and url_spec.scheme.startswith("http"):
            self._assert_url_in_allowed_media_domains(url_spec)

            connection = self.connection
            data = await connection.async_get_bytes(
                url,
                timeout=fetch_timeout,
                allow_redirects=envs.VLLM_MEDIA_URL_ALLOW_REDIRECTS,
            )
            future = loop.run_in_executor(global_thread_pool, media_io.load_bytes, data)
            return await future

        if url_spec.scheme == "data":
            future = loop.run_in_executor(
                global_thread_pool, self._load_data_url, url_spec, media_io
            )
            return await future

        if url_spec.scheme == "file":
            future = loop.run_in_executor(
                global_thread_pool, self._load_file_url, url_spec, media_io
            )
            return await future
        msg = "The URL must be either a HTTP, data or file URL."
        raise ValueError(msg)

    def fetch_audio(
        self,
        audio_url: str,
    ) -> tuple[np.ndarray, int | float]:
        """
        Load audio from a URL.
        """
        audio_io = AudioMediaIO(**self.media_io_kwargs.get("audio", {}))

        return self.load_from_url(
            audio_url,
            audio_io,
            fetch_timeout=envs.VLLM_AUDIO_FETCH_TIMEOUT,
        )

    async def fetch_audio_async(
        self,
        audio_url: str,
    ) -> tuple[np.ndarray, int | float]:
        """
        Asynchronously fetch audio from a URL.
        """
        audio_io = AudioMediaIO(**self.media_io_kwargs.get("audio", {}))

        return await self.load_from_url_async(
            audio_url,
            audio_io,
            fetch_timeout=envs.VLLM_AUDIO_FETCH_TIMEOUT,
        )

    def fetch_image(
        self,
        image_url: str,
        *,
        image_mode: str = "RGB",
    ) -> Image.Image:
        """
        Load a PIL image from an HTTP or base64 data URL.

        By default, the image is converted into RGB format.
        """
        image_io = ImageMediaIO(
            image_mode=image_mode, **self.media_io_kwargs.get("image", {})
        )

        try:
            return self.load_from_url(
                image_url,
                image_io,
                fetch_timeout=envs.VLLM_IMAGE_FETCH_TIMEOUT,
            )
        except UnidentifiedImageError as e:
            # convert to ValueError to be properly caught upstream
            raise ValueError(str(e)) from e

    async def fetch_image_async(
        self,
        image_url: str,
        *,
        image_mode: str = "RGB",
    ) -> Image.Image:
        """
        Asynchronously load a PIL image from an HTTP or base64 data URL.

        By default, the image is converted into RGB format.
        """
        image_io = ImageMediaIO(
            image_mode=image_mode, **self.media_io_kwargs.get("image", {})
        )

        try:
            return await self.load_from_url_async(
                image_url,
                image_io,
                fetch_timeout=envs.VLLM_IMAGE_FETCH_TIMEOUT,
            )
        except UnidentifiedImageError as e:
            # convert to ValueError to be properly caught upstream
            raise ValueError(str(e)) from e

    def fetch_video(
        self,
        video_url: str,
        *,
        image_mode: str = "RGB",
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """
        Load video from an HTTP or base64 data URL.
        """
        image_io = ImageMediaIO(
            image_mode=image_mode, **self.media_io_kwargs.get("image", {})
        )
        video_io = VideoMediaIO(image_io, **self.media_io_kwargs.get("video", {}))

        return self.load_from_url(
            video_url,
            video_io,
            fetch_timeout=envs.VLLM_VIDEO_FETCH_TIMEOUT,
        )

    async def fetch_video_async(
        self,
        video_url: str,
        *,
        image_mode: str = "RGB",
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """
        Asynchronously load video from an HTTP or base64 data URL.

        By default, the image is converted into RGB format.
        """
        image_io = ImageMediaIO(
            image_mode=image_mode, **self.media_io_kwargs.get("image", {})
        )
        video_io = VideoMediaIO(image_io, **self.media_io_kwargs.get("video", {}))

        return await self.load_from_url_async(
            video_url,
            video_io,
            fetch_timeout=envs.VLLM_VIDEO_FETCH_TIMEOUT,
        )

    def fetch_image_embedding(
        self,
        data: str,
    ) -> torch.Tensor:
        """
        Load image embedding from a URL.
        """
        image_embedding_io = ImageEmbeddingMediaIO()

        return image_embedding_io.load_base64("", data)

    def fetch_audio_embedding(
        self,
        data: str,
    ) -> torch.Tensor:
        """
        Load audio embedding from a URL.
        """
        audio_embedding_io = AudioEmbeddingMediaIO()

        return audio_embedding_io.load_base64("", data)
