# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import atexit
from collections.abc import Generator, Set
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import ParseResult, urlparse
from urllib.request import url2pathname

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image, UnidentifiedImageError

import vllm.envs as envs
from vllm.connections import HTTPConnection, global_http_connection
from vllm.logger import init_logger
from vllm.utils.registry import ExtensionManager

from .audio import AudioEmbeddingMediaIO, AudioMediaIO
from .base import MediaIO
from .image import ImageEmbeddingMediaIO, ImageMediaIO
from .video import VideoMediaIO

if TYPE_CHECKING:
    from .inputs import (
        BatchedTensorInputs,
        MultiModalKwargsItem,
        MultiModalPlaceholderDict,
    )
else:
    BatchedTensorInputs = Any
    MultiModalKwargsItem = Any
    MultiModalPlaceholderDict = Any

logger = init_logger(__name__)

global_thread_pool = ThreadPoolExecutor(
    max_workers=envs.VLLM_MEDIA_LOADING_THREAD_COUNT
)
atexit.register(global_thread_pool.shutdown)

_M = TypeVar("_M")

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
        url_spec: ParseResult,
        media_io: MediaIO[_M],
    ) -> _M:  # type: ignore[type-var]
        data_spec, data = url_spec.path.split(",", 1)
        media_type, data_type = data_spec.split(";", 1)

        if data_type != "base64":
            msg = "Only base64 data URLs are supported for now."
            raise NotImplementedError(msg)

        return media_io.load_base64(media_type, data)

    def _load_file_url(
        self,
        url_spec: ParseResult,
        media_io: MediaIO[_M],
    ) -> _M:  # type: ignore[type-var]
        allowed_local_media_path = self.allowed_local_media_path
        if allowed_local_media_path is None:
            raise RuntimeError(
                "Cannot load local files without `--allowed-local-media-path`."
            )

        filepath = Path(url2pathname(url_spec.netloc + url_spec.path))
        if allowed_local_media_path not in filepath.resolve().parents:
            raise ValueError(
                f"The file path {filepath} must be a subpath "
                f"of `--allowed-local-media-path {allowed_local_media_path}`."
            )

        return media_io.load_file(filepath)

    def _assert_url_in_allowed_media_domains(self, url_spec: ParseResult) -> None:
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
        url_spec = urlparse(url)

        if url_spec.scheme.startswith("http"):
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
        url_spec = urlparse(url)
        loop = asyncio.get_running_loop()

        if url_spec.scheme.startswith("http"):
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


def encode_audio_base64(
    audio: np.ndarray,
    sampling_rate: int,
) -> str:
    """Encode audio as base64."""
    audio_io = AudioMediaIO()
    return audio_io.encode_base64((audio, sampling_rate))


def encode_image_base64(
    image: Image.Image,
    *,
    image_mode: str = "RGB",
    format: str = "JPEG",
) -> str:
    """
    Encode a pillow image to base64 format.

    By default, the image is converted into RGB format before being encoded.
    """
    image_io = ImageMediaIO(image_mode=image_mode)
    return image_io.encode_base64(image, image_format=format)


def encode_video_base64(frames: npt.NDArray) -> str:
    image_io = ImageMediaIO()
    video_io = VideoMediaIO(image_io)
    return video_io.encode_base64(frames)


def argsort_mm_positions(
    mm_positions: MultiModalPlaceholderDict,
) -> list[tuple[str, int]]:
    """
    Given a `MultiModalPlaceholderDict`, output a sequence of keys to
    sort the dictionary by `offset` (starting index in the input sequence)
    in ascending order.

    Returns:
        A list of `(modality, idx)`, which can be used to access an item
        by `mm_positions[modality][idx]`.
    """
    flat_items = (
        (modality, idx, item)
        for modality, items in mm_positions.items()
        for idx, item in enumerate(items)
    )

    sorted_flat_items = sorted(flat_items, key=lambda x: x[2].offset)

    return [(modality, idx) for modality, idx, _ in sorted_flat_items]


def group_mm_kwargs_by_modality(
    mm_kwargs: list[MultiModalKwargsItem],
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
    merge_by_field_config: bool | None = None,
    multimodal_cpu_fields: Set[str] | None = None,
) -> Generator[tuple[str, int, BatchedTensorInputs], None, None]:
    """Group consecutive `MultiModalKwargsItem`s from `mm_kwargs` with the same
    modality together into the same `MultiModalKwargs` instance.

    Args:
        mm_kwargs: List of `MultiModalKwargsItem`.
        device: The device to place the grouped tensors on.
        pin_memory: Whether to pin memory for faster host-to-device transfer.

    Yields:
        A tuple `(modality, num_items, grouped_kwargs)`.
    """
    if merge_by_field_config is not None:
        logger.warning_once(
            "The `merge_by_field_config` argument of `group_mm_kwargs_by_modality` "
            "is deprecated and will be removed in v0.14."
        )
    if multimodal_cpu_fields is not None:
        logger.warning_once(
            "The `multimodal_cpu_fields` argument of `group_mm_kwargs_by_modality` "
            "is deprecated and will be removed in v0.14."
        )

    from vllm.multimodal.inputs import MultiModalKwargsItems

    for modality, items in groupby(mm_kwargs, key=lambda item: item.modality):
        items_lst = list(items)
        mm_kwargs_items = MultiModalKwargsItems.from_seq(items_lst)
        mm_kwargs_data = mm_kwargs_items.get_data(
            device=device,
            pin_memory=pin_memory,
        )

        yield modality, len(items_lst), mm_kwargs_data


def fetch_audio(
    audio_url: str,
    audio_io_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, int | float]:
    """
    Args:
        audio_url: URL of the audio file to fetch.
        audio_io_kwargs: Additional kwargs passed to handle audio IO.

    Warning:
        This method has direct access to local files and is only intended
        to be called by user code. Never call this from the online server!
    """
    media_io_kwargs = None if not audio_io_kwargs else {"audio": audio_io_kwargs}
    media_connector = MediaConnector(
        media_io_kwargs=media_io_kwargs,
        allowed_local_media_path="/",
    )
    return media_connector.fetch_audio(audio_url)


def fetch_image(
    image_url: str,
    image_io_kwargs: dict[str, Any] | None = None,
) -> Image.Image:
    """
    Args:
        image_url: URL of the image file to fetch.
        image_io_kwargs: Additional kwargs passed to handle image IO.

    Warning:
        This method has direct access to local files and is only intended
        to be called by user code. Never call this from the online server!
    """
    media_io_kwargs = None if not image_io_kwargs else {"image": image_io_kwargs}
    media_connector = MediaConnector(
        media_io_kwargs=media_io_kwargs,
        allowed_local_media_path="/",
    )
    return media_connector.fetch_image(image_url)


def fetch_video(
    video_url: str,
    video_io_kwargs: dict[str, Any] | None = None,
) -> tuple[npt.NDArray, dict[str, Any]]:
    """
    Args:
        video_url: URL of the video file to fetch.
        video_io_kwargs: Additional kwargs passed to handle video IO.

    Warning:
        This method has direct access to local files and is only intended
        to be called by user code. Never call this from the online server!
    """
    media_io_kwargs = None if not video_io_kwargs else {"video": video_io_kwargs}
    media_connector = MediaConnector(
        media_io_kwargs=media_io_kwargs,
        allowed_local_media_path="/",
    )
    return media_connector.fetch_video(video_url)
