# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import threading
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pybase64
from PIL import Image

from vllm import envs

from ..video import VIDEO_LOADER_REGISTRY
from .base import MediaIO
from .image import ImageMediaIO

_VideoDecodeCacheKey = tuple[str, int, int, int, str, tuple[tuple[str, Any], ...]]
_VideoDecodeCacheValue = tuple[npt.NDArray, dict[str, Any]]


class _InflightVideoDecode:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: _VideoDecodeCacheValue | None = None
        self.error: BaseException | None = None


class _VideoDecodeCache:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cache: OrderedDict[_VideoDecodeCacheKey, _VideoDecodeCacheValue] = (
            OrderedDict()
        )
        self.inflight: dict[_VideoDecodeCacheKey, _InflightVideoDecode] = {}

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.inflight.clear()

    @classmethod
    def _freeze_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(
                sorted((str(k), cls._freeze_value(v)) for k, v in value.items())
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls._freeze_value(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted(cls._freeze_value(v) for v in value))
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return value

    @staticmethod
    def _copy_value(value: _VideoDecodeCacheValue) -> _VideoDecodeCacheValue:
        frames, metadata = value
        return frames.copy(), deepcopy(metadata)

    def key_for_file(
        self,
        filepath: Path,
        num_frames: int,
        video_loader_backend: str,
        kwargs: dict[str, Any],
    ) -> _VideoDecodeCacheKey:
        stat = filepath.stat()
        kwargs_key = tuple(
            sorted((str(k), self._freeze_value(v)) for k, v in kwargs.items())
        )
        return (
            str(filepath.resolve()),
            stat.st_mtime_ns,
            stat.st_size,
            num_frames,
            video_loader_backend,
            kwargs_key,
        )

    def get_or_load(
        self,
        key: _VideoDecodeCacheKey,
        max_size: int,
        load: Callable[[], _VideoDecodeCacheValue],
    ) -> _VideoDecodeCacheValue:
        owner = False
        with self.lock:
            cached = self.cache.get(key)
            if cached is not None:
                self.cache.move_to_end(key)
                return self._copy_value(cached)

            inflight = self.inflight.get(key)
            if inflight is None:
                inflight = _InflightVideoDecode()
                self.inflight[key] = inflight
                owner = True

        if owner:
            try:
                result = load()
                cached_result = self._copy_value(result)
            except BaseException as exc:
                with self.lock:
                    inflight.error = exc
                    inflight.event.set()
                    self.inflight.pop(key, None)
                raise

            with self.lock:
                self.cache[key] = cached_result
                self.cache.move_to_end(key)
                while len(self.cache) > max_size:
                    self.cache.popitem(last=False)
                inflight.result = cached_result
                inflight.event.set()
                self.inflight.pop(key, None)
            return result

        inflight.event.wait()
        if inflight.error is not None:
            raise inflight.error
        assert inflight.result is not None
        return self._copy_value(inflight.result)


_VIDEO_DECODE_CACHE = _VideoDecodeCache()


class VideoMediaIO(MediaIO[tuple[npt.NDArray, dict[str, Any]]]):
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    @classmethod
    def merge_kwargs(
        cls,
        default_kwargs: dict[str, Any] | None,
        runtime_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged = super().merge_kwargs(default_kwargs, runtime_kwargs)
        # fps and num_frames interact with each other, so if either is
        # overridden at request time, wipe the other from defaults to
        # avoid unintuitive cross-field interactions.
        if runtime_kwargs:
            if "num_frames" in runtime_kwargs and "fps" not in runtime_kwargs:
                merged.pop("fps", None)
            elif "fps" in runtime_kwargs and "num_frames" not in runtime_kwargs:
                merged.pop("num_frames", None)
        return merged

    def __init__(
        self,
        image_io: ImageMediaIO,
        num_frames: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()

        self.image_io = image_io
        self.num_frames = num_frames
        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality, merged with
        # per-request runtime media_io_kwargs via merge_kwargs().
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.

        # Allow per-request override of video backend via kwargs.
        # This enables users to specify a different backend than the
        # global VLLM_VIDEO_LOADER_BACKEND env var, e.g.:
        #   --media-io-kwargs '{"video": {"video_backend": "torchcodec"}}'
        video_loader_backend = (
            kwargs.pop("video_backend", None) or envs.VLLM_VIDEO_LOADER_BACKEND
        )
        self.kwargs = kwargs
        self.video_loader = VIDEO_LOADER_REGISTRY.load(video_loader_backend)
        self.video_loader_backend = video_loader_backend

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, dict[str, Any]]:
        return self.video_loader.load_bytes(
            data, num_frames=self.num_frames, **self.kwargs
        )

    def load_base64(
        self, media_type: str, data: str
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        if media_type.lower() == "video/jpeg":
            load_frame = partial(
                self.image_io.load_base64,
                "image/jpeg",
            )

            if self.num_frames > 0:
                frame_parts = data.split(",", self.num_frames)[: self.num_frames]
            elif self.num_frames == 0:
                raise ValueError("num_frames must be greater than 0 or -1")
            else:
                frame_parts = data.split(",")

            frames = np.stack(
                [np.asarray(load_frame(frame_data)) for frame_data in frame_parts]
            )
            total = int(frames.shape[0])
            fps = float(self.kwargs.get("fps", 1))

            # validate and extract frames_indices
            frames_indices = self.kwargs.get("frames_indices")
            if frames_indices is not None:
                if not (
                    isinstance(frames_indices, list)
                    and all(isinstance(i, int) for i in frames_indices)
                ):
                    raise ValueError("frames_indices must be a list of integers")
                if len(frames_indices) != total:
                    raise ValueError(
                        f"frames_indices length ({len(frames_indices)}) must "
                        f"match number of frames sent ({total})"
                    )
            else:
                frames_indices = list(range(total))

            # validate and extract total_num_frames
            total_num_frames = self.kwargs.get("total_num_frames", total)
            if not isinstance(total_num_frames, int) or total_num_frames < 1:
                raise ValueError("total_num_frames must be a positive integer")
            if total_num_frames < total:
                raise ValueError(
                    f"total_num_frames ({total_num_frames}) must be >= "
                    f"number of frames sent ({total})"
                )

            # validate and extract duration
            duration = self.kwargs.get("duration")
            if duration is not None:
                if not isinstance(duration, (int, float)) or duration < 0:
                    raise ValueError("duration must be a non-negative number")
            else:
                duration = total_num_frames / fps if fps > 0 else 0.0

            metadata = {
                "total_num_frames": total_num_frames,
                "fps": fps,
                "duration": duration,
                "video_backend": "jpeg_sequence",
                "frames_indices": frames_indices,
                "do_sample_frames": self.kwargs.get("do_sample_frames", False),
            }
            return frames, metadata

        return self.load_bytes(pybase64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, dict[str, Any]]:
        cache_size = envs.VLLM_VIDEO_DECODE_CACHE_SIZE
        if cache_size <= 0:
            return self._load_file_uncached(filepath)

        return _VIDEO_DECODE_CACHE.get_or_load(
            _VIDEO_DECODE_CACHE.key_for_file(
                filepath,
                self.num_frames,
                self.video_loader_backend,
                self.kwargs,
            ),
            cache_size,
            lambda: self._load_file_uncached(filepath),
        )

    def _load_file_uncached(self, filepath: Path) -> tuple[npt.NDArray, dict[str, Any]]:
        with filepath.open("rb") as f:
            return self.load_bytes(f.read())

    def encode_base64(
        self,
        media: npt.NDArray,
        *,
        video_format: str = "JPEG",
    ) -> str:
        video = media

        if video_format == "JPEG":
            encode_frame = partial(
                self.image_io.encode_base64,
                image_format=video_format,
            )

            return ",".join(encode_frame(Image.fromarray(frame)) for frame in video)

        msg = "Only JPEG format is supported for now."
        raise NotImplementedError(msg)
