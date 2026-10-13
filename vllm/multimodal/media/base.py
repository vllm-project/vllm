# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

import numpy as np

import vllm.envs as envs

_T = TypeVar("_T")


class MediaDecodedBudget:
    """Thread-safe decoded-media byte accumulator for one request."""

    def __init__(self) -> None:
        self.used = 0
        self._reserved = 0
        self._condition = threading.Condition()

    def reserve(self, max_bytes: int) -> int:
        """Reserve decoder capacity, waiting for in-flight decodes if needed."""
        limit = envs.VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES
        if limit <= 0:
            return max_bytes
        max_bytes = max(0, max_bytes)
        with self._condition:
            while True:
                available = limit - self.used - self._reserved
                if available >= max_bytes or self._reserved == 0:
                    if available <= 0:
                        raise ValueError(
                            "Request decoded media exceeds "
                            "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES."
                        )
                    reserved = min(max_bytes, available)
                    self._reserved += reserved
                    return reserved
                self._condition.wait()

    def settle(self, reserved: int, decoded_bytes: int) -> None:
        """Replace a decoder reservation with its actual decoded size."""
        limit = envs.VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES
        if limit <= 0:
            return
        with self._condition:
            self._reserved -= reserved
            try:
                used = self.used + decoded_bytes
                if decoded_bytes > reserved or used > limit:
                    raise ValueError(
                        "Request decoded media exceeds "
                        f"VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES={limit} "
                        f"({used} bytes)."
                    )
                self.used = used
            finally:
                self._condition.notify_all()

    def charge(self, decoded_bytes: int) -> None:
        limit = envs.VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES
        if limit <= 0 or decoded_bytes <= 0:
            return
        with self._condition:
            while (
                self.used + self._reserved + decoded_bytes > limit
                and self._reserved > 0
                and self.used + decoded_bytes <= limit
            ):
                self._condition.wait()
            used = self.used + decoded_bytes
            if used > limit:
                raise ValueError(
                    "Request decoded media exceeds "
                    f"VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES={limit} "
                    f"({used} bytes)."
                )
            self.used = used


@dataclass
class MediaWithBytes(Generic[_T]):
    """Wrapper that couples a media object with its original encoded bytes.

    This ensures the raw bytes and media object remain synchronized,
    preventing cache corruption from in-place modifications.

    The wrapper delegates attribute access to the underlying media object,
    making it behave transparently like the wrapped type (e.g., PIL.Image).

    NOTE: Currently, this wrapper is used only for the image and video
    modalities.
    """

    media: _T
    original_bytes: bytes = field(repr=False)
    io_config: dict[str, Any] | None = None
    """Decode settings that altered the media relative to `original_bytes`
    (e.g. `image_mode` conversion), so they participate in cache hashing."""

    def __array__(self, *args, **kwargs) -> np.ndarray:
        """Allow np.array(obj) to return np.array(obj.media)."""
        return np.array(self.media, *args, **kwargs)

    def __iter__(self) -> Iterator[Any]:
        """Allow unpacking obj to unpack obj.media (e.g. video tuples)."""
        return iter(cast(Iterable[Any], self.media))

    def __getitem__(self, index: Any) -> Any:
        """Allow obj[i] to index obj.media (e.g. video tuples)."""
        return cast(Any, self.media)[index]

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]):
        self.__dict__.update(state)

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying media object."""
        return getattr(self.media, name)


class MediaIO(ABC, Generic[_T]):
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
        """Merge config-level kwargs and request-level kwargs.

        By default this performs a shallow merge where runtime kwargs override
        keys in default kwargs. Subclasses may override to apply modality-
        specific behavior.
        """
        merged = dict(default_kwargs or {})
        if runtime_kwargs:
            merged.update(runtime_kwargs)
        return merged

    def get_max_bytes(self) -> int | None:
        """Return the maximum encoded payload size accepted by this media IO."""
        return None

    @abstractmethod
    def load_bytes(self, data: bytes) -> _T:
        raise NotImplementedError

    @abstractmethod
    def load_base64(self, media_type: str, data: str) -> _T:
        """List of media types:
        https://www.iana.org/assignments/media-types/media-types.xhtml
        """
        raise NotImplementedError

    @abstractmethod
    def load_file(self, filepath: Path) -> _T:
        raise NotImplementedError
