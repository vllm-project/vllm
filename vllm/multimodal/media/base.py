# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

import numpy as np
import pybase64

_T = TypeVar("_T")


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


_UNDECODED: Any = object()


class LazyMedia(Generic[_T]):
    """Holds encoded bytes plus a decode callable; decodes on first access.

    Duck-type compatible with
    [`MediaWithBytes`][vllm.multimodal.media.base.MediaWithBytes]:
    `.media`, `.original_bytes`, `.io_config`, attribute delegation and
    `__array__`/`__iter__`/`__getitem__` passthrough.

    Decoding is thread-safe and happens at most once; the result is cached.
    """

    __slots__ = (
        "_decoder",
        "_lock",
        "_media",
        "_original_bytes",
        "header_image",
        "io_config",
    )

    def __init__(
        self,
        decoder: Callable[[], _T],
        original_bytes: bytes,
        io_config: dict[str, Any] | None = None,
        header_image: Any | None = None,
    ) -> None:
        super().__init__()

        self._decoder = decoder
        self._media: Any = _UNDECODED
        self._lock = threading.Lock()
        self._original_bytes = original_bytes
        self.io_config = io_config
        """Decode settings that altered the media relative to `original_bytes`,
        so they participate in cache hashing."""
        self.header_image = header_image
        """Header-only PIL image opened eagerly at fetch time, letting the
        hasher read EXIF metadata without triggering pixel decoding."""

    @property
    def media(self) -> _T:
        """The decoded media object; the first access triggers decoding."""
        return self.decode()

    @property
    def original_bytes(self) -> bytes:
        """The encoded bytes; `b""` once released by `release_bytes()`."""
        return self._original_bytes

    @property
    def is_decoded(self) -> bool:
        return self._media is not _UNDECODED

    def decode(self) -> _T:
        """Decode explicitly; thread-safe; exceptions propagate unchanged
        (wrapping them is the caller's job)."""
        media = self._media
        if media is _UNDECODED:
            with self._lock:
                media = self._media
                if media is _UNDECODED:
                    media = self._media = self._decoder()
        return cast(_T, media)

    def map(self, transform: Callable[[_T], Any]) -> "LazyMedia[Any]":
        """Return a new LazyMedia decoding to `transform(self.media)`.

        Shares the same `original_bytes` reference (no copy), letting the
        parse layer stack transforms (e.g. audio resample/normalization)
        onto the decode step.
        """
        return LazyMedia(
            lambda: transform(self.media),
            self._original_bytes,
            self.io_config,
            self.header_image,
        )

    def release_bytes(self) -> None:
        """Release the original bytes once hashing is done and the item is
        either cache-hit or decoded."""
        self._original_bytes = b""

    def __array__(self, *args, **kwargs) -> np.ndarray:
        """Allow np.array(obj) to return np.array(obj.media)."""
        return np.array(self.media, *args, **kwargs)

    def __iter__(self) -> Iterator[Any]:
        """Allow unpacking obj to unpack obj.media (e.g. video tuples)."""
        return iter(cast(Iterable[Any], self.media))

    def __getitem__(self, index: Any) -> Any:
        """Allow obj[i] to index obj.media (e.g. video tuples)."""
        return cast(Any, self.media)[index]

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying media object."""
        return getattr(self.media, name)

    def __repr__(self) -> str:
        state = "decoded" if self.is_decoded else "undecoded"
        return f"<LazyMedia {state}, {len(self._original_bytes)} bytes>"

    def __reduce__(self):
        # Materialize before pickling; rebuild as a plain eager
        # MediaWithBytes so no decode closure crosses the pickle boundary.
        media = self.media
        if isinstance(media, MediaWithBytes):
            # Flatten wrappers produced by MediaIO.load_bytes.
            return (
                MediaWithBytes,
                (media.media, media.original_bytes, media.io_config),
            )
        return (MediaWithBytes, (media, self.original_bytes, self.io_config))


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

    def load_bytes_lazy(self, data: bytes) -> LazyMedia[_T]:
        """Return a lazy handle that decodes `data` on first access.

        Decoding errors surface at the use site (e.g. inside the
        multi-modal processor) instead of at the fetch site.
        """
        return LazyMedia(partial(self.load_bytes, data), data)

    def load_base64_lazy(self, media_type: str, data: str) -> LazyMedia[_T]:
        """Lazy variant of `load_base64`; the base64 decoding stays eager."""
        return self.load_bytes_lazy(pybase64.b64decode(data, validate=True))

    def load_file_lazy(self, filepath: Path) -> LazyMedia[_T]:
        """Lazy variant of `load_file`; the file read stays eager."""
        return self.load_bytes_lazy(filepath.read_bytes())

    @abstractmethod
    def load_base64(self, media_type: str, data: str) -> _T:
        """List of media types:
        https://www.iana.org/assignments/media-types/media-types.xhtml
        """
        raise NotImplementedError

    @abstractmethod
    def load_file(self, filepath: Path) -> _T:
        raise NotImplementedError
