# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

import pybase64

_T = TypeVar("_T")
_U = TypeVar("_U")

_LENGTH_BYTES = 8


@dataclass(frozen=True, init=False)
class DecodeSpec:
    """Canonical snapshot of the decode settings included in a media key."""

    _canonical: bytes

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        canonical = json.dumps(
            dict(settings) if settings is not None else {},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        object.__setattr__(self, "_canonical", canonical)

    @property
    def settings(self) -> Mapping[str, Any]:
        """Return a fresh copy of the settings without exposing mutable state."""
        return json.loads(self._canonical)

    def extend(self, **settings: Any) -> "DecodeSpec":
        """Return a copy with `settings` merged over the current ones."""
        return DecodeSpec({**self.settings, **settings})

    def canonical_bytes(self) -> bytes:
        """Return the canonical JSON bytes used for key derivation."""
        return self._canonical


def derive_media_key(data: bytes, spec: DecodeSpec) -> bytes:
    """Derive a cache key from encoded bytes and the decode spec.

    Header-only parsing at most: this never decodes pixels, frames or PCM, so
    the cost of keying media is independent of the cost of decoding it. It is
    still linear in the payload, so callers on an event loop should offload it.

    sha256 rather than the configured `MMHasherAlgorithm`: the key must not
    depend on which optional hash packages happen to be installed.
    """
    hasher = hashlib.sha256()
    for chunk in (data, spec.canonical_bytes()):
        # Each chunk is preceded by its length, so a chunk boundary can never
        # be shifted by the content of an earlier one.
        hasher.update(len(chunk).to_bytes(_LENGTH_BYTES, "little"))
        hasher.update(chunk)
    return hasher.digest()


_UNDECODED: Any = object()


def _media_ref_from_pickle(media: Any, key: bytes, spec: DecodeSpec) -> "MediaRef[Any]":
    """Rebuild a ref pickled by `MediaRef.__reduce__`.

    Neither the decode closure nor the lock is picklable, so a ref pickles as
    its decoded media: the result is already decoded, keeps its key and spec,
    and carries no encoded bytes.
    """
    ref = MediaRef(lambda: media, b"", spec, key=key)
    ref._media = media
    return ref


class MediaRef(Generic[_T]):
    """A reference to one piece of fetched media.

    This is the single value type for media that vLLM fetched and has not
    (yet) decoded. It carries three things:

    - `key`: the cache identity, derived from the encoded bytes and `spec`
      without decoding, so hashing never pays a decode cost.
    - `spec`: the decode configuration that parameterized or mutated the
      media, already folded into `key`.
    - `decode()`: the decoded media, produced at most once and thread-safely.
    """

    __slots__ = ("_data", "_decoder", "_lock", "_media", "_key", "_spec")

    def __init__(
        self,
        decoder: Callable[[], _T],
        data: bytes,
        spec: DecodeSpec | None = None,
        *,
        key: bytes | None = None,
    ) -> None:
        super().__init__()

        self._decoder: Callable[[], _T] | None = decoder
        self._data = data
        self._media: Any = _UNDECODED
        self._lock = threading.Lock()

        self._spec = DecodeSpec() if spec is None else spec
        self._key = derive_media_key(data, self.spec) if key is None else key

    @property
    def key(self) -> bytes:
        return self._key

    @property
    def spec(self) -> DecodeSpec:
        return self._spec

    @property
    def data(self) -> bytes:
        """The encoded bytes; `b""` once `release()` has run."""
        return self._data

    @property
    def is_decoded(self) -> bool:
        return self._media is not _UNDECODED

    def decode(self) -> _T:
        """Return the decoded media, decoding at most once.

        Thread-safe. Exceptions from the decoder propagate unchanged;
        wrapping them is the caller's job.

        Raises:
            RuntimeError: If the ref was released before being decoded.

        """
        media = self._media
        if media is _UNDECODED:
            decoder = self._decoder
            if decoder is None:
                raise RuntimeError(
                    "Cannot decode a MediaRef whose bytes were released; "
                    "release() may only run once the item is decoded or is "
                    "known to be a cache hit."
                )
            with self._lock:
                media = self._media
                if media is _UNDECODED:
                    media = self._media = decoder()
        return cast(_T, media)

    def fork(self) -> "MediaRef[_T]":
        """Borrow this ref through an independently releasable wrapper."""
        return MediaRef(self.decode, self._data, self.spec, key=self.key)

    def map(
        self,
        transform: Callable[[_T], _U],
        **settings: Any,
    ) -> "MediaRef[_U]":
        """Return a ref decoding to `transform(self.decode())`.

        Shares this ref's bytes (no copy) so the parse layer can stack
        transforms (e.g. audio resampling) onto the decode step. `settings`
        extend the spec -- and therefore the key -- so the transform's own
        parameters stay part of the cache identity. The new key is derived
        from this ref's key rather than by re-digesting the payload.

        With no settings, the key is preserved: use this only for transforms
        that preserve cache identity. Output-changing transforms must supply
        settings identifying the transform and every output-affecting parameter.
        """
        if not settings:
            spec, key = self.spec, self.key
        else:
            spec = self.spec.extend(**settings)
            key = derive_media_key(self.key, spec)

        return MediaRef(lambda: transform(self.decode()), self._data, spec, key=key)

    def release(self) -> None:
        """Drop this ref's encoded bytes and decode closure.

        Call once hashing is done and the item is either decoded or known to
        be a cache hit, after any in-flight decode has finished. Decoded media
        remains readable; an undecoded ref cannot decode after release.

        Other owners can retain the payload: a mapped ref does not release a
        separately retained parent, and decoded media may contain encoded bytes.
        """
        self._data = b""
        self._decoder = None

    def __repr__(self) -> str:
        state = "decoded" if self.is_decoded else "undecoded"
        return f"<MediaRef {state}, {len(self._data)} bytes>"

    def __reduce__(self):
        # Pickling materializes the media: the decoder is a closure over the
        # encoded bytes and the MediaIO that produced it, neither of which can
        # cross a process boundary. A ref released before being decoded has
        # nothing to materialize, so decode() raises.
        return _media_ref_from_pickle, (self.decode(), self.key, self.spec)


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

    def get_decode_spec(self) -> DecodeSpec | None:
        """Return the decode configuration to fold into a ref's cache key.

        `None` when this IO has no settings that affect the decoded media.
        """
        return None

    @abstractmethod
    def load_bytes(self, data: bytes) -> _T:
        raise NotImplementedError

    def load_bytes_ref(self, data: bytes) -> MediaRef[_T]:
        """Return a reference that decodes `data` on first access.

        Decoding errors surface at the use site (e.g. inside the
        multi-modal processor) instead of at the fetch site.
        """
        return MediaRef(partial(self.load_bytes, data), data, self.get_decode_spec())

    def load_base64_ref(self, media_type: str, data: str) -> MediaRef[_T]:
        """Ref variant of `load_base64`; the base64 decoding stays eager."""
        return self.load_bytes_ref(pybase64.b64decode(data, validate=True))

    def load_file_ref(self, filepath: Path) -> MediaRef[_T]:
        """Ref variant of `load_file`; the file read stays eager."""
        return self.load_bytes_ref(filepath.read_bytes())

    @abstractmethod
    def load_base64(self, media_type: str, data: str) -> _T:
        """List of media types:
        https://www.iana.org/assignments/media-types/media-types.xhtml
        """
        raise NotImplementedError

    @abstractmethod
    def load_file(self, filepath: Path) -> _T:
        raise NotImplementedError
