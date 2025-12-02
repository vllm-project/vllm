# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np

_T = TypeVar("_T")


@dataclass
class MediaWithBytes(Generic[_T]):
    """
    Wrapper that couples a media object with its original encoded bytes.

    This ensures the raw bytes and media object remain synchronized,
    preventing cache corruption from in-place modifications.

    The wrapper delegates attribute access to the underlying media object,
    making it behave transparently like the wrapped type (e.g., PIL.Image).
    """

    media: _T
    original_bytes: bytes

    def __array__(self, *args, **kwargs) -> np.ndarray:
        """Allow np.array(obj) to return np.array(obj.media)."""
        return np.array(self.media, *args, **kwargs)

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying media object."""
        # This is only called when the attribute is not found on self
        return getattr(self.media, name)


class MediaIO(ABC, Generic[_T]):
    @abstractmethod
    def load_bytes(self, data: bytes) -> _T:
        raise NotImplementedError

    @abstractmethod
    def load_base64(self, media_type: str, data: str) -> _T:
        """
        List of media types:
        https://www.iana.org/assignments/media-types/media-types.xhtml
        """
        raise NotImplementedError

    @abstractmethod
    def load_file(self, filepath: Path) -> _T:
        raise NotImplementedError
