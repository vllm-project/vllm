# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

_T = TypeVar("_T")


class MediaIO(ABC, Generic[_T]):

    @abstractmethod
    def load_bytes(
        self,
        data: bytes,
        *,
        request_overrides: Optional[dict[str, Any]] = None,
    ) -> _T:
        raise NotImplementedError

    @abstractmethod
    def load_base64(
        self,
        media_type: str,
        data: str,
        *,
        request_overrides: Optional[dict[str, Any]] = None,
    ) -> _T:
        """
        List of media types:
        https://www.iana.org/assignments/media-types/media-types.xhtml
        """
        raise NotImplementedError

    @abstractmethod
    def load_file(
        self,
        filepath: Path,
        *,
        request_overrides: Optional[dict[str, Any]] = None,
    ) -> _T:
        raise NotImplementedError
