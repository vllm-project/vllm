# SPDX-License-Identifier: Apache-2.0
"""Shim that exposes hipfile behind compatible names for cufile-python.

This module wraps ``hipfile.Driver``, ``hipfile.FileHandle``, and
``hipfile.Buffer`` so that callers written against the cufile-python API
(``CuFileDriver`` / ``CuFile``) can use hipfile without any code changes.
"""

# Standard
from collections.abc import Callable
from types import TracebackType
from typing import Any
import ctypes
import os

# Third Party
from hipfile import Buffer, Driver, FileHandle


def _os_flags(mode: str) -> int:
    """Translate a Python-style mode string to ``os.open`` flags."""
    flags_map: dict[str, int] = {
        "r": os.O_RDONLY,
        "r+": os.O_RDWR,
        "w": os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
        "w+": os.O_CREAT | os.O_RDWR | os.O_TRUNC,
        "a": os.O_CREAT | os.O_WRONLY | os.O_APPEND,
        "a+": os.O_CREAT | os.O_RDWR | os.O_APPEND,
    }
    return flags_map[mode]


def _singleton(cls: type) -> Callable:
    """Decorator that turns *cls* into a singleton."""
    _instances: dict[type, object] = {}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]

    return wrapper


@_singleton
class CuFileDriver:
    """Singleton wrapper around :class:`hipfile.Driver`.

    Matches the cufile-python ``CuFileDriver`` interface: the driver is opened
    on first instantiation and closed when the singleton is destroyed.
    """

    def __init__(self) -> None:
        self._driver = Driver()
        self._driver.open()

    def __del__(self) -> None:
        self._driver.close()


class CuFile:
    """Wrapper around :class:`hipfile.FileHandle` with cufile-python semantics.

    Args:
        path: Filesystem path to open.
        mode: Python-style mode string (``"r"``, ``"r+"``, ``"w"``, etc.).
        use_direct_io: If ``True``, ``O_DIRECT`` is ORed into the flags.
    """

    def __init__(
        self,
        path: str,
        mode: str = "r",
        use_direct_io: bool = False,
    ) -> None:
        flags = _os_flags(mode)
        if use_direct_io:
            flags |= os.O_DIRECT
        self._file_handle = FileHandle(path, flags)

    @property
    def is_open(self) -> bool:
        """Return ``True`` if the underlying handle has been opened."""
        return self._file_handle.handle is not None

    def open(self) -> None:
        """Open the file and register the hipfile handle."""
        if self.is_open:
            return
        self._file_handle.open()

    def close(self) -> None:
        """Deregister the handle and close the file."""
        if not self.is_open:
            return
        self._file_handle.close()

    def __enter__(self) -> "CuFile":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        if self.is_open:
            self.close()

    def read(
        self,
        dest: ctypes.c_void_p,
        size: int,
        file_offset: int = 0,
        dev_offset: int = 0,
    ) -> int:
        """Read from the file into a GPU buffer.

        Args:
            dest: Device pointer (``ctypes.c_void_p``) to read into.
            size: Number of bytes to read.
            file_offset: Byte offset in the file.
            dev_offset: Byte offset in the device buffer.

        Returns:
            Number of bytes actually read.

        Raises:
            IOError: If the file is not open.
        """
        if not self.is_open:
            raise IOError("File is not open.")
        buf = Buffer.from_ctypes_void_p(dest, size, flags=0)
        return self._file_handle.read(buf, size, file_offset, dev_offset)

    def write(
        self,
        src: ctypes.c_void_p,
        size: int,
        file_offset: int = 0,
        dev_offset: int = 0,
    ) -> int:
        """Write from a GPU buffer into the file.

        Args:
            src: Device pointer (``ctypes.c_void_p``) to write from.
            size: Number of bytes to write.
            file_offset: Byte offset in the file.
            dev_offset: Byte offset in the device buffer.

        Returns:
            Number of bytes actually written.

        Raises:
            IOError: If the file is not open.
        """
        if not self.is_open:
            raise IOError("File is not open.")
        buf = Buffer.from_ctypes_void_p(src, size, flags=0)
        return self._file_handle.write(buf, size, file_offset, dev_offset)


__all__ = ["CuFile", "CuFileDriver"]
