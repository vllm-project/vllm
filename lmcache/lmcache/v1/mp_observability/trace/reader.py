# SPDX-License-Identifier: Apache-2.0

"""Streaming reader for trace files.

The reader yields ``(Header, Iterator[Record])`` pairs.  Records are
yielded lazily so that arbitrarily large traces can be inspected
without loading the whole file into memory.

Trailing partial frames (truncated by SIGKILL or filesystem buffering)
are detected and the iterator stops cleanly with a WARNING log.
"""

# Future
from __future__ import annotations

# Standard
from typing import BinaryIO, Iterator
import struct

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.trace.format import (
    FORMAT_VERSION,
    MAGIC,
    TRACE_SCHEMA_VERSION,
    Header,
    Record,
    decode_header,
    decode_record,
)

logger = init_logger(__name__)

_LEN_STRUCT = struct.Struct(">I")
_LEN_PREFIX = _LEN_STRUCT.size


class TraceReader:
    """Streaming reader for a binary trace file.

    Usage::

        with TraceReader("/tmp/run.lct") as r:
            header = r.header
            for record in r.records():
                ...

    Closing the reader closes the underlying file handle.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._fh: BinaryIO | None = open(path, "rb")
        try:
            self._header = self._read_header()
        except Exception:
            self._fh.close()
            self._fh = None
            raise

    def __enter__(self) -> TraceReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def header(self) -> Header:
        """Return the file header.  Always present; populated by
        ``__init__``."""
        return self._header

    @property
    def path(self) -> str:
        """Path of the trace file."""
        return self._path

    def records(self) -> Iterator[Record]:
        """Yield every record in the file in order.

        Yields each :class:`Record` as it is read.  When the file ends
        cleanly (boundary aligned to a frame), iteration stops without
        error.  When a partial trailing frame is detected, a warning
        is logged and iteration stops.
        """
        if self._fh is None:
            raise RuntimeError("TraceReader is closed")
        while True:
            frame = self._read_frame(strict=False)
            if frame is None:
                return
            try:
                yield decode_record(frame)
            except Exception as e:
                logger.warning(
                    "TraceReader: skipping malformed record at offset %d: %s",
                    self._fh.tell(),
                    e,
                )
                continue

    def close(self) -> None:
        """Close the underlying file.  Idempotent."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # ---- internal -----------------------------------------------------

    def _read_header(self) -> Header:
        frame = self._read_frame(strict=True)
        if frame is None:
            raise ValueError(f"trace file {self._path!r} is empty")
        header = decode_header(frame)
        if header.magic != MAGIC:
            raise ValueError(
                f"trace file {self._path!r}: bad magic "
                f"(got {header.magic!r}, expected {MAGIC!r})"
            )
        if header.format_version != FORMAT_VERSION:
            raise ValueError(
                f"trace file {self._path!r}: unsupported format_version "
                f"{header.format_version} (this build expects {FORMAT_VERSION})"
            )
        if header.trace_schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError(
                f"trace file {self._path!r}: unsupported trace_schema_version "
                f"{header.trace_schema_version} "
                f"(this build expects {TRACE_SCHEMA_VERSION})"
            )
        return header

    def _read_frame(self, strict: bool) -> bytes | None:
        """Read one length-prefixed frame.

        Returns ``None`` on clean EOF (when ``strict=False``).  On
        truncation in the middle of a frame, logs a WARNING and
        returns ``None``.  In ``strict=True`` mode, both partial and
        empty reads raise.
        """
        assert self._fh is not None
        prefix = self._fh.read(_LEN_PREFIX)
        if not prefix:
            if strict:
                raise ValueError("unexpected EOF reading frame length")
            return None
        if len(prefix) < _LEN_PREFIX:
            msg = (
                f"truncated frame length prefix at offset "
                f"{self._fh.tell() - len(prefix)}"
            )
            if strict:
                raise ValueError(msg)
            logger.warning("TraceReader: %s", msg)
            return None
        (length,) = _LEN_STRUCT.unpack(prefix)
        body = self._fh.read(length)
        if len(body) < length:
            msg = (
                f"truncated frame body at offset "
                f"{self._fh.tell() - len(body)} (got {len(body)} of {length})"
            )
            if strict:
                raise ValueError(msg)
            logger.warning("TraceReader: %s", msg)
            return None
        return body
