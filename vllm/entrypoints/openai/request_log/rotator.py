# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Filesystem rotation for the writer subprocess's jsonl output.

Lives in the writer process so it owns the file handle exclusively —
this avoids the lock dance / SIGHUP-reopen contracts that external
``logrotate`` setups require.

Rotation is triggered when *either* the configured size or time
threshold is crossed (whichever comes first). Both are optional; with
neither set, the rotator behaves like a plain append-only ``open()``.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import IO


def parse_duration(value: str | None) -> float | None:
    """Convert ``"30s"`` / ``"30m"`` / ``"1h"`` / ``"1d"`` (or a plain
    integer-as-seconds) into seconds.

    Returns ``None`` for falsy / unparseable inputs so callers can treat
    rotation as disabled.
    """
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    units = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    try:
        if s[-1] in units:
            return float(s[:-1]) * units[s[-1]]
        return float(s)
    except ValueError:
        return None


class JsonlRotator:
    """Append-only writer with optional size + time rotation.

    Rotated files are renamed to ``{path}.{YYYYMMDD-HHMMSS}``. When
    ``backup_count > 0`` the oldest rotated files past that count are
    deleted on each rotation.
    """

    def __init__(
        self,
        path: str,
        max_bytes: int = 0,
        interval_seconds: float | None = None,
        backup_count: int = 0,
    ) -> None:
        self._path = Path(path)
        self._max_bytes = int(max_bytes) if max_bytes else 0
        self._interval = float(interval_seconds) if interval_seconds else None
        self._backup_count = int(backup_count) if backup_count else 0

        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Long-lived file handle owned by this class; closed in close().
        self._fh: IO[bytes] = open(self._path, "ab", buffering=0)  # noqa: SIM115
        # Track size in-process so we don't ``stat`` on every write.
        self._size = self._path.stat().st_size if self._path.exists() else 0
        self._next_rotation: float | None = (
            time.time() + self._interval if self._interval else None
        )

    def write(self, line: bytes) -> None:
        if self._should_rotate(len(line)):
            self._rotate()
        self._fh.write(line)
        self._size += len(line)

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()

    def _should_rotate(self, incoming: int) -> bool:
        if self._max_bytes and self._size + incoming > self._max_bytes:
            return True
        return self._next_rotation is not None and time.time() >= self._next_rotation

    def _rotate(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()

        # Pick a non-colliding suffix. Most of the time the second-grain
        # timestamp is unique; with very fast rotation we tack on ms.
        suffix = time.strftime(".%Y%m%d-%H%M%S")
        rotated = self._path.with_name(self._path.name + suffix)
        if rotated.exists():
            rotated = self._path.with_name(
                f"{self._path.name}{suffix}-{int(time.time() * 1000) % 1000:03d}"
            )

        # Source file may have been deleted out from under us; just
        # carry on with a new file in that case.
        with contextlib.suppress(OSError):
            os.replace(self._path, rotated)

        self._fh = open(self._path, "ab", buffering=0)  # noqa: SIM115
        self._size = 0
        if self._interval is not None:
            self._next_rotation = time.time() + self._interval
        self._cleanup_old_backups()

    def _cleanup_old_backups(self) -> None:
        if self._backup_count <= 0:
            return
        parent = self._path.parent
        prefix = self._path.name + "."
        candidates = [p for p in parent.iterdir() if p.name.startswith(prefix)]
        if len(candidates) <= self._backup_count:
            return
        candidates.sort(key=lambda p: p.stat().st_mtime)
        for old in candidates[: len(candidates) - self._backup_count]:
            with contextlib.suppress(OSError):
                old.unlink()
