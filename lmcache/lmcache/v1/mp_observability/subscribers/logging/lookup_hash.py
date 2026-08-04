# SPDX-License-Identifier: Apache-2.0

"""Lookup hash file-logging subscriber.

Subscribes to ``MP_LOOKUP`` events and writes lookup hash data to
rotating JSONL files for offline analysis.  Because the EventBus
drain thread dispatches callbacks off the hot path, no extra queue
or worker thread is needed — file I/O happens in the EventBus
background thread.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import io
import json

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

logger = init_logger(__name__)

# Pattern for discovering existing log files on disk.
_LOG_FILE_GLOB = "lookup_hashes_*.jsonl"


def _format_timestamp(ts: float) -> str:
    """Format a unix timestamp as a compact datetime string.

    Example: 20260401_143025
    """
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y%m%d_%H%M%S")


@dataclass
class LookupHashLogConfig:
    """Configuration for lookup hash file logging.

    When ``output_dir`` is non-empty, chunk hashes computed during
    lookup are written to rotating JSONL files for offline analysis.
    """

    output_dir: str = ""
    """Directory to write lookup hash JSONL files.
    Empty string disables logging."""

    rotation_interval_sec: int = 6 * 3600
    """Time interval in seconds before rotating to a new file
    (default 6 hours)."""

    rotation_max_size: int = 100 * 1024 * 1024
    """Max file size in bytes before rotating even if the time
    interval has not elapsed (default 100MB)."""

    max_files: int = 100
    """Max number of log files to keep before deleting oldest."""

    @property
    def enabled(self) -> bool:
        """Whether lookup hash logging is enabled."""
        return bool(self.output_dir)


class LookupHashLoggingSubscriber(EventSubscriber):
    """EventBus subscriber that writes lookup hashes to rotating JSONL files.

    Leverages the EventBus drain thread for async I/O instead of maintaining
    its own queue and worker.

    Files rotate when either the time interval or file size limit is
    reached, whichever comes first.  File names include a
    human-readable timestamp, e.g.::

        lookup_hashes_20260401_143025_000003.jsonl

    Each JSONL line has the format::

        {"timestamp": 1711929600.123, "request_id": "req-abc",
         "model_name": "DeepSeek-V3",
         "chunk_size": 256, "seq_len": 1024,
         "dtypes": ["float8_e4m3fn"],
         "shapes": [[32, 256, 128]],
         "chunk_hashes": ["0xab...", ...]}
    """

    def __init__(self, config: LookupHashLogConfig) -> None:
        self._config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File state (only accessed from the EventBus drain thread)
        self._current_file_size: int = 0
        self._current_file: Optional[Path] = None
        self._current_handle: Optional[io.TextIOWrapper] = None
        self._current_file_opened_at: float = 0.0

        # Discover existing log files so max_files limit accounts
        # for files from previous runs.
        self._file_list: list[Path] = sorted(
            self.output_dir.glob(_LOG_FILE_GLOB),
            key=lambda p: p.stat().st_mtime,
        )
        self._file_count: int = len(self._file_list)

        logger.info(
            "LookupHashLoggingSubscriber started: output_dir=%s, "
            "rotation_interval=%ds, "
            "rotation_max_size=%d, max_files=%d, "
            "existing_files=%d",
            self.output_dir,
            config.rotation_interval_sec,
            config.rotation_max_size,
            config.max_files,
            len(self._file_list),
        )

    # -- EventSubscriber interface -----------------------------------------

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_LOOKUP: self._on_lookup_hashes,
        }

    def shutdown(self) -> None:
        """Close the current file handle on EventBus shutdown."""
        if self._current_handle is not None:
            self._current_handle.close()
            self._current_handle = None
        logger.info("LookupHashLoggingSubscriber closed")

    # -- Callback ----------------------------------------------------------

    def _on_lookup_hashes(self, event: Event) -> None:
        """Write a lookup hash event to the current JSONL file."""
        meta = event.metadata
        timestamp = event.timestamp

        if self._needs_rotation(timestamp):
            self._rotate_file(timestamp)

        chunk_hashes_raw = meta.get("chunk_hashes", [])
        data = {
            "timestamp": timestamp,
            "request_id": meta.get("request_id", ""),
            "model_name": meta.get("model_name", ""),
            "chunk_size": meta.get("chunk_size", 0),
            "seq_len": meta.get("seq_len", 0),
            "dtypes": meta.get("dtypes", []),
            "shapes": meta.get("shapes", []),
            "chunk_hashes": [
                "0x" + h.hex() if isinstance(h, bytes) else hex(h)
                for h in chunk_hashes_raw
            ],
        }
        line = json.dumps(data) + "\n"
        if self._current_handle is not None:
            self._current_handle.write(line)
            self._current_handle.flush()
            self._current_file_size = self._current_handle.tell()

    # -- File rotation -----------------------------------------------------

    def _needs_rotation(self, now: float) -> bool:
        """Check if the current file needs rotation."""
        if self._current_handle is None:
            return True
        elapsed = now - self._current_file_opened_at
        if elapsed >= self._config.rotation_interval_sec:
            return True
        if self._current_file_size >= self._config.rotation_max_size:
            return True
        return False

    def _rotate_file(self, now: float) -> None:
        """Close current file and open a new one."""
        if self._current_handle is not None:
            self._current_handle.close()
            self._current_handle = None

        time_str = _format_timestamp(now)
        self._current_file = (
            self.output_dir / f"lookup_hashes_{time_str}_{self._file_count:06d}.jsonl"
        )
        self._current_handle = open(self._current_file, "w", encoding="utf-8")
        self._current_file_opened_at = now
        self._current_file_size = 0
        self._file_count += 1
        self._file_list.append(self._current_file)

        # Enforce max file count
        while len(self._file_list) > self._config.max_files:
            oldest = self._file_list.pop(0)
            try:
                if oldest.exists():
                    oldest.unlink()
            except Exception as e:
                logger.error(
                    "Failed to delete old lookup hash file %s: %s",
                    oldest,
                    e,
                )
