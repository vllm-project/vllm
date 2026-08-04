# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import Optional, cast
import io
import json
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.lookup_client.record_strategies.base import RecordStrategy

logger = init_logger(__name__)


class FileHashStrategy(RecordStrategy):
    """File-based strategy that writes chunk hashes to disk."""

    def __init__(self, config, chunk_size: int):
        super().__init__(chunk_size=chunk_size)
        # File size threshold in bytes for rotation (default: 100MB)
        self.file_rotation_size = config.get_extra_config_value(
            "chunk_statistics_file_rotation_size", 100 * 1024 * 1024
        )
        # Maximum number of files to keep before deleting oldest
        self.file_max_count = config.get_extra_config_value(
            "chunk_statistics_file_max_count", 100
        )
        # Directory path for storing chunk hash files
        self.output_dir = Path(
            config.get_extra_config_value(
                "chunk_statistics_file_output_dir", "./chunk_hashes"
            )
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_count = 0
        self.current_file_size = 0
        self.current_file: Optional[Path] = None
        self.current_file_handle: Optional[io.TextIOWrapper] = None
        self.file_list: list[Path] = []

    def preprocess(self, token_ids: list[int]) -> list[str]:
        """Preprocess token IDs into hex hash strings."""
        return self._compute_chunk_hashes_hex(token_ids)

    def record(self, chunk_hashes: list[str], lookup_id: str) -> None:
        """Record chunk hashes to file."""
        with self.lock:
            if (
                self.current_file is None
                or self.current_file_size >= self.file_rotation_size
            ):
                self._rotate_file()
            data = {
                "timestamp": time.time(),
                "lookup_id": lookup_id,
                "chunk_hashes": chunk_hashes,
            }
            if self.current_file_handle is not None:
                file_handle = cast(io.TextIOWrapper, self.current_file_handle)
                line = json.dumps(data) + "\n"
                file_handle.write(line)
                file_handle.flush()
                self.current_file_size += len(line)
            self.total_chunks += len(chunk_hashes)
            self.unique_chunks_count += len(set(chunk_hashes))

    def _rotate_file(self) -> None:
        if self.current_file_handle is not None:
            cast(io.TextIOWrapper, self.current_file_handle).close()
        timestamp = int(time.time())
        self.current_file = (
            self.output_dir / f"chunk_hashes_{timestamp}_{self.file_count:06d}.jsonl"
        )
        self.current_file_handle = open(self.current_file, "w")
        self.current_file_size = 0
        self.file_count += 1
        self.file_list.append(self.current_file)
        if len(self.file_list) > self.file_max_count:
            oldest_file = self.file_list.pop(0)
            try:
                if oldest_file.exists():
                    oldest_file.unlink()
            except Exception as e:
                logger.error("Failed to delete file %s: %s", oldest_file, e)

    def get_statistics(self) -> dict:
        stats = super().get_statistics()
        stats.update(
            {
                "file_hash": {
                    "file_count": self.file_count,
                    "current_file_size": self.current_file_size,
                    "file_max_count": self.file_max_count,
                    "output_dir": str(self.output_dir),
                }
            }
        )
        return stats

    def setup_metrics(self, prometheus_logger) -> None:
        """Setup file hash specific metrics."""
        super().setup_metrics(prometheus_logger)
        prometheus_logger.chunk_statistics_file_count.set_function(
            lambda: self.file_count
        )
        prometheus_logger.chunk_statistics_current_file_size.set_function(
            lambda: self.current_file_size
        )

    def reset(self) -> None:
        """Reset file writer and statistics."""
        with self.lock:
            if self.current_file_handle is not None:
                cast(io.TextIOWrapper, self.current_file_handle).close()
                self.current_file_handle = None
                self.current_file = None
            self.total_chunks = 0
            self.unique_chunks_count = 0
            self.file_count = 0
            self.current_file_size = 0
            self.file_list.clear()

    def close(self) -> None:
        """Close file handles."""
        if self.current_file_handle is not None:
            cast(io.TextIOWrapper, self.current_file_handle).close()
            self.current_file_handle = None
            self.current_file = None
