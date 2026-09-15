# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CLI configuration helpers for the tiered Responses token store.

This module only defines and validates configuration. It deliberately does not
register with vLLM's global parser, create stores, or start cleanup tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cleanup import PeriodicCleanupConfig

_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ResponsesStoreOptions:
    """User-facing settings accepted by ``--responses-store-config``.

    Capacities and cleanup budgets use MiB. Watermarks are capacity ratios.
    TTLs and intervals use seconds; a zero TTL disables expiration.
    """

    enabled: bool = False
    disk_enabled: bool = True
    memory_capacity_mb: int = 1024
    disk_capacity_mb: int = 10240
    memory_low_watermark: float = 0.4
    memory_high_watermark: float = 0.8
    disk_low_watermark: float = 0.4
    disk_high_watermark: float = 0.9
    memory_ttl_seconds: int = 1800
    disk_ttl_seconds: int = 36000
    cleanup_interval_seconds: float = 300.0
    cleanup_max_candidates: int = 256
    cleanup_max_bytes_mb: int = 2048
    num_shards: int = 64
    disk_write_interval_seconds: float = 0.05

    @classmethod
    def from_dict(cls, value: object) -> ResponsesStoreOptions:
        if not isinstance(value, dict):
            raise ValueError("responses store config must be a JSON object")
        defaults = {field.name: field.default for field in fields(cls)}
        for name, setting in value.items():
            if name not in defaults:
                raise ValueError(f"unknown responses store config field: {name}")
            expected = type(defaults[name])
            valid = (
                type(setting) in (int, float)
                if expected is float
                else type(setting) is expected
            )
            if not valid:
                raise ValueError(
                    f"responses store config {name} must be {expected.__name__}"
                )
            if isinstance(setting, float) and not math.isfinite(setting):
                raise ValueError(f"responses store config {name} must be finite")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ResponsesStoreConfig:
    """Validated configuration for the tiered Responses token store."""

    enabled: bool
    disk_path: str
    memory_capacity_bytes: int
    disk_capacity_bytes: int
    memory_low_watermark_bytes: int
    memory_high_watermark_bytes: int
    disk_low_watermark_bytes: int
    disk_high_watermark_bytes: int
    memory_ttl_seconds: int | None
    disk_ttl_seconds: int | None
    cleanup_interval_seconds: float
    cleanup_max_candidates: int
    cleanup_max_bytes: int
    num_shards: int
    disk_write_interval_seconds: float
    disk_enabled: bool = True
    key_file: str | None = None

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> ResponsesStoreConfig:
        """Convert CLI-friendly units into the store's internal units."""
        value = getattr(args, "responses_store_config", None)
        options = ResponsesStoreOptions.from_dict({} if value is None else value)
        disk_path = getattr(args, "responses_store_disk_path", None)
        if options.memory_capacity_mb <= 0 or options.disk_capacity_mb <= 0:
            raise ValueError("responses store capacities must be greater than 0")
        if not (
            0 <= options.memory_low_watermark < options.memory_high_watermark <= 1
        ) or not (
            0 <= options.disk_low_watermark < options.disk_high_watermark <= 1
        ):
            raise ValueError(
                "responses store watermarks must satisfy 0 <= low < high <= 1"
            )
        if options.memory_ttl_seconds < 0 or options.disk_ttl_seconds < 0:
            raise ValueError("responses store TTL must be non-negative")
        if options.cleanup_interval_seconds <= 0:
            raise ValueError("cleanup interval must be greater than 0")
        if options.cleanup_max_candidates <= 0:
            raise ValueError("cleanup max candidates must be greater than 0")
        if options.cleanup_max_bytes_mb <= 0:
            raise ValueError("cleanup max bytes must be greater than 0")
        if options.num_shards <= 0:
            raise ValueError("responses store num shards must be greater than 0")
        if options.disk_write_interval_seconds <= 0:
            raise ValueError("disk write interval must be greater than 0")

        key_file = getattr(args, "responses_store_key_file", None)
        if key_file is not None:
            if not options.disk_enabled:
                raise ValueError(
                    "responses store key file requires the disk tier to be enabled"
                )
            if not disk_path:
                raise ValueError(
                    "responses store key file requires an explicit disk path"
                )
            if disk_path == ":memory:":
                raise ValueError(
                    "responses store key management requires a persistent disk path"
                )

        memory_capacity_bytes = options.memory_capacity_mb * _MIB
        disk_capacity_bytes = options.disk_capacity_mb * _MIB

        return cls(
            enabled=options.enabled,
            disk_path=disk_path or _default_disk_path(),
            memory_capacity_bytes=memory_capacity_bytes,
            disk_capacity_bytes=disk_capacity_bytes,
            memory_low_watermark_bytes=int(
                memory_capacity_bytes * options.memory_low_watermark
            ),
            memory_high_watermark_bytes=int(
                memory_capacity_bytes * options.memory_high_watermark
            ),
            disk_low_watermark_bytes=int(
                disk_capacity_bytes * options.disk_low_watermark
            ),
            disk_high_watermark_bytes=int(
                disk_capacity_bytes * options.disk_high_watermark
            ),
            memory_ttl_seconds=options.memory_ttl_seconds or None,
            disk_ttl_seconds=options.disk_ttl_seconds or None,
            cleanup_interval_seconds=options.cleanup_interval_seconds,
            cleanup_max_candidates=options.cleanup_max_candidates,
            cleanup_max_bytes=options.cleanup_max_bytes_mb * _MIB,
            num_shards=options.num_shards,
            disk_write_interval_seconds=options.disk_write_interval_seconds,
            disk_enabled=options.disk_enabled,
            key_file=key_file,
        )

    def build_cleanup_config(self) -> PeriodicCleanupConfig:
        """Build the configuration consumed by periodic cleanup."""
        from .cleanup import PeriodicCleanupConfig, TierCleanupConfig
        from .eviction import CapacityWaterMarks, EvictionSelectionBudget

        budget = EvictionSelectionBudget(
            max_candidates=self.cleanup_max_candidates,
            max_bytes=self.cleanup_max_bytes,
        )
        return PeriodicCleanupConfig(
            interval_seconds=self.cleanup_interval_seconds,
            memory=TierCleanupConfig(
                watermarks=CapacityWaterMarks(
                    max_bytes=self.memory_capacity_bytes,
                    high_watermark_bytes=self.memory_high_watermark_bytes,
                    low_watermark_bytes=self.memory_low_watermark_bytes,
                ),
                budget=budget,
            ),
            disk=TierCleanupConfig(
                watermarks=CapacityWaterMarks(
                    max_bytes=self.disk_capacity_bytes,
                    high_watermark_bytes=self.disk_high_watermark_bytes,
                    low_watermark_bytes=self.disk_low_watermark_bytes,
                ),
                budget=budget,
            ),
        )


def add_responses_store_cli_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Register Responses token-store arguments without starting the store."""
    group = parser.add_argument_group("Responses token store")
    group.add_argument(
        "--responses-store-config",
        type=json.loads,
        default=None,
        help=(
            "Responses token store settings as a JSON object. "
            'Use {"enabled": true} to enable the store. '
            "Keys and defaults: "
            + json.dumps(
                {field.name: field.default for field in fields(ResponsesStoreOptions)}
            )
            + ". Capacities and cleanup_max_bytes_mb use MiB; watermarks are "
            "capacity ratios; TTLs and intervals use seconds (TTL 0 disables "
            "expiration). Capacity targets trigger cleanup, not write rejection. "
            "Set disk_path and key_file using their separate CLI options."
        ),
    )
    group.add_argument(
        "--responses-store-disk-path",
        default=None,
        help="SQLite path. Defaults to a process-local file in the temp directory.",
    )
    group.add_argument(
        "--responses-store-key-file",
        default=None,
        help=(
            "File containing a Base64-encoded 32-byte AES-256 key. Providing "
            "this option preserves key metadata across restarts and enables "
            "atomic database-wide key rotation every 90 days. Session data is "
            "cleared on restart. The file and its parent directory must be writable."
        ),
    )
    return parser


def _default_disk_path() -> str:
    return str(
        Path(tempfile.gettempdir()) / f"vllm-responses-store-{os.getpid()}.sqlite3"
    )


__all__ = [
    "ResponsesStoreConfig",
    "ResponsesStoreOptions",
    "add_responses_store_cli_args",
]
