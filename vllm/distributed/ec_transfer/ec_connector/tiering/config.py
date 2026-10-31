# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration for the local encoder-cache filesystem tier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm.v1.kv_offload.tiering.backpressure import EMABackpressureDetector

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_SUPPORTED_FS_KEYS = {
    "type",
    "root_dir",
    "capacity_bytes",
    "n_read_threads",
    "n_write_threads",
    "backpressure",
    "fsync",
}

_SUPPORTED_BACKPRESSURE_KEYS = {
    "enabled",
    "high_water_s",
    "low_water_s",
}

_DEFAULT_READ_THREADS = 4
_DEFAULT_WRITE_THREADS = 2
_DEFAULT_BACKPRESSURE_CONFIG = EMABackpressureDetector.default_config("fs")
assert _DEFAULT_BACKPRESSURE_CONFIG is not None


def _get_positive_int(
    config: dict[str, Any], key: str, default: int | None = None
) -> int:
    """Read a required or defaulted positive integer field."""
    if key not in config:
        if default is None:
            raise ValueError(f"filesystem secondary tier '{key}' is required")
        return default

    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"filesystem secondary tier '{key}' must be a positive integer"
        )
    return value


def _get_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    """Read a boolean field, using the given default when it is absent."""
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"filesystem secondary tier '{key}' must be a boolean")
    return value


def _get_positive_float(
    config: dict[str, Any], key: str, default: float
) -> float:
    """Read a positive numeric field as a float."""
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(
            f"filesystem backpressure '{key}' must be a positive number"
        )
    return float(value)


@dataclass(frozen=True)
class ECFileSystemBackpressureConfig:
    """Validated EMA backpressure settings for filesystem stores."""

    enabled: bool
    high_water_s: float
    low_water_s: float


def _parse_backpressure(raw_config: object) -> ECFileSystemBackpressureConfig:
    """Validate filesystem backpressure settings and resolve KV defaults."""
    if not isinstance(raw_config, dict):
        raise ValueError("filesystem backpressure must be a dictionary")

    unsupported_keys = sorted(set(raw_config) - _SUPPORTED_BACKPRESSURE_KEYS)
    if unsupported_keys:
        unsupported = ", ".join(unsupported_keys)
        raise ValueError(
            f"filesystem backpressure contains unsupported fields: {unsupported}"
        )

    high_water_s = _get_positive_float(
        raw_config,
        "high_water_s",
        _DEFAULT_BACKPRESSURE_CONFIG["high_water_s"],
    )
    low_water_s = _get_positive_float(
        raw_config,
        "low_water_s",
        _DEFAULT_BACKPRESSURE_CONFIG["low_water_s"],
    )
    if low_water_s > high_water_s:
        raise ValueError(
            "filesystem backpressure 'low_water_s' must be less than or "
            "equal to 'high_water_s'"
        )
    return ECFileSystemBackpressureConfig(
        enabled=_get_bool(raw_config, "enabled", True),
        high_water_s=high_water_s,
        low_water_s=low_water_s,
    )


@dataclass(frozen=True)
class ECFileSystemTierConfig:
    """Validated settings for one local encoder-cache filesystem tier."""

    root_dir: Path
    capacity_bytes: int
    n_read_threads: int
    n_write_threads: int
    backpressure: ECFileSystemBackpressureConfig
    fsync: bool


def _parse_filesystem_tier(raw_tier: object) -> ECFileSystemTierConfig:
    """Validate one filesystem tier mapping and return its typed config."""
    if not isinstance(raw_tier, dict):
        raise ValueError("filesystem secondary tier must be a dictionary")

    unsupported_keys = sorted(set(raw_tier) - _SUPPORTED_FS_KEYS)
    if unsupported_keys:
        unsupported = ", ".join(unsupported_keys)
        raise ValueError(
            f"filesystem secondary tier contains unsupported fields: {unsupported}"
        )

    if raw_tier.get("type") != "fs":
        raise ValueError("secondary tier 'type' must be 'fs'")

    root_dir_value = raw_tier.get("root_dir")
    if not isinstance(root_dir_value, str) or not root_dir_value.strip():
        raise ValueError(
            "filesystem secondary tier 'root_dir' must be a non-empty string"
        )
    root_dir = Path(root_dir_value).expanduser()
    if not root_dir.is_absolute():
        raise ValueError(
            "filesystem secondary tier 'root_dir' must be an absolute path"
        )

    return ECFileSystemTierConfig(
        root_dir=root_dir,
        capacity_bytes=_get_positive_int(raw_tier, "capacity_bytes"),
        n_read_threads=_get_positive_int(
            raw_tier, "n_read_threads", _DEFAULT_READ_THREADS
        ),
        n_write_threads=_get_positive_int(
            raw_tier, "n_write_threads", _DEFAULT_WRITE_THREADS
        ),
        backpressure=_parse_backpressure(raw_tier.get("backpressure", {})),
        fsync=_get_bool(raw_tier, "fsync", False),
    )


@dataclass(frozen=True)
class ECTieringConfig:
    """Validated top-level configuration for encoder-cache tiering."""

    filesystem_tier: ECFileSystemTierConfig | None

    @property
    def filesystem_enabled(self) -> bool:
        """Return whether a filesystem secondary tier is configured."""
        return self.filesystem_tier is not None

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> ECTieringConfig:
        """Build tiering configuration from vLLM EC connector settings."""
        ec_config = vllm_config.ec_transfer_config
        if ec_config is None:
            raise ValueError("ECTieringConnector requires ec_transfer_config")

        extra_config = ec_config.ec_connector_extra_config
        if not isinstance(extra_config, dict):
            raise ValueError("ec_connector_extra_config must be a dictionary")

        secondary_tiers = extra_config.get("secondary_tiers", [])
        if not isinstance(secondary_tiers, list):
            raise ValueError("secondary_tiers must be a list")
        if not secondary_tiers:
            return cls(filesystem_tier=None)
        if len(secondary_tiers) > 1:
            raise ValueError(
                "ECTieringConnector currently supports at most one secondary tier"
            )

        return cls(filesystem_tier=_parse_filesystem_tier(secondary_tiers[0]))
