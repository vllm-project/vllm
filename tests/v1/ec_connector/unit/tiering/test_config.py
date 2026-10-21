# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.distributed.ec_transfer.ec_connector.tiering.config import (
    ECFileSystemBackpressureConfig,
    ECFileSystemTierConfig,
    ECTieringConfig,
)


def _vllm_config(extra_config: object) -> SimpleNamespace:
    ec_config = SimpleNamespace(ec_connector_extra_config=extra_config)
    return SimpleNamespace(ec_transfer_config=ec_config)


@pytest.mark.parametrize("secondary_tiers", [None, []])
def test_cpu_only_mode_has_no_filesystem_tier(secondary_tiers: object) -> None:
    extra_config = {}
    if secondary_tiers is not None:
        extra_config["secondary_tiers"] = secondary_tiers

    config = ECTieringConfig.from_vllm_config(_vllm_config(extra_config))

    assert config.filesystem_tier is None
    assert not config.filesystem_enabled


def test_filesystem_tier_uses_thread_defaults() -> None:
    config = ECTieringConfig.from_vllm_config(
        _vllm_config(
            {
                "ec_cpu_bytes": 1024,
                "secondary_tiers": [
                    {
                        "type": "fs",
                        "root_dir": "/var/tmp/ec-cache",
                        "capacity_bytes": 4096,
                    }
                ],
            }
        )
    )

    assert config.filesystem_tier == ECFileSystemTierConfig(
        root_dir=Path("/var/tmp/ec-cache"),
        capacity_bytes=4096,
        n_read_threads=4,
        n_write_threads=2,
        backpressure=ECFileSystemBackpressureConfig(
            enabled=True,
            high_water_s=0.010,
            low_water_s=0.002,
        ),
        fsync=False,
    )


def test_filesystem_tier_accepts_explicit_values() -> None:
    config = ECTieringConfig.from_vllm_config(
        _vllm_config(
            {
                "secondary_tiers": [
                    {
                        "type": "fs",
                        "root_dir": "/mnt/ssd/ec-cache",
                        "capacity_bytes": 8192,
                        "n_read_threads": 8,
                        "n_write_threads": 3,
                        "backpressure": {
                            "enabled": False,
                            "high_water_s": 0.2,
                            "low_water_s": 0.1,
                        },
                        "fsync": True,
                    }
                ]
            }
        )
    )

    assert config.filesystem_tier == ECFileSystemTierConfig(
        root_dir=Path("/mnt/ssd/ec-cache"),
        capacity_bytes=8192,
        n_read_threads=8,
        n_write_threads=3,
        backpressure=ECFileSystemBackpressureConfig(
            enabled=False,
            high_water_s=0.2,
            low_water_s=0.1,
        ),
        fsync=True,
    )


@pytest.mark.parametrize(
    ("secondary_tiers", "error"),
    [
        ({}, "secondary_tiers must be a list"),
        ([{}, {}], "at most one secondary tier"),
        (["fs"], r"filesystem secondary tier must be a dictionary"),
        ([{"type": "memory"}], r"secondary tier 'type' must be 'fs'"),
        (
            [{"type": "fs", "root_dir": "relative", "capacity_bytes": 1}],
            r"'root_dir' must be an absolute path",
        ),
        (
            [{"type": "fs", "root_dir": "/tmp/cache"}],
            r"'capacity_bytes' is required",
        ),
        (
            [
                {
                    "type": "fs",
                    "root_dir": "/tmp/cache",
                    "capacity_bytes": 1,
                    "backpressure": True,
                }
            ],
            r"filesystem backpressure must be a dictionary",
        ),
        (
            [
                {
                    "type": "fs",
                    "root_dir": "/tmp/cache",
                    "capacity_bytes": 1,
                    "backpressure": {
                        "high_water_s": 0.1,
                        "low_water_s": 0.2,
                    },
                }
            ],
            r"'low_water_s' must be less than or equal to 'high_water_s'",
        ),
        (
            [
                {
                    "type": "fs",
                    "root_dir": "/tmp/cache",
                    "capacity_bytes": 1,
                    "backpressure": {"alpha": 0.2},
                }
            ],
            r"backpressure contains unsupported fields: alpha",
        ),
        (
            [
                {
                    "type": "fs",
                    "root_dir": "/tmp/cache",
                    "capacity_bytes": 1,
                    "max_pending_reads": 64,
                    "max_pending_writes": 64,
                }
            ],
            r"unsupported fields: max_pending_reads, max_pending_writes",
        ),
        (
            [
                {
                    "type": "fs",
                    "root_dir": "/tmp/cache",
                    "capacity_bytes": True,
                }
            ],
            r"'capacity_bytes' must be a positive integer",
        ),
        (
            [
                {
                    "type": "fs",
                    "root_dir": "/tmp/cache",
                    "capacity_bytes": 1,
                    "fsync": "false",
                }
            ],
            r"'fsync' must be a boolean",
        ),
        (
            [
                {
                    "type": "fs",
                    "root_dir": "/tmp/cache",
                    "capacity_bytes": 1,
                    "unknown": 2,
                }
            ],
            r"unsupported fields: unknown",
        ),
    ],
)
def test_invalid_secondary_tier_is_rejected(
    secondary_tiers: object, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        ECTieringConfig.from_vllm_config(
            _vllm_config({"secondary_tiers": secondary_tiers})
        )
