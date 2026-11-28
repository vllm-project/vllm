# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from vllm.distributed.ec_transfer.ec_connector.tiering.config import (
    ECFileSystemBackpressureConfig,
    ECFileSystemTierConfig,
    ECTieringConfig,
)

pytestmark = pytest.mark.cpu_test


def test_tiering_factory_registered():
    from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
    from vllm.distributed.ec_transfer.ec_connector.tiering.connector import (
        ECTieringConnector,
    )

    cls = ECConnectorFactory._registry["ECTieringConnector"]()
    assert cls is ECTieringConnector


@pytest.fixture
def _vllm_config(tmp_path):
    from transformers import LlavaConfig

    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.ec_transfer import ECTransferConfig

    LlavaConfig(architectures=["LlavaForConditionalGeneration"]).save_pretrained(
        tmp_path
    )
    model = ModelConfig(
        model=str(tmp_path), skip_tokenizer_init=True, max_model_len=128
    )

    def create(extra_config=None, *, role="ec_both"):
        return VllmConfig(
            model_config=model,
            ec_transfer_config=ECTransferConfig(
                ec_connector="ECTieringConnector",
                ec_role=role,
                ec_connector_extra_config=extra_config or {},
            ),
        )

    return create


@pytest.mark.parametrize("role", ["ec_consumer", "ec_producer"])
def test_tiering_rejects_separate_producer_and_consumer_roles(role, _vllm_config):
    with pytest.raises(
        ValueError, match="ECTieringConnector requires ec_role='ec_both'"
    ):
        ECTieringConfig.from_vllm_config(_vllm_config(role=role))


@pytest.mark.parametrize("secondary_tiers", [None, []])
def test_cpu_only_mode_has_no_filesystem_tier(secondary_tiers, _vllm_config) -> None:
    extra_config = {}
    if secondary_tiers is not None:
        extra_config["secondary_tiers"] = secondary_tiers

    config = ECTieringConfig.from_vllm_config(_vllm_config(extra_config))

    assert config.filesystem_tier is None
    assert not config.filesystem_enabled


def test_filesystem_tier_uses_thread_defaults(_vllm_config) -> None:
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


@pytest.mark.parametrize("key", ["high_water_s", "low_water_s"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_backpressure_rejects_non_finite_thresholds(_vllm_config, key, value):
    config = _vllm_config(
        {
            "secondary_tiers": [
                {
                    "type": "fs",
                    "root_dir": "/tmp/cache",
                    "capacity_bytes": 4096,
                    "backpressure": {key: value},
                }
            ]
        }
    )
    with pytest.raises(ValueError, match=f"'{key}'"):
        ECTieringConfig.from_vllm_config(config)


def test_filesystem_tier_accepts_explicit_values(_vllm_config) -> None:
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
    secondary_tiers: object, error: str, _vllm_config
) -> None:
    with pytest.raises(ValueError, match=error):
        ECTieringConfig.from_vllm_config(
            _vllm_config({"secondary_tiers": secondary_tiers})
        )
