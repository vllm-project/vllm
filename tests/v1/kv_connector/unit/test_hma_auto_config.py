# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests for HMA auto-disable with KV transfer connectors."""

import pytest

from vllm.config import DeviceConfig, KVTransferConfig, SchedulerConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def mock_hybrid_kv_cache_supported(monkeypatch):
    monkeypatch.setattr(current_platform, "support_hybrid_kv_cache", lambda: True)


@pytest.mark.parametrize(
    "kv_transfer_config,expect_disabled",
    [
        (  # HMA-supporting connector → HMA stays enabled
            KVTransferConfig(
                kv_connector="SimpleCPUOffloadConnector",
                kv_role="kv_both",
                kv_connector_extra_config={"cpu_bytes_to_use": 1 << 30},
            ),
            False,
        ),
        (  # Non-HMA connector → HMA is auto-disabled
            KVTransferConfig(kv_connector="ExampleConnector", kv_role="kv_both"),
            True,
        ),
        (  # MultiConnector: all HMA children → HMA stays enabled
            KVTransferConfig(
                kv_connector="MultiConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "connectors": [
                        {
                            "kv_connector": "SimpleCPUOffloadConnector",
                            "kv_role": "kv_both",
                            "kv_connector_extra_config": {"cpu_bytes_to_use": 1 << 30},
                        },
                        {
                            "kv_connector": "OffloadingConnector",
                            "kv_role": "kv_both",
                            "kv_connector_extra_config": {"cpu_bytes_to_use": 1 << 30},
                        },
                    ]
                },
            ),
            False,
        ),
        (  # MultiConnector: mixed children → HMA is auto-disabled
            KVTransferConfig(
                kv_connector="MultiConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "connectors": [
                        {
                            "kv_connector": "SimpleCPUOffloadConnector",
                            "kv_role": "kv_both",
                            "kv_connector_extra_config": {"cpu_bytes_to_use": 1 << 30},
                        },
                        {"kv_connector": "ExampleConnector", "kv_role": "kv_both"},
                    ]
                },
            ),
            True,
        ),
    ],
    ids=["hma_connector", "non_hma_connector", "multi_all_hma", "multi_mixed"],
)
def test_hma_auto_config(kv_transfer_config, expect_disabled):
    vllm_config = VllmConfig(
        device_config=DeviceConfig("cpu"),
        kv_transfer_config=kv_transfer_config,
    )
    assert (
        vllm_config.scheduler_config.disable_hybrid_kv_cache_manager is expect_disabled
    )


def test_explicit_hma_with_non_hma_connector_errors_at_factory():
    vllm_config = VllmConfig(
        device_config=DeviceConfig("cpu"),
        scheduler_config=SchedulerConfig(
            max_model_len=16,
            is_encoder_decoder=False,
            disable_hybrid_kv_cache_manager=False,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
        ),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
    )
    with pytest.raises(ValueError, match="does not support HMA but HMA is enabled"):
        KVConnectorFactory.create_connector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )
