# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

from vllm.config import set_current_vllm_config
from vllm.distributed.kv_events import BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    connector as mooncake_store_connector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    MooncakeStoreConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.metrics import (
    MooncakeStoreConnectorStats,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.outputs import KVConnectorOutput

from .utils import create_vllm_config


def _make_vllm_config():
    return create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
    )


def _make_kv_cache_config() -> KVCacheConfig:
    """Single-group full-attention KVCacheConfig — enough for the connector
    constructor's validate() pass."""
    spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    return KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[KVCacheTensor(size=8192, shared_by=["layer0"])],
        kv_cache_groups=[KVCacheGroupSpec(["layer0"], spec)],
    )


def _make_block_stored() -> BlockStored:
    return BlockStored(
        block_hashes=[b"hash"],
        parent_block_hash=None,
        token_ids=[1, 2, 3],
        block_size=16,
        lora_id=None,
        medium="cpu",
        lora_name=None,
    )


def test_scheduler_role_initializes_store_scheduler_only():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ) as mock_scheduler,
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )

    mock_scheduler.assert_called_once_with(vllm_config, kv_cache_config)
    mock_worker.assert_not_called()
    assert connector.connector_scheduler is mock_scheduler.return_value
    assert connector.connector_worker is None


def test_worker_methods_delegate_to_store_worker():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()
    kv_caches = {"layer0": MagicMock()}
    metadata = MooncakeStoreConnectorMetadata(set(), set())
    finished_req_ids = {"req-1"}

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

    worker = mock_worker_cls.return_value
    worker.get_finished.return_value = ({"req-1"}, {"req-2"})
    connector.bind_connector_metadata(metadata)

    connector.register_kv_caches(kv_caches)
    result = connector.get_finished(finished_req_ids)

    worker.register_kv_caches.assert_called_once_with(kv_caches)
    worker.get_finished.assert_called_once_with(finished_req_ids, metadata)
    assert result == ({"req-1"}, {"req-2"})


def test_get_kv_connector_kv_cache_events_returns_none_when_empty():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

    mock_worker_cls.return_value.get_kv_events.return_value = []
    assert connector.get_kv_connector_kv_cache_events() is None


def test_get_kv_connector_stats_delegates_to_worker():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()
    expected_stats = MooncakeStoreConnectorStats()
    expected_stats.record_operation("save_put", 0.01, 2, num_bytes=1024)

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

    mock_worker_cls.return_value.get_kv_connector_stats.return_value = expected_stats
    stats = connector.get_kv_connector_stats()

    assert stats is expected_stats
    mock_worker_cls.return_value.get_kv_connector_stats.assert_called_once_with()


def test_build_kv_connector_stats_reconstructs_mooncake_stats():
    stats = mooncake_store_connector.MooncakeStoreConnector.build_kv_connector_stats(
        {
            "save_put": [
                {
                    "duration_seconds": 0.02,
                    "num_keys": 4,
                    "num_bytes": 2048,
                    "status": "ok",
                    "num_failed_keys": 0,
                }
            ]
        }
    )

    assert isinstance(stats, MooncakeStoreConnectorStats)
    assert stats.data["save_put"][0]["num_bytes"] == 2048


def test_get_kv_connector_kv_cache_events_wraps_worker_events():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()
    event = _make_block_stored()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

    mock_worker_cls.return_value.get_kv_events.return_value = [event]
    kv_events = connector.get_kv_connector_kv_cache_events()

    assert isinstance(kv_events, mooncake_store_connector.MooncakeStoreKVEvents)
    assert kv_events.get_number_of_workers() == 1
    assert kv_events.get_all_events() == [event]


def test_prefer_cross_layer_blocks_from_config():
    # Default: disabled
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()
    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ),
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )
    assert connector.prefer_cross_layer_blocks is False

    # Enabled via config
    vllm_config_enabled = create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"enable_cross_layers_blocks": "true"},
    )
    with (
        set_current_vllm_config(vllm_config_enabled),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ),
    ):
        connector_enabled = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config_enabled, KVConnectorRole.SCHEDULER, kv_cache_config
        )
    assert connector_enabled.prefer_cross_layer_blocks is True


def test_register_cross_layers_kv_cache_delegates_to_worker():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

    fake_tensor = MagicMock()
    fake_backend = MagicMock()
    connector.register_cross_layers_kv_cache(fake_tensor, fake_backend)

    worker = mock_worker_cls.return_value
    worker.register_cross_layers_kv_caches.assert_called_once_with(fake_tensor)


def test_update_connector_output_and_take_events():
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()
    event = _make_block_stored()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ),
    ):
        connector = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )

    kv_events = mooncake_store_connector.MooncakeStoreKVEvents(num_workers=1)
    kv_events.add_events([event])
    connector.update_connector_output(KVConnectorOutput(kv_cache_events=kv_events))

    assert connector._kv_cache_events is kv_events
    assert list(connector.take_events()) == [event]
    assert connector._kv_cache_events is None
