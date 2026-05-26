# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

from vllm.config import set_current_vllm_config
from vllm.distributed.kv_events import BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    connector,
    worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (  # noqa: E501
    MooncakeStoreConnectorMetadata,
)
from vllm.v1.outputs import KVConnectorOutput

from .utils import create_vllm_config


def _make_vllm_config():
    return create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
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
        conn = connector.MooncakeStoreConnector(vllm_config, KVConnectorRole.SCHEDULER)

    mock_scheduler.assert_called_once_with(vllm_config)
    mock_worker.assert_not_called()
    assert conn.connector_scheduler is mock_scheduler.return_value
    assert conn.connector_worker is None


def test_worker_role_initializes_store_worker_on_rank0():
    vllm_config = _make_vllm_config()

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
        conn = connector.MooncakeStoreConnector(vllm_config, KVConnectorRole.WORKER)

    mock_scheduler.assert_not_called()
    mock_worker.assert_called_once_with(vllm_config)
    assert conn.connector_scheduler is None
    assert conn.connector_worker is mock_worker.return_value


def test_worker_role_initializes_on_nonzero_rank():
    vllm_config = _make_vllm_config()
    vllm_config.parallel_config.rank = 1

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker,
    ):
        connector.MooncakeStoreConnector(vllm_config, KVConnectorRole.WORKER)

    mock_worker.assert_called_once_with(vllm_config)


def test_lookup_rpc_path_uses_data_parallel_index_in_dense_dp():
    vllm_config = _make_vllm_config()
    vllm_config.parallel_config.data_parallel_rank = 0
    vllm_config.parallel_config.data_parallel_index = 3

    path = worker.get_zmq_rpc_path_lookup(vllm_config)

    assert path.endswith("_dp_rank3")


def test_lookup_rpc_path_uses_local_rank_when_local_engines_only():
    vllm_config = _make_vllm_config()
    vllm_config.parallel_config.data_parallel_index = 7
    vllm_config.parallel_config.data_parallel_rank_local = 1
    vllm_config.parallel_config.data_parallel_hybrid_lb = True

    path = worker.get_zmq_rpc_path_lookup(vllm_config)

    assert path.endswith("_dp_rank1")


def test_worker_methods_delegate_to_store_worker():
    vllm_config = _make_vllm_config()
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
        conn = connector.MooncakeStoreConnector(vllm_config, KVConnectorRole.WORKER)

    worker_inst = mock_worker_cls.return_value
    worker_inst.get_finished.return_value = ({"req-1"}, {"req-2"})
    conn.bind_connector_metadata(metadata)

    conn.register_kv_caches(kv_caches)
    result = conn.get_finished(finished_req_ids)

    worker_inst.register_kv_caches.assert_called_once_with(kv_caches)
    worker_inst.get_finished.assert_called_once_with(finished_req_ids, metadata)
    assert result == ({"req-1"}, {"req-2"})


def test_get_kv_connector_kv_cache_events_returns_none_when_empty():
    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
    ):
        conn = connector.MooncakeStoreConnector(vllm_config, KVConnectorRole.WORKER)

    mock_worker_cls.return_value.get_kv_events.return_value = []
    assert conn.get_kv_connector_kv_cache_events() is None


def test_get_kv_connector_kv_cache_events_wraps_worker_events():
    vllm_config = _make_vllm_config()
    event = _make_block_stored()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
    ):
        conn = connector.MooncakeStoreConnector(vllm_config, KVConnectorRole.WORKER)

    mock_worker_cls.return_value.get_kv_events.return_value = [event]
    kv_events = conn.get_kv_connector_kv_cache_events()

    assert isinstance(kv_events, connector.MooncakeStoreKVEvents)
    assert kv_events.get_number_of_workers() == 1
    assert kv_events.get_all_events() == [event]


def test_prefer_cross_layer_blocks_from_config():
    # Default: disabled
    vllm_config = _make_vllm_config()
    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ),
    ):
        conn = connector.MooncakeStoreConnector(vllm_config, KVConnectorRole.SCHEDULER)
    assert conn.prefer_cross_layer_blocks is False

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
        conn_enabled = connector.MooncakeStoreConnector(
            vllm_config_enabled, KVConnectorRole.SCHEDULER
        )
    assert conn_enabled.prefer_cross_layer_blocks is True


def test_register_cross_layers_kv_cache_delegates_to_worker():
    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ) as mock_worker_cls,
    ):
        conn = connector.MooncakeStoreConnector(vllm_config, KVConnectorRole.WORKER)

    fake_tensor = MagicMock()
    fake_backend = MagicMock()
    conn.register_cross_layers_kv_cache(fake_tensor, fake_backend)

    worker_inst = mock_worker_cls.return_value
    worker_inst.register_cross_layers_kv_caches.assert_called_once_with(fake_tensor)


def test_update_connector_output_and_take_events():
    vllm_config = _make_vllm_config()
    event = _make_block_stored()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ),
    ):
        conn = connector.MooncakeStoreConnector(vllm_config, KVConnectorRole.SCHEDULER)

    kv_events = connector.MooncakeStoreKVEvents(num_workers=1)
    kv_events.add_events([event])
    conn.update_connector_output(KVConnectorOutput(kv_cache_events=kv_events))

    assert conn._kv_cache_events is kv_events
    assert list(conn.take_events()) == [event]
    assert conn._kv_cache_events is None
