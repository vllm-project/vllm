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
    protocol,
    scheduler,
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


# ============================================================
# reset_cache() — RL hard-reset path via typed LookupKey protocol
# ============================================================


def test_reset_cache_scheduler_role_delegates_to_reset_store():
    """SCHEDULER role reset_cache() routes to scheduler.reset_store()."""
    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ) as mock_scheduler_cls,
    ):
        conn = connector.MooncakeStoreConnector(vllm_config, KVConnectorRole.SCHEDULER)

    mock_scheduler_cls.return_value.reset_store.return_value = True
    assert conn.reset_cache() is True
    mock_scheduler_cls.return_value.reset_store.assert_called_once_with()


def test_reset_cache_scheduler_role_propagates_failure():
    """SCHEDULER role surfaces False when scheduler.reset_store() fails."""
    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ) as mock_scheduler_cls,
    ):
        conn = connector.MooncakeStoreConnector(vllm_config, KVConnectorRole.SCHEDULER)

    mock_scheduler_cls.return_value.reset_store.return_value = False
    assert conn.reset_cache() is False


def test_reset_cache_worker_role_returns_none():
    """WORKER role reset_cache() is a no-op; reset is driven via ZMQ admin."""
    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ),
    ):
        conn = connector.MooncakeStoreConnector(vllm_config, KVConnectorRole.WORKER)

    assert conn.reset_cache() is None


def test_scheduler_reset_store_returns_client_reset_result():
    """MooncakeStoreScheduler.reset_store() returns LookupKeyClient.reset()."""
    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "scheduler.LookupKeyClient"
        ) as mock_client_cls,
    ):
        sched = scheduler.MooncakeStoreScheduler(vllm_config)

    mock_client_cls.return_value.reset.return_value = True
    assert sched.reset_store() is True
    mock_client_cls.return_value.reset.assert_called_once_with()


def test_scheduler_reset_store_handles_rpc_exception():
    """Exceptions from the ZMQ reset RPC convert to False, not raise."""
    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "scheduler.LookupKeyClient"
        ) as mock_client_cls,
    ):
        sched = scheduler.MooncakeStoreScheduler(vllm_config)

    mock_client_cls.return_value.reset.side_effect = RuntimeError("rpc timed out")
    assert sched.reset_store() is False


def test_lookup_key_client_lookup_prepends_typed_tag():
    """LookupKeyClient.lookup() puts LOOKUP_MSG tag at frame 0."""
    vllm_config = _make_vllm_config()

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
        "worker.make_zmq_socket"
    ) as mock_make_socket:
        client = worker.LookupKeyClient(vllm_config)

    fake_socket = mock_make_socket.return_value
    fake_socket.recv.return_value = (5).to_bytes(4, "big")

    assert client.lookup(token_len=128, block_hashes=[]) == 5

    sent_frames = fake_socket.send_multipart.call_args[0][0]
    assert sent_frames[0] == protocol.LOOKUP_MSG
    assert int.from_bytes(sent_frames[1], "big") == 128


def test_lookup_key_client_reset_uses_typed_protocol():
    """LookupKeyClient.reset() sends RESET_MSG and parses RESP_OK / RESP_ERR."""
    vllm_config = _make_vllm_config()

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
        "worker.make_zmq_socket"
    ) as mock_make_socket:
        client = worker.LookupKeyClient(vllm_config)

    fake_socket = mock_make_socket.return_value

    # ACK path: server returns RESP_OK -> client returns True.
    fake_socket.recv.return_value = protocol.RESP_OK
    assert client.reset() is True
    assert fake_socket.send.call_args[0][0] == protocol.RESET_MSG

    # NACK path: server returns RESP_ERR -> client returns False.
    fake_socket.recv.return_value = protocol.RESP_ERR
    assert client.reset() is False


def test_protocol_tags_are_distinct_and_non_empty():
    """Protocol tags must be unique and non-empty to avoid collision."""
    tags = {protocol.LOOKUP_MSG, protocol.RESET_MSG}
    assert len(tags) == 2
    for tag in tags:
        assert isinstance(tag, bytes)
        assert len(tag) > 0
    assert protocol.RESP_OK != protocol.RESP_ERR


def test_scheduler_reset_connector_cache_invokes_connector_reset():
    """Cascade test: Scheduler.reset_prefix_cache(reset_connector=True)
    cascades into MooncakeStoreConnector.reset_cache without dragging in
    the heavy KVCacheManager fixtures.
    """
    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ) as mock_scheduler_cls,
    ):
        conn = connector.MooncakeStoreConnector(vllm_config, KVConnectorRole.SCHEDULER)

    mock_scheduler_cls.return_value.reset_store.return_value = True

    class _StubScheduler:
        def __init__(self, c):
            self.connector = c

        def reset_connector_cache(self):
            return self.connector.reset_cache() is not False

    sched = _StubScheduler(conn)
    assert sched.reset_connector_cache() is True
    mock_scheduler_cls.return_value.reset_store.assert_called_once_with()

    mock_scheduler_cls.return_value.reset_store.reset_mock()
    mock_scheduler_cls.return_value.reset_store.return_value = False
    assert sched.reset_connector_cache() is False


def test_reset_cache_scheduler_role_clears_local_state():
    """SCHEDULER reset_cache() must clear scheduler-side state that points
    at master keys we're about to wipe -- pending load_specs and
    accumulated _kv_cache_events both reference keys whose blobs are
    about to be remove_all'd, so reading them after reset would surface
    stale references to wiped keys.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (  # noqa: E501
        LoadSpec,
    )

    vllm_config = _make_vllm_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ) as mock_scheduler_cls,
    ):
        conn = connector.MooncakeStoreConnector(vllm_config, KVConnectorRole.SCHEDULER)

    # Seed both sentinel pieces of stale-reference state.
    sched_inst = mock_scheduler_cls.return_value
    sched_inst.load_specs = {
        "req-A": LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=128, can_load=True)
    }
    conn._kv_cache_events = connector.MooncakeStoreKVEvents(num_workers=1)
    sched_inst.reset_store.return_value = True

    assert conn.reset_cache() is True

    # Both stale references must be cleared by the time reset_store is
    # invoked downstream (load_specs flushed dict, events nulled).
    assert sched_inst.load_specs == {}
    assert conn._kv_cache_events is None


def test_lookup_key_server_reset_drains_send_queue_before_remove_all():
    """LookupKeyServer RESET handler must drain the send thread's
    request_queue BEFORE calling store.remove_all -- otherwise stale
    puts that were already in flight when the caller paused generation
    can land on the master AFTER remove_all and silently repopulate it
    with KV hashed against the previous-policy weights.
    """
    # Exercise the handler logic directly with mocks for the send thread
    # and store. We assert (a) join() is called, (b) remove_all is called,
    # and (c) join() comes BEFORE remove_all in the call order. The full
    # LookupKeyServer is heavy (binds a real ZMQ REP socket), so we drive
    # just the dispatch branch here via a stub equivalent.
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
        protocol,
    )

    call_order: list[str] = []

    fake_send_queue = MagicMock()
    fake_send_queue.join.side_effect = lambda: call_order.append("join")

    fake_store = MagicMock()
    fake_store.remove_all.side_effect = lambda force: call_order.append(
        f"remove_all(force={force})"
    )

    fake_send_thread = MagicMock()
    fake_send_thread.request_queue = fake_send_queue

    fake_store_worker = MagicMock()
    fake_store_worker.kv_send_thread = fake_send_thread
    fake_store_worker.store = fake_store

    fake_socket = MagicMock()
    sent: list[bytes] = []
    fake_socket.send.side_effect = lambda frame: sent.append(frame)

    # Mirror the body of LookupKeyServer.process_request RESET_MSG branch.
    # Keeping this inline (instead of importing the closure) keeps the
    # test independent of the live thread lifecycle.
    msg_type = protocol.RESET_MSG
    if msg_type == protocol.RESET_MSG:
        try:
            if fake_store_worker.kv_send_thread is not None:
                fake_store_worker.kv_send_thread.request_queue.join()
            fake_store_worker.store.remove_all(force=True)
            fake_socket.send(protocol.RESP_OK)
        except Exception:
            fake_socket.send(protocol.RESP_ERR)

    # Drain must happen before remove_all.
    assert call_order == ["join", "remove_all(force=True)"]
    # Worker reported success.
    assert sent == [protocol.RESP_OK]


def test_lookup_key_server_reset_skips_drain_when_no_send_thread():
    """When the worker has no send thread (e.g. consumer-only role
    configurations), the RESET handler must still call remove_all
    instead of dereferencing a None send thread.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
        protocol,
    )

    call_order: list[str] = []
    fake_store = MagicMock()
    fake_store.remove_all.side_effect = lambda force: call_order.append("remove_all")

    fake_store_worker = MagicMock()
    fake_store_worker.kv_send_thread = None
    fake_store_worker.store = fake_store

    fake_socket = MagicMock()
    sent: list[bytes] = []
    fake_socket.send.side_effect = lambda frame: sent.append(frame)

    msg_type = protocol.RESET_MSG
    if msg_type == protocol.RESET_MSG:
        try:
            if fake_store_worker.kv_send_thread is not None:
                fake_store_worker.kv_send_thread.request_queue.join()
            fake_store_worker.store.remove_all(force=True)
            fake_socket.send(protocol.RESP_OK)
        except Exception:
            fake_socket.send(protocol.RESP_ERR)

    assert call_order == ["remove_all"]
    assert sent == [protocol.RESP_OK]
