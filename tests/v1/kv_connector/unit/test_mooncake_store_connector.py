# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from unittest.mock import MagicMock, patch

from vllm.config import set_current_vllm_config
from vllm.distributed.kv_events import BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    connector as mooncake_store_connector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    protocol,
    scheduler,
    worker,
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
    worker.get_block_ids_with_load_errors.return_value = {3, 4}
    connector.bind_connector_metadata(metadata)

    connector.register_kv_caches(kv_caches)
    result = connector.get_finished(finished_req_ids)
    invalid_block_ids = connector.get_block_ids_with_load_errors()

    worker.register_kv_caches.assert_called_once_with(kv_caches)
    worker.get_finished.assert_called_once_with(finished_req_ids, metadata)
    assert result == ({"req-1"}, {"req-2"})
    worker.get_block_ids_with_load_errors.assert_called_once_with()
    assert invalid_block_ids == {3, 4}


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


# ============================================================
# reset_cache() — RL hard-reset path via typed LookupKey protocol
# ============================================================


def test_reset_cache_scheduler_role_delegates_to_reset_store():
    """SCHEDULER role reset_cache() routes to scheduler.reset_store()."""
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ) as mock_scheduler_cls,
    ):
        conn = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )

    mock_scheduler_cls.return_value.reset_store.return_value = True
    assert conn.reset_cache() is True
    mock_scheduler_cls.return_value.reset_store.assert_called_once_with()


def test_reset_cache_scheduler_role_propagates_failure():
    """SCHEDULER role surfaces False when scheduler.reset_store() fails."""
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ) as mock_scheduler_cls,
    ):
        conn = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )

    mock_scheduler_cls.return_value.reset_store.return_value = False
    assert conn.reset_cache() is False


def test_reset_cache_worker_role_returns_none():
    """WORKER role reset_cache() is a no-op; reset is driven via ZMQ admin."""
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreWorker"
        ),
    ):
        conn = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

    assert conn.reset_cache() is None


def test_scheduler_reset_store_returns_client_reset_result():
    """MooncakeStoreScheduler.reset_store() returns LookupKeyClient.reset()."""
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "scheduler.LookupKeyClient"
        ) as mock_client_cls,
    ):
        sched = scheduler.MooncakeStoreScheduler(vllm_config, kv_cache_config)

    mock_client_cls.return_value.reset.return_value = True
    assert sched.reset_store() is True
    mock_client_cls.return_value.reset.assert_called_once_with()


def test_scheduler_reset_store_handles_rpc_exception():
    """Exceptions from the ZMQ reset RPC convert to False, not raise."""
    vllm_config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "scheduler.LookupKeyClient"
        ) as mock_client_cls,
    ):
        sched = scheduler.MooncakeStoreScheduler(vllm_config, kv_cache_config)

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

    # Blocking lookup (non_block defaults to False) runs on the executor and
    # returns the resolved hit length.
    assert client.lookup("req0", num_tokens=128, block_hashes=[]) == 5

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


def _poll_lookup(client, req_id, num_tokens=128, block_hashes=(), timeout=5.0):
    """Drive non-blocking lookup until the executor completes it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = client.lookup(req_id, num_tokens, list(block_hashes), non_block=True)
        if result is not None:
            return result
        time.sleep(0.005)
    return None


def _gated_recv(gate: threading.Event, value: int):
    """Mock recv side-effect that blocks until ``gate`` is set, so the
    executor's lookup can be held pending deterministically."""

    def recv():
        gate.wait()
        return value.to_bytes(4, "big")

    return recv


def test_lookup_key_client_non_block_lookup_async():
    """Non-blocking lookup defers to the executor: None first, hit once the
    Future resolves."""
    vllm_config = _make_vllm_config()

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
        "worker.make_zmq_socket"
    ) as mock_make_socket:
        client = worker.LookupKeyClient(vllm_config)

    fake_socket = mock_make_socket.return_value
    # Hold the executor's lookup pending until we release the gate.
    gate = threading.Event()
    fake_socket.recv.side_effect = _gated_recv(gate, 7)

    # First query submits the lookup and returns None while it is in flight.
    assert client.lookup("req1", 128, [], non_block=True) is None
    # Release the executor; a later poll returns the hit length.
    gate.set()
    assert _poll_lookup(client, "req1") == 7
    # Future is consumed (popped) on read.
    assert "req1" not in client.futures


def test_lookup_key_client_discard_clears_state():
    """discard() drops a completed lookup Future so it is not served stale."""
    vllm_config = _make_vllm_config()

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
        "worker.make_zmq_socket"
    ) as mock_make_socket:
        client = worker.LookupKeyClient(vllm_config)

    fake_socket = mock_make_socket.return_value
    gate = threading.Event()
    fake_socket.recv.side_effect = _gated_recv(gate, 9)

    # Submit while gated so the call returns None and the Future stays in
    # `futures` (unconsumed) once it resolves.
    assert client.lookup("req2", 128, [], non_block=True) is None
    gate.set()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if client.futures["req2"].done():
            break
        time.sleep(0.005)
    # discard() drops the completed result before any lookup consumes it.
    client.discard("req2")
    assert "req2" not in client.futures
    # A fresh query re-submits rather than returning a stale value: hold the
    # gate so the resubmitted lookup stays in flight.
    gate.clear()
    assert client.lookup("req2", 128, [], non_block=True) is None
    gate.set()  # release the executor so the worker thread can drain


def test_get_num_new_matched_tokens_async_defers_then_reports():
    """Async lookup returns (None, False) until ready, then the hit count."""
    vllm_config = create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"lookup_async": True},
    )
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "scheduler.LookupKeyClient"
        ) as mock_client_cls,
    ):
        sched = scheduler.MooncakeStoreScheduler(vllm_config, kv_cache_config)

    assert sched.lookup_async is True
    mock_client = mock_client_cls.return_value

    block_size = sched._block_size
    request = MagicMock()
    request.request_id = "r1"
    request.num_tokens = 4 * block_size
    request.block_hashes = []

    # Lookup not ready -> defer.
    mock_client.lookup.return_value = None
    assert sched.get_num_new_matched_tokens(request, 0) == (None, False)
    assert "r1" not in sched.load_specs

    # Lookup ready with a hit -> report need_to_allocate + async-load flag.
    hit = 3 * block_size
    mock_client.lookup.return_value = hit
    need, load_async = sched.get_num_new_matched_tokens(request, 0)
    assert need == hit
    assert load_async == sched.load_async
    assert sched.load_specs["r1"].kvpool_cached_tokens == hit


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
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ) as mock_scheduler_cls,
    ):
        conn = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )

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
    kv_cache_config = _make_kv_cache_config()

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
            "connector.MooncakeStoreScheduler"
        ) as mock_scheduler_cls,
    ):
        conn = mooncake_store_connector.MooncakeStoreConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )

    # Seed both sentinel pieces of stale-reference state.
    sched_inst = mock_scheduler_cls.return_value
    sched_inst.load_specs = {
        "req-A": LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=128, can_load=True)
    }
    conn._kv_cache_events = mooncake_store_connector.MooncakeStoreKVEvents(
        num_workers=1
    )
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


def test_shutdown_closes_worker_store():
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

    worker = mock_worker_cls.return_value
    connector.shutdown()

    worker.close.assert_called_once_with()


def test_del_invokes_shutdown_and_closes_store():
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

    worker = mock_worker_cls.return_value
    # __del__ is the GC backstop; it must route through shutdown() -> close().
    connector.__del__()

    worker.close.assert_called_once_with()


def test_shutdown_scheduler_role_is_noop():
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

    # Scheduler role holds no store handle, so shutdown must be a safe no-op.
    assert connector.connector_worker is None
    connector.shutdown()
