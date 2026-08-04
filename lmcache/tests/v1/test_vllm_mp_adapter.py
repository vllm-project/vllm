# SPDX-License-Identifier: Apache-2.0
"""Public-API unit tests for ``LMCacheMPWorkerAdapter``. The MQ boundary is
stubbed (see ``fake_adapter``); no GPU or live server needed. End-to-end
recovery: ``.buildkite/k3_tests/multiprocess/scripts/run-restart-recovery.sh``."""

# Standard
from typing import Callable, ClassVar
from unittest.mock import MagicMock
import gc
import os
import threading
import time
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_mod
from lmcache.integration.vllm.experimental.dispatcher import Dispatcher
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    HeartbeatThread,
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
    ParallelStrategy,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocol import RequestType


class FakeCudaEvent:
    def ipc_handle(self) -> bytes:
        return b"fake-ipc-handle"


class FakeHeartbeatThread:
    """Test double mirroring ``HeartbeatThread``'s public surface.
    ``start()`` invokes class-level ``start_hook`` when set, otherwise
    simulates a successful first ping. Class state reset per test."""

    instances: ClassVar[list["FakeHeartbeatThread"]] = []
    start_hook: ClassVar[Callable[["FakeHeartbeatThread"], None] | None] = None

    def __init__(
        self,
        mq_client: object = None,
        health_event: threading.Event | None = None,
        interval: float = 0.0,
        instance_id: int | None = None,
    ) -> None:
        self.mq_client = mq_client
        self.health_event = (
            health_event if health_event is not None else threading.Event()
        )
        self.interval = interval
        self.instance_id = instance_id
        # Snapshot of the health event at construction time: lets tests
        # assert the adapter starts the heartbeat healthy (event still set).
        self.health_event_set_at_init = self.health_event.is_set()
        self.recover_callback: Callable[[], bool] | None = None
        # Ordered record of public calls ("register_recover_callback",
        # "start", "stop") for call-order assertions.
        self.calls: list[str] = []
        self.stop_requested = False
        FakeHeartbeatThread.instances.append(self)

    def register_recover_callback(self, callback: Callable[[], bool]) -> None:
        self.calls.append("register_recover_callback")
        self.recover_callback = callback

    def start(self) -> None:
        self.calls.append("start")
        hook = FakeHeartbeatThread.start_hook
        if hook is not None:
            hook(self)
        else:
            self.simulate_successful_ping()

    def stop(self, timeout: float = 5.0) -> None:
        self.calls.append("stop")
        self.stop_requested = True

    def simulate_successful_ping(self) -> None:
        """Mimic one successful heartbeat cycle: on the unhealthy->healthy
        edge the recover callback runs first, and the event is set only
        when the callback returns ``True``."""
        was_healthy = self.health_event.is_set()
        ok = True
        if not was_healthy and self.recover_callback is not None:
            ok = self.recover_callback()
        if ok:
            self.health_event.set()


def _make_worker_adapter(
    extra_config: dict[str, object] | None = None,
) -> LMCacheMPWorkerAdapter:
    """Construct a worker adapter with the standard test arguments; the
    network boundary must already be patched (see ``fake_adapter``).
    ``extra_config`` forwards ``lmcache.mp.*`` overrides."""
    parallel_strategy = ParallelStrategy(
        mla_only=False,
        vllm_world_size=1,
        vllm_worker_id=0,
        tp_size=1,
        pp_size=1,
        n_servers=1,
    )
    return LMCacheMPWorkerAdapter(
        server_url="tcp://127.0.0.1:0",
        context=MagicMock(name="zmq_context"),
        model_name="test-model",
        vllm_block_size=16,
        parallel_strategy=parallel_strategy,
        mq_timeout=5.0,
        extra_config=extra_config,
    )


def _op(block_ids: list[list[int]]) -> LoadStoreOp:
    """Build a minimal four-token ``LoadStoreOp`` over *block_ids*."""
    return LoadStoreOp(token_ids=[1, 2, 3, 4], block_ids=block_ids, start=0, end=4)


def _patch_transfer_context_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> list[MagicMock]:
    """Patch ``create_transfer_context`` to mint recorded MagicMocks,
    returning the list every created context is appended to."""
    contexts: list[MagicMock] = []

    def fake_create_transfer_context(
        kv_caches: dict[str, torch.Tensor], mode: str
    ) -> MagicMock:
        ctx = MagicMock(name=f"transfer_ctx_{len(contexts)}")
        contexts.append(ctx)
        return ctx

    monkeypatch.setattr(
        adapter_mod, "create_transfer_context", fake_create_transfer_context
    )
    return contexts


@pytest.fixture
def fake_adapter(monkeypatch):
    """Build an adapter with the network boundary stubbed. Returns
    ``(adapter, send_mock, future)``; ``future.result()`` defaults to succeed.
    ``HeartbeatThread`` is replaced by ``FakeHeartbeatThread``."""
    # Stub the MQ boundary so __init__'s chunk-size query and any later
    # send_lmcache_request call don't touch a real socket.
    fake_client = MagicMock(name="mq_client")
    monkeypatch.setattr(adapter_mod, "MessageQueueClient", lambda *a, **kw: fake_client)
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", lambda *a, **kw: 256)
    monkeypatch.setattr(adapter_mod, "get_experimental", lambda *a, **kw: set())

    future = MagicMock(name="future")
    future.result.return_value = None
    send_mock = MagicMock(name="send_lmcache_request", return_value=future)
    monkeypatch.setattr(adapter_mod, "send_lmcache_request", send_mock)

    FakeHeartbeatThread.instances.clear()
    FakeHeartbeatThread.start_hook = None
    monkeypatch.setattr(adapter_mod, "HeartbeatThread", FakeHeartbeatThread)

    # KV-cache wrapping pulls in CUDA IPC; bypass for unit tests.
    # First Party
    from lmcache.v1.multiprocess.transfer_context import worker_transfer

    monkeypatch.setattr(
        worker_transfer,
        "wrap_kv_caches",
        lambda kv: list(kv.values()),
    )
    # ``vllm_layout_hints`` returns a ``LayoutHints`` (TypedDict / dict at
    # runtime); stub it with an empty dict.
    monkeypatch.setattr(
        "lmcache.integration.vllm.utils.vllm_layout_hints",
        lambda: {},
    )

    adapter = _make_worker_adapter()
    # __init__ issues exactly one MQ call (the chunk-size query). Reset
    # so individual tests start with a clean call count.
    send_mock.reset_mock()
    return adapter, send_mock, future


def test_register_kv_caches_updates_kv_caches_and_submits(fake_adapter):
    """Public register_kv_caches stores the dict and submits one request."""
    adapter, send_mock, _ = fake_adapter
    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    new_caches = {"layer.0": fake_tensor, "layer.1": fake_tensor}

    adapter.register_kv_caches(new_caches)

    assert adapter.kv_caches is new_caches
    assert send_mock.call_count == 1
    args, _kwargs = send_mock.call_args
    assert args[1] == RequestType.REGISTER_KV_CACHE


def test_register_kv_caches_raises_connection_error_on_timeout(fake_adapter):
    """Public register_kv_caches surfaces ConnectionError on MQ timeout."""
    adapter, _send_mock, future = fake_adapter
    future.result.side_effect = TimeoutError("server down")

    with pytest.raises(ConnectionError, match="did not respond"):
        fake_tensor = MagicMock()
        fake_tensor.device.type = "cuda"
        adapter.register_kv_caches({"layer.0": fake_tensor})


def test_register_kv_caches_cpu_submits_engine_driven_context_registration(
    fake_adapter, monkeypatch
):
    """CPU KV cache registration routes to REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT."""
    adapter, send_mock, _ = fake_adapter
    monkeypatch.setattr(
        "lmcache.integration.vllm.utils.vllm_layout_hints",
        lambda: {},
        raising=False,
    )
    cpu_kv = {"layer.0": torch.randn(2, 8, 4, 2, 8)}

    adapter.register_kv_caches(cpu_kv)

    assert adapter.kv_caches is cpu_kv
    assert send_mock.call_count == 1
    args, _kwargs = send_mock.call_args
    assert args[1] == RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT
    assert len(args[2]) == 1


def test_submit_store_request_tracks_returned_future(fake_adapter, monkeypatch):
    """submit_store_request stores the returned future in store_futures."""
    adapter, _send_mock, _ = fake_adapter
    monkeypatch.setattr(adapter, "_ensure_heartbeat_started", lambda: None)
    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    adapter.kv_caches = {"layer.0": fake_tensor}
    transfer_ctx = MagicMock()
    fake_future = MagicMock()
    transfer_ctx.submit_store.return_value = fake_future
    adapter.transfer_ctx = transfer_ctx
    op = LoadStoreOp(token_ids=[1, 2, 3, 4], block_ids=[[0]], start=0, end=4)

    adapter.submit_store_request("req-1", op, event=MagicMock())

    assert transfer_ctx.submit_store.called
    assert transfer_ctx.submit_store.call_args.kwargs == {}
    assert transfer_ctx.submit_store.call_args.args[4] == [[0]]
    assert adapter.store_futures["req-1"] is fake_future


def test_submit_store_request_expands_block_ids_to_views(fake_adapter, monkeypatch):
    adapter, _send_mock, _ = fake_adapter
    monkeypatch.setattr(adapter, "_ensure_heartbeat_started", lambda: None)
    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    adapter.kv_caches = {"layer.0": fake_tensor}
    adapter.engine_group_infos = [
        EngineGroupInfo(0, (0, 2)),
        EngineGroupInfo(0, (4,)),
        EngineGroupInfo(1, (1, 3)),
    ]
    transfer_ctx = MagicMock()
    fake_future = MagicMock()
    transfer_ctx.submit_store.return_value = fake_future
    adapter.transfer_ctx = transfer_ctx
    op = LoadStoreOp(
        token_ids=[1, 2, 3, 4],
        block_ids=[[0, 1], [10, 11]],
        start=0,
        end=4,
    )

    adapter.submit_store_request("req-1", op, event=MagicMock())

    assert transfer_ctx.submit_store.call_args.args[4] == [
        [0, 1],
        [0, 1],
        [10, 11],
    ]


def test_submit_retrieve_request_tracks_returned_future(fake_adapter, monkeypatch):
    """submit_retrieve_request stores returned future and block IDs."""
    adapter, _send_mock, _ = fake_adapter
    monkeypatch.setattr(adapter, "_ensure_heartbeat_started", lambda: None)
    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    adapter.kv_caches = {"layer.0": fake_tensor}
    transfer_ctx = MagicMock()
    fake_future = MagicMock()
    transfer_ctx.submit_retrieve.return_value = fake_future
    adapter.transfer_ctx = transfer_ctx
    op = LoadStoreOp(
        token_ids=[1, 2, 3, 4],
        block_ids=[[0]],
        start=0,
        end=4,
        skip_first_n_tokens=1,
    )

    adapter.submit_retrieve_request("req-1", op, event=MagicMock())

    assert transfer_ctx.submit_retrieve.called
    assert transfer_ctx.submit_retrieve.call_args.kwargs == {"skip_first_n_tokens": 1}
    assert transfer_ctx.submit_retrieve.call_args.args[4] == [[0]]
    assert adapter.retrieve_futures["req-1"] == (fake_future, [0])


def test_load_store_op_accepts_per_group_block_ids():
    op = LoadStoreOp(
        token_ids=[1, 2, 3, 4],
        block_ids=[[0, 1], [10, 11]],
        start=0,
        end=4,
    )

    assert op.block_ids == [[0, 1], [10, 11]]
    assert op.flat_block_ids == [0, 1, 10, 11]


def test_store_keeps_event_until_future_finishes(fake_adapter):
    """Store requests keep the exported CUDA event alive while pending."""
    adapter, _send_mock, _future = fake_adapter
    cuda_future = MagicMock(name="cuda_future")
    cuda_future.query.return_value = False
    transfer_ctx = MagicMock()
    transfer_ctx.submit_store.return_value = cuda_future
    adapter.transfer_ctx = transfer_ctx

    event = FakeCudaEvent()
    event_ref = weakref.ref(event)
    op = LoadStoreOp(token_ids=[1, 2], block_ids=[[7]], start=0, end=2)

    adapter.submit_store_request("req-1", op, event)
    del event
    gc.collect()
    assert event_ref() is not None

    cuda_future.query.return_value = True
    cuda_future.result.return_value = True
    finished_stores, finished_retrieves = adapter.get_finished({"req-1"})

    assert finished_stores == {"req-1"}
    assert finished_retrieves == set()
    assert "req-1" not in adapter.store_events
    transfer_ctx.reset_mock()
    gc.collect()
    assert event_ref() is None


def test_retrieve_keeps_event_until_future_finishes(fake_adapter):
    """Retrieve requests keep the exported CUDA event alive while pending."""
    adapter, _send_mock, _future = fake_adapter
    cuda_future = MagicMock(name="cuda_future")
    cuda_future.query.return_value = False
    transfer_ctx = MagicMock()
    transfer_ctx.submit_retrieve.return_value = cuda_future
    adapter.transfer_ctx = transfer_ctx

    event = FakeCudaEvent()
    event_ref = weakref.ref(event)
    op = LoadStoreOp(token_ids=[1, 2], block_ids=[[7]], start=0, end=2)

    adapter.submit_retrieve_request("req-1", op, event)
    del event
    gc.collect()
    assert event_ref() is not None

    cuda_future.query.return_value = True
    cuda_future.result.return_value = True
    finished_stores, finished_retrieves = adapter.get_finished(set())

    assert finished_stores == set()
    assert finished_retrieves == {"req-1"}
    assert "req-1" not in adapter.retrieve_events
    transfer_ctx.reset_mock()
    gc.collect()
    assert event_ref() is None


def test_instance_id_is_uuid_derived_63_bit_int(fake_adapter) -> None:
    """instance_id is a 63-bit int, not the PID, and unique per adapter."""
    adapter, _send_mock, _ = fake_adapter

    assert isinstance(adapter.instance_id, int)
    assert not isinstance(adapter.instance_id, bool)
    assert 0 <= adapter.instance_id < 2**63
    assert adapter.instance_id != os.getpid()

    other = _make_worker_adapter()
    assert other.instance_id != adapter.instance_id


def test_instance_id_logged_at_info_on_construction(fake_adapter, monkeypatch) -> None:
    """The constructor logs instance_id at INFO for correlating server-side
    reap warnings. The module logger does not propagate (``propagate=False``),
    so the test spies on it directly instead of using ``caplog``."""
    _adapter, _send_mock, _ = fake_adapter
    messages: list[str] = []

    def spy_info(msg: object, *args: object, **kwargs: object) -> None:
        messages.append(str(msg) % args if args else str(msg))

    monkeypatch.setattr(adapter_mod.logger, "info", spy_info)

    adapter = _make_worker_adapter()

    assert any(str(adapter.instance_id) in msg for msg in messages)


def test_heartbeat_lazy_start_wires_callback_before_start(fake_adapter) -> None:
    """The lazy create path starts the heartbeat healthy (no pessimistic
    clear) and wires the recover callback before ``start()``; the first
    store is not gated. Idempotent on re-entry (no second thread)."""
    adapter, _send_mock, _ = fake_adapter
    adapter.transfer_ctx = MagicMock()
    assert adapter.is_healthy  # the constructor leaves the event set

    adapter.submit_store_request("req-1", _op([[0]]), MagicMock())

    assert len(FakeHeartbeatThread.instances) == 1
    heartbeat = FakeHeartbeatThread.instances[0]
    # Started healthy: the event was NOT cleared before construction, so
    # the first store is not dropped.
    assert heartbeat.health_event_set_at_init is True
    # The recover callback is wired before start() (for genuine recovery).
    assert heartbeat.calls == ["register_recover_callback", "start"]
    assert adapter.is_healthy
    assert adapter.transfer_ctx.submit_store.call_count == 1

    # Re-entry is idempotent: no new thread.
    adapter.submit_store_request("req-2", _op([[1]]), MagicMock())
    assert len(FakeHeartbeatThread.instances) == 1
    assert adapter.transfer_ctx.submit_store.call_count == 2


def test_heartbeat_first_ping_runs_callback_before_setting_event(
    monkeypatch,
) -> None:
    """Real HeartbeatThread: started with the health event cleared, the
    first successful ping invokes the recover callback while the event
    is still cleared, then sets the event."""
    monkeypatch.setattr(
        adapter_mod, "send_ping", lambda mq_client, timeout, instance_id=None: True
    )
    health_event = threading.Event()  # cleared: pessimistic start state
    heartbeat = HeartbeatThread(
        mq_client=MagicMock(name="mq_client"),
        health_event=health_event,
        interval=60.0,
    )
    event_state_during_callback: list[bool] = []

    def recover() -> bool:
        event_state_during_callback.append(health_event.is_set())
        return True

    heartbeat.register_recover_callback(recover)
    try:
        heartbeat.start()
        assert health_event.wait(timeout=10.0)
    finally:
        heartbeat.stop(timeout=10.0)

    assert event_state_during_callback == [False]


def test_dropped_retrieve_reported_once_via_unhealthy_get_finished(
    fake_adapter,
) -> None:
    """A retrieve submitted while unhealthy is dropped (blocks flagged,
    nothing sent) and reported exactly once by the unhealthy branch of
    ``get_finished``."""
    adapter, _send_mock, _ = fake_adapter
    transfer_ctx = MagicMock()
    adapter.transfer_ctx = transfer_ctx
    # Simulate a failed first ping: the heartbeat start clears the event.
    FakeHeartbeatThread.start_hook = lambda hb: hb.health_event.clear()

    adapter.submit_retrieve_request("req-1", _op([[3, 4]]), MagicMock())

    assert not adapter.is_healthy
    assert not transfer_ctx.submit_retrieve.called

    ret_stores, finished_retrieves = adapter.get_finished(set())
    assert ret_stores == set()
    assert finished_retrieves == {"req-1"}
    assert adapter.get_block_ids_with_load_errors() == {3, 4}

    # Exactly once: a second poll must not re-report the request.
    _ret_stores, finished_retrieves = adapter.get_finished(set())
    assert finished_retrieves == set()


def test_dropped_retrieve_reported_once_via_healthy_get_finished(
    fake_adapter,
) -> None:
    """A retrieve dropped while unhealthy is still reported exactly once
    by the healthy branch of ``get_finished`` after the server
    recovers."""
    adapter, _send_mock, _ = fake_adapter
    adapter.transfer_ctx = MagicMock()
    # Simulate a failed first ping: the heartbeat start clears the event.
    FakeHeartbeatThread.start_hook = lambda hb: hb.health_event.clear()

    adapter.submit_retrieve_request("req-1", _op([[5]]), MagicMock())
    assert not adapter.is_healthy

    # Server recovers: the next heartbeat cycle takes the edge.
    FakeHeartbeatThread.instances[0].simulate_successful_ping()
    assert adapter.is_healthy

    _ret_stores, finished_retrieves = adapter.get_finished(set())
    assert finished_retrieves == {"req-1"}
    assert adapter.get_block_ids_with_load_errors() == {5}

    _ret_stores, finished_retrieves = adapter.get_finished(set())
    assert finished_retrieves == set()


def test_shutdown_stops_heartbeat_before_unregister(fake_adapter) -> None:
    """shutdown() stops the heartbeat before sending UNREGISTER, so no
    stray heartbeat ping can race the closing mq_client."""
    adapter, send_mock, future = fake_adapter
    adapter.transfer_ctx = MagicMock()
    adapter.submit_store_request("req-1", _op([[0]]), MagicMock())
    heartbeat = FakeHeartbeatThread.instances[0]

    stop_state_at_unregister: list[bool] = []

    def record_send(
        mq_client: object, request_type: RequestType, payloads: list[object]
    ) -> MagicMock:
        if request_type == RequestType.UNREGISTER_KV_CACHE:
            stop_state_at_unregister.append(heartbeat.stop_requested)
        return future

    send_mock.side_effect = record_send

    adapter.shutdown()

    assert "stop" in heartbeat.calls
    assert stop_state_at_unregister == [True]


def test_shutdown_without_heartbeat_sends_unregister(fake_adapter) -> None:
    """shutdown() on an adapter whose heartbeat was never lazily started
    (cold shutdown before any traffic) still sends UNREGISTER and does
    not raise."""
    adapter, send_mock, _future = fake_adapter

    adapter.shutdown()

    assert FakeHeartbeatThread.instances == []
    assert send_mock.call_count == 1
    args, _kwargs = send_mock.call_args
    assert args[1] == RequestType.UNREGISTER_KV_CACHE
    assert args[2] == [adapter.instance_id]


def test_straggler_cycle_after_stop_skips_callback_and_event(monkeypatch) -> None:
    """Real HeartbeatThread: a ping still in flight when ``stop()`` returns
    completes without firing the recover callback or setting the health
    event — a straggler success must not re-register a ghost context."""
    ping_entered = threading.Event()
    release_ping = threading.Event()

    def slow_ping(
        mq_client: object, timeout: float, instance_id: int | None = None
    ) -> bool:
        ping_entered.set()
        release_ping.wait(timeout=10.0)
        return True

    monkeypatch.setattr(adapter_mod, "send_ping", slow_ping)
    health_event = threading.Event()  # cleared: a success would take the edge
    heartbeat = HeartbeatThread(
        mq_client=MagicMock(name="mq_client"),
        health_event=health_event,
        interval=60.0,
    )
    callback = MagicMock(name="recover_callback", return_value=True)
    heartbeat.register_recover_callback(callback)

    heartbeat.start()
    assert ping_entered.wait(timeout=10.0)
    # The join times out while the ping is still in flight.
    heartbeat.stop(timeout=0.05)
    release_ping.set()

    # Wait for the straggler cycle to complete.
    deadline = time.time() + 10.0
    while heartbeat.total_runs == 0 and time.time() < deadline:
        time.sleep(0.01)

    assert heartbeat.total_runs == 1
    callback.assert_not_called()
    assert not health_event.is_set()


def test_recover_callback_skips_register_after_stop_requested(
    fake_adapter, monkeypatch
) -> None:
    """A recover callback that observes a requested stop bails out before
    submitting REGISTER: a REGISTER submitted after UNREGISTER would
    re-create a ghost server-side context."""
    adapter, _send_mock, _ = fake_adapter
    contexts = _patch_transfer_context_factory(monkeypatch)

    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    adapter.register_kv_caches({"layer.0": fake_tensor})
    adapter.submit_store_request("req-1", _op([[0]]), MagicMock())
    heartbeat = FakeHeartbeatThread.instances[0]
    assert heartbeat.recover_callback is not None
    rebuilds_before = len(contexts)

    # Simulate a stop landing while a recovery cycle is in flight: the
    # pre-submission re-check must refuse to re-register.
    heartbeat.stop()
    assert heartbeat.recover_callback() is False

    assert len(contexts) == rebuilds_before  # no new transfer context
    assert contexts[-1].register.call_count == 1  # no second REGISTER


def test_register_uses_local_context_when_self_transfer_ctx_nulled(
    monkeypatch,
) -> None:
    """register must call register() on the local context, not re-read
    self.transfer_ctx: a concurrent shutdown() can null the attribute
    between publish and the call, which previously raised AttributeError."""

    class _NullingTransferCtxAdapter(LMCacheMPWorkerAdapter):
        # Models self.transfer_ctx already nulled by a racing shutdown():
        # the getter always reports None, so any code re-reading the
        # attribute (rather than the local) hits None.register.
        @property
        def transfer_ctx(self):
            return None

        @transfer_ctx.setter
        def transfer_ctx(self, value):
            pass

    fake_client = MagicMock(name="mq_client")
    monkeypatch.setattr(adapter_mod, "MessageQueueClient", lambda *a, **kw: fake_client)
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", lambda *a, **kw: 256)
    monkeypatch.setattr(adapter_mod, "get_experimental", lambda *a, **kw: set())
    future = MagicMock(name="future")
    future.result.return_value = None
    monkeypatch.setattr(adapter_mod, "send_lmcache_request", lambda *a, **kw: future)
    monkeypatch.setattr(adapter_mod, "HeartbeatThread", FakeHeartbeatThread)
    # First Party
    from lmcache.v1.multiprocess.transfer_context import worker_transfer

    monkeypatch.setattr(
        worker_transfer,
        "wrap_kv_caches",
        lambda kv: list(kv.values()),
    )
    monkeypatch.setattr("lmcache.integration.vllm.utils.vllm_layout_hints", lambda: {})
    local_ctx = MagicMock(name="local_transfer_ctx")
    monkeypatch.setattr(
        adapter_mod, "create_transfer_context", lambda kv, mode: local_ctx
    )

    parallel_strategy = ParallelStrategy(
        mla_only=False,
        vllm_world_size=1,
        vllm_worker_id=0,
        tp_size=1,
        pp_size=1,
        n_servers=1,
    )
    adapter = _NullingTransferCtxAdapter(
        server_url="tcp://127.0.0.1:0",
        context=MagicMock(name="zmq_context"),
        model_name="test-model",
        vllm_block_size=16,
        parallel_strategy=parallel_strategy,
        mq_timeout=5.0,
    )

    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    # Under the bug this raises AttributeError (None.register).
    adapter.register_kv_caches({"layer.0": fake_tensor})

    local_ctx.register.assert_called_once()


def test_startup_warns_when_heartbeat_interval_exceeds_reap_floor(
    fake_adapter, monkeypatch
) -> None:
    """3 x heartbeat_interval > 30 s emits a startup WARNING to raise the
    server's worker reap timeout. The module logger does not propagate
    (``propagate=False``), so the test spies on it instead of ``caplog``."""
    _adapter, _send_mock, _ = fake_adapter
    warnings: list[str] = []
    monkeypatch.setattr(
        adapter_mod.logger,
        "warning",
        lambda msg, *args, **kwargs: warnings.append(str(msg)),
    )

    _make_worker_adapter(extra_config={"lmcache.mp.heartbeat_interval": 15})

    assert any("reap" in msg for msg in warnings)


def test_startup_does_not_warn_for_default_heartbeat_interval(
    fake_adapter, monkeypatch
) -> None:
    """The default 10 s heartbeat interval (3 x 10 s == 30 s floor) must
    not emit the reap-timeout startup WARNING."""
    _adapter, _send_mock, _ = fake_adapter
    warnings: list[str] = []
    monkeypatch.setattr(
        adapter_mod.logger,
        "warning",
        lambda msg, *args, **kwargs: warnings.append(str(msg)),
    )

    _make_worker_adapter()

    assert not any("reap" in msg for msg in warnings)


def test_recover_callback_rebuilds_transfer_ctx_without_closing_previous(
    fake_adapter, monkeypatch
) -> None:
    """Pin current behavior: every recover-callback invocation rebuilds
    ``transfer_ctx`` without closing the previous context (known IPC leak;
    in-flight submissions may still hold a reference to the old context)."""
    adapter, _send_mock, _ = fake_adapter
    contexts = _patch_transfer_context_factory(monkeypatch)

    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    adapter.register_kv_caches({"layer.0": fake_tensor})  # contexts[0]
    # Start the heartbeat (healthy, no recover) so the callback is wired.
    adapter.submit_store_request("req-1", _op([[0]]), MagicMock())
    heartbeat = FakeHeartbeatThread.instances[0]
    assert heartbeat.recover_callback is not None
    assert len(contexts) == 1
    assert adapter.transfer_ctx is contexts[0]

    # Each recover-callback invocation rebuilds transfer_ctx without closing
    # the previous context (known IPC leak; in-flight submissions may still
    # hold a reference to the old context).
    assert heartbeat.recover_callback() is True
    assert len(contexts) == 2
    assert adapter.transfer_ctx is contexts[1]
    contexts[0].close.assert_not_called()

    assert heartbeat.recover_callback() is True
    assert len(contexts) == 3
    assert adapter.transfer_ctx is contexts[2]
    contexts[1].close.assert_not_called()


# For the experimental dispatcher
def test_enabled_feature_receives_reclaim_and_shutdown(fake_adapter):
    """get_finished reclaims ring blocks and shutdown unregisters the ring."""
    adapter, _, _ = fake_adapter
    assert adapter.dispatcher is None

    dispatcher = MagicMock(spec=Dispatcher)
    adapter.dispatcher = dispatcher

    adapter.get_finished(set())
    adapter.shutdown()

    dispatcher.reclaim.assert_called_once_with()
    dispatcher.shutdown.assert_called_once_with()


@pytest.mark.parametrize("ring_ok", [True, False])
def test_recovery_reports_the_ring_re_registration_result(fake_adapter, ring_ok):
    """A worker whose Q ring failed to re-register must not be reported
    healthy: the server would have no Q context, so every later STORE_Q would
    raise. The success path must still report healthy so the health event can
    be set."""
    adapter, _, _ = fake_adapter
    dispatcher = MagicMock(spec=Dispatcher)
    dispatcher.reregister.return_value = ring_ok
    adapter.dispatcher = dispatcher
    fake_tensor = MagicMock()
    fake_tensor.device.type = "cuda"
    adapter.register_kv_caches({"layer.0": fake_tensor})

    assert adapter._reregister_kv_caches_callback() is ring_ok
