# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import queue
import threading
from collections import defaultdict
from types import SimpleNamespace

import pytest

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    LayerTransferPlan,
    MoRIIOConfig,
    MoRIIOConnectorMetadata,
    MoRIIOMode,
    ReqMeta,
    get_moriio_mode,
    get_port_offset,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    _MAX_PENDING_UNMAPPED_DONE_TIDS,
    MoRIIOConnector,
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
    MoRIIOWriter,
)


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


class FakeStatus:
    def __init__(
        self,
        *,
        succeeded: bool = False,
        failed: bool = False,
        succeed_after: int = 0,
    ) -> None:
        self.succeeded = succeeded
        self.failed = failed
        self.succeed_after = succeed_after
        self.succeeded_calls = 0

    def Succeeded(self) -> bool:
        self.succeeded_calls += 1
        if self.succeed_after and self.succeeded_calls >= self.succeed_after:
            self.succeeded = True
        return self.succeeded

    def Failed(self) -> bool:
        return self.failed

    def Message(self) -> str:
        return "fake failure"

    def Code(self) -> int:
        return 123


class FakeWrapper:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.notifies: list[tuple[str, str, str | int]] = []
        self.done_req_ids: list[str] = []
        self.done_remote_allocate_req_dict: dict[str, object] = {}
        self.read_error: Exception | None = None
        self.read_results: list[FakeStatus | Exception] = []
        self.notify_error: Exception | None = None
        self.waited_for_transfer = False

    def send_notify(self, transfer_id: str, host: str, port: str) -> None:
        if self.notify_error is not None:
            error = self.notify_error
            self.notify_error = None
            raise error
        self.notifies.append((transfer_id, host, port))

    def async_wait_reqid(self) -> None:
        pass

    def pop_finished_req_ids(self) -> set[str]:
        return set()

    def read_remote_data(self, *_args):
        if self.read_results:
            result = self.read_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        if self.read_error is not None:
            raise self.read_error
        return FakeStatus()

    def waiting_for_transfer_complete(self) -> None:
        self.waited_for_transfer = True

    def shutdown(self) -> None:
        pass


class FakeTensor:
    def numel(self) -> int:
        return 1

    def element_size(self) -> int:
        return 1


def make_worker() -> MoRIIOConnectorWorker:
    worker = object.__new__(MoRIIOConnectorWorker)
    worker.is_producer = False
    worker.mode = MoRIIOMode.READ
    worker.moriio_config = SimpleNamespace(
        transfer_timeout=1.0,
        defer_timeout=1.0,
        max_inflight_global=0,
        max_inflight_per_transfer=0,
        max_dispatch_layers=0,
    )
    worker.moriio_wrapper = FakeWrapper()
    worker._recving_transfers = defaultdict(dict)
    worker._pending_read_plans = defaultdict(dict)
    worker._recving_transfers_callback_addr = {}
    worker._recving_transfer_local_block_ids = {}
    worker._invalid_block_ids = queue.Queue()
    worker._reqs_to_send = {}
    worker.tp_rank = 0
    worker.tp_size = 1
    worker._pending_unmapped_done_tids = {}
    worker._unmatched_write_completions = set()
    worker.transfer_id_to_request_id = {}
    worker.request_id_to_transfer_id = {}
    worker.transfer_id_to_remote_tp_size = {}
    worker.transfer_id_to_completion_count = {}
    worker._read_completion_ids = defaultdict(set)
    return worker


def test_connector_wait_for_layer_load_forwards_only_for_read_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.moriio import (
        moriio_connector as connector_module,
    )

    connector = object.__new__(MoRIIOConnector)
    calls: list[str] = []
    connector.connector_worker = SimpleNamespace(
        wait_for_layer_load=lambda layer_name: calls.append(layer_name)
    )

    connector.mode = MoRIIOMode.READ
    monkeypatch.setattr(connector_module, "get_role", lambda: ROLE.CONSUMER)
    connector.wait_for_layer_load("layer0")
    assert calls == ["layer0"]

    connector.mode = MoRIIOMode.WRITE
    connector.wait_for_layer_load("layer1")
    assert calls == ["layer0"]


def test_pending_unmapped_done_tids_evicts_oldest_at_cap() -> None:
    worker = make_worker()

    for idx in range(_MAX_PENDING_UNMAPPED_DONE_TIDS):
        worker._buffer_pending_unmapped_done_tid(f"missing-{idx}")

    worker._buffer_pending_unmapped_done_tid("new")

    assert len(worker._pending_unmapped_done_tids) == (_MAX_PENDING_UNMAPPED_DONE_TIDS)
    assert "missing-0" not in worker._pending_unmapped_done_tids
    assert "missing-1" in worker._pending_unmapped_done_tids
    assert "new" in worker._pending_unmapped_done_tids


def test_freed_transfer_ids_drop_pending_unmapped_done_tids() -> None:
    worker = make_worker()
    worker.is_producer = True
    transfer_id = "tx-00000000-0000-0000-0000-000000000001"
    request_id = "req0-deadbeef"
    wrapped_transfer_id = f"wrapped-{transfer_id}-suffix"
    worker.transfer_id_to_request_id[transfer_id] = request_id
    worker.request_id_to_transfer_id[request_id] = transfer_id
    worker._reqs_to_send[request_id] = 1.0
    worker._pending_unmapped_done_tids = {
        transfer_id: None,
        wrapped_transfer_id: None,
        request_id: None,
        "other": None,
    }
    metadata = SimpleNamespace(
        freed_transfer_ids={transfer_id},
        transfer_id_to_request_id={},
    )

    worker.start_load_kv(metadata)

    assert transfer_id not in worker._pending_unmapped_done_tids
    assert wrapped_transfer_id not in worker._pending_unmapped_done_tids
    assert request_id not in worker._pending_unmapped_done_tids
    assert "other" in worker._pending_unmapped_done_tids
    assert transfer_id not in worker.transfer_id_to_request_id
    assert request_id not in worker._reqs_to_send


def test_freed_transfer_ids_filter_stale_worker_metadata_deltas() -> None:
    worker = make_worker()
    worker.is_producer = True
    transfer_id = "tx-00000000-0000-0000-0000-000000000001"
    request_id = "req0-deadbeef"
    metadata = MoRIIOConnectorMetadata()
    metadata.freed_transfer_ids.add(transfer_id)
    metadata.transfer_id_to_request_id[transfer_id] = request_id
    metadata.transfer_id_to_remote_tp_size[transfer_id] = 2
    metadata.reqs_to_send[request_id] = 1.0

    worker.start_load_kv(metadata)

    assert transfer_id not in worker.transfer_id_to_request_id
    assert transfer_id not in worker.transfer_id_to_remote_tp_size
    assert transfer_id not in worker.transfer_id_to_completion_count
    assert request_id not in worker.request_id_to_transfer_id
    assert request_id not in worker._reqs_to_send


def test_consumer_request_finished_does_not_warn_for_unscheduled_read(
    caplog: pytest.LogCaptureFixture,
) -> None:
    connector = object.__new__(MoRIIOConnectorScheduler)
    connector.is_producer = False
    connector.transfer_id_to_request_id = {}
    connector.request_id_to_transfer_id = {}
    connector._reqs_need_recv = {}
    request = SimpleNamespace(
        request_id="tx-00000000-0000-0000-0000-000000000001",
        status=None,
        kv_transfer_params={
            "do_remote_prefill": True,
            "transfer_id": "tx-00000000-0000-0000-0000-000000000001",
        },
    )

    with caplog.at_level("WARNING"):
        delay_free_blocks, kv_params = connector.request_finished(request, [])

    assert delay_free_blocks is False
    assert kv_params is None
    assert request.kv_transfer_params["do_remote_prefill"] is False
    assert "Could not find" not in caplog.text


def test_zmq_ctx_sets_keepalive_before_bind_and_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.moriio import (
        moriio_common as common,
    )

    class FakeSocket:
        def __init__(self) -> None:
            self.events: list[tuple[str, object, object | None]] = []

        def setsockopt(self, option: object, value: object) -> None:
            self.events.append(("setsockopt", option, value))

        def bind(self, path: str) -> None:
            self.events.append(("bind", path, None))

        def connect(self, path: str) -> None:
            self.events.append(("connect", path, None))

    sockets: list[FakeSocket] = []

    class FakeContext:
        def socket(self, _socket_type: object) -> FakeSocket:
            sock = FakeSocket()
            sockets.append(sock)
            return sock

        def destroy(self, linger: int = 0) -> None:
            pass

    monkeypatch.setattr(common.zmq, "Context", FakeContext)
    monkeypatch.setattr(
        common.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=64 * 1024**3, available=32 * 1024**3),
    )

    with common.zmq_ctx(common.zmq.ROUTER, "tcp://127.0.0.1:5555"):
        pass
    router_events = sockets[-1].events
    keepalive = ("setsockopt", common.zmq.TCP_KEEPALIVE, 1)
    assert router_events.index(keepalive) < router_events.index(
        ("bind", "tcp://127.0.0.1:5555", None)
    )

    with common.zmq_ctx(common.zmq.DEALER, "tcp://127.0.0.1:5555"):
        pass
    dealer_events = sockets[-1].events
    assert dealer_events.index(keepalive) < dealer_events.index(
        ("connect", "tcp://127.0.0.1:5555", None)
    )


def test_finished_count_tracks_tensor_parallel_size() -> None:
    connector = object.__new__(MoRIIOConnector)
    connector._vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=8)
    )

    assert connector.get_finished_count() == 1

    connector._vllm_config.parallel_config.tensor_parallel_size = 8
    connector._vllm_config.parallel_config.data_parallel_size = 1

    assert connector.get_finished_count() == 8


def test_get_port_offset_includes_tensor_parallel_size() -> None:
    assert get_port_offset(dp_rank=0, tp_rank=7, tp_size=16) == 7
    assert get_port_offset(dp_rank=1, tp_rank=0, tp_size=16) == 16
    assert get_port_offset(dp_rank=3, tp_rank=5, tp_size=16) == 53


def test_moriio_config_notify_port_uses_tensor_parallel_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common as common

    monkeypatch.setattr(common, "get_tensor_model_parallel_rank", lambda: 3)
    monkeypatch.setattr(common, "get_tensor_model_parallel_world_size", lambda: 16)
    monkeypatch.setattr(common, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(common, "get_open_port", lambda: 12345)

    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "notify_port": 61005,
                "proxy_ip": "127.0.0.1",
                "proxy_ping_port": 36367,
                "http_port": 8100,
                "handshake_port": 6301,
            }
        ),
        parallel_config=SimpleNamespace(
            data_parallel_rank=2,
            data_parallel_size=4,
        ),
    )

    config = MoRIIOConfig.from_vllm_config(vllm_config)

    assert config.notify_port == 61005 + 35
    assert config.tp_size == 16
    assert config.tp_rank == 3
    assert config.dp_rank == 2


def test_moriio_config_reads_flow_control_extra_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common as common

    monkeypatch.setattr(common, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(common, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(common, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(common, "get_open_port", lambda: 12345)

    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "notify_port": 61005,
                "proxy_ip": "127.0.0.1",
                "proxy_ping_port": 36367,
                "http_port": 8100,
                "handshake_port": 6301,
                "handshake_timeout": 120,
                "max_inflight_global": 16,
                "max_inflight_per_transfer": 4,
                "max_dispatch_layers": 4,
            }
        ),
        parallel_config=SimpleNamespace(data_parallel_rank=0, data_parallel_size=1),
    )

    config = MoRIIOConfig.from_vllm_config(vllm_config)

    assert config.handshake_timeout == 120
    assert config.max_inflight_global == 16
    assert config.max_inflight_per_transfer == 4
    assert config.max_dispatch_layers == 4


def test_read_mode_extra_config_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    def make_config(extra_config: dict[str, object]) -> KVTransferConfig:
        return KVTransferConfig(kv_connector_extra_config=extra_config)

    monkeypatch.setenv("VLLM_MORIIO_CONNECTOR_READ_MODE", "False")
    assert get_moriio_mode(make_config({"read_mode": True})) == MoRIIOMode.READ
    assert get_moriio_mode(make_config({"read_mode": "true"})) == MoRIIOMode.READ

    monkeypatch.setenv("VLLM_MORIIO_CONNECTOR_READ_MODE", "True")
    assert get_moriio_mode(make_config({"read_mode": False})) == MoRIIOMode.WRITE
    assert get_moriio_mode(make_config({"read_mode": "False"})) == MoRIIOMode.WRITE
    assert get_moriio_mode(make_config({"read_mode": "0"})) == MoRIIOMode.WRITE
    assert get_moriio_mode(make_config({"read_mode": "off"})) == MoRIIOMode.WRITE
    assert get_moriio_mode(make_config({"read_mode": None})) == MoRIIOMode.WRITE
    assert get_moriio_mode(make_config({})) == MoRIIOMode.WRITE


def test_requires_piecewise_for_cudagraph_rejects_allow_full() -> None:
    for extra_config in (
        {"allow_full_cudagraph": True},
        {"allow_full_cudagraph": "true"},
        {"allow_full_cudagraph": "1"},
    ):
        with pytest.raises(ValueError, match="allow_full_cudagraph"):
            MoRIIOConnector.requires_piecewise_for_cudagraph(extra_config)


def test_add_new_req_uses_remote_zmq_address_without_embedded_request_id() -> None:
    metadata = MoRIIOConnectorMetadata()

    metadata.add_new_req(
        request_id="plain-request-id",
        local_block_ids=[10, 11],
        kv_transfer_params={
            "transfer_id": "tx-1",
            "remote_block_ids": [0, 1],
            "remote_engine_id": "engine-A",
            "remote_zmq_address": "host:prefill.example,handshake:1234,notify:2345",
            "remote_hosts": ["prefill-a", "prefill-b"],
        },
    )

    req_meta = next(iter(metadata.reqs_to_recv.values()))
    assert req_meta.remote_host == "prefill.example"
    assert req_meta.remote_handshake_port == 1234
    assert req_meta.remote_notify_port == 2345
    assert req_meta.remote_hosts == ["prefill-a", "prefill-b"]


def test_add_new_req_rejects_remote_hosts_without_zmq_ports() -> None:
    metadata = MoRIIOConnectorMetadata()

    with pytest.raises(ValueError, match="remote_zmq_address"):
        metadata.add_new_req(
            request_id="plain-request-id",
            local_block_ids=[10, 11],
            kv_transfer_params={
                "transfer_id": "tx-1",
                "remote_block_ids": [0, 1],
                "remote_engine_id": "engine-A",
                "remote_hosts": ["prefill-a", "prefill-b"],
            },
        )


def test_write_remote_blocks_uses_remote_zmq_address_without_embedded_request_id():
    class Blocks:
        def get_block_ids(self):
            return [[1, 2, 3]]

    scheduler = object.__new__(MoRIIOConnectorScheduler)
    scheduler.mode = MoRIIOMode.WRITE
    scheduler.tp_size = 1
    scheduler.transfer_id_to_request_id = {}
    scheduler.request_id_to_transfer_id = {}
    scheduler._pending_transfer_id_to_request_id = {}
    scheduler.transfer_id_to_remote_tp_size = {}
    scheduler._pending_transfer_id_to_remote_tp_size = {}
    sent_notifies = []
    scheduler.send_notify_block = lambda **kwargs: sent_notifies.append(kwargs)
    scheduler._trim_block_ids_to_token_span = lambda block_ids, _tokens: block_ids
    request = SimpleNamespace(
        request_id="plain-request-id",
        num_prompt_tokens=3,
        kv_transfer_params={
            "transfer_id": "tx-1",
            "do_remote_prefill": True,
            "remote_zmq_address": "host:prefill.example,handshake:1234,notify:2345",
            "remote_tp_size": 2,
            "remote_dp_size": 1,
            "remote_hosts": ["prefill-a", "prefill-b"],
        },
    )

    scheduler.update_state_after_alloc(request, Blocks(), num_external_tokens=3)

    assert [notify["host"] for notify in sent_notifies] == [
        "prefill-a",
        "prefill-b",
    ]
    assert [notify["port"] for notify in sent_notifies] == [2345, 2346]


def test_write_producer_done_sending_translates_transfer_ids() -> None:
    worker = make_worker()
    worker.is_producer = True
    worker.mode = MoRIIOMode.WRITE
    worker.moriio_wrapper.pop_finished_req_ids = lambda: {"transfer0"}
    worker.transfer_id_to_request_id["transfer0"] = "req0"
    worker.request_id_to_transfer_id["req0"] = "transfer0"

    done_sending, done_recving = worker.get_finished()

    assert done_sending == {"req0"}
    assert done_recving == set()
    assert "transfer0" not in worker.transfer_id_to_request_id


def test_write_producer_done_sending_accepts_request_id() -> None:
    # Production WRITE path: the engine appends the *request_id* (not the
    # transfer_id) to done_req_ids in _finalize_if_complete / _mark_request_done
    # (moriio_engine.py). get_finished must therefore resolve a raw request_id
    # via request_id_to_transfer_id -> transfer_id_to_request_id (case 3 of
    # _pop_mapped_completion_id), not only the legacy transfer_id form covered
    # by test_write_producer_done_sending_translates_transfer_ids.
    worker = make_worker()
    worker.is_producer = True
    worker.mode = MoRIIOMode.WRITE
    worker.moriio_wrapper.pop_finished_req_ids = lambda: {"req0"}
    worker.transfer_id_to_request_id["transfer0"] = "req0"
    worker.request_id_to_transfer_id["req0"] = "transfer0"

    done_sending, done_recving = worker.get_finished()

    assert done_sending == {"req0"}
    assert done_recving == set()
    assert "transfer0" not in worker.transfer_id_to_request_id
    assert "req0" not in worker.request_id_to_transfer_id


def test_write_producer_completion_waits_for_mapping() -> None:
    worker = make_worker()
    worker.is_producer = True
    worker.mode = MoRIIOMode.WRITE
    completions = [{"transfer0"}, set()]
    worker.moriio_wrapper.pop_finished_req_ids = lambda: completions.pop(0)

    done_sending, _ = worker.get_finished()

    assert done_sending == set()
    assert "transfer0" in worker._pending_unmapped_done_tids

    worker.transfer_id_to_request_id["transfer0"] = "req0"
    worker.request_id_to_transfer_id["req0"] = "transfer0"

    done_sending, _ = worker.get_finished()

    assert done_sending == {"req0"}
    assert worker._pending_unmapped_done_tids == {}


def test_read_producer_completion_waits_for_remote_tp_quorum() -> None:
    worker = make_worker()
    worker.is_producer = True
    worker.mode = MoRIIOMode.READ
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.moriio_wrapper.pop_finished_req_ids = lambda: {"transfer0:tp0"}
    worker.transfer_id_to_request_id["transfer0"] = "req0"
    worker.request_id_to_transfer_id["req0"] = "transfer0"
    worker.transfer_id_to_completion_count["transfer0"] = 2

    done_sending, done_recving = worker.get_finished()

    assert done_sending == set()
    assert done_recving == set()
    assert worker.transfer_id_to_request_id["transfer0"] == "req0"
    assert worker._read_completion_ids["transfer0"] == {"tp0"}


def test_read_producer_completion_ignores_duplicate_until_final_rank() -> None:
    worker = make_worker()
    worker.is_producer = True
    worker.mode = MoRIIOMode.READ
    worker.tp_rank = 0
    worker.tp_size = 1
    completions = [{"transfer0:tp0"}, {"transfer0:tp0"}, {"transfer0:tp1"}]
    worker.moriio_wrapper.pop_finished_req_ids = lambda: completions.pop(0)
    worker.transfer_id_to_request_id["transfer0"] = "req0"
    worker.request_id_to_transfer_id["req0"] = "transfer0"
    worker.transfer_id_to_completion_count["transfer0"] = 2

    done_sending, _ = worker.get_finished()
    assert done_sending == set()

    done_sending, _ = worker.get_finished()
    assert done_sending == set()
    assert worker._read_completion_ids["transfer0"] == {"tp0"}

    done_sending, done_recving = worker.get_finished()

    assert done_sending == {"req0"}
    assert done_recving == set()
    assert "transfer0" not in worker.transfer_id_to_request_id
    assert "transfer0" not in worker._read_completion_ids


def test_read_producer_pending_completion_uses_quorum_after_mapping() -> None:
    worker = make_worker()
    worker.is_producer = True
    worker.mode = MoRIIOMode.READ
    completions = [{"transfer0:tp0", "transfer0:tp1"}, set()]
    worker.moriio_wrapper.pop_finished_req_ids = lambda: completions.pop(0)

    done_sending, _ = worker.get_finished()

    assert done_sending == set()
    assert set(worker._pending_unmapped_done_tids) == {
        "transfer0:tp0",
        "transfer0:tp1",
    }

    worker.transfer_id_to_request_id["transfer0"] = "req0"
    worker.request_id_to_transfer_id["req0"] = "transfer0"
    worker.transfer_id_to_completion_count["transfer0"] = 2

    done_sending, done_recving = worker.get_finished()

    assert done_sending == {"req0"}
    assert done_recving == set()
    assert worker._pending_unmapped_done_tids == {}


def test_read_producer_zero_quorum_rank_finishes_without_notify() -> None:
    worker = make_worker()
    worker.is_producer = True
    worker.mode = MoRIIOMode.READ
    worker.tp_rank = 3
    worker.tp_size = 4
    metadata = MoRIIOConnectorMetadata()
    metadata.transfer_id_to_request_id["transfer0"] = "req0"
    metadata.transfer_id_to_remote_tp_size["transfer0"] = 2
    metadata.reqs_to_send["req0"] = 1.0

    worker.start_load_kv(metadata)
    done_sending, done_recving = worker.get_finished()

    assert done_sending == {"req0"}
    assert done_recving == set()
    assert "req0" not in worker._reqs_to_send
    assert "transfer0" not in worker.transfer_id_to_request_id


def test_write_finalize_reports_request_id_locally_and_transfer_id_remotely() -> None:
    class Worker:
        pass

    worker = Worker()
    worker.moriio_config = SimpleNamespace(defer_timeout=1.0)
    worker.moriio_wrapper = FakeWrapper()
    worker.num_layers = 1
    worker.tp_rank = 2
    worker.tp_size = 8
    worker.moriio_wrapper.done_remote_allocate_req_dict["transfer0"] = object()
    writer = MoRIIOWriter(worker)
    task = SimpleNamespace(
        request_id="req0",
        transfer_id="transfer0",
        remote_notify_port=61005,
        remote_ip="10.0.0.2",
    )
    request_info = SimpleNamespace(writes_done=0, decode_dp_rank=1)

    writer._finalize_if_complete(task, request_info)

    assert worker.moriio_wrapper.notifies == [("transfer0", "10.0.0.2", 61015)]
    assert worker.moriio_wrapper.done_req_ids == ["req0"]
    assert "transfer0" not in worker.moriio_wrapper.done_remote_allocate_req_dict
    assert worker.moriio_wrapper.waited_for_transfer is True


def test_write_deferred_expiry_reports_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Worker:
        pass

    worker = Worker()
    worker.moriio_config = SimpleNamespace(defer_timeout=1.0)
    worker.moriio_wrapper = FakeWrapper()
    worker.moriio_wrapper.done_remote_allocate_req_dict["transfer0"] = object()
    writer = MoRIIOWriter(worker)
    writer._deferred_tasks = [
        SimpleNamespace(
            enqueue_time=0.0,
            request_id="req0",
            transfer_id="transfer0",
        )
    ]

    monkeypatch.setattr("time.perf_counter", lambda: 5.0)

    writer._process_deferred_tasks()

    assert writer._deferred_tasks == []
    assert worker.moriio_wrapper.done_req_ids == ["req0"]
    assert "transfer0" not in worker.moriio_wrapper.done_remote_allocate_req_dict


def test_remote_allocation_keys_ready_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.moriio import (
        moriio_engine as engine_module,
    )

    monkeypatch.setattr(engine_module, "get_role", lambda: ROLE.PRODUCER)
    worker = make_worker()
    writer = MoRIIOWriter(worker)
    transfer_id = "transfer0"
    request_id = "request0-deadbeef"
    stripped_request_id = "request0"
    task = SimpleNamespace(transfer_id=transfer_id, request_id=request_id)

    assert writer._is_remote_ready(task) is False

    MoRIIOWrapper._handle_structured_message(
        worker.moriio_wrapper,
        {
            "type": "remote_blocks",
            "transfer_id": transfer_id,
            "req_id": request_id,
            "block_notify_list": [21, 22],
            "decode_rank": 3,
        },
    )

    allocation_by_key = worker.moriio_wrapper.done_remote_allocate_req_dict
    info = allocation_by_key[transfer_id]
    assert allocation_by_key[request_id] is info
    assert allocation_by_key[stripped_request_id] is info
    assert info.block_ids == [21, 22]
    assert info.decode_dp_rank == 3
    assert writer._is_remote_ready(task) is True

    def assert_lookup_finds(single_key, lookup_task):
        allocation_by_key.clear()
        allocation_by_key[single_key] = info
        assert writer._is_remote_ready(lookup_task) is True
        assert writer._get_remote_alloc_info(lookup_task) is info

    assert_lookup_finds(
        transfer_id,
        SimpleNamespace(transfer_id=transfer_id, request_id="missing-deadbeef"),
    )
    assert_lookup_finds(
        request_id,
        SimpleNamespace(transfer_id="missing", request_id=request_id),
    )
    assert_lookup_finds(
        stripped_request_id,
        SimpleNamespace(transfer_id="missing", request_id=request_id),
    )

    allocation_by_key[transfer_id] = info
    allocation_by_key[request_id] = info
    allocation_by_key[stripped_request_id] = info

    writer._mark_request_done(request_id, transfer_id)

    assert worker.moriio_wrapper.done_req_ids == [request_id]
    assert transfer_id not in allocation_by_key
    assert request_id not in allocation_by_key
    assert stripped_request_id not in allocation_by_key


def make_req_meta(
    *,
    tp_size: int,
    remote_dp_size: int,
    remote_hosts: list[str] | None = None,
) -> ReqMeta:
    remote_hosts = remote_hosts or ["host0", "host1"]
    return ReqMeta(
        transfer_id="transfer0",
        local_block_ids=[1],
        remote_block_ids=[2],
        remote_host="host0",
        remote_port=1234,
        remote_handshake_port=6301,
        remote_notify_port=61005,
        remote_engine_id="host0:6301",
        tp_size=tp_size,
        remote_dp_size=remote_dp_size,
        remote_hosts=remote_hosts,
    )


def test_multi_node_tp_host_selection_prefers_tp_rank() -> None:
    worker = object.__new__(MoRIIOConnectorWorker)
    worker.tp_rank = 9

    meta = make_req_meta(tp_size=16, remote_dp_size=1)

    assert worker._pick_host_for_dp_rank(meta, dp_rank=7) == "host1"


def test_multi_node_dp_host_selection_uses_dp_rank_when_dp_size_gt_one() -> None:
    worker = object.__new__(MoRIIOConnectorWorker)
    worker.tp_rank = 0

    meta = make_req_meta(
        tp_size=8,
        remote_dp_size=8,
        remote_hosts=[f"host{i}" for i in range(8)],
    )

    assert worker._pick_host_for_dp_rank(meta, dp_rank=7) == "host7"


def test_multi_node_dp_host_selection_uses_dp_rank_for_tp1() -> None:
    worker = object.__new__(MoRIIOConnectorWorker)
    worker.tp_rank = 0

    meta = make_req_meta(tp_size=1, remote_dp_size=16)

    assert worker._pick_host_for_dp_rank(meta, dp_rank=12) == "host1"


def test_wait_for_layer_load_waits_until_layer_status_succeeds() -> None:
    worker = make_worker()
    status = FakeStatus(succeed_after=3)
    worker._recving_transfers["req0"]["layer0"] = status

    worker.wait_for_layer_load("layer0")

    assert status.succeeded_calls >= 3


def test_read_blocks_respects_layer_dispatch_window() -> None:
    worker = make_worker()
    worker.moriio_config.max_dispatch_layers = 1
    worker.tp_rank = 0
    worker.dp_rank = 0
    worker.kv_caches = {
        "layer0": FakeTensor(),
        "layer1": FakeTensor(),
        "layer2": FakeTensor(),
    }
    worker.layer_name_to_local_kv_cache_metadata = {
        "layer0": [],
        "layer1": [],
        "layer2": [],
    }
    worker._get_built_session = lambda _engine_id: (
        ["session0", "session1", "session2"],
        SimpleNamespace(num_blocks=1, block_len=1),
    )
    worker._compute_block_transfer_offsets = lambda *_args: ([0], [0], [1])
    worker.moriio_wrapper.read_results = [FakeStatus(), FakeStatus(), FakeStatus()]

    worker._read_blocks(
        local_block_ids=[11, 12],
        remote_block_ids=[21, 22],
        dst_engine_id="remote",
        request_id="req0",
        transfer_id="transfer0",
        remote_host="host",
        remote_notify_port=61005,
        remote_dp_rank=0,
        remote_tp_size=1,
    )

    assert set(worker._recving_transfers["req0"]) == {"layer0"}
    assert set(worker._pending_read_plans["req0"]) == {"layer1", "layer2"}

    worker._recving_transfers["req0"]["layer0"].succeeded = True
    worker.wait_for_layer_load("layer0")

    assert set(worker._recving_transfers["req0"]) == {"layer0", "layer1"}
    assert set(worker._pending_read_plans["req0"]) == {"layer2"}


def test_wait_for_layer_load_dispatches_required_pending_layer() -> None:
    worker = make_worker()
    worker.moriio_config.max_dispatch_layers = 1
    worker.moriio_wrapper.read_results = [FakeStatus(succeeded=True)]
    worker._recving_transfers_callback_addr["req0"] = ("host", "1234", "transfer0")
    worker._recving_transfer_local_block_ids["req0"] = {11, 12}
    worker._pending_read_plans["req0"]["layer1"] = LayerTransferPlan(
        request_id="req0",
        transfer_id="transfer0",
        layer_name="layer1",
        sess_idx=0,
        transfer_local_offsets=[0],
        transfer_remote_offsets=[0],
        transfer_sizes=[1],
        session="session0",
    )

    worker.wait_for_layer_load("layer1")

    assert "layer1" in worker._recving_transfers["req0"]
    assert "req0" not in worker._pending_read_plans


def test_read_blocks_respects_global_active_layer_window() -> None:
    worker = make_worker()
    worker.moriio_config.max_inflight_global = 2
    worker.tp_rank = 0
    worker.dp_rank = 0
    worker.kv_caches = {
        "layer0": FakeTensor(),
        "layer1": FakeTensor(),
        "layer2": FakeTensor(),
    }
    worker.layer_name_to_local_kv_cache_metadata = {
        "layer0": [],
        "layer1": [],
        "layer2": [],
    }
    worker._get_built_session = lambda _engine_id: (
        ["session0", "session1", "session2"],
        SimpleNamespace(num_blocks=1, block_len=1),
    )
    worker._compute_block_transfer_offsets = lambda *_args: ([0], [0], [1])
    worker.moriio_wrapper.read_results = [FakeStatus(), FakeStatus()]

    worker._read_blocks(
        local_block_ids=[11, 12],
        remote_block_ids=[21, 22],
        dst_engine_id="remote0",
        request_id="req0",
        transfer_id="transfer0",
        remote_host="host",
        remote_notify_port=61005,
        remote_dp_rank=0,
        remote_tp_size=1,
    )
    worker._read_blocks(
        local_block_ids=[13, 14],
        remote_block_ids=[23, 24],
        dst_engine_id="remote1",
        request_id="req1",
        transfer_id="transfer1",
        remote_host="host",
        remote_notify_port=61005,
        remote_dp_rank=0,
        remote_tp_size=1,
    )

    assert set(worker._recving_transfers["req0"]) == {"layer0", "layer1"}
    assert "req1" not in worker._recving_transfers
    assert set(worker._pending_read_plans["req0"]) == {"layer2"}
    assert set(worker._pending_read_plans["req1"]) == {
        "layer0",
        "layer1",
        "layer2",
    }


def test_wait_for_layer_load_raises_on_failed_status() -> None:
    worker = make_worker()
    worker._recving_transfers["req0"]["layer0"] = FakeStatus(failed=True)

    with pytest.raises(RuntimeError, match="request req0, layer layer0"):
        worker.wait_for_layer_load("layer0")


def test_wait_for_layer_load_failure_marks_local_blocks_invalid() -> None:
    worker = make_worker()
    worker._recving_transfers["req0"]["layer0"] = FakeStatus(failed=True)
    worker._recving_transfers_callback_addr["req0"] = ("host", "1234", "transfer0")
    worker._recving_transfer_local_block_ids["req0"] = {11, 12}
    worker.transfer_id_to_request_id["transfer0"] = "req0"

    with pytest.raises(RuntimeError, match="request req0, layer layer0"):
        worker.wait_for_layer_load("layer0")

    assert worker.moriio_wrapper.notifies == [("transfer0:tp0", "host", "1234")]
    assert worker.get_block_ids_with_load_errors() == {11, 12}
    assert "req0" not in worker._recving_transfers
    assert "req0" not in worker._recving_transfers_callback_addr
    assert "req0" not in worker._recving_transfer_local_block_ids
    assert "transfer0" not in worker.transfer_id_to_request_id


def test_pop_done_transfers_waits_for_all_layer_statuses() -> None:
    worker = make_worker()
    worker._recving_transfers["req0"]["layer0"] = FakeStatus(succeeded=True)
    worker._recving_transfers["req0"]["layer1"] = FakeStatus()
    worker._recving_transfers_callback_addr["req0"] = ("host", "1234", "transfer0")
    worker._recving_transfer_local_block_ids["req0"] = {11, 12}
    worker.transfer_id_to_request_id["transfer0"] = "req0"

    assert worker._pop_done_transfers() == set()
    assert worker.moriio_wrapper.notifies == []
    assert "req0" in worker._recving_transfers

    worker._recving_transfers["req0"]["layer1"].succeeded = True

    assert worker._pop_done_transfers() == set()
    assert worker.moriio_wrapper.notifies == [("transfer0:tp0", "host", "1234")]
    assert "req0" not in worker._recving_transfers
    assert "req0" not in worker._recving_transfers_callback_addr
    assert "req0" not in worker._recving_transfer_local_block_ids
    assert "transfer0" not in worker.transfer_id_to_request_id


def test_pop_done_transfers_retries_success_notify_failure() -> None:
    worker = make_worker()
    worker._recving_transfers["req0"]["layer0"] = FakeStatus(succeeded=True)
    worker._recving_transfers_callback_addr["req0"] = ("host", "1234", "transfer0")
    worker._recving_transfer_local_block_ids["req0"] = {11, 12}
    worker.transfer_id_to_request_id["transfer0"] = "req0"
    worker.moriio_wrapper.notify_error = RuntimeError("notify failed")

    assert worker._pop_done_transfers() == set()

    assert worker.moriio_wrapper.notifies == []
    assert "req0" in worker._recving_transfers
    assert worker.transfer_id_to_request_id["transfer0"] == "req0"

    assert worker._pop_done_transfers() == set()

    assert worker.moriio_wrapper.notifies == [("transfer0:tp0", "host", "1234")]
    assert "req0" not in worker._recving_transfers
    assert "transfer0" not in worker.transfer_id_to_request_id


def test_failed_read_marks_local_blocks_invalid() -> None:
    worker = make_worker()
    worker._recving_transfers["req0"]["layer0"] = FakeStatus(failed=True)
    worker._recving_transfers_callback_addr["req0"] = ("host", "1234", "transfer0")
    worker._recving_transfer_local_block_ids["req0"] = {11, 12}
    worker.transfer_id_to_request_id["transfer0"] = "req0"

    assert worker._pop_done_transfers() == set()

    assert worker.moriio_wrapper.notifies == [("transfer0:tp0", "host", "1234")]
    assert worker.get_block_ids_with_load_errors() == {11, 12}
    assert worker.get_block_ids_with_load_errors() == set()
    assert "req0" not in worker._recving_transfers
    assert "req0" not in worker._recving_transfers_callback_addr
    assert "req0" not in worker._recving_transfer_local_block_ids
    assert "transfer0" not in worker.transfer_id_to_request_id


def test_read_blocks_partial_setup_exception_marks_local_blocks_invalid() -> None:
    worker = make_worker()
    worker.tp_rank = 2
    worker.dp_rank = 0
    worker.kv_caches = {"layer0": FakeTensor(), "layer1": FakeTensor()}
    worker.layer_name_to_local_kv_cache_metadata = {"layer0": [], "layer1": []}
    worker._get_built_session = lambda _engine_id: (
        ["session0", "session1"],
        SimpleNamespace(num_blocks=1, block_len=1),
    )
    worker._compute_block_transfer_offsets = lambda *_args: ([0], [0], [1])
    worker.moriio_wrapper.read_results = [
        FakeStatus(succeeded=True),
        RuntimeError("layer1 setup failed"),
    ]

    with pytest.raises(RuntimeError, match="layer1 setup failed"):
        worker._read_blocks(
            local_block_ids=[11, 12],
            remote_block_ids=[21, 22],
            dst_engine_id="remote",
            request_id="req0",
            transfer_id="transfer0",
            remote_host="host",
            remote_notify_port=61005,
            remote_dp_rank=1,
            remote_tp_size=16,
        )

    assert worker.moriio_wrapper.notifies == [("transfer0:tp2", "host", "61023")]
    assert worker.get_block_ids_with_load_errors() == {11, 12}
    assert "req0" not in worker._recving_transfers
    assert "req0" not in worker._recving_transfers_callback_addr
    assert "req0" not in worker._recving_transfer_local_block_ids


def test_read_blocks_setup_exception_notifies_without_invalid_blocks() -> None:
    worker = make_worker()
    worker.tp_rank = 2
    worker.dp_rank = 0
    worker.kv_caches = {"layer0": FakeTensor()}
    worker.layer_name_to_local_kv_cache_metadata = {"layer0": []}
    worker._get_built_session = lambda _engine_id: (
        ["session0"],
        SimpleNamespace(num_blocks=1, block_len=1),
    )
    worker._compute_block_transfer_offsets = lambda *_args: ([0], [0], [1])
    worker.moriio_wrapper.read_error = RuntimeError("setup failed")

    with pytest.raises(RuntimeError, match="setup failed"):
        worker._read_blocks(
            local_block_ids=[11, 12],
            remote_block_ids=[21, 22],
            dst_engine_id="remote",
            request_id="req0",
            transfer_id="transfer0",
            remote_host="host",
            remote_notify_port=61005,
            remote_dp_rank=1,
            remote_tp_size=16,
        )

    assert worker.moriio_wrapper.notifies == [("transfer0:tp2", "host", "61023")]
    assert worker.get_block_ids_with_load_errors() == set()
    assert "req0" not in worker._recving_transfers
    assert "req0" not in worker._recving_transfers_callback_addr
    assert "req0" not in worker._recving_transfer_local_block_ids
