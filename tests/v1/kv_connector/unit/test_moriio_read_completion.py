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
    MoRIIOConfig,
    MoRIIOMode,
    ReqMeta,
    get_moriio_mode,
    get_port_offset,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    _MAX_PENDING_UNMAPPED_DONE_TIDS,
    MoRIIOConnector,
    MoRIIOConnectorWorker,
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
        self.notifies: list[tuple[str, str, str]] = []
        self.read_error: Exception | None = None
        self.read_results: list[FakeStatus | Exception] = []
        self.notify_error: Exception | None = None

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
    worker.moriio_config = SimpleNamespace(transfer_timeout=1.0, defer_timeout=1.0)
    worker.moriio_wrapper = FakeWrapper()
    worker._recving_transfers = defaultdict(dict)
    worker._recving_transfers_callback_addr = {}
    worker._recving_transfer_local_block_ids = {}
    worker._invalid_block_ids = queue.Queue()
    worker._reqs_to_send = {}
    worker._pending_unmapped_done_tids = {}
    worker._unmatched_write_completions = set()
    worker.transfer_id_to_request_id = {}
    worker.request_id_to_transfer_id = {}
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

    assert worker.moriio_wrapper.notifies == [("transfer0", "host", "1234")]
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
    assert worker.moriio_wrapper.notifies == [("transfer0", "host", "1234")]
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

    assert worker.moriio_wrapper.notifies == [("transfer0", "host", "1234")]
    assert "req0" not in worker._recving_transfers
    assert "transfer0" not in worker.transfer_id_to_request_id


def test_failed_read_marks_local_blocks_invalid() -> None:
    worker = make_worker()
    worker._recving_transfers["req0"]["layer0"] = FakeStatus(failed=True)
    worker._recving_transfers_callback_addr["req0"] = ("host", "1234", "transfer0")
    worker._recving_transfer_local_block_ids["req0"] = {11, 12}
    worker.transfer_id_to_request_id["transfer0"] = "req0"

    assert worker._pop_done_transfers() == set()

    assert worker.moriio_wrapper.notifies == [("transfer0", "host", "1234")]
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

    assert worker.moriio_wrapper.notifies == [("transfer0", "host", "61023")]
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

    assert worker.moriio_wrapper.notifies == [("transfer0", "host", "61023")]
    assert worker.get_block_ids_with_load_errors() == set()
    assert "req0" not in worker._recving_transfers
    assert "req0" not in worker._recving_transfers_callback_addr
    assert "req0" not in worker._recving_transfer_local_block_ids
