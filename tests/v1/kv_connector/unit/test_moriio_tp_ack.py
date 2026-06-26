# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOMode,
    MoRIIOTransferAck,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker,
    get_moriio_expected_ack_count,
    get_moriio_remote_tp_rank,
    resolve_moriio_transfer_ack,
    validate_moriio_heterogeneous_tp_kv_heads,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.skip_global_cleanup


def test_remote_tp_rank_same_tp_maps_to_self():
    assert [get_moriio_remote_tp_rank(rank, 4, 4) for rank in range(4)] == [
        0,
        1,
        2,
        3,
    ]


def test_remote_tp_rank_p4_d8_floor_maps_decode_to_prefill():
    assert [get_moriio_remote_tp_rank(rank, 8, 4) for rank in range(8)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]


def test_remote_tp_rank_p8_d4_maps_to_first_prefill_rank_per_pair():
    assert [get_moriio_remote_tp_rank(rank, 4, 8) for rank in range(4)] == [
        0,
        2,
        4,
        6,
    ]


@pytest.mark.parametrize(
    ("local_tp_rank", "local_tp_size", "remote_tp_size"),
    [
        (0, 6, 4),
        (0, 4, 6),
    ],
)
def test_remote_tp_rank_invalid_non_multiple_tp_raises(
    local_tp_rank: int, local_tp_size: int, remote_tp_size: int
):
    with pytest.raises(ValueError, match="multiple"):
        get_moriio_remote_tp_rank(local_tp_rank, local_tp_size, remote_tp_size)


@pytest.mark.parametrize(
    ("local_tp_size", "remote_tp_size", "total_num_kv_heads"),
    [
        (4, 4, 8),
        (8, 4, 4),
        (4, 8, 4),
    ],
)
def test_heterogeneous_tp_head_guard_allows_supported_layouts(
    local_tp_size: int, remote_tp_size: int, total_num_kv_heads: int
):
    validate_moriio_heterogeneous_tp_kv_heads(
        local_tp_size,
        remote_tp_size,
        total_num_kv_heads,
        is_mla=False,
    )


def test_heterogeneous_tp_head_guard_allows_mla_layouts():
    validate_moriio_heterogeneous_tp_kv_heads(
        local_tp_size=2,
        remote_tp_size=4,
        total_num_kv_heads=4,
        is_mla=True,
    )


@pytest.mark.parametrize(
    ("local_tp_size", "remote_tp_size", "total_num_kv_heads"),
    [
        (4, 2, 4),
        (2, 4, 4),
    ],
)
def test_heterogeneous_tp_head_guard_rejects_split_kv_heads(
    local_tp_size: int, remote_tp_size: int, total_num_kv_heads: int
):
    with pytest.raises(NotImplementedError, match="replicated KV heads"):
        validate_moriio_heterogeneous_tp_kv_heads(
            local_tp_size,
            remote_tp_size,
            total_num_kv_heads,
            is_mla=False,
        )


def test_expected_ack_count_for_homogeneous_or_smaller_consumer_tp_is_one():
    assert get_moriio_expected_ack_count(4, 4) == 1
    assert get_moriio_expected_ack_count(8, 4) == 1


def test_expected_ack_count_for_decode_fan_in():
    assert get_moriio_expected_ack_count(4, 8) == 2


def test_expected_ack_count_rejects_non_multiple_fan_in():
    with pytest.raises(ValueError, match="multiple"):
        get_moriio_expected_ack_count(4, 6)


def test_plain_string_ack_is_backward_compatible_single_ack():
    notification_counts: dict[str, int] = {}
    completed_transfer_ids: set[str] = set()

    assert (
        resolve_moriio_transfer_ack(
            "tx-plain",
            producer_tp_size=4,
            live_transfer_ids={"tx-plain"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        == "tx-plain"
    )
    assert notification_counts == {}
    assert completed_transfer_ids == {"tx-plain"}


def test_structured_release_ack_waits_for_all_expected_acks():
    ack = MoRIIOTransferAck("tx-fanin", consumer_tp_size=8)
    notification_counts: dict[str, int] = {}
    completed_transfer_ids: set[str] = set()

    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-fanin"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        is None
    )
    assert notification_counts == {"tx-fanin": 1}
    assert completed_transfer_ids == set()

    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-fanin"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        == "tx-fanin"
    )
    assert notification_counts == {}
    assert completed_transfer_ids == {"tx-fanin"}


def test_duplicate_ack_after_completion_does_not_resolve_twice():
    ack = MoRIIOTransferAck("tx-dup", consumer_tp_size=8)
    notification_counts: dict[str, int] = {}
    completed_transfer_ids: set[str] = set()

    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-dup"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        is None
    )
    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-dup"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        == "tx-dup"
    )
    assert (
        resolve_moriio_transfer_ack(
            ack,
            producer_tp_size=4,
            live_transfer_ids={"tx-dup"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        is None
    )
    assert notification_counts == {}
    assert completed_transfer_ids == {"tx-dup"}


def test_ack_for_non_live_transfer_is_ignored():
    notification_counts: dict[str, int] = {}
    completed_transfer_ids: set[str] = set()

    assert (
        resolve_moriio_transfer_ack(
            MoRIIOTransferAck("tx-stale", consumer_tp_size=8),
            producer_tp_size=4,
            live_transfer_ids={"tx-live"},
            notification_counts=notification_counts,
            completed_transfer_ids=completed_transfer_ids,
        )
        is None
    )
    assert notification_counts == {}
    assert completed_transfer_ids == set()


def _bare_scheduler(is_producer: bool = True) -> MoRIIOConnectorScheduler:
    scheduler = object.__new__(MoRIIOConnectorScheduler)
    scheduler.is_producer = is_producer
    scheduler.transfer_id_to_request_id = {}
    scheduler.request_id_to_transfer_id = {}
    scheduler._reqs_need_recv = {}
    scheduler._reqs_need_send = {}
    scheduler._deferred_send_deadlines = {}
    scheduler._defer_timeout = 60.0
    scheduler.engine_id = "127.0.0.1:6301"
    scheduler.host_ip = "127.0.0.1"
    scheduler.handshake_port = 6301
    scheduler.side_notify_port = 61005
    scheduler.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=1,
        )
    )
    return scheduler


def test_transfer_id_remap_does_not_let_stale_unmap_delete_new_mapping():
    scheduler = _bare_scheduler()

    scheduler.map_request_id("old-req", "tx0")
    scheduler.map_request_id("new-req", "tx0")
    scheduler.unmap_request_id("old-req")

    assert scheduler.transfer_id_to_request_id == {"tx0": "new-req"}
    assert scheduler.request_id_to_transfer_id == {"new-req": "tx0"}


def test_request_remap_removes_stale_transfer_mapping():
    scheduler = _bare_scheduler()

    scheduler.map_request_id("req0", "tx0")
    scheduler.map_request_id("req0", "tx1")

    assert scheduler.transfer_id_to_request_id == {"tx1": "req0"}
    assert scheduler.request_id_to_transfer_id == {"req0": "tx1"}


def test_producer_request_finished_unmaps_when_blocks_not_deferred():
    scheduler = _bare_scheduler(is_producer=True)
    scheduler.map_request_id("req0", "tx0")
    request = SimpleNamespace(
        request_id="req0",
        kv_transfer_params={"transfer_id": "tx0", "do_remote_decode": False},
        status=RequestStatus.FINISHED_STOPPED,
    )

    delay_free, new_params = scheduler.request_finished(request, block_ids=[])

    assert delay_free is False
    assert new_params is None
    assert scheduler.transfer_id_to_request_id == {}
    assert scheduler.request_id_to_transfer_id == {}


def test_producer_request_finished_keeps_mapping_when_blocks_deferred():
    scheduler = _bare_scheduler(is_producer=True)
    scheduler.map_request_id("req0", "tx0")
    request = SimpleNamespace(
        request_id="req0",
        kv_transfer_params={"transfer_id": "tx0", "do_remote_decode": True},
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
    )

    delay_free, new_params = scheduler.request_finished(request, block_ids=[1])

    assert delay_free is True
    assert new_params is not None
    assert scheduler.transfer_id_to_request_id == {"tx0": "req0"}
    assert scheduler.request_id_to_transfer_id == {"req0": "tx0"}
    assert scheduler._reqs_need_send.keys() == {"req0"}
    assert scheduler._deferred_send_deadlines.keys() == {"req0"}


def test_deferred_send_timeout_marks_finished_and_clears_mapping():
    scheduler = _bare_scheduler(is_producer=True)
    scheduler.map_request_id("req0", "tx0")
    scheduler._deferred_send_deadlines["req0"] = 0.0
    connector_output = KVConnectorOutput()

    scheduler.update_connector_output(connector_output)

    assert connector_output.finished_sending == {"req0"}
    assert scheduler._deferred_send_deadlines == {}
    assert scheduler.transfer_id_to_request_id == {}
    assert scheduler.request_id_to_transfer_id == {}


def test_write_completion_before_transfer_mapping_is_retried():
    class FakeWrapper:
        def __init__(self):
            self.batches = [{"tx-early"}, set()]

        def pop_finished_write_req_ids(self):
            return self.batches.pop(0)

        def shutdown(self):
            pass

    worker = object.__new__(MoRIIOConnectorWorker)
    worker.is_producer = False
    worker.mode = MoRIIOMode.WRITE
    worker.moriio_wrapper = FakeWrapper()
    worker.transfer_id_to_request_id = {}
    worker._unmatched_write_completions = set()

    assert worker.get_finished() == (set(), set())
    assert worker._unmatched_write_completions == {"tx-early"}

    worker.transfer_id_to_request_id = {"tx-early": "req0"}

    assert worker.get_finished() == (set(), {"req0"})
    assert worker._unmatched_write_completions == set()


def test_worker_get_finished_counts_structured_release_fan_in():
    class FakeWrapper:
        def __init__(self):
            self.batches = [
                [MoRIIOTransferAck("tx-fanin", consumer_tp_size=8)],
                [MoRIIOTransferAck("tx-fanin", consumer_tp_size=8)],
            ]

        def pop_finished_req_ids(self):
            return self.batches.pop(0)

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.is_producer = True
    worker.mode = MoRIIOMode.READ
    worker.world_size = 4
    worker.moriio_wrapper = FakeWrapper()
    worker.transfer_id_to_request_id = {"tx-fanin": "req-fanin"}
    worker._consumer_notification_counts = {}
    worker._completed_consumer_notifications = set()

    assert worker.get_finished() == (set(), set())
    assert worker._consumer_notification_counts == {"tx-fanin": 1}

    assert worker.get_finished() == ({"req-fanin"}, set())
    assert worker._consumer_notification_counts == {}
    assert worker._completed_consumer_notifications == {"tx-fanin"}


def test_read_completion_sends_structured_release_with_consumer_tp_size():
    class DoneStatus:
        def Succeeded(self):
            return True

        def Failed(self):
            return False

    class FakeWrapper:
        def __init__(self):
            self.lock = threading.Lock()
            self.sent = []

        def send_notify(
            self,
            transfer_id,
            host,
            port,
            message_type=None,
            message_fields=None,
        ):
            self.sent.append((transfer_id, host, port, message_type, message_fields))

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.world_size = 8
    worker.moriio_wrapper = FakeWrapper()
    worker._recving_transfers = {"req": [DoneStatus()]}
    worker._recving_transfers_callback_addr = {
        "req": ("127.0.0.1", "7000", "tx-release")
    }

    assert worker._pop_done_transfers() == {"tx-release"}
    assert worker.moriio_wrapper.sent == [
        (
            "tx-release",
            "127.0.0.1",
            "7000",
            "release",
            {"consumer_tp_size": 8},
        )
    ]
    assert worker._recving_transfers == {}
    assert worker._recving_transfers_callback_addr == {}
