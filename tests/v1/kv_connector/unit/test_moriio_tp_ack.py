# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOMode,
    MoRIIOTransferAck,
    RemoteAllocInfo,
    WriteTask,
    get_port_offset,
    resolve_peer_tp_size,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnector,
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker,
    get_moriio_expected_ack_count,
    get_moriio_remote_tp_rank,
    resolve_moriio_transfer_ack,
    validate_moriio_heterogeneous_tp_kv_heads,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWriter,
)


def test_remote_tp_rank_same_tp_maps_to_self():
    assert [get_moriio_remote_tp_rank(rank, 4, 4) for rank in range(4)] == [
        0,
        1,
        2,
        3,
    ]


@pytest.mark.parametrize(("dp_size", "tp_size"), [(2, 2), (2, 4), (4, 2)])
def test_dp_tp_port_offsets_are_injective(dp_size, tp_size):
    offsets = {
        get_port_offset(dp_rank, tp_rank, tp_size)
        for dp_rank in range(dp_size)
        for tp_rank in range(tp_size)
    }

    assert len(offsets) == dp_size * tp_size


def test_port_offset_rejects_unknown_tp_size():
    """0 is the "unknown peer TP" sentinel; it must not silently drop the DP term."""
    with pytest.raises(ValueError, match="tp_size must be positive"):
        get_port_offset(1, 0, 0)


def test_resolve_peer_tp_size_falls_back_when_unadvertised():
    assert resolve_peer_tp_size({}, 4) == 4
    assert resolve_peer_tp_size({"tp_size": 0}, 4) == 4
    assert resolve_peer_tp_size({"tp_size": 2}, 4) == 2
    assert resolve_peer_tp_size({"remote_tp_size": 8, "tp_size": 2}, 4) == 8


def test_early_write_release_uses_producer_tp_size():
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.tp_size = 4
    sent = []
    scheduler._send_transfer_release = lambda *args: sent.append(args)

    scheduler._release_write_prefill_blocks(
        "req",
        {
            "transfer_id": "tx",
            "remote_dp_rank": 1,
            "remote_host": "producer",
            "remote_notify_port": 7000,
            "tp_size": 2,
        },
    )

    assert sent == [
        ("tx", "producer", 7000 + get_port_offset(1, 0, 2)),
        ("tx", "producer", 7000 + get_port_offset(1, 1, 2)),
    ]


def _writer_with_stub_worker(tp_rank, world_size):
    sent = []
    wrapper = SimpleNamespace(
        lock=threading.Lock(),
        done_req_ids=[],
        done_remote_allocate_req_dict={},
        waiting_for_transfer_complete=lambda _: None,
        send_notify=lambda _, host, port, **kwargs: sent.append((host, port)),
        _mark_transfer_terminal_locked=lambda _: None,
    )
    worker = SimpleNamespace(
        moriio_wrapper=wrapper,
        tp_rank=tp_rank,
        world_size=world_size,
    )
    writer = MoRIIOWriter.__new__(MoRIIOWriter)
    writer._worker_ref = lambda: worker
    writer._write_state_lock = threading.Lock()
    writer._sealed_writes = {}
    writer._clear_transfer_state = lambda _: None
    return writer, worker, sent


def _complete_write(
    writer,
    worker,
    transfer_id,
    decode_dp_rank,
    notify_port,
    decode_tp_size,
    remote_ip="127.0.0.1",
):
    info = RemoteAllocInfo(block_ids=None)  # type: ignore[arg-type]
    info.decode_dp_rank = decode_dp_rank
    worker.moriio_wrapper.done_remote_allocate_req_dict[transfer_id] = info
    writer._execute_write_task(
        WriteTask(
            request_id="req",
            transfer_id=transfer_id,
            dst_engine_id="decode",
            local_block_ids=[0],
            remote_block_ids_hint=None,
            layer_name="layer",
            event=None,  # type: ignore[arg-type]
            remote_notify_port=notify_port,
            remote_ip=remote_ip,
            remote_tp_size=decode_tp_size,
        )
    )
    info.block_ids = [0]
    info.writes_expected = 1
    info.writes_done = 1
    writer._finalize_if_complete(transfer_id, info)


@pytest.mark.parametrize("decode_dp_rank", [0, 1])
def test_write_done_targets_the_exact_decode_rank_that_bound_the_port(decode_dp_rank):
    base, producer_tp, decode_tp = 7000, 4, 2
    bound = {
        base + get_port_offset(dp, tp, decode_tp): (dp, tp)
        for dp in range(2)
        for tp in range(decode_tp)
    }

    for producer_rank in range(producer_tp):
        writer, worker, sent = _writer_with_stub_worker(producer_rank, producer_tp)
        _complete_write(
            writer,
            worker,
            f"tx-{producer_rank}",
            decode_dp_rank,
            base,
            decode_tp,
        )
        ((_, port),) = sent
        expected_tp = get_moriio_remote_tp_rank(producer_rank, producer_tp, decode_tp)
        assert bound[port] == (decode_dp_rank, expected_tp)


def test_write_done_uses_per_request_decode_tp_size():
    base = 7000
    writer, worker, sent = _writer_with_stub_worker(tp_rank=0, world_size=8)

    _complete_write(writer, worker, "tx-a", 1, base, 2, remote_ip="host-a")
    _complete_write(writer, worker, "tx-b", 1, base, 4, remote_ip="host-b")

    assert sent == [
        ("host-a", base + get_port_offset(1, 0, 2)),
        ("host-b", base + get_port_offset(1, 0, 4)),
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
    worker._pending_unmapped_acks = []

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
    worker._recving_transfers = {"req": {"layer0": DoneStatus()}}
    worker._recving_transfers_callback_addr = {
        "req": ("127.0.0.1", "7000", "tx-release")
    }
    # Transfer-timeout reaping state consulted by _pop_done_transfers.
    worker._recving_transfers_start = {}

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


def test_requested_cudagraph_mode_is_never_overridden():
    # The configured cudagraph mode is always honored: the barrier fires when
    # the operator sets cudagraph_mode=PIECEWISE, and READ mode with full
    # graphs only warns instead of silently forcing PIECEWISE.
    assert (
        MoRIIOConnector.requires_piecewise_for_cudagraph({"read_mode": True}) is False
    )
    assert (
        MoRIIOConnector.requires_piecewise_for_cudagraph({"read_mode": False}) is False
    )
