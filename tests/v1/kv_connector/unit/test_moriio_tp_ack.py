# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOMode,
    MoRIIOTransferAck,
    TransferError,
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
    MoRIIOWrapper,
)
from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    MambaConvSplitInfo,
)
from vllm.v1.kv_cache_interface import MambaSpec


@pytest.mark.parametrize(
    "params,expected",
    [
        (None, (0, False)),
        ({}, (0, False)),
        ({"do_remote_prefill": False}, (0, False)),
        ({"do_remote_prefill": True}, (31, False)),
    ],
)
def test_read_matches_only_requests_with_pending_remote_prefill(params, expected):
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.is_producer = False
    scheduler.mode = MoRIIOMode.READ
    request = SimpleNamespace(num_prompt_tokens=32, kv_transfer_params=params)

    assert scheduler.get_num_new_matched_tokens(request, 0) == expected


@pytest.mark.parametrize(
    "mode,consumer,has_mamba,layout,pending,expected",
    [
        (MoRIIOMode.READ, True, True, "LBNHC", [[7, 8], [90]], [7, 8]),
        (MoRIIOMode.READ, True, True, "LBNHC", [[], [90]], []),
        (MoRIIOMode.READ, True, True, "LBNHC", None, []),
        (MoRIIOMode.READ, False, True, "LBNHC", [[7], [90]], []),
        (MoRIIOMode.READ, True, False, "LBNHC", [[7]], []),
        (MoRIIOMode.READ, True, True, "NBLHC", [[7], [90]], []),
        (MoRIIOMode.WRITE, True, True, "LBNHC", [[7], [90]], []),
    ],
)
def test_sync_read_initializes_only_supported_attention_destinations(
    mode, consumer, has_mamba, layout, pending, expected
):
    connector = MoRIIOConnector.__new__(MoRIIOConnector)
    connector.mode = mode
    connector.kv_transfer_config = SimpleNamespace(is_kv_consumer=consumer)
    connector._vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: SimpleNamespace(name=layout)
        )
    )
    request = SimpleNamespace(request_id="req")
    connector.connector_scheduler = SimpleNamespace(
        _has_mamba=has_mamba,
        _reqs_need_recv={} if pending is None else {"req": (request, pending)},
    )

    assert connector.get_sync_load_block_ids(request) == expected


@pytest.mark.parametrize("mode", [MoRIIOMode.READ, MoRIIOMode.WRITE])
def test_read_requires_completion_of_draft_kv_writes(mode):
    connector = MoRIIOConnector.__new__(MoRIIOConnector)
    connector.mode = mode

    assert connector.requires_full_step_completion is (mode == MoRIIOMode.READ)


def _unaligned_cpu_backing():
    raw = bytearray(12288)
    original = torch.frombuffer(raw, dtype=torch.uint8)
    offset = (128 - original.data_ptr()) % 4096
    return torch.frombuffer(raw, dtype=torch.uint8, offset=offset, count=8192)


def _mamba_registration_worker(caches):
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.kv_caches = caches
    spec = MambaSpec(
        block_size=16,
        shapes=((4, 1), (2, 2)),
        dtypes=(torch.float32, torch.float32),
    )
    worker.layer_to_spec = dict.fromkeys(caches, spec)
    return worker


def test_shared_registration_covers_final_ssm_with_unaligned_first_enabled_layer():
    backing = _unaligned_cpu_backing()
    # Earlier layers can be transfer-disabled. The first enabled layer need
    # not begin at the aligned start of the common allocation.
    cache = backing[-64:].view(2, 1, 1, 32)
    worker = _mamba_registration_worker({"kda": cache})

    registration, offsets = worker._build_shared_kv_mr(worker.kv_caches)

    expected_base = cache.data_ptr() // 4096 * 4096
    assert registration.data_ptr() == expected_base
    assert registration.data_ptr() >= backing.data_ptr()
    assert registration.data_ptr() + registration.numel() == (
        backing.data_ptr() + backing.numel()
    )
    assert registration.untyped_storage().data_ptr() == backing.data_ptr()
    assert offsets == {"kda": cache.data_ptr() - expected_base}


def test_shared_registration_refuses_alignment_before_storage():
    cache = _unaligned_cpu_backing()[:64].view(2, 1, 1, 32)
    worker = _mamba_registration_worker({"kda": cache})

    with pytest.raises(ValueError, match="outside.*storage"):
        worker._build_shared_kv_mr(worker.kv_caches)


def test_shared_registration_refuses_regions_from_other_storage():
    backing = _unaligned_cpu_backing()
    other = _unaligned_cpu_backing()
    worker = _mamba_registration_worker(
        {
            "first": backing[-64:].view(2, 1, 1, 32),
            "other": other[-64:].view(2, 1, 1, 32),
        }
    )

    with pytest.raises(ValueError, match="same storage"):
        worker._build_shared_kv_mr(worker.kv_caches)


def test_mamba_reads_address_each_region_relative_to_shared_registration(monkeypatch):
    cache = torch.zeros((2, 1, 1, 32), dtype=torch.uint8)
    worker = _mamba_registration_worker({"kda": cache})
    worker.world_size = 1
    worker._conv_decomp = MambaConvSplitInfo(1, (1, 1, 2), 4, (16, 16))
    worker._mamba_offset_templates = {}
    worker.kv_region_mr_offsets = {"kda": [128, 144]}
    worker.layer_base_addr_index = {"kda": 0}
    worker.layer_name_to_remote_kv_cache_metadata = {"peer": {"kda": [4096, 4096]}}
    worker.moriio_wrapper = MoRIIOWrapper(
        moriio_engine=SimpleNamespace(allocate_transfer_uid=lambda: 1)
    )
    worker.moriio_wrapper.local_memory_registered = True
    monkeypatch.setattr(
        worker.moriio_wrapper,
        "get_unpack_memory_metadata",
        lambda address: SimpleNamespace(data=address),
    )
    posted = []

    class Session:
        def batch_read(self, local, remote, sizes, uid):
            posted.append((local, remote, sizes))
            return SimpleNamespace(Failed=lambda: False)

    statuses = worker._post_mamba_reads(
        "kda",
        [Session(), Session()],
        [0, 1],
        [1],
        [0],
        1,
        SimpleNamespace(kv_caches_base_addr=[4352, 4368]),
        "peer",
        "req",
        float("inf"),
    )

    assert len(statuses) == 2
    assert posted == [
        ([160, 164, 168], [256, 260, 264], [4, 4, 8]),
        ([176], [272], [16]),
    ]


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

        # Exercise the real batch verdict rather than restating it here.
        poll_transfer_batch = MoRIIOWrapper.poll_transfer_batch

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
    worker.moriio_config = SimpleNamespace(recv_abort_timeout=600.0)
    worker.moriio_wrapper = FakeWrapper()
    # A layer maps to the list of reads posted for it (a KDA layer posts two).
    worker._recving_transfers = {"req": {"layer0": [DoneStatus(), DoneStatus()]}}
    worker._recving_transfers_callback_addr = {
        "req": ("127.0.0.1", "7000", "tx-release")
    }
    # Transfer-timeout reaping state consulted by _pop_done_transfers.
    worker._recving_transfers_start = {}
    # Load-error bookkeeping cleared alongside a completed transfer.
    worker._recving_local_blocks = {}
    worker._invalid_block_ids = set()

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


def test_aborted_read_without_local_blocks_only_releases_remote_blocks():
    class FakeWrapper:
        def __init__(self):
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
    worker.mode = MoRIIOMode.READ
    worker.world_size = 8
    worker.moriio_wrapper = FakeWrapper()

    worker._read_blocks(
        local_block_ids=[],
        remote_block_ids=[[10, 11], [90]],
        dst_engine_id="prefill",
        request_id="req-aborted",
        transfer_id="tx-aborted",
        remote_host="127.0.0.1",
        remote_notify_port=7000,
        remote_tp_size=8,
        remote_dp_rank=0,
        chosen_tp=2,
    )

    assert worker.moriio_wrapper.sent == [
        (
            "tx-aborted",
            "127.0.0.1",
            "7002",
            "release",
            {"consumer_tp_size": 8},
        )
    ]


def test_read_completion_waits_for_every_posted_read():
    class Status:
        def __init__(self, done: bool):
            self.done = done

        def Succeeded(self):
            return self.done

        def Failed(self):
            return False

    class FakeWrapper:
        def __init__(self):
            self.lock = threading.Lock()
            self.sent = []

        poll_transfer_batch = MoRIIOWrapper.poll_transfer_batch

        def send_notify(self, *args, **kwargs):
            self.sent.append((args, kwargs))

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.world_size = 8
    worker.moriio_config = SimpleNamespace(recv_abort_timeout=600.0)
    worker.moriio_wrapper = FakeWrapper()
    worker._recving_transfers = {
        "req": {"kda_layer": [Status(done=True), Status(done=False)]}
    }
    worker._recving_transfers_callback_addr = {
        "req": ("127.0.0.1", "7000", "tx-pending")
    }
    worker._recving_transfers_start = {"req": float("inf")}
    worker._recving_local_blocks = {"req": [1, 2]}
    worker._invalid_block_ids = set()

    assert worker._pop_done_transfers() == set()
    assert "req" in worker._recving_transfers
    assert worker.moriio_wrapper.sent == []


@pytest.mark.parametrize(
    ("has_mamba", "expected_invalid"),
    [(False, {1, 2}), (True, set())],
)
def test_failed_read_reports_blocks_only_without_hma(has_mamba, expected_invalid):
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker._has_mamba = has_mamba
    worker._recving_local_blocks = {"req": [1, 2]}
    worker._invalid_block_ids = set()

    worker._record_failed_recv("req")

    assert worker.get_block_ids_with_load_errors() == expected_invalid


@pytest.mark.parametrize(
    "states,elapsed,has_mamba,error",
    [
        (["failed", "pending"], 1, True, "failed"),
        (["pending", "failed"], 1, True, "failed"),
        (["pending"], 121, True, "timed out"),
        (["done", "done"], 121, True, None),
        (["pending"], 120, True, None),
        (["failed", "pending"], 1, False, None),
    ],
)
def test_read_completion_preserves_hybrid_blocks_on_error(
    monkeypatch, states, elapsed, has_mamba, error
):
    class Status:
        def __init__(self, state):
            self.state = state

        def Succeeded(self):
            return self.state == "done"

        def Failed(self):
            return self.state == "failed"

        def Message(self):
            return self.state

        def Code(self):
            return self.state

    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector.time.monotonic",
        lambda: 1000.0,
    )
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.is_producer = False
    worker.mode = MoRIIOMode.READ
    worker.world_size = 8
    worker._has_mamba = has_mamba
    worker.moriio_config = SimpleNamespace(recv_abort_timeout=120.0)
    worker.moriio_wrapper = MoRIIOWrapper()
    notifications = []
    monkeypatch.setattr(
        worker.moriio_wrapper,
        "send_notify",
        lambda *args, **kwargs: notifications.append((args, kwargs)),
    )
    worker._recving_transfers = {"req": {"layer": [Status(state) for state in states]}}
    worker._recving_transfers_callback_addr = {"req": ("host", "7000", "tx")}
    worker._recving_transfers_start = {"req": 1000.0 - elapsed}
    worker._recving_local_blocks = {"req": [7, 8]}
    worker._invalid_block_ids = set()

    if error is not None:
        with pytest.raises(TransferError, match=error):
            worker.get_finished()
        assert notifications == []
        assert "req" in worker._recving_transfers
        assert worker._recving_transfers_callback_addr == {
            "req": ("host", "7000", "tx")
        }
        assert worker._recving_transfers_start == {"req": 1000.0 - elapsed}
        assert worker._recving_local_blocks == {"req": [7, 8]}
        assert worker.get_block_ids_with_load_errors() == set()
    else:
        assert worker.get_finished() == (set(), set())
        completed = states == ["done", "done"] or not has_mamba
        assert len(notifications) == int(completed)
        assert bool(worker._recving_transfers) is not completed
        assert worker.get_block_ids_with_load_errors() == (
            set() if has_mamba else {7, 8}
        )


def test_hybrid_step_barrier_fails_closed(monkeypatch):
    class FailingWrapper:
        def waiting_for_transfer_complete(self, _statuses):
            raise TransferError("failed")

        def shutdown(self):
            pass

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker._has_mamba = True
    worker._reads_issued_this_step = [object()]
    worker._mamba_reads_this_step = [object()]
    worker.moriio_wrapper = FailingWrapper()
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector.get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=None),
    )

    with pytest.raises(TransferError, match="failed"):
        worker._await_reads_issued_this_step()


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
