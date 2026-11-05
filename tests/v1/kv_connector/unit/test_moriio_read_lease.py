# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""READ mode: prefill frees a request's blocks when decode's lease on them
expires, and decode renews that lease while the request waits."""

import threading
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio import (
    moriio_connector,
    moriio_engine,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    MoRIIOConnectorMetadata,
    MoRIIOMode,
    ReadLeaseTarget,
    get_port_offset,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
)
from vllm.v1.core.sched.interface import PauseState

from .test_moriio_connector import _setup_kv_transfer_request, create_vllm_config
from .test_moriio_routing import _producer, _req, _request_finished
from .utils import create_request, create_scheduler

LEASE = 60.0


class _Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    clock = _Clock()
    with patch.object(moriio_connector.time, "perf_counter", clock):
        yield clock


def _send(wrapper: MoRIIOWrapper, message: dict) -> None:
    """Deliver a decode message through prefill's notify handler."""
    with patch.object(moriio_engine, "get_role", return_value=ROLE.PRODUCER):
        wrapper._handle_structured_message(message)


def _prefill_worker() -> MoRIIOConnectorWorker:
    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper.done_req_ids = []
    wrapper.renewed_lease_transfer_ids = set()
    wrapper.done_remote_allocate_req_dict = {}
    wrapper._terminal_transfer_ids = OrderedDict()
    wrapper.paths = {}
    wrapper.async_wait_reqid = lambda: None

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.is_producer = True
    worker.mode = MoRIIOMode.READ
    worker.world_size = 1
    worker.moriio_wrapper = wrapper
    worker.transfer_id_to_request_id = {}
    worker._consumer_notification_counts = {}
    worker._completed_consumer_notifications = set()
    worker._pending_unmapped_acks = []
    worker._reqs_to_send = {}
    worker._read_lease_extension = LEASE * 2 / 3
    worker._expired_read_leases = OrderedDict()
    worker._num_late_read_releases = 0
    return worker


def _step(worker, held: dict[str, str], scheduler_clock=0.0, new=None) -> None:
    """A scheduler step: ``held`` maps transfer ids to the requests the
    scheduler still holds; ``new`` the lease expiries of newly finished ones."""
    meta = MoRIIOConnectorMetadata()
    meta.transfer_id_to_request_id = held
    meta.reqs_to_send = new or {}
    meta.scheduler_clock = scheduler_clock
    worker.start_load_kv(meta)


def test_prefill_frees_blocks_once_renewals_stop(clock):
    worker = _prefill_worker()
    # The scheduler's perf_counter epoch is unrelated to this worker's.
    _step(worker, {"tx-1": "req-1"}, scheduler_clock=5.0, new={"req-1": 5.0 + LEASE})

    # Renewed every LEASE / 6, the blocks outlive the original lease.
    for _ in range(12):
        clock.now += LEASE / 6
        _send(worker.moriio_wrapper, {"type": "renew_lease", "transfer_ids": ["tx-1"]})
        assert worker.get_finished() == (set(), set())

    clock.now += LEASE * 2 / 3
    assert worker.get_finished() == ({"req-1"}, set())

    # Decode reads after all. Its release is counted, never reported again, and
    # not buffered as an unmapped ACK, even while the scheduler still maps it.
    _send(worker.moriio_wrapper, {"type": "release", "transfer_id": "tx-1"})
    assert worker.get_finished() == (set(), set())
    assert worker._num_late_read_releases == 1
    assert worker._pending_unmapped_acks == []


def test_prefill_frees_blocks_on_release_and_forgets_freed_requests(clock):
    worker = _prefill_worker()
    held = {"tx-1": "req-1", "tx-2": "req-2"}
    _step(worker, held, new={"req-1": LEASE, "req-2": LEASE})

    _send(worker.moriio_wrapper, {"type": "release", "transfer_id": "tx-1"})
    assert worker.get_finished() == ({"req-1"}, set())

    # The scheduler freed req-2 itself (renewal cap); the worker drops it too.
    _step(worker, {})
    assert worker.get_finished() == (set(), set())
    assert worker._reqs_to_send == {}
    assert worker._expired_read_leases == {}


def test_prefill_advertises_its_lease_and_caps_renewals(clock):
    scheduler = _producer(global_dp_rank=0, dp_size=1, dp_size_local=1)
    scheduler.mode = MoRIIOMode.READ
    scheduler._defer_timeout = LEASE
    scheduler._reqs_need_send = {}
    scheduler._deferred_send_deadlines = {}
    scheduler._pending_sent_acks = {}
    scheduler.unmap_request_id = MagicMock()
    scheduler.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[None], select_transfer_block_ids=lambda block_ids: block_ids
    )
    params = {"do_remote_decode": True, "transfer_id": "tx-1"}

    held, out = _request_finished(scheduler, _req("r1", dict(params)), [[1, 2]])
    assert held and out["remote_lease_duration"] == LEASE
    assert scheduler._reqs_need_send == {"r1": clock.now + LEASE}
    held, out = _request_finished(scheduler, _req("r2", dict(params)), [[]])
    assert not held and "remote_lease_duration" not in out

    # The scheduler leaves expiry to the lease and only caps renewals.
    output = SimpleNamespace(finished_sending=set())
    MoRIIOConnectorScheduler.update_connector_output(scheduler, output)
    assert not output.finished_sending
    deadline, transfer_id = scheduler._deferred_send_deadlines["r1"]
    scheduler._deferred_send_deadlines["r1"] = (deadline - 2 * 3600, transfer_id)
    MoRIIOConnectorScheduler.update_connector_output(scheduler, output)
    assert output.finished_sending == {"r1"}


def _decode_scheduler():
    vllm_config = create_vllm_config(role="kv_consumer", read_mode=True)
    vllm_config.kv_transfer_config.kv_connector_extra_config["defer_timeout"] = LEASE
    return create_scheduler(vllm_config)


def _waiting_request(scheduler, request_id: int, lease: float | None):
    block_size = scheduler.vllm_config.cache_config.block_size
    request = create_request(
        request_id=request_id,
        block_size=block_size,
        num_tokens=block_size * 2,
        do_remote_prefill=True,
    )
    request = _setup_kv_transfer_request(request, fake_transfer_id=f"tx-{request_id}")
    request.kv_transfer_params.update(remote_dp_rank=1, tp_size=2)
    if lease is not None:
        request.kv_transfer_params["remote_lease_duration"] = lease
    return request


def test_decode_renews_waiting_requests_until_it_reads_them(clock):
    scheduler = _decode_scheduler()
    connector = scheduler.connector.connector_scheduler
    target = ReadLeaseTarget("127.0.0.1", 4789, 1, 2, LEASE)

    def renewals():
        meta = MoRIIOConnectorMetadata()
        connector._package_read_lease_renewals(meta)
        return meta.read_lease_renewals

    # A prefill that does not advertise a lease predates renewals.
    scheduler.add_request(_waiting_request(scheduler, 1, lease=LEASE))
    scheduler.add_request(_waiting_request(scheduler, 2, lease=None))
    assert renewals() == {target: ["tx-1"]}

    # Throttled to one renewal per LEASE / 6.
    clock.now += LEASE / 6 - 1
    assert renewals() == {}
    clock.now += 1
    assert renewals() == {target: ["tx-1"]}

    # Scheduling allocates the request and posts its read: one last renewal
    # covers the read, then none.
    assert scheduler.schedule().kv_connector_metadata.read_lease_renewals == {
        target: ["tx-1"]
    }
    clock.now += LEASE
    assert scheduler.schedule().kv_connector_metadata.read_lease_renewals == {}


def test_paused_decode_keeps_stepping_to_renew_its_leases():
    scheduler = _decode_scheduler()
    scheduler.add_request(_waiting_request(scheduler, 1, lease=LEASE))
    scheduler.set_pause_state(PauseState.PAUSED_ALL)
    assert scheduler.get_num_unfinished_requests() == 0
    assert scheduler.has_requests()


def test_decode_tp_rank_0_renews_on_every_prefill_tp_rank():
    target = ReadLeaseTarget("10.0.0.1", 7000, dp_rank=1, tp_size=4, lease_duration=60)
    sent = {}
    for tp_rank in (0, 1):
        worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
        worker.tp_rank = tp_rank
        worker.world_size = 2
        worker.moriio_wrapper = MagicMock()
        worker._send_read_lease_renewals({target: ["tx-1", "tx-2"]})
        calls = worker.moriio_wrapper.send_lease_renewal.call_args_list
        sent[tp_rank] = [c.args for c in calls]

    assert sent[0] == [
        (["tx-1", "tx-2"], "10.0.0.1", 7000 + get_port_offset(1, rank, 4))
        for rank in range(4)
    ]
    assert sent[1] == []
