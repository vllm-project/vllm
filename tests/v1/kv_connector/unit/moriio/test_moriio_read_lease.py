# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""READ mode: prefill frees a request's blocks when decode's lease on them
expires, and decode renews that lease while the request waits."""

import threading
import time
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
    MoRIIOConstants,
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
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.request import RequestStatus

from ..utils import create_request, create_scheduler
from .test_moriio_connector import _setup_kv_transfer_request, create_vllm_config
from .test_moriio_routing import _producer, _req, _request_finished

LEASE = 60.0


class _Clock:
    """The connector's `time`: perf_counter and monotonic read `now`."""

    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def perf_counter(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


@pytest.fixture
def clock():
    clock = _Clock()
    with patch.object(moriio_connector, "time", clock):
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
    """A scheduler step: `held` maps transfer ids to the requests the
    scheduler still holds; `new` the lease expiries of newly finished ones."""
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
    clock.now += 2 * LEASE
    MoRIIOConnectorScheduler.update_connector_output(scheduler, output)
    assert not output.finished_sending
    clock.now += MoRIIOConstants.VLLM_MORI_READ_ABORT_REQUEST_TIMEOUT
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
    request.kv_transfer_params.update(
        remote_dp_rank=1, tp_size=2, remote_block_ids=[[7, 8]]
    )
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


def test_paused_decode_goes_idle_and_computes_lapsed_requests_locally(clock):
    """A pause completes only once the engine has no work; queued requests are
    never allocated while paused ("wait", checkpoint_prepare's default, or
    "keep"), so their leases must not keep it busy."""
    scheduler = _decode_scheduler()
    request = _waiting_request(scheduler, 1, lease=LEASE)
    scheduler.add_request(request)
    engine = EngineCoreProc.__new__(EngineCoreProc)
    engine.scheduler = scheduler
    engine.engines_running = False
    engine.batch_queue = None
    for pause in (PauseState.PAUSED_ALL, PauseState.PAUSED_NEW):  # keep, wait
        scheduler.set_pause_state(pause)
        assert engine._pause_complete()

    # Not renewed while paused, the prefill's lease lapses: after the pause the
    # request is computed locally, reads nothing and only releases the blocks.
    clock.now += LEASE
    scheduler.set_pause_state(PauseState.UNPAUSED)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens[request.request_id] == request.num_tokens
    meta = out.kv_connector_metadata
    assert meta.read_lease_renewals == {}
    assert not any(meta.reqs_to_recv[request.request_id].local_block_ids)


def test_decode_reads_a_request_whose_lease_was_renewed(clock):
    scheduler = _decode_scheduler()
    request = _waiting_request(scheduler, 1, lease=LEASE)
    scheduler.add_request(request)
    connector = scheduler.connector.connector_scheduler
    for _ in range(12):
        clock.now += LEASE / 6
        connector._package_read_lease_renewals(MoRIIOConnectorMetadata())

    out = scheduler.schedule()
    # READ recomputes only the last token.
    assert out.num_scheduled_tokens[request.request_id] == 1
    meta = out.kv_connector_metadata
    assert any(meta.reqs_to_recv[request.request_id].local_block_ids)


def test_decode_stops_relying_on_a_lease_near_the_renewal_cap(clock):
    """The prefill frees a request at its renewal cap however often it is
    renewed, so a request held that long is computed locally."""
    scheduler = _decode_scheduler()
    request = _waiting_request(scheduler, 1, lease=LEASE)
    scheduler.add_request(request)
    connector = scheduler.connector.connector_scheduler
    clock.now += MoRIIOConstants.VLLM_MORI_READ_ABORT_REQUEST_TIMEOUT - LEASE
    connector._package_read_lease_renewals(MoRIIOConnectorMetadata())

    out = scheduler.schedule()
    assert out.num_scheduled_tokens[request.request_id] == request.num_tokens


def test_decode_renews_a_new_lease_in_its_next_step(clock):
    """Prefill's lease started when the prefill finished, before decode
    admitted the request, so decode renews it at once, not an interval later."""
    scheduler = _decode_scheduler()
    connector = scheduler.connector.connector_scheduler
    target = ReadLeaseTarget("127.0.0.1", 4789, 1, 2, LEASE)
    scheduler.add_request(_waiting_request(scheduler, 1, lease=LEASE))
    connector._package_read_lease_renewals(MoRIIOConnectorMetadata())

    clock.now += 1
    scheduler.add_request(_waiting_request(scheduler, 2, lease=LEASE))
    meta = MoRIIOConnectorMetadata()
    connector._package_read_lease_renewals(meta)
    assert meta.read_lease_renewals == {target: ["tx-2"]}


def test_decode_reads_a_new_request_whose_lease_exceeds_the_cap(clock):
    scheduler = _decode_scheduler()
    lease = 2 * MoRIIOConstants.VLLM_MORI_READ_ABORT_REQUEST_TIMEOUT
    request = _waiting_request(scheduler, 1, lease=lease)
    scheduler.add_request(request)
    assert scheduler.schedule().num_scheduled_tokens[request.request_id] == 1


def test_decode_abort_while_queued_stops_renewing(clock):
    scheduler = _decode_scheduler()
    request = _waiting_request(scheduler, 1, lease=LEASE)
    scheduler.add_request(request)
    connector = scheduler.connector.connector_scheduler
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)

    assert connector._read_leases == {}
    clock.now += LEASE
    meta = MoRIIOConnectorMetadata()
    connector._package_read_lease_renewals(meta)
    assert meta.read_lease_renewals == {}


def test_decode_does_not_track_an_infinite_lease():
    scheduler = _decode_scheduler()
    scheduler.add_request(_waiting_request(scheduler, 1, lease=float("inf")))
    assert scheduler.connector.connector_scheduler._read_leases == {}


def test_malformed_renewal_does_not_stop_the_notify_listener():
    wrapper = _prefill_worker().moriio_wrapper
    _send(wrapper, {"type": "renew_lease", "transfer_ids": [["tx-1"], "tx-2"]})
    assert wrapper.pop_renewed_lease_transfer_ids() == {"tx-2"}
    # Misrouted to a decode instance.
    with patch.object(moriio_engine, "get_role", return_value=ROLE.CONSUMER):
        wrapper._handle_structured_message(
            {"type": "renew_lease", "transfer_ids": ["tx-3"]}
        )
    assert wrapper.pop_renewed_lease_transfer_ids() == set()


def test_decode_tracks_a_lease_with_a_null_dp_rank():
    scheduler = _decode_scheduler()
    request = _waiting_request(scheduler, 1, lease=LEASE)
    request.kv_transfer_params["remote_dp_rank"] = None
    scheduler.add_request(request)
    target, _ = scheduler.connector.connector_scheduler._read_leases[request.request_id]
    assert target.dp_rank == 0


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
