# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WRITE-mode consumer receive timeout of the MoRIIO connector.

In WRITE mode the decode side parks a request in ``WAITING_FOR_REMOTE_KVS``
until the producer's ``write_done`` arrives. If the producer dies that never
happens, so the scheduler arms a per-request timer, configured by
``kv_connector_extra_config["write_recv_timeout"]`` (seconds, default 540).
These tests pin the configuration and the lifecycle:

  * the timeout is read from ``kv_connector_extra_config``, validated at
    startup;
  * the timer is armed with the request's *own* (unhashed) blocks only, so a
    prefix-cache hit shared with another request is never reported;
  * a timeout fails a live request but never frees its blocks -- RDMA writes
    are one-sided, so a late write can still land (vllm-project/vllm#59382);
  * a client abort keeps the timer running, and at the deadline the request is
    logged as orphaned instead of failed a second time (#59114);
  * the timer stops when the receive completes, not when the request finishes,
    so a request that already has its KV is never failed while it decodes.

They run on the scheduler alone, with no GPU and no RDMA.
"""

import importlib
import math
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.request import RequestStatus

from .test_moriio_connector import _setup_kv_transfer_request
from .test_moriio_connector import create_vllm_config as create_moriio_config
from .utils import create_model_runner_output, create_request, create_scheduler

# Scheduler-side logic only: no GPU, no RDMA, no ``mori`` package needed, so
# (unlike test_moriio_connector.py) this runs in the CPU CI job.
pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

moriio_common = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common"
)
moriio_connector = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector"
)

MoRIIOMode = moriio_common.MoRIIOMode
MoRIIOConnectorScheduler = moriio_connector.MoRIIOConnectorScheduler
MoRIIOConnectorWorker = moriio_connector.MoRIIOConnectorWorker

TIMEOUT_S = 60.0
REQ_ID = "decode-req-0"


@pytest.fixture(autouse=True)
def restore_moriio_role():
    """``build_connector_meta`` reads the process-global role."""
    role = moriio_common.get_role()
    moriio_common.set_role(moriio_common.ROLE.CONSUMER)
    yield
    moriio_common.set_role(role)


@pytest.fixture
def log(monkeypatch):
    """Replace the module logger so tests can assert on what was reported."""
    mock_logger = MagicMock()
    monkeypatch.setattr(moriio_connector, "logger", mock_logger)
    return mock_logger


def _make_scheduler(timeout: float = TIMEOUT_S) -> MoRIIOConnectorScheduler:
    """A WRITE-mode consumer scheduler without the heavy real ``__init__``."""
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.is_producer = False
    scheduler.mode = MoRIIOMode.WRITE
    scheduler.tp_size = 1
    scheduler._is_kv_master = True
    scheduler._global_dp_rank = 0
    scheduler._has_mamba = False
    scheduler._reqs_need_recv = {}
    scheduler._reqs_need_save = {}
    scheduler._reqs_need_send = {}
    scheduler._reqs_need_pending_save = {}
    scheduler._req_kv_params = {}
    scheduler.transfer_id_to_request_id = {}
    scheduler._write_recv_deadline = {}
    scheduler._write_recv_block_ids = {}
    scheduler._write_recv_orphaned = set()
    scheduler._write_load_failed_block_ids = []
    scheduler._write_recv_timeout = timeout
    scheduler._write_recv_single_group = True
    scheduler.send_notify_block = MagicMock()
    scheduler._release_write_prefill_blocks = MagicMock()
    scheduler.map_request_id = MagicMock()
    scheduler.unmap_request_id = MagicMock()
    return scheduler


def _make_blocks(
    own_ids: list[int], shared_ids: list[int] | None = None
) -> KVCacheBlocks:
    """Allocation of one KV cache group.

    ``shared_ids`` are prefix-cache hits: full blocks that carry a hash and are
    referenced by other requests too. ``own_ids`` are the request's own
    unhashed blocks.
    """
    shared = [KVCacheBlock(block_id=i) for i in shared_ids or []]
    for block in shared:
        block.set_block_hash(
            make_block_hash_with_group_id(BlockHash(b"hash-%d" % block.block_id), 0)
        )
    own = [KVCacheBlock(block_id=i) for i in own_ids]
    return KVCacheBlocks(blocks=(shared + own,))


def _make_request(request_id: str = REQ_ID):
    # A plain request id carries no router-embedded ZMQ address, so the notify
    # target comes from kv_transfer_params, as with the routing sidecar.
    return SimpleNamespace(
        request_id=request_id,
        status=RequestStatus.WAITING_FOR_REMOTE_KVS,
        kv_transfer_params={
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "transfer_id": f"xfer-{request_id}",
            "remote_host": "127.0.0.1",
            "remote_notify_port": 7000,
        },
    )


def _arm(
    scheduler: MoRIIOConnectorScheduler,
    request=None,
    blocks: KVCacheBlocks | None = None,
    num_external_tokens: int = 32,
):
    request = request or _make_request()
    blocks = blocks or _make_blocks(own_ids=[10, 11])
    scheduler.update_state_after_alloc(request, blocks, num_external_tokens)
    return request


def _abort(scheduler: MoRIIOConnectorScheduler, request):
    """The client gave up while the request was still waiting for its KV."""
    request.status = RequestStatus.FINISHED_ABORTED
    return scheduler.request_finished(request, block_ids=[[10, 11]])


def _expire(scheduler: MoRIIOConnectorScheduler, request_id: str = REQ_ID):
    """Move a request's deadline into the past instead of sleeping."""
    scheduler._write_recv_deadline[request_id] = time.monotonic() - 1.0


# --- configuration --------------------------------------------------------


def _build_real_scheduler(
    extra_config: dict | None = None,
    kv_load_failure_policy: str = "fail",
    num_groups: int = 1,
):
    """Run the real ``__init__`` (full-attention groups, WRITE mode)."""
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    groups = [
        SimpleNamespace(enable_kv_transfer=True, kv_cache_spec=spec)
        for _ in range(num_groups)
    ]
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=groups, transfer_groups=tuple(groups)
    )
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "read_mode": False,
                "host_ip": "127.0.0.1",
                "handshake_port": 6001,
                "notify_port": 6002,
                **(extra_config or {}),
            },
            kv_role="kv_consumer",
            kv_load_failure_policy=kv_load_failure_policy,
        ),
        speculative_config=None,
        cache_config=SimpleNamespace(block_size=16, mamba_cache_mode="all"),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
        num_lookahead_tokens=0,
        num_speculative_tokens=0,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_rank=0,
            data_parallel_size_local=1,
            data_parallel_size=1,
        ),
    )
    return MoRIIOConnectorScheduler(vllm_config, "engine", kv_cache_config)


def test_default_timeout_is_the_documented_constant():
    scheduler = _build_real_scheduler()
    assert scheduler._write_recv_timeout == 540.0
    assert (
        scheduler._write_recv_timeout
        == moriio_common.MoRIIOConstants.DEFAULT_WRITE_RECV_TIMEOUT
    )


@pytest.mark.parametrize("value", [60, "60"])
def test_timeout_is_read_from_kv_connector_extra_config(value):
    scheduler = _build_real_scheduler({"write_recv_timeout": value})
    assert scheduler._write_recv_timeout == float(value)


@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan])
def test_invalid_timeout_is_rejected_at_startup(value):
    """A non-finite or non-positive value would make every request time out."""
    with pytest.raises(ValueError, match="write_recv_timeout"):
        _build_real_scheduler({"write_recv_timeout": value})


def test_timer_is_disabled_with_several_kv_cache_groups(log):
    """The scheduler raises on block-level load errors with several groups."""
    scheduler = _build_real_scheduler(num_groups=2)
    messages = [str(c.args[0]) for c in log.warning.call_args_list]
    assert any("write_recv_timeout is disabled" in m for m in messages)

    scheduler._arm_write_recv_timer(REQ_ID, _make_blocks(own_ids=[10, 11]), 32)
    assert REQ_ID not in scheduler._write_recv_deadline


def test_timer_is_armed_with_a_single_kv_cache_group(log):
    scheduler = _build_real_scheduler()
    assert not any(
        "write_recv_timeout is disabled" in str(c.args[0])
        for c in log.warning.call_args_list
    )

    scheduler._arm_write_recv_timer(REQ_ID, _make_blocks(own_ids=[10, 11]), 32)
    assert scheduler._write_recv_block_ids[REQ_ID] == [10, 11]


# --- arming ---------------------------------------------------------------


def test_timer_records_only_the_requests_own_blocks():
    """Prefix-cache hits are shared with other requests.

    Reporting them at the deadline would fail those requests too.
    """
    scheduler = _make_scheduler()
    blocks = _make_blocks(own_ids=[30, 31], shared_ids=[1, 2, 3])

    _arm(scheduler, blocks=blocks)

    assert scheduler._write_recv_block_ids[REQ_ID] == [30, 31]
    # The producer is still told to write every block it has to fill.
    notified = scheduler.send_notify_block.call_args.kwargs["block_notify_list"]
    assert notified == [1, 2, 3, 30, 31]


def test_timer_is_armed_on_a_scheduler_that_does_not_send_the_notify():
    """Only the DP leader notifies the producer, but every scheduler holding
    the request still waits for write_done, so each needs its own timer."""
    scheduler = _make_scheduler()
    scheduler._is_kv_master = False
    _arm(scheduler)
    scheduler.send_notify_block.assert_not_called()
    assert REQ_ID in scheduler._write_recv_deadline
    assert scheduler._write_recv_block_ids[REQ_ID] == [10, 11]


def test_fully_cached_request_arms_no_timer():
    """With no own blocks there is nothing to wait for or to fail."""
    scheduler = _make_scheduler()

    _arm(scheduler, blocks=_make_blocks(own_ids=[], shared_ids=[1, 2]))

    assert REQ_ID not in scheduler._write_recv_deadline
    assert REQ_ID not in scheduler._write_recv_block_ids


def test_no_external_tokens_arms_no_timer():
    """A full local prefix hit releases the producer instead of waiting."""
    scheduler = _make_scheduler()

    _arm(scheduler, num_external_tokens=0)

    assert scheduler._write_recv_deadline == {}
    scheduler._release_write_prefill_blocks.assert_called_once()
    scheduler.send_notify_block.assert_not_called()


# --- deadline fires on a live request ---------------------------------------


def test_deadline_fails_a_live_request_and_reports_its_blocks(log):
    scheduler = _make_scheduler()
    _arm(scheduler, blocks=_make_blocks(own_ids=[10, 11]))
    _expire(scheduler)

    scheduler._reap_stale_write_recvs()

    assert scheduler._write_load_failed_block_ids == [10, 11]
    assert REQ_ID not in scheduler._write_recv_deadline
    log.error.assert_called_once()
    assert "TIMED OUT" in log.error.call_args.args[0]
    log.warning.assert_not_called()


def test_request_before_its_deadline_is_left_alone(log):
    scheduler = _make_scheduler()
    _arm(scheduler)

    scheduler._reap_stale_write_recvs()

    assert scheduler._write_load_failed_block_ids == []
    assert REQ_ID in scheduler._write_recv_deadline
    log.error.assert_not_called()
    log.warning.assert_not_called()


def test_expired_request_is_reaped_exactly_once():
    scheduler = _make_scheduler()
    _arm(scheduler)
    _expire(scheduler)

    scheduler._reap_stale_write_recvs()
    scheduler._reap_stale_write_recvs()

    assert scheduler._write_load_failed_block_ids == [10, 11]


# --- client abort keeps the timer (#59114) ----------------------------------


def test_client_abort_keeps_the_timer_and_marks_the_request_orphaned():
    scheduler = _make_scheduler()
    request = _arm(scheduler)

    delay_free, params = _abort(scheduler, request)

    assert not delay_free
    assert params is None
    assert REQ_ID in scheduler._write_recv_deadline
    assert REQ_ID in scheduler._write_recv_block_ids
    assert REQ_ID in scheduler._write_recv_orphaned


def test_orphaned_request_at_deadline_is_logged_not_failed_again(log):
    scheduler = _make_scheduler()
    request = _arm(scheduler)
    _abort(scheduler, request)
    _expire(scheduler)

    scheduler._reap_stale_write_recvs()

    # The client is already gone, so a second terminal outcome would be wrong.
    assert scheduler._write_load_failed_block_ids == []
    log.error.assert_not_called()
    log.warning.assert_called_once()
    assert "still incomplete" in log.warning.call_args.args[0]
    assert "stay allocated" in log.warning.call_args.args[0]
    # Fully forgotten afterwards.
    assert REQ_ID not in scheduler._write_recv_orphaned
    assert REQ_ID not in scheduler._write_recv_deadline


# --- receive completion stops the timer -------------------------------------


def test_finished_recving_stops_the_timer():
    scheduler = _make_scheduler()
    _arm(scheduler)

    scheduler.update_connector_output(
        SimpleNamespace(finished_recving={REQ_ID}, finished_sending=None)
    )

    assert REQ_ID not in scheduler._write_recv_deadline
    assert REQ_ID not in scheduler._write_recv_block_ids


def test_request_that_received_its_kv_is_not_failed_while_decoding(log):
    """A healthy request may sit past its original deadline while decoding.

    Failing it then would roll back a request that is working.
    """
    scheduler = _make_scheduler()
    _arm(scheduler)
    # The deadline passes, but the receive completed first.
    _expire(scheduler)
    scheduler.update_connector_output(
        SimpleNamespace(finished_recving={REQ_ID}, finished_sending=None)
    )

    scheduler._reap_stale_write_recvs()

    assert scheduler._write_load_failed_block_ids == []
    log.error.assert_not_called()
    log.warning.assert_not_called()


def test_finished_recving_for_an_orphan_clears_the_orphan_mark():
    scheduler = _make_scheduler()
    request = _arm(scheduler)
    _abort(scheduler, request)

    scheduler.update_connector_output(
        SimpleNamespace(finished_recving={REQ_ID}, finished_sending=None)
    )

    assert scheduler._write_recv_orphaned == set()
    assert scheduler._write_recv_deadline == {}


# --- scheduler -> worker hand-off -------------------------------------------


def test_build_connector_meta_ships_timed_out_blocks_once():
    scheduler = _make_scheduler()
    _arm(scheduler, blocks=_make_blocks(own_ids=[10, 11]))
    _expire(scheduler)

    first = scheduler.build_connector_meta(SimpleNamespace())
    second = scheduler.build_connector_meta(SimpleNamespace())

    assert first.write_load_failed_block_ids == [10, 11]
    # Drained: the worker must not be told about the same blocks again.
    assert second.write_load_failed_block_ids == []


def test_worker_drains_timed_out_blocks_through_the_load_error_set():
    """Timed-out blocks share the READ load-failure drain."""
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker._invalid_block_ids = {10, 11}

    assert worker.get_block_ids_with_load_errors() == {10, 11}
    # Drained: a second call reports nothing new.
    assert worker.get_block_ids_with_load_errors() == set()


# --- through the real scheduler -------------------------------------------
#
# The tests above call connector methods directly. These run the real vLLM
# ``Scheduler``, block pool and ``MoRIIOConnectorScheduler``, so they also pin
# what the scheduler does with a reported timeout: the request is failed, but
# its blocks stay allocated until a real ``finished_recving`` arrives.


class _Clock:
    """``time`` stand-in whose ``monotonic()`` a test can advance.

    Only the connector module sees it, so a test can cross a deadline without
    sleeping and without skewing any other user of ``time.monotonic``.
    """

    def __init__(self):
        self.now = time.monotonic()

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __getattr__(self, name):
        return getattr(time, name)


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(moriio_connector, "time", fake)
    return fake


@pytest.fixture
def consumer(monkeypatch, clock):
    """Real WRITE-consumer scheduler with a 60s receive timeout."""
    config = create_moriio_config(role="kv_consumer")
    config.kv_transfer_config.kv_connector_extra_config["write_recv_timeout"] = (
        TIMEOUT_S
    )
    scheduler = create_scheduler(config)
    connector = scheduler.connector.connector_scheduler
    # The notify goes to the producer over ZMQ; there is none here.
    monkeypatch.setattr(connector, "send_notify_block", lambda **kwargs: None)
    return scheduler, connector


def _step(
    scheduler,
    finished_recving: set[str] | None = None,
    failed_recving: set[str] | None = None,
) -> set[int]:
    """One engine step. Returns the block ids reported as timed out.

    The worker is simulated the way the real one behaves: block ids the
    scheduler connector put in the metadata come back as ``invalid_block_ids``
    (the worker adds them to ``_invalid_block_ids``, and
    ``get_block_ids_with_load_errors`` returns them in the same step).
    """
    out = scheduler.schedule()
    timed_out = set(out.kv_connector_metadata.write_load_failed_block_ids)
    scheduled = [scheduler.requests[i] for i in out.num_scheduled_tokens]
    if finished_recving or timed_out or scheduled:
        runner_output = create_model_runner_output(
            scheduled,
            finished_recving=finished_recving,
            invalid_block_ids=timed_out or None,
        )
    else:
        runner_output = EMPTY_MODEL_RUNNER_OUTPUT
    if failed_recving:
        runner_output.kv_connector_output.failed_recving = failed_recving
    scheduler.update_from_output(out, runner_output)
    return timed_out


def _add_waiting_request(scheduler, transfer_id: str = "xfer-0"):
    """A decode request that has allocated its blocks and awaits the KV."""
    request = create_request(
        num_tokens=4 * scheduler.block_size, do_remote_prefill=True
    )
    _setup_kv_transfer_request(request, fake_transfer_id=transfer_id)
    scheduler.add_request(request)
    _step(scheduler)
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    return request


def test_scheduler_fails_a_timed_out_request_and_keeps_its_blocks(consumer, clock):
    scheduler, connector = consumer
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()
    request = _add_waiting_request(scheduler)
    request_id = request.request_id
    held = pool.get_num_free_blocks()
    assert held < free_before
    assert request_id in connector._write_recv_deadline

    # Not yet due: the request keeps waiting.
    clock.advance(TIMEOUT_S - 1)
    assert _step(scheduler) == set()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    # Due: the connector reports its blocks and the scheduler fails it.
    clock.advance(2)
    assert _step(scheduler)
    assert request.status == RequestStatus.FINISHED_ERROR
    # Failing the request must not free its blocks: the producer may still
    # write into them (vllm-project/vllm#59382).
    assert request_id in scheduler.requests
    assert pool.get_num_free_blocks() == held
    # It is reported once; later steps do not report it again.
    assert _step(scheduler) == set()
    assert pool.get_num_free_blocks() == held

    # Only a real finished_recving releases them, exactly once.
    _step(scheduler, finished_recving={request_id})
    assert request_id not in scheduler.requests
    assert pool.get_num_free_blocks() == free_before


def test_scheduler_does_not_fail_an_aborted_request_a_second_time(consumer, clock, log):
    scheduler, connector = consumer
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()
    request = _add_waiting_request(scheduler)
    request_id = request.request_id
    held = pool.get_num_free_blocks()

    # The client gives up while the request waits (#59114). The scheduler keeps
    # the blocks, and the connector must keep timing the receive.
    scheduler.finish_requests(request_id, RequestStatus.FINISHED_ABORTED)
    _step(scheduler)
    assert request.status == RequestStatus.FINISHED_ABORTED
    assert request_id in scheduler.requests
    assert pool.get_num_free_blocks() == held
    assert request_id in connector._write_recv_deadline
    assert request_id in connector._write_recv_orphaned

    clock.advance(TIMEOUT_S + 1)
    assert _step(scheduler) == set()
    assert request.status == RequestStatus.FINISHED_ABORTED
    assert pool.get_num_free_blocks() == held
    assert not log.error.called
    assert any("still incomplete" in str(c.args[0]) for c in log.warning.call_args_list)
    assert request_id not in connector._write_recv_deadline
    assert request_id not in connector._write_recv_orphaned

    _step(scheduler, finished_recving={request_id})
    assert request_id not in scheduler.requests
    assert pool.get_num_free_blocks() == free_before


def test_scheduler_never_fails_a_request_that_received_its_kv(consumer, clock):
    scheduler, connector = consumer
    request = _add_waiting_request(scheduler)
    request_id = request.request_id

    _step(scheduler, finished_recving={request_id})
    assert request_id not in connector._write_recv_deadline

    # Well past the deadline the request is still decoding, untouched.
    clock.advance(10 * TIMEOUT_S)
    for _ in range(3):
        assert _step(scheduler) == set()
        assert request.status == RequestStatus.RUNNING
        assert request.num_output_tokens > 0


def test_scheduler_fails_only_the_request_whose_kv_never_arrived(consumer, clock):
    scheduler, connector = consumer
    pool = scheduler.kv_cache_manager.block_pool
    received = _add_waiting_request(scheduler, "xfer-received")
    stuck = _add_waiting_request(scheduler, "xfer-stuck")
    stuck_blocks = {
        b.block_id
        for b in scheduler.kv_cache_manager.coordinator.single_type_managers[
            0
        ].req_to_blocks[stuck.request_id]
    }
    held = pool.get_num_free_blocks()

    _step(scheduler, finished_recving={received.request_id})
    clock.advance(TIMEOUT_S + 1)
    timed_out = _step(scheduler)

    assert timed_out and timed_out <= stuck_blocks
    assert stuck.status == RequestStatus.FINISHED_ERROR
    assert received.status == RequestStatus.RUNNING
    assert received.request_id in scheduler.requests
    # The stuck request's blocks stay held; the healthy one is still decoding.
    assert pool.get_num_free_blocks() <= held


def test_scheduler_releases_an_aborted_request_whose_kv_arrives_in_time(
    consumer, clock, log
):
    """Abort, then the producer's write_done lands before the deadline."""
    scheduler, connector = consumer
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()
    request = _add_waiting_request(scheduler)
    request_id = request.request_id

    scheduler.finish_requests(request_id, RequestStatus.FINISHED_ABORTED)
    _step(scheduler)
    assert request_id in connector._write_recv_orphaned

    clock.advance(TIMEOUT_S - 1)
    _step(scheduler, finished_recving={request_id})
    # The late completion releases the blocks exactly once and ends the timer,
    # so the deadline passing later is silent.
    assert request_id not in scheduler.requests
    assert pool.get_num_free_blocks() == free_before
    assert request_id not in connector._write_recv_deadline
    assert request_id not in connector._write_recv_orphaned

    clock.advance(10 * TIMEOUT_S)
    assert _step(scheduler) == set()
    assert not log.warning.called
    assert not log.error.called


def test_scheduler_does_not_time_out_a_request_the_producer_reported_failed(
    consumer, clock, log
):
    """A producer ``write_failed`` (#57705) is reported as finished_recving
    plus failed_recving, which ends the receive and so stops the timer."""
    scheduler, connector = consumer
    request = _add_waiting_request(scheduler)
    request_id = request.request_id

    _step(scheduler, finished_recving={request_id}, failed_recving={request_id})
    assert request_id not in connector._write_recv_deadline
    assert request.status == RequestStatus.FINISHED_ERROR

    clock.advance(10 * TIMEOUT_S)
    assert _step(scheduler) == set()
    assert not log.error.called
    assert not any(
        "still incomplete" in str(c.args[0]) for c in log.warning.call_args_list
    )
