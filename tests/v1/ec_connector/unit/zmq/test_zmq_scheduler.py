# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the scheduler side of the ECZmqConnector.

The scheduler decides two things that the engine cannot recover from if they are
wrong: whether an item may be treated as already encoded (`has_cache_item`), and
whether a request may run before its embeddings arrived
(`ensure_cache_available`). Both are driven by the worker reports, so the tests
here drive them the same way the engine does.
"""

import pytest

from vllm.distributed.ec_transfer.ec_connector.zmq.common import (
    ECZmqWorkerMetadata,
    ZmqDst,
)
from vllm.distributed.ec_transfer.ec_connector.zmq.scheduler import ECZmqScheduler
from vllm.v1.outputs import ECConnectorOutput

pytestmark = pytest.mark.cpu_test


class _Feature:
    def __init__(self, identifier: str):
        self.identifier = identifier
        self.modality = "image"


class _Request:
    def __init__(self, *mm_hashes: str, ec_transfer_params=None, request_id="req0"):
        self.mm_features = [_Feature(mm_hash) for mm_hash in mm_hashes]
        self.ec_transfer_params = ec_transfer_params
        self.request_id = request_id


def _scheduler(vllm_config, role="ec_consumer", **kwargs) -> ECZmqScheduler:
    return ECZmqScheduler(vllm_config(role, **kwargs))


def _report_arrival(scheduler: ECZmqScheduler, *mm_hashes: str, ranks: int = 1) -> None:
    """Feed the scheduler what `ranks` worker ranks reported for `mm_hashes`."""
    for _ in range(ranks):
        scheduler.update_connector_output(
            ECConnectorOutput(
                ec_connector_worker_meta=ECZmqWorkerMetadata(
                    staged={mm_hash: 1 for mm_hash in mm_hashes}
                )
            )
        )


def test_item_is_available_only_once_every_rank_has_it(vllm_config):
    """Loading on a rank that is still waiting would read an empty staging."""
    scheduler = _scheduler(vllm_config, tensor_parallel_size=2)

    _report_arrival(scheduler, "mm0")
    assert scheduler.has_cache_item("mm0") is False

    _report_arrival(scheduler, "mm0")
    assert scheduler.has_cache_item("mm0") is True


def test_producer_never_claims_to_have_a_cached_item(vllm_config):
    scheduler = _scheduler(vllm_config, role="ec_producer")

    _report_arrival(scheduler, "mm0")

    assert scheduler.has_cache_item("mm0") is False


def test_declared_item_holds_the_request_back_until_it_arrives(vllm_config):
    scheduler = _scheduler(vllm_config)
    request = _Request("mm0", ec_transfer_params={"ec_items": [{"mm_hash": "mm0"}]})

    assert scheduler.ensure_cache_available(request, 0) is False

    _report_arrival(scheduler, "mm0")

    assert scheduler.ensure_cache_available(request, 0) is True


def test_an_undeclared_item_is_never_waited_for(vllm_config):
    """Without a declaration the engine encodes locally, as it always did."""
    scheduler = _scheduler(vllm_config)

    assert scheduler.ensure_cache_available(_Request("mm0"), 0) is True


def test_waiting_is_bounded(vllm_config):
    """A push that never comes must not park the request forever."""
    scheduler = _scheduler(vllm_config, extra_config={"ec_zmq_recv_timeout_s": 0.0})
    request = _Request("mm0", ec_transfer_params={"ec_items": [{"mm_hash": "mm0"}]})

    assert scheduler.ensure_cache_available(request, 0) is True


def test_wait_for_all_remote_holds_back_undeclared_items(vllm_config):
    scheduler = _scheduler(
        vllm_config, extra_config={"ec_zmq_wait_for_all_remote": True}
    )

    assert scheduler.ensure_cache_available(_Request("mm0"), 0) is False


def test_a_producer_does_not_hold_requests_back(vllm_config):
    scheduler = _scheduler(
        vllm_config,
        role="ec_producer",
        extra_config={"ec_zmq_wait_for_all_remote": True},
    )

    assert scheduler.ensure_cache_available(_Request("mm0"), 0) is True


def test_an_arrived_item_is_loaded_exactly_once(vllm_config):
    """The worker hands the embedding over once, so a second load would find
    nothing staged."""
    scheduler = _scheduler(vllm_config)
    request = _Request("mm0")
    _report_arrival(scheduler, "mm0")

    scheduler.update_state_after_alloc(request, 0)
    meta = scheduler.build_connector_meta(scheduler_output=None)

    assert meta.loads == ["mm0"]
    assert scheduler.has_cache_item("mm0") is False
    assert scheduler.build_connector_meta(scheduler_output=None).loads == []


def test_two_requests_in_one_step_load_the_item_once(vllm_config):
    scheduler = _scheduler(vllm_config)
    _report_arrival(scheduler, "mm0")

    scheduler.update_state_after_alloc(_Request("mm0", request_id="a"), 0)
    scheduler.update_state_after_alloc(_Request("mm0", request_id="b"), 0)

    assert scheduler.build_connector_meta(scheduler_output=None).loads == ["mm0"]


def test_producer_sends_to_the_configured_consumers(vllm_config):
    scheduler = _scheduler(
        vllm_config,
        role="ec_producer",
        extra_config={
            "ec_zmq_consumers": [
                {"host": "10.0.0.1", "port": 5000, "num_ranks": 2},
            ]
        },
    )

    scheduler.update_state_after_alloc(_Request("mm0"), 0)
    meta = scheduler.build_connector_meta(scheduler_output=None)

    assert meta.sends == {"mm0": [ZmqDst(host="10.0.0.1", port=5000, num_ranks=2)]}


def test_a_request_can_name_its_own_consumer(vllm_config):
    """One encoder fleet serving several consumers needs per-request routing."""
    scheduler = _scheduler(vllm_config, role="ec_producer")
    request = _Request(
        "mm0", ec_transfer_params={"ec_dst": {"host": "10.0.0.9", "port": 6000}}
    )

    scheduler.update_state_after_alloc(request, 0)
    meta = scheduler.build_connector_meta(scheduler_output=None)

    assert meta.sends == {"mm0": [ZmqDst(host="10.0.0.9", port=6000)]}


def test_a_malformed_destination_falls_back_to_the_configured_consumers(
    vllm_config, caplog
):
    scheduler = _scheduler(
        vllm_config,
        role="ec_producer",
        extra_config={"ec_zmq_consumers": [{"host": "10.0.0.1", "port": 5000}]},
    )
    request = _Request("mm0", ec_transfer_params={"ec_dst": {"host": "10.0.0.9"}})

    scheduler.update_state_after_alloc(request, 0)
    meta = scheduler.build_connector_meta(scheduler_output=None)

    assert meta.sends == {"mm0": [ZmqDst(host="10.0.0.1", port=5000)]}


def test_the_engine_keeps_stepping_until_pushes_complete(vllm_config):
    """Otherwise the engine could quiesce with an embedding still in flight."""
    scheduler = _scheduler(vllm_config, role="ec_producer")

    assert scheduler.has_pending_push_work() is False

    scheduler.update_state_after_alloc(_Request("mm0"), 0)
    assert scheduler.has_pending_push_work() is True

    scheduler.build_connector_meta(scheduler_output=None)
    assert scheduler.has_pending_push_work() is True

    scheduler.update_connector_output(ECConnectorOutput(finished_sending={"mm0"}))
    assert scheduler.has_pending_push_work() is False


def test_an_item_is_queued_for_delivery_once(vllm_config):
    scheduler = _scheduler(vllm_config, role="ec_producer")

    scheduler.update_state_after_alloc(_Request("mm0", request_id="a"), 0)
    scheduler.update_state_after_alloc(_Request("mm0", request_id="b"), 0)
    meta = scheduler.build_connector_meta(scheduler_output=None)

    assert list(meta.sends) == ["mm0"]


def test_an_ec_both_instance_delivers_what_it_encodes_and_loads_what_it_gets(
    vllm_config,
):
    scheduler = _scheduler(vllm_config, role="ec_both")
    _report_arrival(scheduler, "mm_remote")

    scheduler.update_state_after_alloc(_Request("mm_remote"), 0)
    scheduler.update_state_after_alloc(_Request("mm_local"), 0)
    meta = scheduler.build_connector_meta(scheduler_output=None)

    assert meta.loads == ["mm_remote"]
    assert list(meta.sends) == ["mm_local"]


def test_shutdown_forgets_everything(vllm_config):
    scheduler = _scheduler(vllm_config, role="ec_both")
    _report_arrival(scheduler, "mm0")
    scheduler.update_state_after_alloc(_Request("mm1"), 0)

    scheduler.shutdown()

    assert scheduler.has_cache_item("mm0") is False
    assert scheduler.has_pending_push_work() is False
