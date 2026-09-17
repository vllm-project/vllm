# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual producer bindings must survive until consumer completion."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeLoadSpec,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.scheduler import (
    ECMooncakeScheduler,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.state import (
    SchedulerTransferState,
    SchedulerTransferTable,
)


def scheduler(producer):
    s = object.__new__(ECMooncakeScheduler)
    s._is_producer, s._is_consumer = producer, not producer
    s._prepared_pushes = {}
    s._pushes_to_prepare = {}
    s._local_cache = set()
    s._pending_load_hashes = {}
    s._failed_saves = set()

    s._model_config = SimpleNamespace(dtype=torch.float16)
    s._encoder_cache_hidden_dim = 4
    s._unresolved_transfer_ids = 0
    s._metadata_resolver = None  # Features lack placeholder data.
    s._transfers = SchedulerTransferTable(1024, 600)
    s._pending_cancels = {}
    s._event_ready_shards = {}
    s._control_addr = "tcp://consumer"
    s._control_executor = Mock()
    s._drain_push_notifications = lambda: None
    s._push_wait_timeout = 30
    return s


def request(hashes, params):
    return SimpleNamespace(
        request_id="r",
        mm_features=[
            SimpleNamespace(
                identifier=h, data=None, mm_position=SimpleNamespace(offset=0, length=2)
            )
            for h in hashes
        ],
        ec_transfer_params=params,
        get_num_encoder_embeds=lambda index: 2,
    )


@pytest.mark.parametrize("hashes", [["video", "audio"], ["same", "same"]])
def test_response_reports_every_prepared_transfer(hashes):
    s = scheduler(True)
    req = request(
        hashes,
        {
            "consumer_zmq": "tcp://consumer",
            "ec_items": [{"transfer_id": "proxy-transfer"}],
        },
    )
    for i in range(len(hashes)):
        s.update_state_after_alloc(req, i)
    expected = [
        {"mm_hash": spec.mm_hash, "transfer_id": spec.transfer_id}
        for spec in s._pushes_to_prepare.values()
    ]
    # Dispatch removes the pending list; completion must retain the bindings.
    s._pushes_to_prepare.clear()
    _, params = s.request_finished(req)
    assert params["ec_items"] == expected
    assert not s._prepared_pushes
    consumer = scheduler(False)
    pd = request(list(reversed(hashes)), params)
    for item in expected:
        consumer._transfers.observe_ready(
            ECMooncakeLoadSpec(
                item["mm_hash"], 16, (2, 4), "float16", item["transfer_id"]
            ),
            1000,
        )
    assert not consumer.ensure_cache_available(pd, 0)
    loads = consumer._transfers.take_loads_to_dispatch()
    assert {record.mm_hash for record in loads} == set(hashes)
    assert {record.transfer_id for record in loads} <= {
        item["transfer_id"] for item in expected
    }


@pytest.mark.parametrize("consumed", [False, True])
def test_finish_cancels_unconsumed_bindings_even_without_matching_feature(consumed):
    s = scheduler(False)
    bindings = [{"mm_hash": "same", "transfer_id": t} for t in ("used", "extra")]
    req = request(["same"], {"ec_items": bindings})
    if consumed:
        s._transfers.observe_ready(
            ECMooncakeLoadSpec("same", 16, (2, 4), "float16", "used"), 1000
        )
        s._transfers.begin_load("same", "used", "r", 1000)
        s._transfers.complete_load("same")
    s.request_finished(req)
    assert s._transfers.get("extra").state is SchedulerTransferState.CANCELLED
    cancelled = {call.args[2] for call in s._control_executor.submit.call_args_list}
    assert cancelled == ({"extra"} if consumed else {"used", "extra"})
    if consumed:
        assert s._transfers.get("used").state is SchedulerTransferState.READY
    else:
        assert s._transfers.get("used").state is SchedulerTransferState.CANCELLED


def test_repeated_transfer_binding_is_prepared_once():
    s = scheduler(True)
    req = request(
        ["same", "same"],
        {
            "consumer_zmq": "tcp://consumer",
            "ec_items": [{"mm_hash": "same", "transfer_id": "shared"}],
        },
    )
    s.update_state_after_alloc(req, 0)
    s.update_state_after_alloc(req, 1)
    assert len(s._pushes_to_prepare) == 1
    _, params = s.request_finished(req)
    assert params["ec_items"] == [{"mm_hash": "same", "transfer_id": "shared"}]


def test_finish_does_not_report_an_unprepared_request_binding():
    s = scheduler(True)
    req = request(["image"], {"ec_items": [{"transfer_id": "not-prepared"}]})
    _, params = s.request_finished(req)
    assert params == {"image": {"metadata": {}}}


@pytest.mark.parametrize("complete", [True, False])
def test_finish_preserves_extra_transfer_loading_for_another_request(complete):
    s = scheduler(False)
    owner = request(
        ["same"],
        {"ec_items": [{"mm_hash": "same", "transfer_id": t} for t in ("own", "extra")]},
    )
    borrower = request(
        ["same"],
        {"ec_items": [{"mm_hash": "same", "transfer_id": "borrower-transfer"}]},
    )
    borrower.request_id = "borrower"
    s._transfers.observe_ready(
        ECMooncakeLoadSpec("same", 16, (2, 4), "float16", "extra"), 1000
    )
    assert not s.ensure_cache_available(borrower, 0)
    assert s._transfers.get("extra").state is SchedulerTransferState.LOADING

    s.request_finished(owner)
    assert s._transfers.get("extra").state is SchedulerTransferState.LOADING
    assert {call.args[2] for call in s._control_executor.submit.call_args_list} == {
        "own"
    }
    if complete:
        assert s._transfers.complete_load("same")
    else:
        s._transfers.get("extra").deadline = 0
        s._expire_transfers()
        assert s._transfers.get("extra").state is SchedulerTransferState.CANCELLED
        assert s.take_unavailable_requests() == {"borrower"}
        assert {call.args[2] for call in s._control_executor.submit.call_args_list} == {
            "own",
            "extra",
        }


@pytest.mark.parametrize("first_finished", [0, 1])
def test_shared_transfer_is_reported_to_both_requests(first_finished):
    s = scheduler(True)
    binding = {"mm_hash": "same", "transfer_id": "shared"}
    requests = [
        request(
            ["same"],
            {
                "consumer_zmq": "tcp://consumer",
                "ec_items": [binding],
            },
        )
        for _ in range(2)
    ]
    for i, req in enumerate(requests):
        req.request_id = f"r{i}"
        s.update_state_after_alloc(req, 0)
    assert len(s._pushes_to_prepare) == 1
    s._pushes_to_prepare.clear()

    _, params = s.request_finished(requests[first_finished])
    assert params["ec_items"] == [binding]
    remaining = requests[1 - first_finished]
    s.update_state_after_alloc(remaining, 0)
    assert not s._pushes_to_prepare
    _, params = s.request_finished(remaining)
    assert params["ec_items"] == [binding]
    assert not s._prepared_pushes


def test_conflicting_transfer_is_not_reported_as_a_shared_binding():
    s = scheduler(True)
    first = request(
        ["first"],
        {
            "consumer_zmq": "tcp://consumer",
            "ec_items": [{"transfer_id": "shared"}],
        },
    )
    second = request(["different"], first.ec_transfer_params)
    second.request_id = "second"
    s.update_state_after_alloc(first, 0)
    s.update_state_after_alloc(second, 0)
    assert len(s._pushes_to_prepare) == 1
    assert s.take_unavailable_requests() == {"second"}
    _, params = s.request_finished(second)
    assert params == {"different": {"metadata": {}}}
    _, params = s.request_finished(first)
    assert params["ec_items"] == [{"mm_hash": "first", "transfer_id": "shared"}]


@pytest.mark.parametrize("cleanup", ["finish", "free"])
def test_primary_binding_borrowed_by_another_request_survives_cleanup(cleanup):
    s = scheduler(False)
    owner = request(["same"], {"ec_items": [{"transfer_id": "own"}]})
    borrower = request(["same"], {"ec_items": [{"transfer_id": "borrower"}]})
    borrower.request_id = "borrower"
    s._transfers.observe_ready(
        ECMooncakeLoadSpec("same", 16, (2, 4), "float16", "own"), 1000
    )
    assert not s.ensure_cache_available(borrower, 0)
    if cleanup == "finish":
        s.request_finished(owner)
    else:
        s.update_state_after_free(owner, 0)
    assert s._transfers.get("own").state is SchedulerTransferState.LOADING
    s._control_executor.submit.assert_not_called()
    assert s._transfers.complete_load("same")


def consumer_worker(capacity=256):
    from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import (
        ConsumerMemoryPool,
    )
    from vllm.distributed.ec_transfer.ec_connector.mooncake.reservation import (
        ConsumerReservationManager,
    )
    from vllm.distributed.ec_transfer.ec_connector.mooncake.worker import (
        ECMooncakeWorker,
    )

    transfer = Mock()
    transfer.register_memory.return_value = 0
    pool = ConsumerMemoryPool(capacity, transfer)
    pool.prepare(torch.device("cpu"))
    reservations = ConsumerReservationManager(pool, 600, 16)
    worker = object.__new__(ECMooncakeWorker)
    worker._resolve_consumer_rank = lambda: None
    worker._is_receiving_rank = True
    worker._buffer_device = "cpu"
    worker._transfer = transfer
    worker._reservations = reservations
    worker._consumer_memory = pool
    worker._completed_loads, worker._failed_loads = set(), set()
    cache: dict[str, torch.Tensor] = {}

    return worker, pool, reservations, cache


@pytest.mark.parametrize("allocated", [False, True])
def test_completed_load_has_bounded_residency_after_abort_or_eviction(allocated):
    from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
        ECMooncakeWorkerMetadata,
    )

    worker, pool, reservations, cache = consumer_worker()
    reservation, _ = reservations.reserve(
        "own", "same", 16, (2, 4), "float16", torch.float16
    )
    reservation.allocation.tensor.fill_(3)
    reservations.complete("own", reservation.reservation_id)
    s = scheduler(False)
    req = request(["same"], {"ec_items": [{"transfer_id": "own"}]})
    s._transfers.observe_ready(
        ECMooncakeLoadSpec("same", 16, (2, 4), "float16", "own"), 1000
    )
    assert not s.ensure_cache_available(req, 0)
    output = SimpleNamespace(free_encoder_mm_hashes=[])
    loads = s.build_connector_meta(output)
    if not allocated:
        s.request_finished(req)  # Abort after dispatch, before Worker takes it.
    assert s._transfers.get("own").state is SchedulerTransferState.LOADING
    worker.start_load_caches(loads, cache)
    assert torch.equal(cache["same"], torch.full((2, 4), 3, dtype=torch.float16))
    s.update_connector_output(
        SimpleNamespace(
            ec_connector_worker_meta=ECMooncakeWorkerMetadata(loaded={"same"})
        )
    )
    if allocated:
        s.update_state_after_alloc(req, 0)
    cleanup = s.build_connector_meta(output)
    assert (
        output.free_encoder_mm_hashes == []
    )  # Connector cannot mutate SchedulerOutput.
    worker.start_load_caches(cleanup, cache)
    if allocated:
        assert cleanup.freed == []
        assert "same" in cache
        assert pool.reclaim_and_allocate(16, (2, 4), torch.float16) is None
        s.request_finished(req)
        cleanup = s.build_connector_meta(
            SimpleNamespace(free_encoder_mm_hashes=["same"])
        )
        cache.pop("same")  # Model runner applies scheduler-requested eviction.
        worker.start_load_caches(cleanup, cache)
    assert cleanup.freed == ["same"]
    assert not cache
    assert s._transfers.get("own").state is SchedulerTransferState.RESIDENT
    # Resident reuse is allowed, but pressure must be able to reclaim the slab.
    assert pool.reclaim_and_allocate(16, (2, 4), torch.float16) is not None
    assert pool.drain_reclaimed() == {"same"}
    s.update_connector_output(
        SimpleNamespace(
            ec_connector_worker_meta=ECMooncakeWorkerMetadata(reclaimed={"same"})
        )
    )
    assert s._transfers.get("own").state is SchedulerTransferState.EXPIRED
    pool.close()


def publish_ready(s, reservations, mm_hash, transfer_id, value):
    reservation, _ = reservations.reserve(
        transfer_id, mm_hash, 16, (2, 4), "float16", torch.float16
    )
    reservation.allocation.tensor.fill_(value)
    reservations.complete(transfer_id, reservation.reservation_id)
    s._transfers.observe_ready(
        ECMooncakeLoadSpec(mm_hash, 16, (2, 4), "float16", transfer_id), 1000
    )


def run_worker_loads(s, worker, cache):
    from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
        ECMooncakeWorkerMetadata,
    )

    meta = s.build_connector_meta(SimpleNamespace(free_encoder_mm_hashes=[]))
    worker.start_load_caches(meta, cache)
    s.update_connector_output(
        SimpleNamespace(
            ec_connector_worker_meta=ECMooncakeWorkerMetadata(
                loaded=worker._completed_loads, failed_loads=worker._failed_loads
            )
        )
    )
    worker._completed_loads, worker._failed_loads = set(), set()
    return meta


@pytest.mark.parametrize("staggered", [False, True])
def test_active_request_accumulates_ready_features_before_allocation(staggered):
    s = scheduler(False)
    worker, pool, reservations, cache = consumer_worker(512)
    req = request(
        ["a", "b"], {"ec_items": [{"mm_hash": h, "transfer_id": h} for h in ("a", "b")]}
    )
    publish_ready(s, reservations, "a", "a", 1)
    if not staggered:
        publish_ready(s, reservations, "b", "b", 2)
    loads: list[str] = []
    for step in range(3):
        if staggered and step == 1:
            publish_ready(s, reservations, "b", "b", 2)
        ready = s.ensure_cache_available(req, 0)
        if ready:  # Match the engine's gate -> allocation order.
            for index in range(2):
                s.update_state_after_alloc(req, index)
        meta = run_worker_loads(s, worker, cache)
        loads.extend(spec.mm_hash for spec in meta.loads)
        if ready:
            break
    assert ready, "Partially READY outputs must survive the all-feature gate"
    assert sorted(loads) == ["a", "b"]
    assert not s._pending_load_hashes
    assert all(
        torch.equal(cache[h], torch.full((2, 4), value, dtype=torch.float16))
        for h, value in [("a", 1), ("b", 2)]
    )
    pool.close()


def test_pending_borrower_keeps_ready_output_after_first_request_finishes():
    s = scheduler(False)
    worker, pool, reservations, cache = consumer_worker()
    owner = request(["same"], {"ec_items": [{"transfer_id": "own"}]})
    borrower = request(
        ["same", "later"],
        {
            "ec_items": [
                {"mm_hash": "same", "transfer_id": "borrower"},
                {"mm_hash": "later", "transfer_id": "later"},
            ]
        },
    )
    borrower.request_id = "borrower"
    publish_ready(s, reservations, "same", "own", 3)
    assert not s.ensure_cache_available(owner, 0)
    assert not s.ensure_cache_available(borrower, 0)
    s.request_finished(owner)
    run_worker_loads(s, worker, cache)
    assert not s.ensure_cache_available(borrower, 0)  # Still waiting for later.
    assert run_worker_loads(s, worker, cache).freed == []
    assert "same" in cache
    s.request_finished(borrower)
    assert not s._pending_load_hashes
    assert run_worker_loads(s, worker, cache).freed == ["same"]
    assert not cache
    pool.close()
