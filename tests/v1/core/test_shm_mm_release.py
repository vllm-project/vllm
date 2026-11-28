# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Requests dropped before dispatch must release their SHM cache references."""

import copy
import multiprocessing
import queue
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.cache.shm import (
    ShmObjectStoreReceiverCache,
    ShmObjectStoreSenderCache,
)
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
    PlaceholderRange,
)
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker.worker_base import WorkerWrapperBase

from .utils import create_requests, create_scheduler

_MM_MODEL = "llava-hf/llava-1.5-7b-hf"


def _shm_handle_item() -> MultiModalKwargsItem:
    return MultiModalKwargsItem(
        {
            name: MultiModalFieldElem(data=data, field=MultiModalBatchedField())
            for name, data in (
                ("address", 0),
                ("monotonic_id", 1),
                ("signature", [0] * 32),
            )
        }
    )


def _make_requests(num_requests: int, *, shm: bool = True) -> list[Request]:
    requests = create_requests(
        num_requests=num_requests,
        num_tokens=10,
        mm_hashes_list=[[f"hash{i}"] for i in range(num_requests)],
        mm_positions=[[PlaceholderRange(offset=0, length=4)]] * num_requests,
    )
    if shm:
        for request in requests:
            request.mm_features[0].data = _shm_handle_item()
    return requests


def test_aborted_waiting_request_hands_shm_features_to_workers():
    scheduler = create_scheduler(model=_MM_MODEL, max_num_seqs=1)
    running, waiting = _make_requests(2)
    scheduler.add_request(running)
    scheduler.add_request(waiting)

    # Only `running` fits; `waiting` is never dispatched to the workers.
    output = scheduler.schedule()
    assert [r.req_id for r in output.scheduled_new_reqs] == [running.request_id]
    assert not output.unread_mm_features

    scheduler.finish_requests(waiting.request_id, RequestStatus.FINISHED_ABORTED)
    output = scheduler.schedule()
    assert output.unread_mm_features == waiting.mm_features

    # The hand-off happens once.
    assert not scheduler.schedule().unread_mm_features


def test_aborted_dispatched_request_is_not_released_twice():
    scheduler = create_scheduler(model=_MM_MODEL)
    (request,) = _make_requests(1)
    scheduler.add_request(request)
    assert scheduler.schedule().scheduled_new_reqs  # workers read it here

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert not scheduler.schedule().unread_mm_features


def test_non_shm_features_are_not_sent_to_workers():
    scheduler = create_scheduler(model=_MM_MODEL, max_num_seqs=1)
    running, waiting = _make_requests(2, shm=False)
    scheduler.add_request(running)
    scheduler.add_request(waiting)
    scheduler.schedule()

    scheduler.finish_requests(waiting.request_id, RequestStatus.FINISHED_ABORTED)
    assert not scheduler.schedule().unread_mm_features


def test_idle_engine_still_runs_a_step_to_release_aborted_features():
    # The engine core only steps while has_requests(), so the hand-off must
    # not depend on other requests being alive.
    scheduler = create_scheduler(model=_MM_MODEL, max_num_seqs=1)
    (request,) = _make_requests(1)
    scheduler.add_request(request)
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)

    assert not scheduler.has_unfinished_requests()
    assert scheduler.has_requests()
    assert scheduler.schedule().unread_mm_features == request.mm_features
    assert not scheduler.has_requests()


N_WORKERS = 2
ITEM_BYTES = 700 * 1024
N_FILLERS = 12


def _payload_item(fill: int = 0) -> MultiModalKwargsItem:
    elem = MultiModalFieldElem(
        data=torch.full((ITEM_BYTES,), fill, dtype=torch.int8),
        field=MultiModalSharedField(batch_size=1),
    )
    return MultiModalKwargsItem({"pixel_values": elem})


def _make_caches():
    """A real SHM sender cache and `N_WORKERS` receiver caches (one per worker)."""
    mm_config = SimpleNamespace(
        mm_processor_cache_gb=4 / 1024,  # ~5 payloads fit in the ring
        mm_shm_cache_max_object_size_mb=1,
        mm_shm_cache_buffer_name=f"test_shm_{uuid.uuid4().hex[:12]}",
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(world_size=N_WORKERS),
        model_config=SimpleNamespace(get_multimodal_config=lambda: mm_config),
    )
    lock = multiprocessing.get_context().Lock()
    sender = ShmObjectStoreSenderCache(vllm_config)  # type: ignore[arg-type]
    receivers = [
        ShmObjectStoreReceiverCache(vllm_config, lock)  # type: ignore[arg-type]
        for _ in range(N_WORKERS)
    ]
    return sender, receivers


def _close_caches(sender, receivers) -> None:
    for receiver in receivers:
        receiver._shm_cache.close()
    sender.close()


def _run_abort_scenario(*, release: bool) -> list[bool]:
    """Real sender cache + scheduler + worker-side cache code, no model.

    A queued image request is aborted before it is scheduled, then more
    images than the ring can hold are sent, each read by every worker.
    Returns, per later image, whether it was stored in the ring.
    """
    sender, receivers = _make_caches()
    try:
        scheduler = create_scheduler(model=_MM_MODEL, max_num_seqs=1)
        blocker = create_requests(1, req_ids=["blocker"])[0]
        aborted = create_requests(
            1,
            mm_hashes_list=[["aborted_hash"]],
            mm_positions=[[PlaceholderRange(offset=0, length=4)]],
            req_ids=["aborted"],
        )[0]
        handle, _ = sender.get_and_update_item((_payload_item(), []), "aborted_hash")
        assert "address" in handle  # written to the ring by the frontend
        aborted.mm_features[0].data = handle

        scheduler.add_request(blocker)
        scheduler.add_request(aborted)
        scheduler.schedule()  # only the blocker fits; `aborted` stays queued
        scheduler.finish_requests("aborted", RequestStatus.FINISHED_ABORTED)
        output = scheduler.schedule()

        if release:
            for receiver in receivers:
                worker = SimpleNamespace(mm_receiver_cache=receiver)
                WorkerWrapperBase._apply_mm_cache(worker, output)  # type: ignore

        stored = []
        for i in range(N_FILLERS):
            key = f"filler_{i}"
            item, _ = sender.get_and_update_item((_payload_item(), []), key)
            is_stored = "address" in item
            stored.append(is_stored)
            if is_stored:  # every worker reads it when the request runs
                for receiver in receivers:
                    receiver.get_and_update_item(item, key)
        return stored
    finally:
        _close_caches(sender, receivers)


def test_aborted_request_without_release_stalls_the_ring():
    stored = _run_abort_scenario(release=False)
    # The unread item pins the ring: it fills up and later images are uncached.
    assert not all(stored)
    assert not any(stored[-3:])


def test_aborted_request_release_keeps_the_ring_usable():
    assert all(_run_abort_scenario(release=True))


def test_request_aborted_after_dispatch_is_still_read_under_ring_pressure():
    """The window an abort-time release cannot see: the request was already
    dispatched to the workers, which have not read it yet, when it is aborted.

    Its item must stay readable however much traffic cycles through the ring
    in between, and it must be reclaimable once the workers have read it.
    """
    sender, receivers = _make_caches()
    try:
        scheduler = create_scheduler(model=_MM_MODEL, max_num_seqs=1)
        (request,) = create_requests(
            1,
            mm_hashes_list=[["victim_hash"]],
            mm_positions=[[PlaceholderRange(offset=0, length=4)]],
            req_ids=["victim"],
        )
        handle, _ = sender.get_and_update_item((_payload_item(7), []), "victim_hash")
        assert "address" in handle
        request.mm_features[0].data = handle

        scheduler.add_request(request)
        dispatched = scheduler.schedule()  # sent to the workers, not read yet
        assert [r.req_id for r in dispatched.scheduled_new_reqs] == ["victim"]

        # The client disconnects before any worker runs the step.
        scheduler.finish_requests("victim", RequestStatus.FINISHED_ABORTED)
        assert not scheduler.schedule().unread_mm_features

        # Other traffic cycles through the ring while the step is in flight.
        for i in range(N_FILLERS):
            key = f"filler_{i}"
            item, _ = sender.get_and_update_item((_payload_item(10 + i), []), key)
            if "address" in item:
                for receiver in receivers:
                    receiver.get_and_update_item(item, key)

        # The workers now run the step: each reads the item that was handed out.
        for receiver in receivers:
            worker = SimpleNamespace(mm_receiver_cache=receiver)
            step = copy.deepcopy(dispatched)  # each worker gets its own copy
            WorkerWrapperBase._apply_mm_cache(worker, step)  # type: ignore
            loaded = step.scheduled_new_reqs[0].mm_features[0].data
            assert loaded is not None
            assert bool((loaded["pixel_values"].data == 7).all())

        sender._shm_cache.free_unused()
        assert not sender.is_cached_item("victim_hash")  # reclaimable once read
    finally:
        _close_caches(sender, receivers)


# ---- requests dropped before they reach the scheduler ----


def _unscheduled_request(req_id: str = "dropped") -> Request:
    (request,) = create_requests(
        1,
        mm_hashes_list=[[f"{req_id}_hash"]],
        mm_positions=[[PlaceholderRange(offset=0, length=4)]],
        req_ids=[req_id],
    )
    request.mm_features[0].data = _shm_handle_item()
    return request


def _bare_engine_core(scheduler) -> EngineCoreProc:
    """An engine core with just the state the dropped-request paths touch."""
    core = EngineCoreProc.__new__(EngineCoreProc)
    core.scheduler = scheduler
    core._dropped_mm_features = queue.SimpleQueue()
    core.input_queue = queue.Queue()
    core._send_finish_outputs_to_client = MagicMock()
    core._send_error_outputs_to_client = MagicMock()
    return core


def _run_queued_wakeup(core: EngineCoreProc) -> None:
    """Run what the busy loop does with the wakeup an input thread queued."""
    request_type, payload = core.input_queue.get_nowait()
    assert request_type == EngineCoreRequestType.WAKEUP
    core._handle_client_request(request_type, payload)


def test_scheduler_release_unread_features_steps_an_idle_engine():
    scheduler = create_scheduler(model=_MM_MODEL, max_num_seqs=1)
    request = _unscheduled_request()
    assert not scheduler.has_requests()

    scheduler.release_unread_mm_features(request.mm_features)

    assert scheduler.has_requests()  # the engine loop will run a step for it
    assert scheduler.schedule().unread_mm_features == request.mm_features
    assert not scheduler.has_requests()


def test_request_rejected_while_paused_releases_its_features():
    scheduler = create_scheduler(model=_MM_MODEL, max_num_seqs=1)
    core = _bare_engine_core(scheduler)
    core.reject_new_requests = True
    request = _unscheduled_request()

    assert core._reject_add_while_paused(request)

    core._send_finish_outputs_to_client.assert_called_once()
    assert scheduler.schedule().unread_mm_features == request.mm_features


def test_request_rejected_while_paused_with_kv_params_is_released_once():
    # That path re-adds the request as an immediate abort; the scheduler then
    # releases its features when it finishes it, so the engine core must not.
    scheduler = create_scheduler(model=_MM_MODEL, max_num_seqs=1)
    core = _bare_engine_core(scheduler)
    core.reject_new_requests = True
    core.add_request = MagicMock()
    request = _unscheduled_request()
    request.kv_transfer_params = {"remote": "blocks"}

    assert core._reject_add_while_paused(request)

    core.add_request.assert_called_once_with(request)
    assert not scheduler.schedule().unread_mm_features


@pytest.mark.parametrize("failure", ["validation", "preprocessing"])
def test_request_failing_on_the_input_thread_releases_its_features(failure):
    scheduler = create_scheduler(model=_MM_MODEL, max_num_seqs=1)
    core = _bare_engine_core(scheduler)
    request = _unscheduled_request()
    engine_request = SimpleNamespace(
        request_id="dropped", client_index=0, mm_features=request.mm_features
    )

    if failure == "validation":
        core._handle_request_validation_error(
            engine_request, VLLMValidationError("bad request")
        )
    else:
        core._handle_request_preproc_error(engine_request)

    # The input thread must not touch the scheduler; the busy loop does.
    assert not scheduler.has_requests()
    _run_queued_wakeup(core)
    assert scheduler.schedule().unread_mm_features == request.mm_features


def test_request_without_multimodal_features_queues_no_wakeup():
    core = _bare_engine_core(create_scheduler(model=_MM_MODEL, max_num_seqs=1))
    core._drop_mm_features_from_input_thread(SimpleNamespace(mm_features=None))
    assert core.input_queue.empty()


# ---- requests rejected by the frontend after their items were cached ----


def test_frontend_rejection_releases_only_items_that_were_handed_out():
    handed_out = _shm_handle_item()
    uncached = _payload_item()  # ring full or oversize: sent with its payload
    engine_input = {
        "type": "multimodal",
        "mm_hashes": {"image": ["stored", "uncached", "stored", "dropped"]},
        "mm_kwargs": {"image": [handed_out, uncached, handed_out, None]},
    }
    processor = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                data_parallel_size_local=1,
                local_engines_only=False,
            )
        ),
        _validate_params=MagicMock(return_value=False),
        _validate_lora=MagicMock(),
        renderer=MagicMock(),
    )

    with patch("vllm.v1.engine.input_processor.current_platform") as platform:
        platform.validate_request.side_effect = ValueError("rejected")
        with pytest.raises(ValueError, match="rejected"):
            InputProcessor.process_inputs(
                processor,  # type: ignore[arg-type]
                "req",
                engine_input,  # type: ignore[arg-type]
                SamplingParams(),
                supported_tasks=("generate",),
            )

    # One release per hand-out: an item referenced twice is released twice, and
    # items that were not written to the ring are not released at all.
    processor.renderer.queue_mm_cache_releases.assert_called_once_with(
        ["stored", "stored"]
    )
    processor.renderer.queue_mm_cache_invalidations.assert_called_once()
