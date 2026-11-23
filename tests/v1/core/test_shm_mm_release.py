# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Requests dropped before dispatch must release their SHM cache references."""

import multiprocessing
import uuid
from types import SimpleNamespace

import torch

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


def _payload_item() -> MultiModalKwargsItem:
    elem = MultiModalFieldElem(
        data=torch.zeros(ITEM_BYTES, dtype=torch.int8),
        field=MultiModalSharedField(batch_size=1),
    )
    return MultiModalKwargsItem({"pixel_values": elem})


def _run_abort_scenario(*, release: bool) -> list[bool]:
    """Real sender cache + scheduler + worker-side cache code, no model.

    A queued image request is aborted before it is scheduled, then more
    images than the ring can hold are sent, each read by every worker.
    Returns, per later image, whether it was stored in the ring.
    """
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
        for receiver in receivers:
            receiver._shm_cache.close()
        sender.close()


def test_aborted_request_without_release_stalls_the_ring():
    stored = _run_abort_scenario(release=False)
    # The unread item pins the ring: it fills up and later images are uncached.
    assert not all(stored)
    assert not any(stored[-3:])


def test_aborted_request_release_keeps_the_ring_usable():
    assert all(_run_abort_scenario(release=True))
