# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import pytest

from vllm.v1.core.sched.session_affinity_scheduler import (
    SessionAffinityScheduler,
)
from vllm.v1.request import Request, RequestStatus

from .utils import create_requests, create_scheduler


def make_scheduler(**kwargs) -> SessionAffinityScheduler:
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        async_scheduling=True,
        scheduler_cls=SessionAffinityScheduler,
        **kwargs,
    )
    assert isinstance(scheduler, SessionAffinityScheduler)
    return scheduler


def test_requires_async_scheduling():
    with pytest.raises(
        ValueError,
        match="requires asynchronous scheduling",
    ):
        create_scheduler(
            enable_prefix_caching=True,
            async_scheduling=False,
            scheduler_cls=SessionAffinityScheduler,
        )


def add_requests(
    scheduler: SessionAffinityScheduler,
    session_ids: list[str | None],
) -> list[Request]:
    requests = create_requests(
        num_requests=len(session_ids),
        num_tokens=128,
        block_size=scheduler.block_size,
    )
    now = time.time()
    for index, (request, session_id) in enumerate(zip(requests, session_ids)):
        request.session_id = session_id
        request.arrival_time = now + index * 0.01
        scheduler.add_request(request)
    return requests


def waiting_ids(scheduler: SessionAffinityScheduler) -> list[str]:
    return [request.request_id for request in scheduler.waiting]


def test_promotes_cache_warm_recent_session(monkeypatch: pytest.MonkeyPatch):
    scheduler = make_scheduler()
    requests = add_requests(scheduler, ["cold", "warm", "other"])
    now = time.time()
    scheduler._session_last_scheduled_at["warm"] = now - 1
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "peek_num_cached_tokens",
        lambda request: 64 if request is requests[1] else 0,
    )

    scheduler._promote_session_continuation(now)

    assert waiting_ids(scheduler) == ["1", "0", "2"]


def test_does_not_promote_session_without_cache_hit(
    monkeypatch: pytest.MonkeyPatch,
):
    scheduler = make_scheduler()
    add_requests(scheduler, ["cold", "warm"])
    now = time.time()
    scheduler._session_last_scheduled_at["warm"] = now - 1
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "peek_num_cached_tokens",
        lambda request: 0,
    )

    scheduler._promote_session_continuation(now)

    assert waiting_ids(scheduler) == ["0", "1"]


def test_head_wait_deadline_prevents_reordering(monkeypatch: pytest.MonkeyPatch):
    scheduler = make_scheduler()
    requests = add_requests(scheduler, ["cold", "warm"])
    now = time.time()
    requests[0].arrival_time = now - scheduler._affinity_max_wait_s
    scheduler._session_last_scheduled_at["warm"] = now - 1
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "peek_num_cached_tokens",
        lambda request: 64,
    )

    scheduler._promote_session_continuation(now)

    assert waiting_ids(scheduler) == ["0", "1"]


def test_preempted_head_is_not_bypassed(monkeypatch: pytest.MonkeyPatch):
    scheduler = make_scheduler()
    requests = add_requests(scheduler, ["preempted", "warm"])
    now = time.time()
    requests[0].status = RequestStatus.PREEMPTED
    scheduler._session_last_scheduled_at["warm"] = now - 1
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "peek_num_cached_tokens",
        lambda request: 64,
    )

    scheduler._promote_session_continuation(now)

    assert waiting_ids(scheduler) == ["0", "1"]


def test_expired_session_is_not_promoted(monkeypatch: pytest.MonkeyPatch):
    scheduler = make_scheduler()
    add_requests(scheduler, ["cold", "stale"])
    now = time.time()
    scheduler._session_last_scheduled_at["stale"] = now - scheduler._affinity_ttl_s - 1
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "peek_num_cached_tokens",
        lambda request: 64,
    )

    scheduler._prune_expired_sessions(now)
    scheduler._promote_session_continuation(now)

    assert waiting_ids(scheduler) == ["0", "1"]
    assert "stale" not in scheduler._session_last_scheduled_at


def test_schedule_records_active_session():
    scheduler = make_scheduler(max_num_seqs=1)
    request = add_requests(scheduler, ["active"])[0]

    output = scheduler.schedule()

    assert request.request_id in output.num_scheduled_tokens
    assert "active" in scheduler._session_last_scheduled_at
