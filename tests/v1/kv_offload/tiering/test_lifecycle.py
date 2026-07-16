# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    _select_store_candidates,
)
from vllm.v1.kv_offload.base import ReqContext, ScheduleEndContext, make_offload_key
from vllm.v1.kv_offload.tiering.lifecycle import (
    LifecycleConfig,
    LifecycleStatus,
    SessionLifecycleManager,
    get_session_id,
)
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)


def _key(value: int) -> bytes:
    return make_offload_key(value.to_bytes(8, "big"), 0)


def _clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        "vllm.v1.kv_offload.tiering.lifecycle.time.monotonic", lambda: now[0]
    )
    monkeypatch.setattr(
        "vllm.v1.kv_offload.tiering.residency.time.monotonic", lambda: now[0]
    )
    return now


def _mock_mmap_region(num_blocks: int, row_bytes: int = 16):
    region = MagicMock()
    view = memoryview(torch.zeros((num_blocks, row_bytes), dtype=torch.int8).numpy())
    region.create_kv_memoryview.return_value = view
    return region


def test_session_id_precedence_and_fallback():
    assert get_session_id(ReqContext(req_id="request")) == "request"
    assert (
        get_session_id(
            ReqContext(
                req_id="request",
                kv_transfer_params={
                    "session_id": "session",
                    "conversation_id": "conversation",
                },
            )
        )
        == "session"
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"idle_ttl_sec": -1}, "lifecycle_idle_ttl_sec"),
        ({"cpu_demote_after_sec": float("inf")}, "cpu_demote"),
        (
            {"cpu_low_watermark": 0.9, "cpu_high_watermark": 0.8},
            "low <= high",
        ),
        ({"reclaim_batch_size": 0}, "reclaim_batch_size"),
        ({"residency_max_entries": 0}, "residency_max_entries"),
        ({"max_sessions": 0}, "lifecycle_max_sessions"),
    ],
)
def test_lifecycle_config_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LifecycleConfig(**kwargs)


def test_disabled_lifecycle_does_not_retain_request_metadata():
    lifecycle = SessionLifecycleManager(LifecycleConfig())
    ctx = ReqContext(req_id="request")
    lifecycle.on_new_request(ctx)
    lifecycle.record_request_keys(ctx, [_key(1)])
    lifecycle.on_request_finished(ctx)

    assert lifecycle.snapshot()["sessions"] == 0
    assert lifecycle.residency.snapshot()["tracked_blocks"] == 0
    assert not lifecycle.has_pending_expiration()


def test_lifecycle_transitions_and_reactivation(monkeypatch):
    now = _clock(monkeypatch)
    lifecycle = SessionLifecycleManager(LifecycleConfig(idle_ttl_sec=5))
    first = ReqContext(req_id="r1", kv_transfer_params={"session_id": "s1"})

    lifecycle.on_new_request(first)
    lifecycle.record_request_keys(first, [_key(1), _key(2)])
    assert lifecycle.snapshot() == {
        "sessions": 1,
        "active_sessions": 1,
        "idle_sessions": 0,
        "retained_blocks": 2,
    }

    lifecycle.on_request_finished(first)
    state = lifecycle._sessions["s1"]
    assert state.status is LifecycleStatus.IDLE_RETAINED
    assert state.ttl_deadline == 105.0

    now[0] = 103.0
    second = ReqContext(req_id="r2", kv_transfer_params={"session_id": "s1"})
    lifecycle.on_new_request(second)
    state = lifecycle._sessions["s1"]
    assert state.status is LifecycleStatus.ACTIVE
    assert state.ttl_deadline is None
    assert state.block_keys == {_key(1), _key(2)}


def test_session_heat_tracks_revisits_and_external_hits(monkeypatch):
    _clock(monkeypatch)
    lifecycle = SessionLifecycleManager(LifecycleConfig())
    first = ReqContext(req_id="r1", kv_transfer_params={"session_id": "s1"})
    second = ReqContext(req_id="r2", kv_transfer_params={"session_id": "s1"})

    lifecycle.on_new_request(first, track_heat=True)
    assert lifecycle.get_session_heat(first) == (1, 0.0)
    lifecycle.on_request_finished(first, track_heat=True)

    lifecycle.on_new_request(second, track_heat=True)
    assert lifecycle.get_session_heat(second) == (2, 0.5)
    lifecycle.record_reuse_hit(second, 3)
    lifecycle.record_reuse_hit(second, 2)

    state = lifecycle._sessions["s1"]
    assert state.reuse_request_count == 1
    assert state.reuse_hit_blocks == 5
    assert state.last_reuse_at == 100.0


def test_request_finalization_releases_fallback_session_metadata(monkeypatch):
    _clock(monkeypatch)
    lifecycle = SessionLifecycleManager(
        LifecycleConfig(residency_tracking_enabled=True)
    )
    ctx = ReqContext(req_id="one-shot")
    key = _key(6)

    lifecycle.on_new_request(ctx)
    lifecycle.record_request_keys(ctx, [key])
    lifecycle.on_request_finished(ctx)
    lifecycle.on_request_finalized(ctx)

    assert lifecycle.snapshot()["sessions"] == 0
    assert lifecycle._req_to_session == {}
    assert lifecycle.residency.get_session_keys("one-shot") == set()


def test_request_finalization_keeps_stable_session_but_not_request_ids(monkeypatch):
    _clock(monkeypatch)
    lifecycle = SessionLifecycleManager(LifecycleConfig())
    ctx = ReqContext(req_id="r1", kv_transfer_params={"session_id": "s1"})

    lifecycle.on_new_request(ctx, track_heat=True)
    lifecycle.on_request_finished(ctx, track_heat=True)
    lifecycle.on_request_finalized(ctx)

    assert lifecycle.snapshot()["sessions"] == 1
    assert lifecycle._req_to_session == {}
    assert lifecycle._sessions["s1"].retained_req_ids == set()


def test_idle_session_metadata_is_bounded(monkeypatch):
    now = _clock(monkeypatch)
    lifecycle = SessionLifecycleManager(LifecycleConfig(max_sessions=2))

    for index in range(3):
        ctx = ReqContext(
            req_id=f"r{index}",
            kv_transfer_params={"session_id": f"s{index}"},
        )
        lifecycle.on_new_request(ctx, track_heat=True)
        lifecycle.on_request_finished(ctx, track_heat=True)
        lifecycle.on_request_finalized(ctx)
        now[0] += 1

    result = lifecycle.prune_idle_sessions()

    assert result.pruned_sessions == 1
    assert lifecycle.snapshot()["sessions"] == 2
    assert "s0" not in lifecycle._sessions


def test_store_budget_selection_advances_only_selected_group_prefixes():
    keys = [_key(i) for i in range(6)]
    selected, frontiers = _select_store_candidates(
        [
            (0, 2, keys[2]),
            (0, 1, keys[1]),
            (1, 1, keys[4]),
            (1, 2, keys[5]),
        ],
        [1, 1],
        2,
    )

    assert [(group, block) for group, block, _ in selected] == [(0, 1), (1, 1)]
    assert frontiers == [2, 2]


def test_shared_block_becomes_unreferenced_after_final_session_expires(monkeypatch):
    now = _clock(monkeypatch)
    lifecycle = SessionLifecycleManager(LifecycleConfig(idle_ttl_sec=5))
    shared_key = _key(7)
    first = ReqContext(req_id="r1", kv_transfer_params={"session_id": "s1"})
    second = ReqContext(req_id="r2", kv_transfer_params={"session_id": "s2"})

    for ctx in (first, second):
        lifecycle.on_new_request(ctx)
        lifecycle.record_request_keys(ctx, [shared_key])
        lifecycle.on_request_finished(ctx)

    now[0] = 106.0
    lifecycle._sessions["s2"].ttl_deadline = 110.0
    first_result = lifecycle.expire_idle_sessions()
    assert first_result.expired_sessions == 1
    assert first_result.unreferenced_keys == set()
    assert lifecycle.residency.snapshot()["shared_blocks"] == 0

    now[0] = 111.0
    second_result = lifecycle.expire_idle_sessions()
    assert second_result.expired_sessions == 1
    assert second_result.unreferenced_keys == {shared_key}


def test_expiration_waits_for_inflight_transfer(monkeypatch):
    now = _clock(monkeypatch)
    lifecycle = SessionLifecycleManager(LifecycleConfig(idle_ttl_sec=5))
    key = _key(8)
    ctx = ReqContext(req_id="request")
    lifecycle.on_new_request(ctx)
    lifecycle.record_request_keys(ctx, [key])
    lifecycle.on_request_finished(ctx)

    now[0] = 106.0
    result = lifecycle.expire_idle_sessions(protected_keys=[key])
    assert result.expired_sessions == 0
    assert lifecycle.has_pending_expiration()

    result = lifecycle.expire_idle_sessions()
    assert result.expired_sessions == 1
    assert result.unreferenced_keys == {key}


def test_idle_cpu_candidate_requires_durable_unshared_copy(monkeypatch):
    now = _clock(monkeypatch)
    lifecycle = SessionLifecycleManager(
        LifecycleConfig(cpu_demote_after_sec=5, residency_tracking_enabled=True)
    )
    ctx = ReqContext(req_id="request")
    key = _key(9)
    lifecycle.on_new_request(ctx)
    lifecycle.record_request_keys(ctx, [key])
    lifecycle.residency.mark_cpu_resident([key])
    lifecycle.on_request_finished(ctx)

    now[0] = 106.0
    assert lifecycle.get_idle_cpu_candidates(limit=1, require_idle_age=True) == []

    lifecycle.residency.mark_secondary_resident([key], "fs:0")
    lifecycle.residency.start_transfer([key], "cpu", "fs:0")
    assert lifecycle.get_idle_cpu_candidates(limit=1, require_idle_age=True) == []

    lifecycle.residency.finish_transfer([key], "cpu", "fs:0")
    assert lifecycle.get_idle_cpu_candidates(limit=1, require_idle_age=True) == [key]


def test_residency_prunes_old_unowned_entries(monkeypatch):
    _clock(monkeypatch)
    lifecycle = SessionLifecycleManager(
        LifecycleConfig(residency_max_entries=2, residency_tracking_enabled=True)
    )
    keys = [_key(1), _key(2), _key(3)]
    lifecycle.residency.mark_secondary_resident(keys, "fs:0")

    assert lifecycle.residency.prune() == 1
    assert lifecycle.residency.snapshot()["tracked_blocks"] == 2


def test_tiering_manager_tracks_and_expires_session(monkeypatch):
    now = _clock(monkeypatch)
    primary = CPUPrimaryTierOffloadingManager(
        num_blocks=4, mmap_region=_mock_mmap_region(4)
    )
    manager = TieringOffloadingManager(
        primary_tier=primary,
        lifecycle_config=LifecycleConfig(idle_ttl_sec=5),
    )
    ctx = ReqContext(req_id="r1", kv_transfer_params={"session_id": "s1"})
    keys = [_key(1), _key(2)]

    manager.on_new_request(ctx)
    prepared = manager.prepare_store(keys, ctx)
    assert prepared is not None
    manager.complete_store(prepared.keys_to_store, ctx)
    manager.on_request_finished(ctx)
    assert manager.get_lifecycle_snapshot()["idle_sessions"] == 1

    now[0] = 106.0
    manager.on_schedule_end(ScheduleEndContext([], ()))
    assert manager.get_lifecycle_snapshot()["sessions"] == 0
    assert primary.resident_blocks == 0
    manager.shutdown()
