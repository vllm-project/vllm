# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from unittest.mock import MagicMock

import torch

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
    return now


class _FileMapper:
    def __init__(self, root: Path):
        self.root = root

    def get_file_name(self, key: bytes) -> str:
        return str(self.root / f"{key.hex()}.bin")


class _FileTier:
    def __init__(self, root: Path):
        self.file_mapper = _FileMapper(root)


def _create_file(tier: _FileTier, key: bytes) -> Path:
    path = Path(tier.file_mapper.get_file_name(key))
    path.write_bytes(b"kv")
    return path


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


def test_disabled_lifecycle_does_not_retain_request_metadata():
    lifecycle = SessionLifecycleManager(LifecycleConfig())
    ctx = ReqContext(req_id="request")
    lifecycle.on_new_request(ctx)
    lifecycle.record_request_keys(ctx, [_key(1)])
    lifecycle.on_request_finished(ctx)

    assert lifecycle.snapshot()["sessions"] == 0
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


def test_expiration_preserves_blocks_referenced_by_active_session(
    monkeypatch, tmp_path
):
    now = _clock(monkeypatch)
    lifecycle = SessionLifecycleManager(
        LifecycleConfig(idle_ttl_sec=5, delete_expired_secondary=True)
    )
    tier = _FileTier(tmp_path)
    shared_key = _key(7)
    path = _create_file(tier, shared_key)

    idle = ReqContext(req_id="idle", kv_transfer_params={"session_id": "idle-s"})
    active = ReqContext(req_id="active", kv_transfer_params={"session_id": "active-s"})
    lifecycle.on_new_request(idle)
    lifecycle.record_request_keys(idle, [shared_key])
    lifecycle.on_request_finished(idle)
    lifecycle.on_new_request(active)
    lifecycle.record_request_keys(active, [shared_key])

    now[0] = 106.0
    assert lifecycle.expire_idle_sessions([tier]) == 1
    assert path.exists()
    assert "idle-s" not in lifecycle._sessions

    lifecycle.on_request_finished(active)
    now[0] = 112.0
    assert lifecycle.expire_idle_sessions([tier]) == 1
    assert not path.exists()


def test_expiration_waits_for_inflight_transfer(monkeypatch, tmp_path):
    now = _clock(monkeypatch)
    lifecycle = SessionLifecycleManager(
        LifecycleConfig(idle_ttl_sec=5, delete_expired_secondary=True)
    )
    tier = _FileTier(tmp_path)
    key = _key(8)
    path = _create_file(tier, key)
    ctx = ReqContext(req_id="request")
    lifecycle.on_new_request(ctx)
    lifecycle.record_request_keys(ctx, [key])
    lifecycle.on_request_finished(ctx)

    now[0] = 106.0
    assert lifecycle.expire_idle_sessions([tier], protected_keys=[key]) == 0
    assert path.exists()
    assert lifecycle.has_pending_expiration()

    assert lifecycle.expire_idle_sessions([tier]) == 1
    assert not path.exists()


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
    manager.shutdown()
