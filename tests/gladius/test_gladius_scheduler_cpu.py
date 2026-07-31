"""CPU-only tests for GladiusScheduler -- no GPU, no executor, real vLLM
Scheduler/Request objects via tests/v1/core/utils.py's create_scheduler()/
create_requests() helpers (upstream's own vehicle for scheduler-only unit
tests). GladiusScheduler isn't reachable through create_scheduler() directly
(it hardcodes Scheduler/AsyncScheduler), so _build_gladius_scheduler() below
is a small local wrapper that reuses a vanilla-built Scheduler's already-
constructed VllmConfig/KVCacheConfig/StructuredOutputManager to build a
GladiusScheduler with an identical configuration -- it does not modify the
upstream helper file.
"""

import json
import os
from datetime import datetime, timedelta, timezone

# Requires network-derived HF metadata to be avoided: the cached model below
# is already on disk, but ModelConfig construction still eagerly builds an
# HTTP client that fails validation against a SOCKS proxy URL scheme unless
# these are set before any vllm config object is constructed.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from gladius_vllm.scheduler import GladiusScheduler

MODEL = "Qwen/Qwen3-1.7B"  # already present in the local HF cache


def _write_snapshot(
    directory,
    *,
    engine_id: str,
    model_id: str,
    generation: int,
    max_num_seqs=None,
    max_num_batched_tokens=None,
    ttl_seconds: float = 30.0,
    created_at=None,
):
    created_at = created_at or datetime.now(timezone.utc)
    expires_at = created_at + timedelta(seconds=ttl_seconds)
    payload = {
        "schema_version": "1.0.0",
        "generation": generation,
        "policy_id": f"policy-{generation}",
        "model_id": model_id,
        "engine_id": engine_id,
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
        "admission": {
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
        },
        "notes": None,
    }
    path = directory / "policy_snapshot.json"
    tmp = directory / ".policy_snapshot.tmp"
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)
    return path


def _build_gladius_scheduler(vanilla, block_size=16):
    return GladiusScheduler(
        vllm_config=vanilla.vllm_config,
        kv_cache_config=vanilla.kv_cache_config,
        structured_output_manager=vanilla.structured_output_manager,
        block_size=block_size,
        log_stats=True,
    )


@pytest.fixture(autouse=True)
def _fixed_engine_id(monkeypatch):
    monkeypatch.setenv("GLADIUS_ENGINE_ID", "test-engine")
    monkeypatch.setenv("GLADIUS_POLICY_POLL_INTERVAL_MS", "0")


def test_no_policy_file_matches_vanilla_scheduler_decisions(tmp_path, monkeypatch):
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(tmp_path))  # dir exists, no file in it

    vanilla = create_scheduler(model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192)
    gladius = _build_gladius_scheduler(vanilla)

    for req in create_requests(num_requests=20, num_tokens=50, max_tokens=8):
        vanilla.add_request(req)
    for req in create_requests(num_requests=20, num_tokens=50, max_tokens=8):
        gladius.add_request(req)

    vanilla_output = vanilla.schedule()
    gladius_output = gladius.schedule()

    assert gladius.max_num_running_reqs == vanilla.max_num_running_reqs
    assert gladius.max_num_scheduled_tokens == vanilla.max_num_scheduled_tokens
    assert sorted(gladius_output.num_scheduled_tokens.items()) == sorted(
        vanilla_output.num_scheduled_tokens.items()
    )
    assert gladius_output.total_num_scheduled_tokens == vanilla_output.total_num_scheduled_tokens
    assert {r.req_id for r in gladius_output.scheduled_new_reqs} == {
        r.req_id for r in vanilla_output.scheduled_new_reqs
    }


def test_no_policy_dir_configured_at_all_matches_vanilla(monkeypatch):
    monkeypatch.delenv("GLADIUS_POLICY_DIR", raising=False)

    vanilla = create_scheduler(model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192)
    gladius = _build_gladius_scheduler(vanilla)

    for req in create_requests(num_requests=5, num_tokens=50, max_tokens=8):
        vanilla.add_request(req)
    for req in create_requests(num_requests=5, num_tokens=50, max_tokens=8):
        gladius.add_request(req)

    vanilla_output = vanilla.schedule()
    gladius_output = gladius.schedule()
    assert gladius_output.total_num_scheduled_tokens == vanilla_output.total_num_scheduled_tokens


def test_policy_lowering_max_num_seqs_changes_admission(tmp_path, monkeypatch):
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(tmp_path))

    vanilla = create_scheduler(model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192)
    gladius = _build_gladius_scheduler(vanilla)
    _write_snapshot(
        tmp_path,
        engine_id=gladius.engine_id,
        model_id=gladius.model_id,
        generation=1,
        max_num_seqs=4,
    )

    for req in create_requests(num_requests=20, num_tokens=10, max_tokens=8):
        gladius.add_request(req)

    output = gladius.schedule()
    assert gladius.max_num_running_reqs == 4
    assert len(gladius.running) == 4
    assert len(output.scheduled_new_reqs) == 4


def test_policy_lowering_token_budget_changes_scheduled_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(tmp_path))

    vanilla = create_scheduler(model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192)
    gladius = _build_gladius_scheduler(vanilla)
    _write_snapshot(
        tmp_path,
        engine_id=gladius.engine_id,
        model_id=gladius.model_id,
        generation=1,
        max_num_batched_tokens=64,
    )

    for req in create_requests(num_requests=10, num_tokens=50, max_tokens=8):
        gladius.add_request(req)

    output = gladius.schedule()
    assert gladius.max_num_scheduled_tokens == 64
    assert output.total_num_scheduled_tokens <= 64


def test_policy_above_startup_ceiling_is_clamped_not_applied(tmp_path, monkeypatch):
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(tmp_path))

    vanilla = create_scheduler(model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192)
    gladius = _build_gladius_scheduler(vanilla)
    _write_snapshot(
        tmp_path,
        engine_id=gladius.engine_id,
        model_id=gladius.model_id,
        generation=1,
        max_num_seqs=999,
        max_num_batched_tokens=999999,
    )

    for req in create_requests(num_requests=20, num_tokens=10, max_tokens=8):
        gladius.add_request(req)
    gladius.schedule()

    # Effective ceiling never exceeds startup values, even though the policy
    # requested far more.
    assert gladius.max_num_running_reqs == gladius.startup_max_num_seqs == 16
    assert (
        gladius.max_num_scheduled_tokens == gladius.startup_max_num_batched_tokens == 8192
    )


def test_corrupt_then_valid_then_stale_generation_sequence(tmp_path, monkeypatch):
    # The lowered ceiling (4) is published *before* the first schedule() call
    # so the running count never starts out above it -- shrinking the
    # ceiling below an already-larger running count is a distinct, separate
    # safety behavior covered by test_running_count_floor_when_ceiling_drops
    # below (the base Scheduler has no preemption path to evict already-
    # admitted requests, so this scheduler floors the ceiling at the current
    # running count rather than violating vLLM's own admission invariant).
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(tmp_path))

    vanilla = create_scheduler(model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192)
    gladius = _build_gladius_scheduler(vanilla)

    for req in create_requests(num_requests=20, num_tokens=10, max_tokens=8):
        gladius.add_request(req)

    # Step 1: valid snapshot, generation 5, lowers ceiling to 4 from the start.
    _write_snapshot(
        tmp_path, engine_id=gladius.engine_id, model_id=gladius.model_id,
        generation=5, max_num_seqs=4,
    )
    gladius.schedule()
    assert gladius.max_num_running_reqs == 4

    # Step 2: corrupt write -> keep last-good (still 4).
    (tmp_path / "policy_snapshot.json").write_text("{not valid json")
    gladius.schedule()
    assert gladius.max_num_running_reqs == 4

    # Step 3: a regressed generation (3 < 5) -> rejected, still 4.
    _write_snapshot(
        tmp_path, engine_id=gladius.engine_id, model_id=gladius.model_id,
        generation=3, max_num_seqs=8,
    )
    gladius.schedule()
    assert gladius.max_num_running_reqs == 4

    # Step 4: a genuinely newer generation (6) -> accepted, ceiling raised to 8.
    _write_snapshot(
        tmp_path, engine_id=gladius.engine_id, model_id=gladius.model_id,
        generation=6, max_num_seqs=8,
    )
    gladius.schedule()
    assert gladius.max_num_running_reqs == 8


def test_running_count_floor_when_ceiling_drops_below_already_admitted(tmp_path, monkeypatch):
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(tmp_path))

    vanilla = create_scheduler(model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192)
    gladius = _build_gladius_scheduler(vanilla)

    for req in create_requests(num_requests=20, num_tokens=10, max_tokens=8):
        gladius.add_request(req)

    # No policy yet: admits up to the startup ceiling (16).
    gladius.schedule()
    assert len(gladius.running) == 16

    # Now lower the ceiling to 4 -- fewer than what's already running. This
    # must not crash (no forced eviction/preemption in phase 1 scope); the
    # scheduler floors the effective ceiling at the current running count
    # instead of violating the base Scheduler's own admission invariant.
    _write_snapshot(
        tmp_path, engine_id=gladius.engine_id, model_id=gladius.model_id,
        generation=1, max_num_seqs=4,
    )
    gladius.schedule()
    assert gladius.max_num_running_reqs == 16
    assert len(gladius.running) == 16


def test_engine_degrades_when_policy_expires_mid_run(tmp_path, monkeypatch):
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(tmp_path))

    vanilla = create_scheduler(model=MODEL, max_num_seqs=16, max_num_batched_tokens=8192)
    gladius = _build_gladius_scheduler(vanilla)

    for req in create_requests(num_requests=20, num_tokens=10, max_tokens=8):
        gladius.add_request(req)

    _write_snapshot(
        tmp_path, engine_id=gladius.engine_id, model_id=gladius.model_id,
        generation=1, max_num_seqs=4, ttl_seconds=0.05,
    )
    gladius.schedule()
    assert gladius.max_num_running_reqs == 4

    import time

    time.sleep(0.15)
    gladius.schedule()
    assert gladius.max_num_running_reqs == gladius.startup_max_num_seqs == 16
