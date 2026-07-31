"""Real-vLLM contract test for GladiusScheduler.

Mirrors tests/plugins_tests/test_scheduler_plugins.py's in-process pattern
(VLLM_ENABLE_V1_MULTIPROCESSING=0, enforce_eager=True) but drives a real
LLMEngine end to end with a real (small, locally-cached) model and a real
policy_snapshot.json produced by the gladius-side policy_writer-equivalent
helper below, asserting the full loop: no-policy startup, a published
snapshot actually lowering admission, and telemetry.jsonl reflecting it.

Needs a real model load (~GB-scale weights) and a working CPU/GPU execution
backend -- this is the one test in this suite that cannot run in a
network/model-cache-less CI lane. Run manually:

    pytest tests/gladius/test_gladius_contract_e2e.py -v -s
"""

import json
import os
from datetime import datetime, timedelta, timezone

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

import pytest

MODEL = "Qwen/Qwen3-1.7B"


def _write_snapshot(directory, *, engine_id, model_id, generation, max_num_seqs, ttl_seconds=60.0):
    now = datetime.now(timezone.utc)
    payload = {
        "schema_version": "1.0.0",
        "generation": generation,
        "policy_id": f"policy-{generation}",
        "model_id": model_id,
        "engine_id": engine_id,
        "created_at": now.isoformat().replace("+00:00", "Z"),
        "expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat().replace("+00:00", "Z"),
        "admission": {"max_num_seqs": max_num_seqs, "max_num_batched_tokens": None},
        "notes": None,
    }
    path = directory / "policy_snapshot.json"
    tmp = directory / ".policy_snapshot.tmp"
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)
    return path


@pytest.mark.slow_test
def test_gladius_scheduler_real_engine_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("GLADIUS_ENGINE_ID", "contract-test-engine")
    monkeypatch.setenv("GLADIUS_POLICY_DIR", str(tmp_path))
    monkeypatch.setenv("GLADIUS_POLICY_POLL_INTERVAL_MS", "0")

    from vllm.engine.arg_utils import EngineArgs
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.llm_engine import LLMEngine

    from gladius_vllm.scheduler import GladiusScheduler

    engine_args = EngineArgs(
        model=MODEL,
        enforce_eager=True,
        scheduler_cls=GladiusScheduler,
        max_num_seqs=16,
        max_model_len=2048,
        gpu_memory_utilization=0.3,
    )
    engine = LLMEngine.from_engine_args(engine_args=engine_args)
    scheduler = engine.engine_core.engine_core.scheduler
    assert isinstance(scheduler, GladiusScheduler)

    # Phase 1: no snapshot published yet -- native-equivalent behavior.
    sampling_params = SamplingParams(max_tokens=4)
    for i in range(4):
        engine.add_request(str(i), "Hello, world! " * 4, sampling_params)
    engine.step()
    assert scheduler.max_num_running_reqs == scheduler.startup_max_num_seqs == 16

    # Phase 2: publish a snapshot lowering the ceiling to 2, matching this
    # engine's real engine_id/model_id, and confirm it takes effect without
    # restarting the process.
    _write_snapshot(
        tmp_path,
        engine_id=scheduler.engine_id,
        model_id=scheduler.model_id,
        generation=1,
        max_num_seqs=2,
    )
    for i in range(4, 8):
        engine.add_request(str(i), "Hello, world! " * 4, sampling_params)
    engine.step()

    # The 4 phase-1 requests are still running, above the new cap of 2 --
    # this must not crash (no forced eviction of already-admitted requests
    # in phase 1 scope; see the equivalent CPU-only test for the isolated
    # case). Drain until they finish naturally; the cap then takes hold for
    # new admissions.
    for _ in range(20):
        if not engine.has_unfinished_requests():
            break
        engine.step()

    assert scheduler.max_num_running_reqs == 2

    telemetry_path = tmp_path / "telemetry.jsonl"
    assert telemetry_path.exists()
    lines = [json.loads(line) for line in telemetry_path.read_text().splitlines() if line]
    assert len(lines) >= 2
    assert lines[0]["policy_status"] == "no_policy"
    assert any(line["policy_status"] == "active" for line in lines)
    assert any(line["generation"] == 1 for line in lines)
