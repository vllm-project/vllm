"""Pure-Python tests for TelemetryWriter -- hand-built fixtures, no vllm
executor/GPU, and no real Scheduler/SchedulerOutput instances needed since
TelemetryWriter only duck-types on a handful of attributes.
"""

import json
from types import SimpleNamespace

from gladius_vllm.policy import PolicyDecision
from gladius_vllm.telemetry import TelemetryWriter

EXPECTED_FIELDS = {
    "schema_version",
    "generation",
    "policy_id",
    "model_id",
    "engine_id",
    "created_at",
    "expires_at",
    "step",
    "num_running_reqs",
    "num_waiting_reqs",
    "num_skipped_waiting_reqs",
    "num_scheduled_reqs",
    "num_scheduled_tokens",
    "num_prefill_reqs",
    "num_decode_reqs",
    "kv_cache_usage",
    "policy_status",
    "policy_source",
    "clamped",
}


def _fake_request(num_computed_tokens: int, num_prompt_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_computed_tokens=num_computed_tokens, num_prompt_tokens=num_prompt_tokens
    )


def _fake_scheduler(
    *,
    requests: dict,
    running: list,
    waiting: list,
    skipped_waiting: list,
    startup_max_num_seqs: int = 64,
    startup_max_num_batched_tokens: int = 4096,
) -> SimpleNamespace:
    stats = SimpleNamespace(
        num_running_reqs=len(running),
        num_waiting_reqs=len(waiting),
        num_skipped_waiting_reqs=len(skipped_waiting),
        kv_cache_usage=0.42,
    )
    return SimpleNamespace(
        requests=requests,
        running=running,
        waiting=waiting,
        skipped_waiting=skipped_waiting,
        startup_max_num_seqs=startup_max_num_seqs,
        startup_max_num_batched_tokens=startup_max_num_batched_tokens,
        make_stats=lambda: stats,
    )


def _fake_output(new_req_ids: list, scheduled_tokens: dict) -> SimpleNamespace:
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=rid) for rid in new_req_ids],
        num_scheduled_tokens=scheduled_tokens,
        total_num_scheduled_tokens=sum(scheduled_tokens.values()),
    )


def _decision(**overrides) -> PolicyDecision:
    defaults = dict(
        max_num_seqs=64,
        max_num_batched_tokens=4096,
        policy_id="policy-1",
        generation=3,
        status="active",
        source="file",
    )
    defaults.update(overrides)
    return PolicyDecision(**defaults)


def test_writes_one_valid_json_line_with_all_fields(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="engine-1", model_id="model-1")
    scheduler = _fake_scheduler(
        requests={"r1": _fake_request(10, 10)},
        running=["r1"],
        waiting=[],
        skipped_waiting=[],
    )
    output = _fake_output(new_req_ids=["r1"], scheduled_tokens={"r1": 1})
    writer.record(scheduler, output, _decision())
    writer.close()

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record.keys()) == EXPECTED_FIELDS
    assert record["policy_status"] == "active"
    assert record["policy_source"] == "file"
    assert record["clamped"] == {"max_num_seqs": False, "max_num_batched_tokens": False}


def test_prefill_decode_split_new_request_counts_as_prefill(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="e", model_id="m")
    scheduler = _fake_scheduler(
        requests={"new1": _fake_request(0, 50)},
        running=[],
        waiting=[],
        skipped_waiting=[],
    )
    output = _fake_output(new_req_ids=["new1"], scheduled_tokens={"new1": 50})
    writer.record(scheduler, output, _decision())
    writer.close()
    record = json.loads(path.read_text().splitlines()[0])
    assert record["num_prefill_reqs"] == 1
    assert record["num_decode_reqs"] == 0


def test_prefill_decode_split_cached_mid_prompt_counts_as_prefill(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="e", model_id="m")
    scheduler = _fake_scheduler(
        requests={"cached1": _fake_request(num_computed_tokens=20, num_prompt_tokens=50)},
        running=["cached1"],
        waiting=[],
        skipped_waiting=[],
    )
    output = _fake_output(new_req_ids=[], scheduled_tokens={"cached1": 30})
    writer.record(scheduler, output, _decision())
    writer.close()
    record = json.loads(path.read_text().splitlines()[0])
    assert record["num_prefill_reqs"] == 1
    assert record["num_decode_reqs"] == 0


def test_prefill_decode_split_cached_finished_prompt_counts_as_decode(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="e", model_id="m")
    scheduler = _fake_scheduler(
        requests={"cached1": _fake_request(num_computed_tokens=50, num_prompt_tokens=50)},
        running=["cached1"],
        waiting=[],
        skipped_waiting=[],
    )
    output = _fake_output(new_req_ids=[], scheduled_tokens={"cached1": 1})
    writer.record(scheduler, output, _decision())
    writer.close()
    record = json.loads(path.read_text().splitlines()[0])
    assert record["num_prefill_reqs"] == 0
    assert record["num_decode_reqs"] == 1


def test_clamped_flag_true_when_decision_exceeds_startup_ceiling(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="e", model_id="m")
    scheduler = _fake_scheduler(
        requests={},
        running=[],
        waiting=[],
        skipped_waiting=[],
        startup_max_num_seqs=16,
        startup_max_num_batched_tokens=2048,
    )
    output = _fake_output(new_req_ids=[], scheduled_tokens={})
    decision = _decision(max_num_seqs=999, max_num_batched_tokens=999999)
    writer.record(scheduler, output, decision)
    writer.close()
    record = json.loads(path.read_text().splitlines()[0])
    assert record["clamped"] == {"max_num_seqs": True, "max_num_batched_tokens": True}


def test_no_path_configured_is_a_silent_noop():
    writer = TelemetryWriter(path=None, engine_id="e", model_id="m")
    scheduler = _fake_scheduler(requests={}, running=[], waiting=[], skipped_waiting=[])
    output = _fake_output(new_req_ids=[], scheduled_tokens={})
    writer.record(scheduler, output, _decision())  # must not raise
    writer.close()


def test_sample_every_n_steps_downsamples(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(
        path=path, engine_id="e", model_id="m", sample_every_n_steps=3
    )
    scheduler = _fake_scheduler(requests={}, running=[], waiting=[], skipped_waiting=[])
    output = _fake_output(new_req_ids=[], scheduled_tokens={})
    for _ in range(7):
        writer.record(scheduler, output, _decision())
    writer.close()
    lines = path.read_text().splitlines()
    assert len(lines) == 2  # steps 3 and 6
    assert json.loads(lines[0])["step"] == 3
    assert json.loads(lines[1])["step"] == 6


def test_expires_at_is_always_null(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(path=path, engine_id="e", model_id="m")
    scheduler = _fake_scheduler(requests={}, running=[], waiting=[], skipped_waiting=[])
    output = _fake_output(new_req_ids=[], scheduled_tokens={})
    writer.record(scheduler, output, _decision())
    writer.close()
    record = json.loads(path.read_text().splitlines()[0])
    assert record["expires_at"] is None
