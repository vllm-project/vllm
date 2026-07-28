from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _load_primary_usdt():
    path = Path(__file__).parents[2] / "vllm" / "primary_usdt.py"
    spec = importlib.util.spec_from_file_location("pre_g4_primary_usdt", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load_primary_usdt()


def _configure(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    control = tmp_path / "control"
    denominator = tmp_path / "denominator"
    monkeypatch.setenv("VLLM_PRIMARY_USDT_CONTROL_DIR", str(control))
    monkeypatch.setenv("VLLM_PRIMARY_USDT_DENOMINATOR_DIR", str(denominator))
    monkeypatch.setenv("VLLM_PRIMARY_USDT_REQUIRE_READINESS", "1")
    monkeypatch.setenv("VLLM_PRIMARY_USDT_REQUIRE_DENOMINATOR", "1")
    monkeypatch.setenv("VLLM_PRIMARY_RUN_ID", "04000000000000000000000000000004")
    monkeypatch.setenv("VLLM_PRIMARY_ATTEMPT_ID", "PRE-G4-CPU-A01")
    return control, denominator


def _records(path: Path) -> list[tuple]:
    data = path.read_bytes()
    assert len(data) % M.DENOMINATOR_RECORD.size == 0
    return [
        M.DENOMINATOR_RECORD.unpack_from(data, offset)
        for offset in range(0, len(data), M.DENOMINATOR_RECORD.size)
    ]


def test_eager_readiness_precedes_first_semantic_event(monkeypatch, tmp_path):
    control, denominator = _configure(monkeypatch, tmp_path)
    emitter = M.PrimaryUSDTEmitter(M.CaptureBackend(), enabled=True)

    readiness = emitter.prepare("frontend_driver")

    assert readiness.ready
    assert readiness.eager_loaded_before_first_semantic_event
    payload = json.loads((control / "readiness" / f"{readiness.pid}.json").read_text())
    assert payload["semantic_events_attempted_at_handshake"] == 0
    assert payload["physical_probe_count"] == 185
    assert payload["registry_sha256"] == M.AUTHORITATIVE_REGISTRY_SHA256
    denominator_path = next(denominator.glob("*.source-denominator.bin"))
    assert denominator_path.stat().st_size == 0

    result = emitter.emit(
        "request_arrival_v2",
        engine_instance_id=(1, 2),
        submission_attempt_id=0,
        lifecycle_state=1,
    )

    assert result.source_event_sequence == 1
    records = _records(denominator_path)
    assert [row[2] for row in records] == [
        M.DENOMINATOR_LOGICAL_ATTEMPT,
        *([M.DENOMINATOR_FRAGMENT_ATTEMPT] * 4),
    ]
    assert {row[9] for row in records} == {readiness.pid}
    assert [row[13] for row in records[1:]] == [1, 2, 3, 4]
    assert [row[3] for row in records[1:]] == [0, 1, 2, 3]


def test_native_provider_is_loaded_and_discoverable_before_first_event(
    monkeypatch, tmp_path
):
    control, denominator = _configure(monkeypatch, tmp_path)
    try:
        backend = M._LibStapSDTBackend()
    except OSError as exc:
        pytest.skip(f"libstapsdt unavailable to the CPU contract test: {exc}")
    emitter = M.PrimaryUSDTEmitter(backend, enabled=True)

    readiness = emitter.prepare("engine_core")

    provider_path = Path(readiness.provider_path)
    assert provider_path.is_file()
    assert len(readiness.provider_sha256) == 64
    assert len(backend._probes) == M.PHYSICAL_PROBE_COUNT
    payload = json.loads((control / "readiness" / f"{readiness.pid}.json").read_text())
    assert payload["semantic_events_attempted_at_handshake"] == 0
    assert payload["provider_path"] == str(provider_path)
    assert next(denominator.glob("*.source-denominator.bin")).stat().st_size == 0


def test_required_readiness_fails_closed_before_provider_or_sequence(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    backend = M.CaptureBackend()
    emitter = M.PrimaryUSDTEmitter(backend, enabled=True)
    with pytest.raises(M.PrimaryUSDTError, match="before readiness"):
        emitter.emit(
            "request_arrival_v2",
            engine_instance_id=(1, 2),
            submission_attempt_id=0,
            lifecycle_state=1,
        )
    assert not backend.probes
    assert not backend.records


def test_failed_readiness_publication_stays_fail_closed(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    backend = M.CaptureBackend()
    emitter = M.PrimaryUSDTEmitter(backend, enabled=True)
    monkeypatch.setattr(
        emitter,
        "_write_readiness",
        lambda readiness: (_ for _ in ()).throw(OSError("injected readiness failure")),
    )

    with pytest.raises(OSError, match="injected readiness failure"):
        emitter.prepare("engine_core")
    with pytest.raises(M.PrimaryUSDTError, match="before readiness"):
        emitter.emit(
            "request_arrival_v2",
            engine_instance_id=(1, 2),
            submission_attempt_id=0,
            lifecycle_state=1,
        )
    assert backend.probes
    assert not backend.records


def test_fragment_fire_failure_has_independent_exact_denominator(
    monkeypatch, tmp_path
):
    _, denominator = _configure(monkeypatch, tmp_path)
    backend = M.CaptureBackend(fail_after=2)
    emitter = M.PrimaryUSDTEmitter(backend, enabled=True, strict_failure=False)
    emitter.prepare("engine_core")

    result = emitter.emit(
        "worker_slot_mapping_entry_v2",
        engine_instance_id=(1, 2),
        **{
            name: 0
            for name in M.EVENT_TAIL_FIELDS["worker_slot_mapping_entry_v2"]
        },
    )

    assert not result.complete and result.emitted_fragments == 2
    records = _records(next(denominator.glob("*.source-denominator.bin")))
    kinds = [row[2] for row in records]
    assert kinds == [
        M.DENOMINATOR_LOGICAL_ATTEMPT,
        M.DENOMINATOR_FRAGMENT_ATTEMPT,
        M.DENOMINATOR_FRAGMENT_ATTEMPT,
        M.DENOMINATOR_FRAGMENT_ATTEMPT,
        M.DENOMINATOR_FRAGMENT_FIRE_FAILURE,
    ]
    failure = records[-1]
    assert failure[3] == 2
    assert failure[13] == M.PHYSICAL_PROBE_IDS[("worker_slot_mapping_entry_v2", 2)]
    assert failure[14] == 1


def test_entrypoints_prepare_each_expected_process_role_before_events():
    root = Path(__file__).parents[2] / "vllm"
    llm = (root / "entrypoints" / "llm.py").read_text()
    async_llm = (root / "v1" / "engine" / "async_llm.py").read_text()
    core = (root / "v1" / "engine" / "core.py").read_text()
    worker = (root / "v1" / "worker" / "gpu_model_runner.py").read_text()
    assert 'prepare_primary_usdt_provider("frontend_driver")' in llm
    assert 'prepare_primary_usdt_provider("frontend_driver")' in async_llm
    assert 'prepare_primary_usdt_provider("engine_core")' in core
    assert "prepare_primary_usdt_provider(f\"worker_rank_" in worker
