from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_primary_usdt():
    path = Path(__file__).parents[2] / "vllm" / "primary_usdt.py"
    spec = importlib.util.spec_from_file_location("g2_primary_usdt_abi", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load_primary_usdt()


def _tail(event_name: str) -> dict[str, int]:
    return {name: 0 for name in M.EVENT_TAIL_FIELDS[event_name]}


def test_a01_all_event_fragments_have_fixed_six_scalar_arguments():
    assert M.ABI_VERSION == 0x0002_0004
    assert len(M.EVENT_TAIL_FIELDS) == 31
    assert "request_cleanup_v3" in M.EVENT_TAIL_FIELDS
    assert "request_cleanup_v2" not in M.EVENT_TAIL_FIELDS
    assert "execution_span_created_v1" in M.EVENT_TAIL_FIELDS
    assert M.EVENT_TAIL_FIELDS["scheduler_state_transition_v3"][-4:] == (
        "transition_reason",
        "transition_initiator",
        "transition_action_id_hi",
        "transition_action_id_lo",
    )
    assert M.EVENT_TAIL_FIELDS["request_terminal_v3"][:6] == (
        "terminal_state",
        "terminal_reason",
        "transition_reason",
        "transition_initiator",
        "transition_action_id_hi",
        "transition_action_id_lo",
    )
    for event_name in M.EVENT_TAIL_FIELDS:
        layout = M.fragment_layout(event_name)
        assert layout
        assert all(1 <= len(fields) <= M.VALUES_PER_FRAGMENT for _, fields in layout)
        assert len({probe for probe, _ in layout}) == len(layout)
        assert all(probe.startswith(event_name + "__f") for probe, _ in layout)


def test_a14_capture_backend_emits_every_declared_fragment_without_truncation(
    monkeypatch,
):
    monkeypatch.setenv(
        "VLLM_PRIMARY_PROFILE", "primary_current_plus_execution_span"
    )
    backend = M.CaptureBackend()
    emitter = M.PrimaryUSDTEmitter(backend, enabled=True)
    expected = 0
    for event_name in M.EVENT_TAIL_FIELDS:
        tail = _tail(event_name)
        if "block_generation" in tail:
            tail["block_generation"] = 1
        result = emitter.emit(event_name, engine_instance_id=(1, 2), **tail)
        assert result.complete
        assert result.emitted_fragments == result.expected_fragments
        expected += result.expected_fragments
    assert len(backend.records) == expected
    assert all(len(arguments) == M.MAX_PROVIDER_ARGUMENTS for _, arguments in backend.records)
    assert all(all(isinstance(value, int) for value in arguments) for _, arguments in backend.records)


def test_queue_member_amendment_has_fixed_identity_membership_and_fragments():
    assert M.EVENT_TAIL_FIELDS["scheduler_queue_member_v2"] == (
        "scheduler_step_id",
        "scheduler_output_id",
        "scheduler_batch_id",
        "queue_snapshot_id",
        "queue_name",
        "queue_position",
        "member_count",
        "request_state",
    )
    assert "queue_snapshot_id" in M.EVENT_TAIL_FIELDS["scheduler_queue_snapshot_v2"]
    member_layout = M.fragment_layout("scheduler_queue_member_v2")
    snapshot_layout = M.fragment_layout("scheduler_queue_snapshot_v2")
    assert len(member_layout) == 6
    assert len(snapshot_layout) == 6
    assert all(len(fields) == 3 for _, fields in member_layout)
    assert all(len(fields) == 3 for _, fields in snapshot_layout)


def test_a15_mid_record_failure_is_explicit_and_never_marked_complete():
    backend = M.CaptureBackend(fail_after=2)
    emitter = M.PrimaryUSDTEmitter(
        backend, enabled=True, strict_failure=False
    )
    result = emitter.emit(
        "worker_slot_mapping_entry_v2",
        engine_instance_id=(1, 2),
        **_tail("worker_slot_mapping_entry_v2"),
    )
    assert not result.complete
    assert result.emitted_fragments == 2
    assert result.failure
    assert emitter.failures["worker_slot_mapping_entry_v2"] == 1


def test_a14_rejects_missing_extra_dynamic_and_out_of_range_fields():
    emitter = M.PrimaryUSDTEmitter(M.CaptureBackend(), enabled=True)
    with pytest.raises(M.PrimaryUSDTError, match="tail mismatch"):
        emitter.emit("request_arrival_v2", engine_instance_id=(1, 2))
    with pytest.raises(M.PrimaryUSDTError, match="tail mismatch"):
        emitter.emit(
            "request_arrival_v2",
            engine_instance_id=(1, 2),
            submission_attempt_id=0,
            lifecycle_state=1,
            invented=3,
        )
    with pytest.raises(M.PrimaryUSDTError, match="outside uint64"):
        emitter.emit(
            "request_arrival_v2",
            engine_instance_id=(1, 2),
            submission_attempt_id=-1,
            lifecycle_state=1,
        )


def test_a14_native_libstapsdt_provider_loads_and_fires_without_attach():
    emitter = M.PrimaryUSDTEmitter(enabled=True, strict_failure=True)
    result = emitter.emit(
        "request_arrival_v2",
        engine_instance_id=(1, 2),
        submission_attempt_id=0,
        lifecycle_state=1,
    )
    assert result.complete
    assert result.emitted_fragments == result.expected_fragments
