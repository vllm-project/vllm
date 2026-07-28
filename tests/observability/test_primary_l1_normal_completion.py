# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.primary_usdt import (
    PrimaryUSDTError,
    TransitionInitiator,
    TransitionReason,
)
from vllm.v1.core.sched import scheduler as scheduler_module
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus


ROOT = Path(__file__).parents[2]
SCHEDULER_PATH = ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py"
PRODUCER_CALL = "self._transition_request_state("
RELEASE_CALL = "kv_transfer_params = self._free_request(request)"


def _producer_block(source: str) -> str:
    ast.parse(source)
    start = source.index("                if finished:\n")
    end = source.index("\n\n                if status_before_stop", start)
    return source[start:end]


def _validate_normal_completion_projection(source: str) -> None:
    block = _producer_block(source)
    producer_count = block.count(PRODUCER_CALL)
    if producer_count == 0:
        raise AssertionError("NORMAL_COMPLETION_TRANSITION_MISSING")
    if producer_count != 1:
        raise AssertionError("DUPLICATE_NORMAL_COMPLETION_TRANSITION")
    if block.index(PRODUCER_CALL) > block.index(RELEASE_CALL):
        raise AssertionError("STATE_TRANSITION_NOT_BEFORE_ENGINE_RELEASE")


def test_normal_completion_projection_is_exactly_once_before_release():
    source = SCHEDULER_PATH.read_text()
    _validate_normal_completion_projection(source)
    block = _producer_block(source)
    assert "stop_status" in block
    assert "scheduler_output" in block


def test_normal_completion_producer_uses_authoritative_status_and_identities():
    source = SCHEDULER_PATH.read_text()
    tree = ast.parse(source)
    helper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_transition_request_state"
    )
    helper_source = ast.get_source_segment(source, helper)
    assert helper_source is not None
    required = (
        "previous_state = request.status",
        "request.status = next_state",
        "primary_request_id_hash(request.request_id)",
        "scheduler_step_id=scheduler_step_id",
        "scheduler_output_id=scheduler_output_id",
        "scheduler_batch_id=scheduler_batch_id",
        "state_before=int(previous_state.value)",
        "state_after=int(next_state.value)",
        "transition_reason=int(reason)",
        "transition_initiator=int(initiator)",
        "transition_action_id_hi=action_id[0]",
        "transition_action_id_lo=action_id[1]",
    )
    assert all(token in helper_source for token in required)
    forbidden = (
        "request_cleanup",
        "terminal",
        "nearest",
        "timestamp",
        "_free_request(",
    )
    assert all(token not in helper_source for token in forbidden)


def test_normal_completion_projection_carries_real_final_state_and_step_identity(
    monkeypatch,
):
    emitted = []
    monkeypatch.setattr(
        scheduler_module,
        "emit_primary_usdt",
        lambda event_name, **fields: emitted.append((event_name, fields)),
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.vllm_config = SimpleNamespace(
        observability_config=SimpleNamespace(
            primary_engine_instance_id_hi=17,
            primary_engine_instance_id_lo=19,
        )
    )
    scheduler._primary_transition_actions = set()
    request = SimpleNamespace(
        request_id="normal-completion-request",
        status=RequestStatus.RUNNING,
    )

    scheduler._transition_request_state(
        request,
        RequestStatus.FINISHED_STOPPED,
        TransitionReason.NORMAL_COMPLETION,
        TransitionInitiator.MODEL_EXECUTION,
        transition_action_id=(37, 41),
        scheduler_step_id=23,
        scheduler_output_id=29,
        scheduler_batch_id=31,
    )

    assert len(emitted) == 1
    event_name, fields = emitted[0]
    assert event_name == "scheduler_state_transition_v3"
    assert fields["engine_instance_id"] == (17, 19)
    assert fields["request_id_hash"] != 0
    assert fields["scheduler_step_id"] == 23
    assert fields["scheduler_output_id"] == 29
    assert fields["scheduler_batch_id"] == 31
    assert fields["state_before"] == RequestStatus.RUNNING.value
    assert fields["state_after"] == RequestStatus.FINISHED_STOPPED.value
    assert fields["transition_reason"] == TransitionReason.NORMAL_COMPLETION
    assert fields["transition_initiator"] == TransitionInitiator.MODEL_EXECUTION
    assert fields["transition_action_id_hi"] == 37
    assert fields["transition_action_id_lo"] == 41
    assert request.status == RequestStatus.FINISHED_STOPPED


def test_normal_completion_projection_rejects_no_status_change(monkeypatch):
    monkeypatch.setattr(
        scheduler_module,
        "emit_primary_usdt",
        lambda *args, **kwargs: pytest.fail("invalid transition emitted"),
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.vllm_config = SimpleNamespace(observability_config=SimpleNamespace())
    scheduler._primary_transition_actions = set()
    request = SimpleNamespace(
        request_id="not-finished",
        status=RequestStatus.RUNNING,
    )
    with pytest.raises(PrimaryUSDTError, match="Q2_TRANSITION_NO_CHANGE"):
        scheduler._transition_request_state(
            request,
            RequestStatus.RUNNING,
            TransitionReason.NORMAL_COMPLETION,
            TransitionInitiator.MODEL_EXECUTION,
            transition_action_id=(1, 2),
        )


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            lambda block: block.replace(
                PRODUCER_CALL,
                "self._deleted_transition_request_state(",
                1,
            ),
            "NORMAL_COMPLETION_TRANSITION_MISSING",
        ),
        (
            lambda block: block.replace(RELEASE_CALL, "", 1).replace(
                PRODUCER_CALL,
                RELEASE_CALL + "\n                    " + PRODUCER_CALL,
                1,
            ),
            "STATE_TRANSITION_NOT_BEFORE_ENGINE_RELEASE",
        ),
        (
            lambda block: block.replace(
                "                if finished:\n",
                "                if finished:\n                    # "
                + PRODUCER_CALL
                + "\n",
                1,
            ),
            "DUPLICATE_NORMAL_COMPLETION_TRANSITION",
        ),
    ],
)
def test_required_mutations_fail_closed(mutation, expected_error):
    source = SCHEDULER_PATH.read_text()
    original_block = _producer_block(source)
    mutated_block = mutation(original_block)
    mutated_source = source.replace(original_block, mutated_block, 1)
    with pytest.raises(AssertionError, match=expected_error):
        _validate_normal_completion_projection(mutated_source)
