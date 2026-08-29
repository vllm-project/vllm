# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for scheduler.update_from_output KV-connector guard.

Covers the fix for KeyError when a KV-connector (e.g. MoRIIO WideEP)
schedules synthetic health-probe requests whose compound req_ids appear
in num_scheduled_tokens but are absent from
model_runner_output.req_id_to_index.

See: vllm/v1/core/sched/scheduler.py update_from_output()
"""
from unittest.mock import MagicMock

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_scheduler

pytestmark = pytest.mark.cpu_test


def _make_scheduler_output(num_tokens: dict) -> SchedulerOutput:
    """Create a minimal SchedulerOutput with the given req_id → num_tokens map."""
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[],
        num_scheduled_tokens=num_tokens,
        total_num_scheduled_tokens=sum(num_tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_requests_ids=[],
        free_encoder_input_ids=[],
        preempted_reqs=[],
    )


def _make_model_runner_output(req_ids: list[str]) -> ModelRunnerOutput:
    """Create a minimal ModelRunnerOutput containing only the given req_ids."""
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[[42]] * len(req_ids),
    )


def _add_mock_request(scheduler, req_id: str, num_tokens: int = 4) -> None:
    """Register a mock in-flight request with the scheduler."""
    mock_req = MagicMock()
    mock_req.num_in_flight_tokens = num_tokens
    mock_req.num_stale_output_tokens = 0
    mock_req.is_finished.return_value = False
    mock_req.drop_stale_output = False
    scheduler.requests[req_id] = mock_req


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MORIIO_SYNTHETIC_REQ = (
    "cmpl-___prefill_addr_host:10.0.0.1,handshake:8405,"
    "notify:61005___decode_addr_host:10.0.0.2,handshake:8405,"
    "notify:61005_abc123def456-0-789xyz"
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_keyerror_with_single_synthetic_req_id():
    """A synthetic KV-connector req_id absent from the output is silently skipped.

    Reproduces the crash reported for MoRIIO WideEP PD-disaggregated mode
    where health-probe requests use compound req_ids that are scheduled but
    not included in model_runner_output.req_id_to_index.
    """
    scheduler = create_scheduler()

    NORMAL = "real-req-0001"
    _add_mock_request(scheduler, NORMAL)

    sched_out = _make_scheduler_output({NORMAL: 4, _MORIIO_SYNTHETIC_REQ: 1})
    # Synthetic req_id intentionally absent from model runner output
    model_out = _make_model_runner_output([NORMAL])

    # Must not raise KeyError
    try:
        scheduler.update_from_output(sched_out, model_out)
    except KeyError as exc:
        pytest.fail(
            f"KeyError for synthetic KV-connector req_id: {exc!r}. "
            "Add `if req_id not in model_runner_output.req_id_to_index: continue` "
            "before the req_id_to_index lookup in update_from_output()."
        )


def test_normal_requests_processed_alongside_synthetic():
    """Normal req_ids are still updated when synthetic req_ids are present."""
    scheduler = create_scheduler()

    NORMAL = "real-req-0002"
    _add_mock_request(scheduler, NORMAL)

    sched_out = _make_scheduler_output({NORMAL: 4, _MORIIO_SYNTHETIC_REQ: 1})
    model_out = _make_model_runner_output([NORMAL])

    # Should not raise and the normal request should be updated
    scheduler.update_from_output(sched_out, model_out)


def test_multiple_synthetic_req_ids_skipped():
    """Multiple synthetic req_ids in one batch are all skipped without error."""
    scheduler = create_scheduler()

    NORMAL = "real-req-0003"
    _add_mock_request(scheduler, NORMAL)

    synthetics = [
        f"cmpl-___p:10.0.0.{i},h:8405___d:10.0.0.{i+10},h:8405_uuid{i}"
        for i in range(5)
    ]
    num_tokens = {NORMAL: 4}
    num_tokens.update({s: 1 for s in synthetics})

    sched_out = _make_scheduler_output(num_tokens)
    model_out = _make_model_runner_output([NORMAL])

    try:
        scheduler.update_from_output(sched_out, model_out)
    except KeyError as exc:
        pytest.fail(f"KeyError with {len(synthetics)} synthetic req_ids: {exc!r}")


def test_purely_synthetic_batch_does_not_crash():
    """A batch with only synthetic req_ids and no real output is safe."""
    scheduler = create_scheduler()

    sched_out = _make_scheduler_output({_MORIIO_SYNTHETIC_REQ: 1})
    model_out = _make_model_runner_output([])  # empty output

    try:
        scheduler.update_from_output(sched_out, model_out)
    except KeyError as exc:
        pytest.fail(f"KeyError for all-synthetic batch: {exc!r}")
