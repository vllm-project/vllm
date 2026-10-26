# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config import SchedulerConfig
from vllm.v1.core.sched.prefill_delayer import (
    PARTIAL_MAX_HOLD_STEPS,
    STALL_STEPS,
    TTFT_MAX_HOLD_STEPS,
    PrefillDelayer,
    local_state,
)
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

BUDGET = 100


def _delayer(dp_size: int = 2, protected_steps: int = 10) -> PrefillDelayer:
    delayer = PrefillDelayer(dp_size, BUDGET, protected_steps)
    assert delayer.decide(0, 0, 0, 0, 0, 0) == (True, "first")
    return delayer


def _state(prefillable=2, pending=10, decodes=4, kv_high=0, partial=0, ran=0):
    return (prefillable, pending, decodes, kv_high, partial, ran)


@pytest.mark.parametrize("protected_steps", [0, 3])
def test_protected_steps_after_prefill(protected_steps: int):
    delayer = _delayer(protected_steps=protected_steps)
    full = _state(pending=BUDGET * 2)
    for i in range(protected_steps):
        assert delayer.decide(*full[:5], ran=int(i == 0)) == (False, "protected")
    assert delayer.decide(*full[:5], ran=int(protected_steps == 0)) == (True, "fill")


def test_protection_needs_running_decodes():
    delayer = _delayer()
    assert delayer.decide(*_state(decodes=0, ran=1)) == (True, "no_decode")


@pytest.mark.parametrize(
    "state, reason",
    [
        (_state(prefillable=0, pending=0), "no_prefill"),
        (_state(decodes=0), "no_decode"),
        (_state(prefillable=1, kv_high=1), "kv_high"),
        (_state(pending=int(0.9 * 2 * BUDGET)), "fill"),
    ],
)
def test_fire_rules(state, reason):
    assert _delayer().decide(*state) == (True, reason)


@pytest.mark.parametrize(
    "partial, bound, reason",
    [(0, TTFT_MAX_HOLD_STEPS, "ttft"), (1, PARTIAL_MAX_HOLD_STEPS, "partial")],
)
def test_alignment_holds_until_bound(partial: int, bound: int, reason: str):
    """One of two ranks has prefill: HOLD for alignment, then FIRE at the
    bound (in-progress chunks have the shorter one)."""
    delayer = _delayer()
    state = _state(prefillable=1, partial=partial)
    for _ in range(bound):
        assert delayer.decide(*state) == (False, "align")
    assert delayer.decide(*state) == (True, reason)
    assert delayer.decide(*state) == (False, "align")


def test_fill_hold_until_stall():
    delayer = _delayer()
    assert delayer.decide(*_state(pending=10)) == (False, "fill")
    assert delayer.decide(*_state(pending=20)) == (False, "fill")
    for _ in range(STALL_STEPS - 1):
        assert delayer.decide(*_state(pending=20)) == (False, "fill")
    assert delayer.decide(*_state(pending=20)) == (True, "stall")


def test_reset_drops_protection():
    delayer = _delayer()
    assert delayer.decide(*_state(ran=1)) == (False, "protected")
    delayer.reset()
    assert delayer.decide(*_state(pending=BUDGET * 2)) == (True, "fill")


def test_local_state():
    """Report waiting, chunked, and decode work from real scheduler transitions."""
    scheduler = create_scheduler(
        max_num_batched_tokens=BUDGET,
        max_model_len=512,
        dp_prefill_delayer=True,
    )
    (long,) = create_requests(num_requests=1, num_tokens=150, req_ids=["long"])
    (short,) = create_requests(num_requests=1, num_tokens=30, req_ids=["short"])
    scheduler.add_request(long)
    scheduler.add_request(short)
    assert local_state(scheduler, BUDGET) == [1, BUDGET, 0, 0, 0, 0]

    output = scheduler.schedule()
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=["long"], req_id_to_index={"long": 0}, sampled_token_ids=[[]]
        ),
    )
    assert local_state(scheduler, BUDGET) == [1, 80, 0, 0, 1, 1]
    assert local_state(scheduler, BUDGET) == [1, 80, 0, 0, 1, 0]

    output = scheduler.schedule()
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=["long", "short"],
            req_id_to_index={"long": 0, "short": 1},
            sampled_token_ids=[[42], [42]],
        ),
    )
    assert local_state(scheduler, BUDGET) == [0, 0, 2, 0, 0, 1]


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_scheduler_holds_prefill_without_local_decodes(async_scheduling: bool):
    """Unlike the cadence, a HOLD defers prefill on a rank with no running
    requests, so that it runs with the other ranks' prefill."""
    scheduler = create_scheduler(
        dp_prefill_delayer=True, async_scheduling=async_scheduling
    )
    (request,) = create_requests(num_requests=1, num_tokens=8, req_ids=["r0"])
    scheduler.add_request(request)

    output = scheduler.schedule(throttle_prefills=True)
    assert "r0" not in output.num_scheduled_tokens
    assert not scheduler.ran_prefill

    output = scheduler.schedule()
    assert "r0" in output.num_scheduled_tokens
    assert scheduler.ran_prefill


def test_config_rejects_prefill_schedule_interval():
    with pytest.raises(ValueError, match="prefill_schedule_interval"):
        SchedulerConfig(
            max_num_batched_tokens=8192,
            max_model_len=8192,
            is_encoder_decoder=False,
            dp_prefill_delayer=True,
            prefill_schedule_interval=2,
        )
