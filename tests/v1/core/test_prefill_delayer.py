# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.config import SchedulerConfig
from vllm.v1.core.sched.prefill_delayer import (
    PARTIAL_MAX_HOLD_STEPS,
    STALL_STEPS,
    TTFT_MAX_HOLD_STEPS,
    PrefillDelayer,
    local_state,
)

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


class _Req:
    def __init__(self, num_tokens: int, num_computed: int = 0, chunk: bool = False):
        self.num_tokens = num_tokens
        self.num_computed_tokens = num_computed
        self.is_prefill_chunk = chunk


def test_local_state():
    deferred = _Req(500)
    scheduler = SimpleNamespace(
        running=[_Req(60, 20, chunk=True), _Req(9, 8), _Req(9, 8)],
        waiting=[deferred, _Req(30), _Req(80)],
        deferred_waiting={deferred},
        kv_cache_manager=SimpleNamespace(usage=0.95),
        ran_prefill=True,
    )
    # 40 chunk tokens + 30 + 80 waiting, capped at the budget; the deferred
    # request does not count.
    assert local_state(scheduler, BUDGET) == [1, BUDGET, 2, 1, 1, 1]
    assert not scheduler.ran_prefill

    scheduler.running, scheduler.waiting = [_Req(9, 8)], []
    assert local_state(scheduler, BUDGET) == [0, 0, 1, 0, 0, 0]


def test_scheduler_hold_skips_rank_without_local_decodes():
    """A HOLD defers prefill only on a rank with a decode to run. A rank with
    nothing else to schedule prefills anyway: left with an empty batch while it
    still has requests, it runs both the empty batch and a dummy batch in one
    step while its DP peers run one, and the engine hangs."""
    scheduler = create_scheduler()
    scheduler.dp_prefill_delayer = True
    (request,) = create_requests(num_requests=1, num_tokens=8, req_ids=["r0"])
    scheduler.add_request(request)

    output = scheduler.schedule(throttle_prefills=True)
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
