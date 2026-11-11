# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.engine import dp_prefill_hold as dph
from vllm.v1.engine.dp_prefill_hold import PrefillHold
from vllm.v1.outputs import ModelRunnerOutput

pytestmark = pytest.mark.cpu_test

T = 15.0
T_DEC = 0.025
RANKS = 8
# Extra ms of a mixed step: fixed, per busiest-rank Ktok, per total Ktok and
# per busiest-rank Gtok^2 of attention work.
TRUTH = (15.0, 12.0, 3.0, 40.0)


def _row(**fields: float) -> list[float]:
    row = [0.0] * dph.ROW_SIZE
    row[dph._SLACK] = T
    for name, value in fields.items():
        row[getattr(dph, "_" + name.upper())] = value
    return row


def _extra_ms(tokens: list[int], attn: list[float]) -> float:
    f, a, b, c = TRUTH
    return f + a * max(tokens) / 1e3 + b * sum(tokens) / 1e3 + c * max(attn) / 1e9


def _measure(hold: PrefillHold, rng: random.Random, steps: int, max_work: int):
    """Feeds `steps` measured mixed steps with 1 to `max_work` ranks prefilling."""
    for _ in range(steps):
        n_work = rng.randint(1, max_work)
        tokens = [rng.randint(256, 8192) if r < n_work else 0 for r in range(RANKS)]
        attn = [t * (rng.uniform(0, 50e3) + t / 2) for t in tokens]
        period = T_DEC + _extra_ms(tokens, attn) / 1e3 * rng.uniform(0.99, 1.01)
        hold.update(
            [
                _row(done=1, period=period, done_tokens=t, done_attn=a, decodes=16)
                for t, a in zip(tokens, attn)
            ]
        )


def _warm(hold: PrefillHold, rng: random.Random, max_work: int = RANKS) -> None:
    decode_only = [_row(done=1, period=T_DEC, decodes=16)] * RANKS
    hold.update(decode_only)  # The first sync has no step to time.
    hold.update(decode_only)
    _measure(hold, rng, 100, max_work)
    # 50 req/s of 4K uncached tokens each.
    hold.update([_row(arrived=1500 / RANKS, arrived_uncached=4000 * 1500 / RANKS)] * 8)


def _one_rank_queued(slack: float = T) -> list[list[float]]:
    """2K tokens queued on one rank, 16 decodes on every rank."""
    busy = _row(
        period=T_DEC,
        decodes=16,
        work=1,
        x=2048,
        x_attn=2048 * (1000 + 1024),
        queued=1,
        slack=slack,
    )
    return [busy] + [_row(period=T_DEC, decodes=16)] * (RANKS - 1)


def test_releases_until_steps_are_measured():
    """With no measured steps there is no cost model, so it fails open."""
    hold = PrefillHold(None, T, 1.0, 8192)
    assert hold.update(_one_rank_queued()) == "cold"
    assert not hold.hold


def test_holds_while_peers_are_expected_and_waiting_is_cheap():
    """At 50 req/s, one more decode step is likely to bring prefill to other
    ranks, which costs 128 decodes less than it costs one queued request."""
    hold = PrefillHold(None, T, 1.0, 8192)
    _warm(hold, random.Random(0))
    assert hold.update(_one_rank_queued()) == "wait"
    assert hold.hold

    hold.alpha = 1e4
    assert hold.update(_one_rank_queued()) == "cost"
    assert not hold.hold


def test_ttft_target_bounds_the_hold():
    """A request whose remaining allowance cannot cover a mixed step is
    released even though waiting would pay."""
    hold = PrefillHold(None, T, 1.0, 8192)
    _warm(hold, random.Random(0))
    assert hold.update(_one_rank_queued(slack=0.01)) == "deadline"
    assert not hold.hold


def test_unsure_until_aligned_steps_are_measured():
    """Steps with one rank prefilling cannot tell the busiest-rank cost from
    the all-rank cost, which is what waiting for peers hinges on: it releases
    until steps with several ranks prefilling pin that down."""
    hold = PrefillHold(None, T, 1.0, 8192)
    rng = random.Random(0)
    _warm(hold, rng, max_work=1)
    assert hold.update(_one_rank_queued()) == "unsure"
    _measure(hold, rng, 100, RANKS)
    assert hold.update(_one_rank_queued()) == "wait"


def test_idle_holds_only_where_a_lone_arrival_would_be_held():
    """With nothing queued, the next step may admit new arrivals before the
    next sync sees them: hold if a typical request would be held."""
    hold = PrefillHold(None, T, 1.0, 8192)
    _warm(hold, random.Random(0))
    idle = [_row(period=T_DEC, decodes=16)] * RANKS
    assert hold.update(idle) == "idle"
    assert hold.hold

    hold.alpha = 1e4
    assert hold.update(idle) == "nothing"
    assert not hold.hold


def _finish_step(scheduler, output) -> None:
    """Completes a step, sampling a token for every request done prefilling."""
    req_ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            sampled_token_ids=[
                [] if scheduler.requests[r].is_prefill_chunk else [0] for r in req_ids
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def test_local_row_offers_one_budget_and_scales_slack_by_releases_left():
    """A rank offers at most one budget of prefill, and a request needing
    three releases may wait a third of its target per hold."""
    scheduler = create_scheduler(max_num_batched_tokens=500, max_model_len=2048)
    (req,) = create_requests(num_requests=1, num_tokens=1500)
    scheduler.add_request(req)
    hold = PrefillHold(scheduler, 10.0, 1.0, 500)
    hold.on_arrival(req)
    row = hold.local_row()
    assert row[dph._ARRIVED] == 1 and row[dph._ARRIVED_UNCACHED] == 1500
    assert row[dph._X] == 500
    assert row[dph._X_ATTN] == 500 * 250
    assert row[dph._QUEUED] == 1
    assert row[dph._SLACK] == pytest.approx(10.0 / 3, abs=0.1)


def test_rank_without_a_free_slot_is_blocked():
    """A rank whose slots are all taken cannot start its queued prefill, so it
    offers none. Its last finished step is still reported for timing."""
    scheduler = create_scheduler(max_num_seqs=1)
    running, queued = create_requests(num_requests=2, num_tokens=8)
    scheduler.add_request(running)
    _finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(queued)
    row = PrefillHold(scheduler, T, 1.0, 8192).local_row()
    assert row[dph._DONE] == 1 and row[dph._DONE_TOKENS] == 8
    assert row[dph._DECODES] == 1
    assert row[dph._BLOCKED] == 1 and row[dph._X] == 0


def test_hold_overrides_per_rank_capacity_escape():
    """With `prefill_schedule_interval`, a rank whose last release left a
    backlog keeps prefilling on throttled steps. A hold applies to every rank
    alike, so that rank holds too."""
    scheduler = create_scheduler(max_num_batched_tokens=200)
    dec, a, b = create_requests(num_requests=3, num_tokens=200, req_ids=["d", "a", "b"])
    scheduler.add_request(dec)
    _finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(a)
    scheduler.add_request(b)
    _finish_step(scheduler, scheduler.schedule())
    assert scheduler.prefill_capacity_bound

    hold = PrefillHold(scheduler, T, 1.0, 200)
    hold.hold = True
    output = scheduler.schedule(throttle_prefills=hold.throttle())
    assert set(output.num_scheduled_tokens) == {"d"}
