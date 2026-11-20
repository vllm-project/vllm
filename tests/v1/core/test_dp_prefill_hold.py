# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
import time
from unittest.mock import Mock

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from tests.v1.kv_connector.unit.utils import MockKVConfig
from vllm.config import ModelConfig
from vllm.v1.core.sched import dp_prefill_hold as dph
from vllm.v1.core.sched.dp_prefill_hold import PrefillHold
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test

T = 15.0
T_DEC = 0.025
RANKS = 8
# Extra ms of a mixed step: fixed, per busiest-rank Ktok, per total Ktok and
# per busiest-rank Gtok^2 of attention work.
TRUTH = (15.0, 12.0, 3.0, 40.0)


@pytest.fixture(autouse=True)
def _moe_model(monkeypatch):
    """The hold needs DP on a MoE model; the dense test model stands in."""
    monkeypatch.setattr(ModelConfig, "is_moe", property(lambda self: True))


def _row(**fields: float) -> list[float]:
    row = [0.0] * dph.ROW_SIZE
    row[dph._SLACK] = T
    for name, value in fields.items():
        row[getattr(dph, "_" + name.upper())] = value
    return row


def _extra_ms(tokens: list[int], attn: list[float]) -> float:
    f, a, b, c = TRUTH
    return f + a * max(tokens) / 1e3 + b * sum(tokens) / 1e3 + c * max(attn) / 1e9


def _measure(
    hold: PrefillHold,
    rng: random.Random,
    steps: int,
    max_work: int,
    prompt_len: int | None = None,
):
    """Feeds `steps` measured mixed steps with 1 to `max_work` ranks prefilling,
    each one or two whole `prompt_len`-token prompts with no cached prefix if
    set."""
    for _ in range(steps):
        n_work = rng.randint(1, max_work)
        if prompt_len is None:
            tokens = [rng.randint(256, 8192) if r < n_work else 0 for r in range(RANKS)]
            attn = [t * (rng.uniform(0, 50e3) + t / 2) for t in tokens]
        else:
            tokens = [
                prompt_len * rng.randint(1, 2) if r < n_work else 0
                for r in range(RANKS)
            ]
            attn = [t * prompt_len / 2 for t in tokens]
        period = T_DEC + _extra_ms(tokens, attn) / 1e3 * rng.uniform(0.99, 1.01)
        hold.update(
            [
                _row(done=1, period=period, done_tokens=t, done_attn=a, decodes=16)
                for t, a in zip(tokens, attn)
            ]
        )


def _warm(
    hold: PrefillHold,
    rng: random.Random,
    max_work: int = RANKS,
    prompt_len: int | None = None,
) -> None:
    decode_only = [_row(done=1, period=T_DEC, decodes=16)] * RANKS
    hold.update(decode_only)  # The first sync has no step to time.
    hold.update(decode_only)
    _measure(hold, rng, 100, max_work, prompt_len)
    # 50 req/s of 4K (or `prompt_len`) uncached tokens each.
    uncached = prompt_len or 4000
    n = 1500 / RANKS
    hold.update([_row(arrived=n, admitted=n, admitted_uncached=uncached * n)] * RANKS)


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
    hold = PrefillHold(Mock(), T, 1.0, 8192)
    assert hold.update(_one_rank_queued()) == "cold"
    assert not hold.hold


def test_holds_while_peers_are_expected_and_waiting_is_cheap():
    """At 50 req/s, one more decode step is likely to bring prefill to other
    ranks, which costs 128 decodes less than it costs one queued request."""
    hold = PrefillHold(Mock(), T, 1.0, 8192)
    _warm(hold, random.Random(0))
    assert hold.update(_one_rank_queued()) == "wait"
    assert hold.hold

    hold.alpha = 1e4
    assert hold.update(_one_rank_queued()) == "cost"
    assert not hold.hold


def test_ttft_target_bounds_the_hold():
    """A request whose remaining allowance cannot cover a mixed step is
    released even though waiting would pay."""
    hold = PrefillHold(Mock(), T, 1.0, 8192)
    _warm(hold, random.Random(0))
    assert hold.update(_one_rank_queued(slack=0.01)) == "deadline"
    assert not hold.hold


def test_unsure_until_aligned_steps_are_measured():
    """Steps with one rank prefilling cannot tell the busiest-rank cost from
    the all-rank cost, which is what waiting for peers hinges on: it releases
    until steps with several ranks prefilling pin that down."""
    hold = PrefillHold(Mock(), T, 1.0, 8192)
    rng = random.Random(0)
    _warm(hold, rng, max_work=1)
    assert hold.update(_one_rank_queued()) == "unsure"
    _measure(hold, rng, 100, RANKS)
    assert hold.update(_one_rank_queued()) == "wait"


def test_holds_when_every_prompt_has_one_length():
    """Prompts of one length with no cached prefix, as in fixed-length
    benchmarks, make attention work proportional to tokens in every measured
    step, so the fit cannot split the per-token and attention costs. Expected
    arrivals are whole prompts of that length, whose cost needs no such split:
    it holds rather than releasing as unsure."""
    hold = PrefillHold(Mock(), T, 1.0, 8192)
    _warm(hold, random.Random(0), prompt_len=1024)
    idle = _row(period=T_DEC, decodes=16)
    queued = _row(period=T_DEC, decodes=16, work=1, x=1024, x_attn=1024 * 512, queued=1)
    assert hold.update([queued] + [idle] * (RANKS - 1)) == "wait"
    assert hold.update([idle] * RANKS) == "idle"


def test_idle_holds_only_where_a_lone_arrival_would_be_held():
    """With nothing queued, the next step may admit new arrivals before the
    next sync sees them: hold if a typical request would be held."""
    hold = PrefillHold(Mock(), T, 1.0, 8192)
    _warm(hold, random.Random(0))
    idle = [_row(period=T_DEC, decodes=16)] * RANKS
    assert hold.update(idle) == "idle"
    assert hold.hold

    hold.alpha = 1e4
    assert hold.update(idle) == "nothing"
    assert not hold.hold


def _finish_step(scheduler, output, finished_recving: set[str] | None = None) -> None:
    """Completes a step, sampling a token for every request done prefilling,
    in which the async KV loads of `finished_recving` finished."""
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
            kv_connector_output=KVConnectorOutput(finished_recving=finished_recving)
            if finished_recving
            else None,
        ),
    )


def _hold_scheduler(ttft_target: float = T, **kwargs):
    """A scheduler with the hold enabled, and its hold."""
    scheduler = create_scheduler(
        prefill_hold_ttft_target=ttft_target, data_parallel_size=2, **kwargs
    )
    hold = scheduler.prefill_hold
    assert hold is not None
    return scheduler, hold


@pytest.mark.parametrize(("dp_size", "is_moe"), [(1, True), (2, False)])
def test_requires_dp_on_a_moe_model(monkeypatch, dp_size: int, is_moe: bool):
    """Only the DP engine core of a MoE model feeds the hold; anywhere else it
    would silently never hold, so the config is rejected."""
    monkeypatch.setattr(ModelConfig, "is_moe", property(lambda self: is_moe))
    with pytest.raises(ValueError, match="requires data parallelism"):
        create_scheduler(prefill_hold_ttft_target=T, data_parallel_size=dp_size)


def test_local_row_offers_one_budget_and_scales_slack_by_releases_left():
    """A rank offers at most one budget of prefill, and a request needing
    three releases may wait a third of its target per hold."""
    scheduler, hold = _hold_scheduler(
        10.0, max_num_batched_tokens=500, max_model_len=2048
    )
    (req,) = create_requests(num_requests=1, num_tokens=1500)
    scheduler.add_request(req)
    row = hold.local_row()
    assert row[dph._ARRIVED] == 1
    assert row[dph._X] == 500
    assert row[dph._X_ATTN] == 500 * 250
    assert row[dph._QUEUED] == 1
    assert row[dph._SLACK] == pytest.approx(10.0 / 3, abs=0.1)


def test_sizes_arrivals_past_external_hits():
    """An arrival counts toward the rate at once, but its size is taken when it
    is first scheduled: tokens a KV connector loads are no prefill, and the
    scheduler learns of them only at its lookup."""
    scheduler, hold = _hold_scheduler(
        use_kv_connector=MockKVConfig(matched_tokens=128, is_async=True)
    )
    (req,) = create_requests(num_requests=1, num_tokens=200)
    scheduler.add_request(req)
    row = hold.local_row()
    assert row[dph._ARRIVED] == 1 and row[dph._ADMITTED] == 0

    output = scheduler.schedule()
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    _finish_step(scheduler, output, finished_recving={req.request_id})
    scheduler.schedule()
    row = hold.local_row()
    assert row[dph._ARRIVED] == 0 and row[dph._ADMITTED] == 1
    assert row[dph._ADMITTED_UNCACHED] == 200 - 128
    assert row[dph._ADMITTED_CACHED] == 128


def test_queue_scan_takes_the_hit_of_a_deferred_lookup():
    """A request looked up but deferred keeps no computed tokens, so the queue
    scan would fall back to the local hit probed at arrival and miss a
    synchronous external load; it takes the scheduler's fresh hit instead."""
    scheduler, hold = _hold_scheduler(
        use_kv_connector=MockKVConfig(matched_tokens=128, is_async=False)
    )
    (dec,) = create_requests(num_requests=1, num_tokens=200, req_ids=["d"])
    scheduler.add_request(dec)
    _finish_step(scheduler, scheduler.schedule())
    assert not dec.is_prefill_chunk
    (req,) = create_requests(num_requests=1, num_tokens=200, req_ids=["r"])
    scheduler.add_request(req)
    assert hold.local_row()[dph._X] == 200

    hold.hold = True
    output = scheduler.schedule()
    assert "r" not in output.num_scheduled_tokens
    assert hold.local_row()[dph._X] == 200 - 128


class _Clock:
    """Stands in for `time` in the hold module."""

    def __init__(self) -> None:
        self.wall = time.time()
        self.mono = 0.0

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono


@pytest.mark.parametrize("holding", [True, False])
def test_prepared_scan_matches_a_fresh_one(monkeypatch, holding: bool):
    """The engine core scans the queue while the GPU runs a step. By the sync,
    every allowance has shrunk by the time since while holding and by at most
    that otherwise, so the prepared slack is exact or a lower bound."""
    clock = _Clock()
    monkeypatch.setattr(dph, "time", clock)
    scheduler, hold = _hold_scheduler(
        10.0, max_num_batched_tokens=100, max_model_len=1024
    )
    reqs = create_requests(num_requests=3, num_tokens=150)
    for req, age in zip(reqs, (2.0, 1.0, 0.1)):
        req.arrival_time = clock.wall - age
        scheduler.add_request(req)
    hold._last_sync = clock.mono
    if holding:
        # Held for 0.5 s: the first two requests waited all of it.
        hold._hold_start, hold._clock = 0.0, 0.5

    scheduler.prepare_dp_row()
    clock.wall += 0.2
    clock.mono += 0.2
    prepared = hold.local_row()
    hold._last_sync = 0.0
    fresh = hold.local_row()

    for field in (dph._X, dph._X_ATTN, dph._QUEUED):
        assert prepared[field] == fresh[field]
    if holding:
        assert prepared[dph._SLACK] == pytest.approx(fresh[dph._SLACK])
    else:
        assert fresh[dph._SLACK] - 0.2 <= prepared[dph._SLACK] < fresh[dph._SLACK]


def test_queue_change_after_prepare_is_rescanned():
    """A request aborted while the GPU runs is gone from the row at the sync."""
    scheduler, hold = _hold_scheduler()
    reqs = create_requests(num_requests=2, num_tokens=8)
    for req in reqs:
        scheduler.add_request(req)
    scheduler.prepare_dp_row()
    scheduler.finish_requests(reqs[1].request_id, RequestStatus.FINISHED_ABORTED)
    assert hold.local_row()[dph._QUEUED] == 1


def test_finished_step_reports_prefill_chunks_only():
    """A finished step is reported once, with its prefill tokens and attention
    work `n * (ctx + n / 2)` summed over prefill chunks, decodes left out."""
    scheduler, hold = _hold_scheduler(
        max_num_seqs=16, max_num_batched_tokens=50, enable_chunked_prefill=True
    )

    def done() -> tuple[float, float, float]:
        row = hold.local_row()
        return row[dph._DONE], row[dph._DONE_TOKENS], row[dph._DONE_ATTN]

    (dec,) = create_requests(num_requests=1, num_tokens=4, req_ids=["dec0"])
    scheduler.add_request(dec)
    output = scheduler.schedule()
    assert done() == (0, 0, 0)
    _finish_step(scheduler, output)
    assert done() == (1, 4, 4 * 2)
    assert done() == (0, 0, 0)

    (chunk,) = create_requests(num_requests=1, num_tokens=80, req_ids=["chk0"])
    scheduler.add_request(chunk)
    first = scheduler.schedule()
    n = first.num_scheduled_tokens["chk0"]
    _finish_step(scheduler, first)
    assert done() == (1, n, n * n / 2)
    second = scheduler.schedule()
    m = second.num_scheduled_tokens["chk0"]
    assert "dec0" in second.num_scheduled_tokens
    _finish_step(scheduler, second)
    assert done() == (1, m, m * (n + m / 2))


def test_rank_without_a_free_slot_is_blocked():
    """A rank whose slots are all taken cannot start its queued prefill, so it
    offers none. Its last finished step is still reported for timing."""
    scheduler, hold = _hold_scheduler(max_num_seqs=1)
    running, queued = create_requests(num_requests=2, num_tokens=8)
    scheduler.add_request(running)
    _finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(queued)
    row = hold.local_row()
    assert row[dph._DONE] == 1 and row[dph._DONE_TOKENS] == 8
    assert row[dph._DECODES] == 1
    assert row[dph._BLOCKED] == 1 and row[dph._X] == 0


def _kv_stopped_scheduler(ttft_target: float = T):
    """A hold scheduler with a running decode that leaves too few free KV
    blocks for a 300-token request, and the decode."""
    scheduler, hold = _hold_scheduler(
        ttft_target, num_blocks=30, block_size=16, max_model_len=1024
    )
    (dec,) = create_requests(num_requests=1, num_tokens=200, req_ids=["d"])
    scheduler.add_request(dec)
    _finish_step(scheduler, scheduler.schedule())
    return scheduler, hold, dec


def test_queue_scan_stops_where_kv_runs_out():
    """Once a release stops for lack of KV blocks, the rank offers only what
    admission could take: its queue ends at the first request that does not
    fit, as admission does."""
    scheduler, hold, dec = _kv_stopped_scheduler()
    (big,) = create_requests(num_requests=1, num_tokens=300, req_ids=["b"])
    (small,) = create_requests(num_requests=1, num_tokens=16, req_ids=["s"])
    scheduler.add_request(big)
    scheduler.add_request(small)
    assert hold.local_row()[dph._QUEUED] == 2

    output = scheduler.schedule()
    assert output.num_scheduled_tokens.keys() == {"d"}
    assert scheduler.prefill_kv_stopped
    row = hold.local_row()
    assert row[dph._WORK] == 1
    assert row[dph._X] == 0 and row[dph._QUEUED] == 0

    scheduler.finish_requests(dec.request_id, RequestStatus.FINISHED_ABORTED)
    row = hold.local_row()
    assert row[dph._X] == 316 and row[dph._QUEUED] == 2


def test_queue_scan_stops_where_running_slots_run_out():
    """A release that stops on a full running set counts as one that stopped
    for lack of KV blocks: the rank then offers only as many queued requests
    as it has free slots."""
    scheduler, hold = _hold_scheduler(max_num_seqs=2)
    reqs = create_requests(num_requests=4, num_tokens=8)
    scheduler.add_request(reqs[0])
    _finish_step(scheduler, scheduler.schedule())
    for req in reqs[1:]:
        scheduler.add_request(req)
    assert hold.local_row()[dph._QUEUED] == 3

    _finish_step(scheduler, scheduler.schedule())
    assert scheduler.prefill_kv_stopped
    scheduler.finish_requests(reqs[0].request_id, RequestStatus.FINISHED_ABORTED)
    row = hold.local_row()
    assert row[dph._X] == 8 and row[dph._QUEUED] == 1


def test_waits_count_from_a_release_that_stopped_on_kv(monkeypatch):
    """A request waiting behind a release that stopped for lack of KV blocks
    could not start before it, so its age does not force a release: the hold
    may add up to the target to its wait from then on."""
    clock = _Clock()
    monkeypatch.setattr(dph, "time", clock)
    scheduler, hold, dec = _kv_stopped_scheduler(10.0)
    (big,) = create_requests(num_requests=1, num_tokens=300, req_ids=["b"])
    big.arrival_time = clock.wall - 20.0
    scheduler.add_request(big)
    assert hold.local_row()[dph._SLACK] == pytest.approx(-10.0)

    scheduler.schedule()
    scheduler.finish_requests(dec.request_id, RequestStatus.FINISHED_ABORTED)
    assert hold.local_row()[dph._SLACK] == pytest.approx(10.0)
    # Held for 0.5 s since that release.
    hold._hold_start, hold._clock = 0.0, 0.5
    assert hold.local_row()[dph._SLACK] == pytest.approx(9.5)


def test_finished_kv_load_is_queued_work():
    """A request whose async KV load finished waits in `kv_holding_waiting`,
    and the hold defers it like any other. Until it is released, it is queued
    work whose age bounds the hold, from the sync right after the step its
    load finished in; a load still in progress is not."""
    scheduler, hold = _hold_scheduler(
        use_kv_connector=MockKVConfig(matched_tokens=64, is_async=True)
    )
    dec, req = create_requests(num_requests=2, num_tokens=200, req_ids=["d", "r"])
    scheduler.add_request(dec)
    _finish_step(scheduler, scheduler.schedule(), finished_recving={"d"})
    _finish_step(scheduler, scheduler.schedule())

    req.arrival_time -= 2 * T
    scheduler.add_request(req)
    hold.hold = True
    output = scheduler.schedule()
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert hold.local_row()[dph._WORK] == 0
    scheduler.prepare_dp_row()
    _finish_step(scheduler, output, finished_recving={"r"})
    # Loaded, then promoted by a step that defers it.
    for _ in range(2):
        row = hold.local_row()
        assert row[dph._WORK] == 1 and row[dph._QUEUED] == 1
        assert row[dph._X] == 200 - 64 and row[dph._SLACK] < 0
        output = scheduler.schedule()
        assert set(output.num_scheduled_tokens) == {"d"}
        scheduler.prepare_dp_row()
        _finish_step(scheduler, output)

    hold.hold = False
    assert scheduler.schedule().num_scheduled_tokens["r"] == 200 - 64


def test_paused_streaming_session_is_not_work():
    """A streaming session waiting for its next input sits in
    `kv_holding_waiting` with nothing to prefill. Its arrival time is its last
    input's, so counting it would release every step."""
    scheduler, hold = _hold_scheduler()
    dec, session = create_requests(
        num_requests=2, num_tokens=8, stop_token_ids=[7], req_ids=["d", "s"]
    )
    session.resumable = True
    scheduler.add_request(dec)
    scheduler.add_request(session)
    scheduler.update_from_output(
        scheduler.schedule(),
        ModelRunnerOutput(req_ids=["d", "s"], sampled_token_ids=[[0], [7]]),
    )
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    session.arrival_time -= 2 * T
    row = hold.local_row()
    assert row[dph._WORK] == 0 and row[dph._DECODES] == 1


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_scans_waiting_queues_in_admission_order(policy):
    """Requests that may load KV asynchronously wait in `async_kv_waiting`
    beside `waiting`; the hold scans the two in the order the scheduler admits
    from them."""
    scheduler, _ = _hold_scheduler(
        use_kv_connector=MockKVConfig(), scheduling_policy=policy
    )
    scheduler.scheduler_config.max_num_inflight_kv_loads = 8
    scheduler.connector.may_load_async = lambda r: int(r.request_id) % 2 == 0
    reqs = create_requests(num_requests=6, num_tokens=8)
    for i, (req, priority) in enumerate(zip(reqs, (2, 0, 1, 0, 2, 1))):
        req.arrival_time, req.priority = float(i), priority
        scheduler.add_request(req)
    assert len(scheduler.async_kv_waiting) == len(scheduler.waiting) == 3

    scanned = [r.request_id for r in scheduler._prefill_queue_state().waiting()]
    admitted = [r.req_id for r in scheduler.schedule().scheduled_new_reqs]
    assert scanned == admitted


def test_hold_overrides_per_rank_capacity_escape():
    """With `prefill_schedule_interval`, a rank whose last release left a
    backlog keeps prefilling on throttled steps. A hold applies to every rank
    alike, so that rank holds too, and releases when the hold lifts."""
    scheduler, hold = _hold_scheduler(max_num_batched_tokens=200)
    dec, a, b = create_requests(num_requests=3, num_tokens=200, req_ids=["d", "a", "b"])
    scheduler.add_request(dec)
    _finish_step(scheduler, scheduler.schedule())
    scheduler.add_request(a)
    scheduler.add_request(b)
    _finish_step(scheduler, scheduler.schedule())
    assert scheduler.prefill_capacity_bound

    hold.hold = True
    output = scheduler.schedule()
    assert set(output.num_scheduled_tokens) == {"d"}
    _finish_step(scheduler, output)

    hold.hold = False
    output = scheduler.schedule()
    assert set(output.num_scheduled_tokens) > {"d"}


def test_scheduler_relays_dp_rows_to_the_hold():
    """The scheduler shares the hold's row in the DP state sync and hands the
    gathered rows back to it; a sync that finds every rank idle releases the
    hold. DP rank 0 logs."""
    assert not create_scheduler().shares_dp_row
    scheduler, hold = _hold_scheduler()
    assert scheduler.shares_dp_row and hold.log_stats
    assert len(scheduler.dp_row()) == dph.ROW_SIZE

    _warm(hold, random.Random(0))
    scheduler.update_from_dp_rows(_one_rank_queued(), idle=False)
    assert hold.hold
    scheduler.update_from_dp_rows(_one_rank_queued(), idle=True)
    assert not hold.hold
