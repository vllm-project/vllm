# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import random
import time

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from tests.v1.kv_connector.unit.utils import MockKVConfig
from vllm.config import ModelConfig
from vllm.config.parallel import SyncedDPState
from vllm.v1.core.sched import dp_prefill_hold as dph
from vllm.v1.core.sched.dp_prefill_hold import PrefillHold, _RankRow
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test

T = 15.0
T_DEC = 0.025
RANKS = 8
# Extra ms of a mixed step: fixed F, slowest-rank a * M, total b * S.
TRUTH = (15.0, 12.0, 3.0)


@pytest.fixture(autouse=True)
def _moe_model(monkeypatch):
    """The hold needs DP on a MoE model; the dense test model stands in."""
    monkeypatch.setattr(ModelConfig, "is_moe", property(lambda self: True))


def _row(
    *,
    done: float = 0.0,
    period: float = 0.0,
    done_tokens: float = 0.0,
    work: float = 0.0,
    x: float = 0.0,
    decodes: float = 0.0,
    queued: float = 0.0,
    slack: float = T,
) -> _RankRow:
    return _RankRow(done, period, done_tokens, work, x, decodes, queued, slack)


def _extra_ms(tokens: list[int], truth: tuple[float, float, float] = TRUTH) -> float:
    f, a, b = truth
    return f + a * max(tokens) / 1e3 + b * sum(tokens) / 1e3


def _measure(
    hold: PrefillHold,
    rng: random.Random,
    steps: int,
    max_work: int = RANKS,
    truth: tuple[float, float, float] = TRUTH,
) -> None:
    """Feed `steps` measured mixed steps with 1 to `max_work` ranks prefilling."""
    for _ in range(steps):
        n_work = rng.randint(1, max_work)
        tokens = [rng.randint(256, 8192) if r < n_work else 0 for r in range(RANKS)]
        period = T_DEC + _extra_ms(tokens, truth) / 1e3 * rng.uniform(0.99, 1.01)
        hold.update(
            [_row(done=1, period=period, done_tokens=t, decodes=16) for t in tokens]
        )


def _warm(
    hold: PrefillHold,
    rng: random.Random,
    max_work: int = RANKS,
    truth: tuple[float, float, float] = TRUTH,
) -> None:
    decode_only = [_row(done=1, period=T_DEC, decodes=16)] * RANKS
    for _ in range(4):  # 1 skipped + 3 cold-start samples for median
        hold.update(decode_only)
    _measure(hold, rng, 100, max_work, truth)


def _one_rank_queued(
    slack: float = T, x: float = 2048, decodes: float = 16
) -> list[_RankRow]:
    """2K tokens queued on one rank, `decodes` decodes on every rank."""
    busy = _row(
        period=T_DEC,
        decodes=decodes,
        work=1,
        x=x,
        queued=1,
        slack=slack,
    )
    return [busy] + [_row(period=T_DEC, decodes=decodes)] * (RANKS - 1)


def test_cold_start_median_and_stall_recovery():
    """Cold-start median drops an initial compile spike; a single stall is
    rejected, while `_MAX_STALL_REJECTS` consecutive high steps adapt to a true
    regime shift."""
    _, hold = _hold_scheduler()
    # First sync skipped; next 3 samples include a 0.5s compile spike.
    hold.update([_row(done=1, period=0.02, decodes=16)] * RANKS)
    hold.update([_row(done=1, period=0.50, decodes=16)] * RANKS)
    assert hold._t_dec == pytest.approx(0.50)
    # Sample 2 takes min, immediately dropping the compile spike.
    hold.update([_row(done=1, period=T_DEC, decodes=16)] * RANKS)
    assert hold._t_dec == pytest.approx(T_DEC)
    hold.update([_row(done=1, period=T_DEC, decodes=16)] * RANKS)
    assert hold._t_dec == pytest.approx(T_DEC)
    assert hold._t_dec_warm

    _measure(hold, random.Random(0), 80)
    t_dec = hold._t_dec
    cost_before = (hold._cost.f, hold._cost.a, hold._cost.b)

    # Single pure-decode stall is rejected.
    hold.update([_row(done=1, period=10.0, decodes=16)] * RANKS)
    assert hold._t_dec == t_dec

    # Single mixed-step stall is rejected.
    tokens = [8192] * RANKS
    mixed = lambda p: [
        _row(done=1, period=p, done_tokens=t, decodes=16) for t in tokens
    ]
    hold.update(mixed(10.0))
    assert (hold._cost.f, hold._cost.a, hold._cost.b) == cost_before

    # Consecutive high decode steps indicate a real regime shift and recover.
    for _ in range(dph._MAX_STALL_REJECTS):
        hold.update([_row(done=1, period=0.20, decodes=16)] * RANKS)
    assert hold._t_dec == pytest.approx(0.20)

    # Consecutive high mixed steps also force-accept instead of locking out.
    w_before = hold._cost.w
    for _ in range(dph._MAX_STALL_REJECTS):
        hold.update(mixed(5.0))
    assert hold._cost.w != w_before


@pytest.mark.parametrize("bad_val", [math.nan, math.inf, -math.inf])
def test_non_finite_row_fails_open(bad_val: float):
    """Any NaN or Inf in gathered rows immediately releases and skips learning."""
    _, hold = _hold_scheduler()
    _warm(hold, random.Random(0))
    assert hold.update(_one_rank_queued()) == "wait"
    assert hold.hold

    t_dec = hold._t_dec
    bad_rows = _one_rank_queued()
    bad_rows[0] = bad_rows[0]._replace(period=bad_val)
    assert hold.update(bad_rows) == "invalid"
    assert not hold.hold
    assert hold._t_dec == t_dec


def test_steady_state_collinearity_and_elastic_ep_reset():
    """Repeated full-budget batches preserve learned (F, a, b) ratios instead of
    collapsing; changing rank count (Elastic EP) resets the cost model."""
    _, hold = _hold_scheduler()
    _warm(hold, random.Random(0))
    assert hold._cost.has_3d
    f0, a0, b0 = hold._cost.f, hold._cost.a, hold._cost.b
    assert f0 > 0 and a0 > 0 and b0 > 0

    # Feed 2500 identical full batches (collinear M = 8, S = 64) until the old
    # variation decays away and _solve_3x3 goes ill-conditioned; ratios hold.
    full = [8192] * RANKS
    period = T_DEC + _extra_ms(full) / 1e3
    for _ in range(2500):
        hold.update(
            [_row(done=1, period=period, done_tokens=t, decodes=16) for t in full]
        )
    assert hold._cost.f > 0 and hold._cost.a > 0 and hold._cost.b > 0
    assert hold._cost.a / hold._cost.f == pytest.approx(a0 / f0, rel=0.15)
    assert hold._cost.b / hold._cost.f == pytest.approx(b0 / f0, rel=0.15)
    # In this collinear steady state, a 2x step-time shift rescales uniformly.
    period_2x = T_DEC + 2.0 * _extra_ms(full) / 1e3
    for _ in range(500):
        hold.update(
            [_row(done=1, period=period_2x, done_tokens=t, decodes=16) for t in full]
        )
    assert hold._cost.a / hold._cost.f == pytest.approx(a0 / f0, rel=0.15)
    assert hold._cost.predict_ms(8.192, 8 * 8.192) == pytest.approx(
        2.0 * _extra_ms(full), rel=0.1
    )

    # Elastic EP resize: rank count changes from 8 to 4 -> reset model.
    assert hold.update([_row(period=T_DEC, decodes=16)] * 4) == "cold"
    assert hold._cost.w == 0.0 and hold._t_dec is None


def test_ski_rental_holds_then_releases_and_ignores_non_amortizable_b():
    """Online Ski-Rental holds while accumulated rent < 0.5 * D * waste_s, then
    releases; purely non-amortizable `b * S` cost does not justify holding."""
    _, hold = _hold_scheduler()
    _warm(hold, random.Random(0))
    assert hold.update(_one_rank_queued()) == "wait"
    assert hold.hold

    # Accumulate rent until Ski-Rental releases via "cost".
    reason = ""
    for _ in range(200):
        reason = hold.update(_one_rank_queued())
        if reason != "wait":
            break
    assert reason == "cost"
    assert not hold.hold

    # High ttft_weight releases immediately.
    hold.ttft_weight = 1e4
    assert hold.update(_one_rank_queued()) == "cost"

    # Pure b * S hardware (F = 0, a = 0, b = 10 ms/ktok): zero amortizable waste.
    _, b_only = _hold_scheduler()
    _warm(b_only, random.Random(0), truth=(0.0, 0.0, 10.0))
    assert b_only.update(_one_rank_queued()) == "cost"


def test_ttft_target_and_lag_steps_bound_the_hold():
    """Tight slack or async `lag_steps` releases via deadline."""
    _, hold = _hold_scheduler()
    hold.lag_steps = 2
    _warm(hold, random.Random(0))
    assert hold.update(_one_rank_queued(slack=0.01)) == "deadline"
    assert not hold.hold
    # Slack barely covers t_mix (~0.065s) without lag, but not + 2 * t_dec.
    assert hold.update(_one_rank_queued(slack=0.09)) == "deadline"


def test_idle_n_avail_and_no_local_decode_boundaries():
    """Covers idle hold, n_avail == 0/1, and no_local_decode piggyback."""
    _, hold = _hold_scheduler()
    _warm(hold, random.Random(0))

    # All ranks blocked (work=1, x=0) -> "blocked".
    all_blocked = [_row(period=T_DEC, decodes=16, work=1, x=0)] * RANKS
    assert hold.update(all_blocked) == "blocked"

    # Idle hold when 0.5 * D * amort_s > ttft_weight * t_dec.
    idle = [_row(period=T_DEC, decodes=16)] * RANKS
    assert hold.update(idle) == "idle"
    assert hold.hold

    # n_avail == 1 with no work uses F only; high weight turns idle into nothing.
    hold.ttft_weight = 1e4
    assert hold.update(idle) == "nothing"
    assert not hold.hold

    # Any rank with decodes=0 and runnable prefill forces piggyback release.
    hold.ttft_weight = 1.0
    rows = [_row(period=T_DEC, decodes=0, work=1, x=2048, queued=1)] + [
        _row(period=T_DEC, decodes=16)
    ] * (RANKS - 1)
    assert hold.update(rows) == "no_local_decode"
    assert not hold.hold


def _finish_step(scheduler, output, finished_recving: set[str] | None = None) -> None:
    """Complete a step, sampling a token for every request done prefilling."""
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
    """Only the DP engine core of a MoE model feeds the hold."""
    monkeypatch.setattr(ModelConfig, "is_moe", property(lambda self: is_moe))
    with pytest.raises(ValueError, match="requires data parallelism"):
        create_scheduler(prefill_hold_ttft_target=T, data_parallel_size=dp_size)


def test_local_row_effective_budget_and_long_prefill_threshold():
    """Effective budget deducts running decodes, and `long_prefill_token_threshold`
    caps single-step `x` and increases the release count `k`."""
    scheduler, hold = _hold_scheduler(
        10.0,
        max_num_batched_tokens=500,
        max_model_len=2048,
        long_prefill_token_threshold=200,
        enable_chunked_prefill=True,
    )
    (dec,) = create_requests(num_requests=1, num_tokens=16, req_ids=["d"])
    scheduler.add_request(dec)
    _finish_step(scheduler, scheduler.schedule())

    (req,) = create_requests(num_requests=1, num_tokens=1000, req_ids=["r"])
    scheduler.add_request(req)
    row = hold.local_row()
    # 1 running decode leaves budget=499, but long_prefill_token_threshold=200
    # caps single-step x to 200 and needs ceil(1000 / 200) = 5 releases.
    assert row.decodes == 1
    assert row.x == 200
    assert row.queued == 1
    assert row.slack == pytest.approx(10.0 / 5, abs=0.1)


def test_full_prefix_hit_excluded():
    """A full prefix hit leaves 1 token (never deferred) and is not reported as
    held prefill work."""
    scheduler, hold = _hold_scheduler(
        num_blocks=40,
        block_size=16,
        max_model_len=512,
        enable_prefix_caching=True,
    )
    # 33 tokens = 2 full 16-token blocks + 1 tail token -> 32 cached, 1 left.
    first, dup = create_requests(
        num_requests=2, num_tokens=33, req_ids=["r0", "r1"], same_prompt=True
    )
    scheduler.add_request(first)
    _finish_step(scheduler, scheduler.schedule())

    scheduler.add_request(dup)
    row = hold.local_row()
    assert row.queued == 0 and row.x == 0


def test_external_hits_are_no_prefill():
    """Deferred requests remember connector lookup hits across steps."""
    scheduler, hold = _hold_scheduler(
        use_kv_connector=MockKVConfig(matched_tokens=128, is_async=False)
    )
    (dec,) = create_requests(num_requests=1, num_tokens=200, req_ids=["d"])
    scheduler.add_request(dec)
    _finish_step(scheduler, scheduler.schedule())
    hold.local_row()

    (req,) = create_requests(num_requests=1, num_tokens=200, req_ids=["r"])
    scheduler.add_request(req)
    assert hold.local_row().x == 200

    hold.hold = True
    output = scheduler.schedule()
    assert "r" not in output.num_scheduled_tokens
    _finish_step(scheduler, output)
    assert hold.local_row().x == 200 - 128

    hold.hold = False
    assert scheduler.schedule().num_scheduled_tokens["r"] == 200 - 128


class _Clock:
    """Stands in for `time` in the hold module."""

    def __init__(self) -> None:
        self.wall = time.time()
        self.mono = 0.0

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono


def test_finished_step_reports_prefill_chunks_only():
    """A finished step is reported once with its prefill tokens, decodes excluded."""
    scheduler, hold = _hold_scheduler(
        max_num_seqs=16, max_num_batched_tokens=50, enable_chunked_prefill=True
    )

    def done() -> tuple[float, float]:
        row = hold.local_row()
        return row.done, row.done_tokens

    (dec,) = create_requests(num_requests=1, num_tokens=4, req_ids=["dec0"])
    scheduler.add_request(dec)
    output = scheduler.schedule()
    assert done() == (0, 0)
    _finish_step(scheduler, output)
    assert done() == (1, 4)
    assert done() == (0, 0)

    (chunk,) = create_requests(num_requests=1, num_tokens=80, req_ids=["chk0"])
    scheduler.add_request(chunk)
    first = scheduler.schedule()
    n = first.num_scheduled_tokens["chk0"]
    _finish_step(scheduler, first)
    assert done() == (1, n)
    second = scheduler.schedule()
    m = second.num_scheduled_tokens["chk0"]
    assert "dec0" in second.num_scheduled_tokens
    _finish_step(scheduler, second)
    assert done() == (1, m)


def test_queue_scan_stops_at_kv_without_double_probe(monkeypatch):
    """Once a release stops for lack of KV blocks, the scan stops where KV runs
    out and probes the prefix cache at most once per waiting request."""
    scheduler, hold = _hold_scheduler(num_blocks=30, block_size=16, max_model_len=1024)
    (dec,) = create_requests(num_requests=1, num_tokens=200, req_ids=["d"])
    scheduler.add_request(dec)
    _finish_step(scheduler, scheduler.schedule())

    (big,) = create_requests(num_requests=1, num_tokens=300, req_ids=["b"])
    (small,) = create_requests(num_requests=1, num_tokens=16, req_ids=["s"])
    scheduler.add_request(big)
    scheduler.add_request(small)

    calls = 0
    orig_probe = scheduler.kv_cache_manager.get_num_cached_tokens

    def counted_probe(req):
        nonlocal calls
        calls += 1
        return orig_probe(req)

    monkeypatch.setattr(
        scheduler.kv_cache_manager, "get_num_cached_tokens", counted_probe
    )

    assert hold.local_row().queued == 2
    assert calls == 2

    output = scheduler.schedule()
    assert output.num_scheduled_tokens.keys() == {"d"}
    assert scheduler.prefill_kv_stopped
    row = hold.local_row()
    assert row.work == 1 and row.x == 0 and row.queued == 0
    # Cached hits were reused; no extra probe calls on KV-stopped scan.
    assert calls == 2

    scheduler.finish_requests(dec.request_id, RequestStatus.FINISHED_ABORTED)
    row = hold.local_row()
    assert row.x == 316 and row.queued == 2


def test_queue_scan_stops_at_free_slots():
    """Queue scan stops at free running slots."""
    scheduler, hold = _hold_scheduler(max_num_seqs=2)
    reqs = create_requests(num_requests=4, num_tokens=8)
    scheduler.add_request(reqs[0])
    _finish_step(scheduler, scheduler.schedule())
    for req in reqs[1:]:
        scheduler.add_request(req)
    row = hold.local_row()
    assert row.x == 8 and row.queued == 1

    _finish_step(scheduler, scheduler.schedule())
    row = hold.local_row()
    assert row.decodes == 2
    assert row.work == 1 and row.x == 0 and row.queued == 0


def test_past_target_bounds_only_waiting_not_in_progress_chunks(monkeypatch):
    """Waiting requests past `T` clamp age to `held`, while an in-progress
    prefill chunk past `T` keeps its real age so it is not re-held."""
    clock = _Clock()
    monkeypatch.setattr(dph, "time", clock)
    scheduler, hold = _hold_scheduler(
        10.0, max_num_batched_tokens=100, enable_chunked_prefill=True
    )
    (req,) = create_requests(num_requests=1, num_tokens=200, req_ids=["r"])
    scheduler.add_request(req)

    req.arrival_time = clock.wall - 4.0
    assert hold.local_row().slack == pytest.approx(3.0)  # (10 - 4) / 2

    # Waiting past target: bounds only the current hold (age clamped to held=0).
    req.arrival_time = clock.wall - 20.0
    assert hold.local_row().slack == pytest.approx(5.0)  # 10 / 2

    # Once admitted and partially prefilled (in_progress), past-target age is
    # NOT reset, so slack <= 0 and the remaining chunk releases immediately.
    _finish_step(scheduler, scheduler.schedule())
    assert req.is_prefill_chunk
    assert hold.local_row().slack <= 0.0


def test_finished_kv_load_is_queued_work():
    """A finished async KV load waits in `kv_holding_waiting` and counts as
    queued work; an in-progress load does not."""
    scheduler, hold = _hold_scheduler(
        use_kv_connector=MockKVConfig(matched_tokens=64, is_async=True)
    )
    dec, req = create_requests(num_requests=2, num_tokens=200, req_ids=["d", "r"])
    scheduler.add_request(dec)
    _finish_step(scheduler, scheduler.schedule(), finished_recving={"d"})
    _finish_step(scheduler, scheduler.schedule())

    req.arrival_time -= T / 2
    scheduler.add_request(req)
    hold.hold = True
    output = scheduler.schedule()
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert hold.local_row().work == 0
    _finish_step(scheduler, output, finished_recving={"r"})
    for _ in range(2):
        row = hold.local_row()
        assert row.work == 1 and row.queued == 1
        assert row.x == 200 - 64 and row.slack < T / 2
        output = scheduler.schedule()
        assert set(output.num_scheduled_tokens) == {"d"}
        _finish_step(scheduler, output)

    hold.hold = False
    assert scheduler.schedule().num_scheduled_tokens["r"] == 200 - 64


def test_paused_streaming_and_stale_output_not_work():
    """A paused streaming session or a request with pending stale output is
    not startable prefill work."""
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
    row = hold.local_row()
    assert row.work == 0 and row.decodes == 1

    (stale,) = create_requests(num_requests=1, num_tokens=16, req_ids=["stale"])
    stale.num_stale_output_tokens = 1
    scheduler.add_request(stale)
    assert hold.local_row().work == 0


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_scans_waiting_queues_in_admission_order(policy):
    """Requests in `async_kv_waiting` and `waiting` are scanned in admission order."""
    scheduler, hold = _hold_scheduler(
        use_kv_connector=MockKVConfig(), scheduling_policy=policy
    )
    scheduler.scheduler_config.max_num_inflight_kv_loads = 8
    scheduler.connector.may_load_async = lambda r: int(r.request_id) % 2 == 0
    reqs = create_requests(num_requests=6, num_tokens=8)
    for i, (req, priority) in enumerate(zip(reqs, (2, 0, 1, 0, 2, 1))):
        req.arrival_time, req.priority = float(i), priority
        scheduler.add_request(req)
    assert len(scheduler.async_kv_waiting) == len(scheduler.waiting) == 3

    scanned = [r.request_id for r in hold._startable_waiting()]
    admitted = [r.req_id for r in scheduler.schedule().scheduled_new_reqs]
    assert scanned == admitted


def test_sync_that_finds_every_rank_idle_releases():
    """A sync that finds every rank idle resets the hold; without the hold no
    DP state is shared."""
    assert create_scheduler().get_dp_state() is None
    scheduler, hold = _hold_scheduler()
    _warm(hold, random.Random(0))
    scheduler.update_dp_state(
        SyncedDPState(
            has_unfinished=True,
            pause_consensus=False,
            scheduler_states=_one_rank_queued(),
        )
    )
    assert hold.hold
    scheduler.update_dp_state(
        SyncedDPState(
            has_unfinished=False,
            pause_consensus=False,
            scheduler_states=_one_rank_queued(),
        )
    )
    assert not hold.hold
