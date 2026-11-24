# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import time

import pytest

from tests.v1.core.utils import create_requests, create_scheduler, mock_kv
from tests.v1.kv_connector.unit.utils import create_model_runner_output
from vllm.entrypoints.serve.instrumentator.metrics import (
    _RefreshBeforeMetrics,
    forecast_new_token_knots,
)
from vllm.v1.core.sched.forecast import (
    Forecaster,
    IngestModel,
    PaceModel,
    Snapshot,
    StepTimeModel,
    curve_value,
)
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test

# Step time used by the tests: 0.1 s fixed plus 1 ms per token.
FIXED = 0.1
PER_TOKEN = 0.001


def make(budget=100, seats=4, threshold=0, queue_size=1) -> Forecaster:
    f = Forecaster(
        token_budget=budget,
        max_seats=seats,
        long_prefill_threshold=threshold,
        adaptive_threshold=False,
        max_model_len=1000,
        queue_size=queue_size,
    )
    f.model.coef = (FIXED, PER_TOKEN, 0.0, 0.0)
    f.model.num_samples = 8
    return f


def step(tokens: int) -> float:
    return FIXED + PER_TOKEN * tokens


def arrival(request_id: str, prompt_tokens: int, in_server: float = 0.0) -> Request:
    """A request the server received `in_server` seconds ago."""
    request = create_requests(
        1,
        num_tokens=prompt_tokens,
        max_tokens=1,
        block_size=1024,
        req_ids=[request_id],
    )[0]
    request.arrival_time = time.time() - in_server
    return request


@pytest.fixture
def forecasting(monkeypatch):
    monkeypatch.setenv("VLLM_TTFT_FORECAST", "1")


def set_state(f: Forecaster, **kw) -> None:
    now = time.monotonic()
    f.snapshot = Snapshot(time=now, exec_started=kw.pop("exec_started", now), **kw)


def test_idle_engine_runs_the_prompt_in_full_steps():
    f = make()
    set_state(f)
    assert f.ttft(250) == pytest.approx(step(100) + step(100) + step(50), abs=1e-3)


def test_single_token_costs_one_small_step():
    f = make()
    set_state(f)
    assert f.ttft(1) == pytest.approx(step(1), abs=1e-3)


def test_running_prefill_shares_the_budget_under_the_cap():
    # One running request with 150 tokens left; cap 50 while two are present.
    f = make(threshold=50)
    set_state(f, prefills=((150, 0, True, 0),))
    # Steps 1-3: 50 + 50 each. Step 3 finishes both the running request and a
    # 150-token probe.
    assert f.ttft(150) == pytest.approx(3 * step(100), abs=1e-3)
    # A 50-token probe finishes in the first shared step.
    assert f.ttft(50) == pytest.approx(step(100), abs=1e-3)


def test_probe_waits_for_a_seat():
    f = make(seats=1)
    set_state(f, prefills=((200, 0, True, 0),))
    # The running request takes two full steps; the probe starts after them.
    assert f.ttft(10) == pytest.approx(2 * step(100) + step(10), abs=1e-3)


def test_waiting_requests_go_first():
    f = make(seats=1)
    set_state(f, prefills=((100, 0, False, 0), (100, 0, False, 0)))
    assert f.ttft(10) == pytest.approx(2 * step(100) + step(10), abs=1e-3)


def test_probe_joins_after_the_step_in_flight():
    f = make()
    now = time.monotonic()
    features = StepTimeModel.features(100, 0, 0)
    # A 0.2 s step started 0.05 s ago, so 0.15 s of it is left.
    set_state(f, exec_started=now - 0.05, inflight=((features, 0),))
    assert f.ttft(10) == pytest.approx(0.15 + step(10), abs=5e-3)


def test_batch_queue_adds_the_step_already_scheduled():
    f = make(queue_size=2)
    now = time.monotonic()
    features = StepTimeModel.features(100, 0, 0)
    # Two steps are out: one executing since `now`, one queued behind it. The
    # probe is scheduled when the first returns and runs after the second.
    set_state(f, exec_started=now, inflight=((features, 0), (features, 0)))
    assert f.ttft(10) == pytest.approx(2 * step(100) + step(10), abs=5e-3)


def test_decoding_requests_take_one_token_per_step():
    f = make()
    set_state(f, decodes=((500, 1000), (500, 1000)))
    assert f.ttft(98) == pytest.approx(step(100), abs=1e-3)
    # 99 new tokens no longer fit beside the two decode tokens.
    assert f.ttft(99) == pytest.approx(step(100) + step(3), abs=1e-3)


def test_arrivals_not_yet_scheduled_are_counted_until_seen():
    f = make(seats=1)
    set_state(f)
    request = arrival("a", 100)
    f.note_arrival(request, cached_tokens=0)
    assert f.ttft(10) == pytest.approx(step(100) + step(10), abs=1e-3)
    # Once a snapshot includes the arrival, it is no longer added on top.
    f.note_added(request)
    set_state(f, seen_seq=1)
    assert f.ttft(10) == pytest.approx(step(10), abs=1e-3)


@pytest.mark.parametrize("new_tokens", [1, 37, 100, 101, 250, 640, 999])
def test_curve_matches_single_forecasts(new_tokens):
    f = make(threshold=50)
    set_state(f, prefills=((320, 0, True, 0), (75, 0, False, 0)))
    curve = f.curve()
    assert curve and curve[0][0] == 0
    seg = next(s for s in curve if new_tokens <= s[2])
    u0, t0, u1, t1 = seg
    at = t0 + (t1 - t0) * (new_tokens - u0) / (u1 - u0)
    assert at == pytest.approx(f.ttft(new_tokens), rel=0.03, abs=2e-3)


def test_curve_is_non_decreasing():
    f = make(threshold=50)
    set_state(f, prefills=((320, 0, True, 0),), decodes=((100, 5),))
    curve = f.curve()
    ends = [s[3] for s in curve]
    assert ends == sorted(ends)
    assert all(s[1] <= s[3] for s in curve)


def test_step_time_fit_recovers_the_step_cost():
    model = StepTimeModel()
    for tokens in (1, 8, 64, 100, 300, 512, 1000, 2000) * 4:
        ctx = 10 * tokens
        seconds = 0.02 + 5e-5 * tokens + 1e-6 * ctx
        model.observe(StepTimeModel.features(tokens, ctx, 0), seconds)
    assert model.coef is not None
    assert all(c >= 0 for c in model.coef)
    assert model.predict(StepTimeModel.features(700, 7000, 0)) == pytest.approx(
        0.02 + 5e-5 * 700 + 1e-6 * 7000, rel=0.02
    )


def test_no_forecast_before_any_step_is_measured():
    f = Forecaster(100, 4, 0, False, 1000)
    assert f.ttft(10) is None
    assert f.curve() == []


def test_floor_ignores_the_queue():
    f = make(threshold=50)
    set_state(f, prefills=((320, 0, True, 0), (75, 0, False, 0)))
    floor = f.floor()
    seg = next(s for s in floor if s[2] >= 250)
    at = seg[1] + (seg[3] - seg[1]) * (250 - seg[0]) / (seg[2] - seg[0])
    assert at == pytest.approx(step(100) + step(100) + step(50), rel=0.03)
    # The loaded curve is never below it.
    assert f.ttft(250) >= at


def one_seat_with_a_cached_prompt(num_waiting: int):
    """A scheduler with one seat taken, and `num_waiting` requests queued whose
    160-token prompt is cached except for its last block."""
    scheduler = create_scheduler(enable_prefix_caching=True, max_num_seqs=1)
    first = create_requests(1, num_tokens=160, max_tokens=1, req_ids=["first"])[0]
    scheduler.add_request(first)
    output = scheduler.schedule()
    scheduler.update_from_output(
        output, create_model_runner_output([first], use_eos=True)
    )
    busy = create_requests(2, num_tokens=160, max_tokens=64, req_ids=["x", "busy"])[1]
    scheduler.add_request(busy)
    output = scheduler.schedule()
    scheduler.update_from_output(
        output, create_model_runner_output([busy], token_id=1000)
    )
    waiting = create_requests(
        num_waiting,
        num_tokens=160,
        max_tokens=1,
        same_prompt=True,
        req_ids=[f"waiting-{i}" for i in range(num_waiting)],
    )
    for request in waiting:
        # The engine core reports an arrival before its loop takes it in.
        scheduler.notify_forecaster("note_arrival", request)
        assert request.forecast_remaining == 16
        scheduler.add_request(request)
    return scheduler, busy


def step_once(scheduler, busy: Request) -> None:
    output = scheduler.schedule()
    scheduler.update_from_output(
        output, create_model_runner_output([busy], token_id=1000)
    )


def test_waiting_request_is_recounted_after_its_prefix_is_evicted(forecasting):
    scheduler, busy = one_seat_with_a_cached_prompt(num_waiting=2)
    forecaster = scheduler.forecaster
    forecaster.refresh_budget = 10.0
    step_once(scheduler, busy)
    # While the prefix is intact each waiting request counts its last block.
    assert [p[0] for p in forecaster.snapshot.prefills] == [16, 16]
    assert forecaster.describe()["refresh_changes"] == 0

    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    step_once(scheduler, busy)
    assert [p[0] for p in forecaster.snapshot.prefills] == [160, 160]
    assert forecaster.describe()["refresh_changes"] == 2


def test_refresh_continues_where_the_budget_ran_out(forecasting):
    scheduler, busy = one_seat_with_a_cached_prompt(num_waiting=3)
    forecaster = scheduler.forecaster
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    # A budget this small allows one request per step.
    forecaster.refresh_budget = 1e-9
    for expected in ([160, 16, 16], [160, 160, 16], [160, 160, 160]):
        step_once(scheduler, busy)
        assert [p[0] for p in forecaster.snapshot.prefills] == expected


def test_busy_share_decays_while_idle(forecasting):
    scheduler = create_scheduler()
    forecaster = scheduler.forecaster
    request = create_requests(1, num_tokens=20, max_tokens=1)[0]
    scheduler.add_request(request)
    output = scheduler.schedule()
    # A step in flight means the engine is not idle.
    forecaster.busy_share = 0.8
    assert forecaster.busy_now() == 0.8
    scheduler.update_from_output(
        output, create_model_runner_output([request], use_eos=True)
    )
    forecaster.busy_share = 0.8
    forecaster.load_window = 0.01
    deadline = time.monotonic() + 5.0
    while forecaster.busy_now() > 0.4 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert forecaster.busy_now() <= 0.4


def test_recent_steps_slower_than_the_fit_stretch_the_forecast():
    f = make()
    set_state(f)
    fitted = f.ttft(250)
    now = time.monotonic()
    for _ in range(3):
        f.pace_model.observe(2.0, now)
    assert f.pace() == pytest.approx(2.0)
    assert f.ttft(250) == pytest.approx(2 * fitted, abs=1e-3)
    assert curve_value(f.floor(), 250) == pytest.approx(2 * fitted, rel=0.03)
    assert f.decode_step() == pytest.approx(2 * step(1), abs=1e-3)


def test_pace_ignores_one_slow_step_and_steps_long_ago():
    now = time.monotonic()
    pace = PaceModel()
    for ratio in (1.0, 1.0, 6.0):
        pace.observe(ratio, now)
    assert pace.value(now) == 1.0
    old = PaceModel()
    for _ in range(3):
        old.observe(2.0, now - 2 * old.window)
    assert old.value(now) == 1.0


def test_refit_is_limited_by_time_after_warmup():
    model = StepTimeModel(refit_every=1, refit_seconds=3600.0, warmup=16)
    for i in range(16):
        model.observe(StepTimeModel.features(1 + i % 4, 0, 0), 0.01)
    fitted = model.coef
    # Slower steps arrive, but the next fit is an hour away.
    for i in range(64):
        model.observe(StepTimeModel.features(1 + i % 4, 0, 0), 0.05)
    assert model.coef == fitted


def test_real_scheduler_feeds_the_forecaster(forecasting):
    scheduler = create_scheduler()
    assert scheduler.forecaster is not None
    for request in create_requests(num_requests=2, num_tokens=20):
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert sum(output.num_scheduled_tokens.values()) == 40
    assert scheduler.forecaster.describe()["hook_calls"] == 1


def test_scheduler_has_no_forecaster_by_default():
    assert create_scheduler().forecaster is None


class FailingForecaster(Forecaster):
    def on_schedule(self, scheduler, scheduler_output) -> None:
        raise RuntimeError("forecaster bug")


def test_failing_forecaster_is_switched_off(forecasting):
    scheduler = create_scheduler()
    scheduler.forecaster = FailingForecaster(100, 4, 0, False, 1000)
    for request in create_requests(num_requests=1, num_tokens=20):
        scheduler.add_request(request)
    # The engine keeps scheduling; only the forecast is lost.
    output = scheduler.schedule()
    assert sum(output.num_scheduled_tokens.values()) == 20
    assert scheduler.forecaster is None


def test_scrapes_close_together_share_one_refresh():
    calls = 0

    async def refresh():
        nonlocal calls
        calls += 1
        return True

    async def app(scope, receive, send):
        return None

    middleware = _RefreshBeforeMetrics(app, refresh, min_interval=60.0)
    scope = {"type": "http", "path": "/metrics"}

    async def scrape_twice():
        await asyncio.gather(
            middleware(scope, None, None), middleware(scope, None, None)
        )

    asyncio.run(scrape_twice())
    assert calls == 1


def test_scrape_does_not_wait_for_an_engine_that_is_slow_to_answer():
    calls = 0
    answered = 0
    served = 0
    ages: list[float] = []

    async def scrapes():
        nonlocal calls, answered, served
        engine_answers = asyncio.Event()
        engine_answers.set()

        async def refresh():
            nonlocal calls, answered
            calls += 1
            await engine_answers.wait()
            answered += 1
            return True

        async def app(scope, receive, send):
            nonlocal served
            served += 1

        timeout = 0.05
        middleware = _RefreshBeforeMetrics(
            app, refresh, ages.append, min_interval=0.0, timeout=timeout
        )
        scope = {"type": "http", "path": "/metrics"}
        await middleware(scope, None, None)
        assert calls == 1 and served == 1

        engine_answers.clear()
        started = time.monotonic()
        await middleware(scope, None, None)
        await middleware(scope, None, None)
        # Both were served the earlier values, each after one timeout, and the
        # second did not start another refresh behind the one in flight.
        waited = time.monotonic() - started
        assert waited >= 1.8 * timeout
        assert served == 3 and calls == 2
        assert ages[-1] >= 1.8 * timeout

        engine_answers.set()

        async def refresh_in_flight_is_done():
            while answered < 2:
                await asyncio.sleep(0)

        await asyncio.wait_for(refresh_in_flight_is_done(), 5.0)
        await middleware(scope, None, None)
        assert calls == 3 and ages[-1] < waited

    asyncio.run(scrapes())


def test_probe_waits_for_the_idle_step_in_flight():
    f = make()
    set_state(f)
    f.idle_step_seconds = 0.2
    f.on_idle_step_start()
    assert f.ttft(10) == pytest.approx(0.2 + step(10), abs=5e-3)
    f.on_idle_step_end()
    assert f.ttft(10) == pytest.approx(step(10), abs=5e-3)
    assert f.describe()["idle_steps"] == 1


def test_gauge_knots_sit_on_both_sides_of_a_step_boundary():
    knots = forecast_new_token_knots(max_len=100_000, token_budget=8192)
    assert knots == sorted(set(knots))
    for boundary in (8192, 16384, 65536):
        assert boundary in knots and boundary + 1 in knots
    assert knots[-1] == 100_000
    # No budget reported: powers of two only.
    assert forecast_new_token_knots(1000, 0) == [16, 32, 64, 128, 256, 512, 1000]


def test_restore_wait_is_timed_and_the_request_counts_what_is_left(forecasting):
    # A connector loads 128 of the 160 prompt tokens; 32 are left to compute.
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(matched_tokens=128, is_async=True),
    )
    forecaster = scheduler.forecaster
    request = create_requests(1, num_tokens=160, max_tokens=1)[0]
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert [p[0] for p in forecaster.snapshot.prefills] == [32]
    assert forecaster.describe()["restores"] == 0

    scheduler.update_from_output(
        output,
        create_model_runner_output([], finished_recving={request.request_id}),
    )
    scheduler.schedule()
    assert request.status == RequestStatus.RUNNING
    assert forecaster.describe()["restores"] == 1
    assert forecaster.restore_seconds > 0.0


def test_forecast_counts_from_the_server_arrival_stamp():
    f = make(budget=100_000)
    f.max_model_len = 200_000
    set_state(f)
    bare = f.ttft(50_000)

    # The server holds a request 20 ms plus 1 us per prompt token before the
    # scheduler's input thread has it.
    for i in range(40):
        tokens = 10_000 + 2_000 * i
        request = arrival(f"w{i}", tokens, 0.020 + 1e-6 * tokens)
        f.note_arrival(request, cached_tokens=0)
        f.note_added(request)
    assert f.ingest.base == pytest.approx(0.020, abs=2e-3)
    assert f.ingest.per_token == pytest.approx(1e-6, rel=0.1)

    # The scheduler has taken all 40 in, so none of them is queued work.
    set_state(f, seen_seq=40)
    assert f.ttft(50_000) == pytest.approx(bare + 0.070, abs=3e-3)
    # The curve of a request with a cached prefix carries the whole prompt.
    fitted, f.ingest = f.ingest, IngestModel()
    without = curve_value(f.curve(40_000), 10_000), curve_value(f.floor(), 50_000)
    f.ingest = fitted
    assert curve_value(f.curve(40_000), 10_000) - without[0] == pytest.approx(
        0.070, abs=3e-3
    )
    assert curve_value(f.floor(), 50_000) - without[1] == pytest.approx(0.070, abs=3e-3)

    # The engine's own check starts at the stamp as well.
    request = arrival("scored", 50_000, 0.070)
    f.note_arrival(request, cached_tokens=0)
    made_at, forecast, _ = request.forecast_made
    assert time.monotonic() - made_at == pytest.approx(0.070, abs=5e-3)
    assert forecast == pytest.approx(bare + 0.070, abs=5e-3)


def test_ingest_without_a_spread_of_prompt_sizes_is_the_mean():
    m = IngestModel()
    for _ in range(20):
        m.observe(50_000, 0.060)
    assert m.per_token == 0.0
    assert m.predict(10) == pytest.approx(0.060)
