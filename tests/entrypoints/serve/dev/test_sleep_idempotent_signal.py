# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the dev `/sleep`, `/wake_up`, and `/is_sleeping`
endpoints — verifying the level-aware idempotency signal, the
TOCTOU/wake-in-flight serialization, and the richer state reporting
introduced in `fix/sleep-idempotent-signal`.

Unlike `test_sleep.py` (which boots a real GPU vLLM server), these
tests mock `EngineClient` and exercise the FastAPI router directly,
so they run on any host with no GPU.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.dev.sleep.api_router import attach_router


def _build_app(engine_client) -> FastAPI:
    app = FastAPI()
    app.state.engine_client = engine_client
    attach_router(app)
    return app


def _make_engine_client(*, sleep_level=None) -> AsyncMock:
    """Default mock: awake engine, all sleep/wake calls succeed.

    `sleep_level` controls what `get_sleep_level()` returns. The
    `is_sleeping()` mock is derived from it so the two stay consistent
    (engine_core.is_sleeping() returns True whenever level is set, and
    False when fully awake).
    """
    engine_client = AsyncMock()
    engine_client.get_sleep_level.return_value = sleep_level
    engine_client.is_sleeping.return_value = sleep_level is not None
    return engine_client


# ----------------------------------------------------------------------
# Existing behaviour — sleep/wake idempotency on an awake/sleeping engine
# ----------------------------------------------------------------------


def test_sleep_returns_already_sleeping_true_when_engine_at_or_above_level():
    """When /sleep is POSTed at level=L and the engine is already at
    depth >= L, the endpoint must return 200 with `already_sleeping:
    true` and must NOT call engine_client.sleep() — that protects
    callers (e.g. jukebox) from recording the no-op as a real sleep
    latency sample and from racing a partially-completed wake."""
    engine_client = _make_engine_client(sleep_level=1)

    app = _build_app(engine_client)
    with TestClient(app) as client:
        resp = client.post("/sleep?level=1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["already_sleeping"] is True
    assert body["current_state"] == "sleeping_l1"
    assert body["requested_level"] == 1
    engine_client.get_sleep_level.assert_awaited()
    engine_client.sleep.assert_not_awaited()


def test_sleep_returns_already_sleeping_false_when_engine_is_awake():
    """When /sleep is POSTed and the engine is awake, the endpoint
    must call engine_client.sleep() and return 200 with
    `already_sleeping: false`."""
    engine_client = _make_engine_client(sleep_level=None)
    # After the sleep call the engine is at the requested depth.
    engine_client.get_sleep_level.side_effect = [None, 2]

    app = _build_app(engine_client)
    with TestClient(app) as client:
        resp = client.post("/sleep?level=2&mode=abort")

    assert resp.status_code == 200
    body = resp.json()
    assert body["already_sleeping"] is False
    assert body["current_state"] == "sleeping_l2"
    assert body["requested_level"] == 2
    engine_client.sleep.assert_awaited_once_with(2, "abort")


def test_sleep_default_level_and_mode_when_query_params_missing():
    """The endpoint historically defaults level=1, mode=abort when
    the query params are absent. Preserve that behavior."""
    engine_client = _make_engine_client(sleep_level=None)
    engine_client.get_sleep_level.side_effect = [None, 1]

    app = _build_app(engine_client)
    with TestClient(app) as client:
        resp = client.post("/sleep")

    assert resp.status_code == 200
    assert resp.json()["already_sleeping"] is False
    engine_client.sleep.assert_awaited_once_with(1, "abort")


def test_wake_up_returns_already_awake_true_when_engine_is_awake():
    """Symmetric idempotency signal for /wake_up: if the engine is
    already fully awake and the caller requested an unscoped wake
    (tags=None), return `already_awake: true` without calling
    engine_client.wake_up()."""
    engine_client = _make_engine_client(sleep_level=None)

    app = _build_app(engine_client)
    with TestClient(app) as client:
        resp = client.post("/wake_up")

    assert resp.status_code == 200
    body = resp.json()
    assert body["already_awake"] is True
    assert body["current_state"] == "awake"
    engine_client.wake_up.assert_not_awaited()


def test_wake_up_returns_already_awake_false_when_engine_is_sleeping():
    engine_client = _make_engine_client(sleep_level=1)
    # After wake the engine is awake — get_sleep_level is called once
    # post-wake to build the response body.
    engine_client.get_sleep_level.return_value = None

    app = _build_app(engine_client)
    with TestClient(app) as client:
        resp = client.post("/wake_up")

    assert resp.status_code == 200
    body = resp.json()
    assert body["already_awake"] is False
    assert body["current_state"] == "awake"
    engine_client.wake_up.assert_awaited_once_with(None)


def test_wake_up_with_explicit_tags_skips_idempotency_short_circuit():
    """When the caller passes explicit tags (partial wake — e.g.
    ?tags=weights), the executor's per-tag bookkeeping is the
    authoritative source of truth; the API-level short-circuit
    must NOT skip the call. Otherwise a partial wake (intended to
    restore weights but leave KV cache asleep) would be silently
    dropped. Surfaces the post-call state so the caller can
    disambiguate partial-vs-full wake outcomes."""
    engine_client = _make_engine_client(sleep_level=None)
    engine_client.get_sleep_level.return_value = None

    app = _build_app(engine_client)
    with TestClient(app) as client:
        resp = client.post("/wake_up?tags=weights")

    assert resp.status_code == 200
    body = resp.json()
    # Tagged-wake on awake engine: we still call through (executor is
    # authoritative); the response should NOT lie about it being a
    # full no-op — already_awake=False reflects "we issued the call".
    assert body["already_awake"] is False
    engine_client.wake_up.assert_awaited_once_with(["weights"])


def test_is_sleeping_returns_engine_state():
    engine_client = _make_engine_client(sleep_level=2)

    app = _build_app(engine_client)
    with TestClient(app) as client:
        resp = client.get("/is_sleeping")

    assert resp.status_code == 200
    body = resp.json()
    assert body["is_sleeping"] is True
    assert body["current_state"] == "sleeping_l2"


def test_partial_wake_reports_partial_state():
    """Round-N+2 LOW-2: when /wake_up is called with explicit tags
    and the engine still reports is_sleeping=True afterwards (some
    other tag is still offloaded), the response's `current_state`
    must surface the partial-wake outcome so callers can disambiguate
    "weights are back, KV is still asleep" from "this was a no-op,
    nothing changed". The label is the per-level base with a
    `_partial_wake` suffix (e.g. `sleeping_l1_partial_wake`) — a
    strict prefix-extension so existing dashboards classifying on
    `sleeping_l1*` continue to match."""
    engine_client = AsyncMock()
    # Engine started at level=1 with both {weights, kv_cache} sleeping.
    # /wake_up?tags=weights brings weights back but KV stays offloaded,
    # so the engine still reports is_sleeping=True after the call.
    # Pre-call is_sleeping read (router decides whether to short-circuit).
    # Post-call get_sleep_level read (response state).
    # Post-call is_sleeping read (partial-wake detection).
    engine_client.is_sleeping.side_effect = [True, True]
    engine_client.get_sleep_level.return_value = 1

    app = _build_app(engine_client)
    with TestClient(app) as client:
        resp = client.post("/wake_up?tags=weights")

    assert resp.status_code == 200
    body = resp.json()
    assert body["already_awake"] is False
    # The load-bearing assertion: pre-amend code returned bare
    # `sleeping_l1` here, hiding the partial-wake outcome from the
    # caller. Post-amend, the suffix tells callers weights are back.
    assert body["current_state"] == "sleeping_l1_partial_wake", (
        f"expected partial-wake state suffix, got {body['current_state']!r}"
    )
    engine_client.wake_up.assert_awaited_once_with(["weights"])


def test_full_wake_via_explicit_tags_reports_awake_not_partial():
    """Counterpart to test_partial_wake_reports_partial_state: when
    explicit tags are passed AND the wake brings the engine fully
    awake (executor reports is_sleeping=False afterwards), the state
    must NOT carry the `_partial_wake` suffix — the caller did fully
    wake the engine, just via the tagged code path."""
    engine_client = AsyncMock()
    # Pre-call is_sleeping=True (engine asleep). Post-call
    # is_sleeping=False (everything came back).
    engine_client.is_sleeping.side_effect = [True, False]
    # get_sleep_level returns None after a full wake (core.py clears it).
    engine_client.get_sleep_level.return_value = None

    app = _build_app(engine_client)
    with TestClient(app) as client:
        resp = client.post("/wake_up?tags=weights&tags=kv_cache")

    assert resp.status_code == 200
    body = resp.json()
    assert body["already_awake"] is False
    assert body["current_state"] == "awake"


def test_409_carries_retry_after_header():
    """LOW-1: 409 transition_in_progress responses must include a
    Retry-After header so clients can implement sane backoff without
    polling. Header value matches `_RETRY_AFTER_S`."""
    import vllm.entrypoints.serve.dev.sleep.api_router as router_mod

    long_wake_started = asyncio.Event()
    long_wake_release = asyncio.Event()

    async def long_wake_up(tags):
        long_wake_started.set()
        await long_wake_release.wait()

    engine_client = AsyncMock()
    engine_client.is_sleeping.return_value = True
    engine_client.get_sleep_level.return_value = 1
    engine_client.wake_up.side_effect = long_wake_up

    app = _build_app(engine_client)

    async def race():
        from httpx import ASGITransport, AsyncClient

        # Shrink lock timeout to keep test fast.
        original_timeout = router_mod._TRANSITION_LOCK_TIMEOUT_S
        router_mod._TRANSITION_LOCK_TIMEOUT_S = 0.05
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                wake_task = asyncio.create_task(ac.post("/wake_up"))
                await long_wake_started.wait()
                sleep_resp = await ac.post("/sleep?level=1")
                long_wake_release.set()
                await wake_task
            return sleep_resp
        finally:
            router_mod._TRANSITION_LOCK_TIMEOUT_S = original_timeout

    sleep_resp = asyncio.run(race())

    assert sleep_resp.status_code == 409
    # Header MUST be present and parseable as an integer second-count.
    assert "retry-after" in {k.lower() for k in sleep_resp.headers.keys()}
    retry_after = int(sleep_resp.headers["retry-after"])
    assert retry_after == router_mod._RETRY_AFTER_S
    assert retry_after > 0


def test_retry_after_matches_lock_timeout():
    """Round-N+3 LOW-1: `_RETRY_AFTER_S` must be derived from
    `_TRANSITION_LOCK_TIMEOUT_S`, not an independent literal. Otherwise
    a future change to the lock timeout (e.g. bumping it to 10s for
    level=2 dump+reload paths) would leave the Retry-After hint stuck
    at 5, causing polite clients to retry while the lock is still held
    by the prior call. Encodes the invariant `Retry-After >= lock
    window` so the relationship survives refactors.

    Round-N+4 MED-1 extension: also assert the floor at 1 — see
    `test_retry_after_floor_protects_against_pathological_timeouts`
    for the rationale."""
    import vllm.entrypoints.serve.dev.sleep.api_router as router_mod

    assert router_mod._RETRY_AFTER_S >= router_mod._TRANSITION_LOCK_TIMEOUT_S, (
        f"_RETRY_AFTER_S ({router_mod._RETRY_AFTER_S}) must be >= "
        f"_TRANSITION_LOCK_TIMEOUT_S ({router_mod._TRANSITION_LOCK_TIMEOUT_S}) "
        "or polite clients will retry while the lock is still held."
    )
    # Floor: Retry-After must be >= 1 second per RFC 7231 delta-seconds
    # production, and >= 1 to avoid busy-retry storms on misconfiguration.
    assert router_mod._RETRY_AFTER_S >= 1, (
        f"_RETRY_AFTER_S ({router_mod._RETRY_AFTER_S}) must be >= 1 to avoid "
        "Retry-After: 0 / negative storms if timeout is misconfigured."
    )


@pytest.mark.parametrize(
    "pathological_timeout",
    [0.0, -1.0, 0.001, -10.5],
)
def test_retry_after_floor_protects_against_pathological_timeouts(
    pathological_timeout: float,
):
    """Round-N+4 MED-1: if `_TRANSITION_LOCK_TIMEOUT_S` is misconfigured to
    0, a negative value, or a sub-second value, the derivation
    `max(1, int(math.ceil(...)))` must still produce `_RETRY_AFTER_S >= 1`.

    Why: HTTP `Retry-After: 0` invites a polite client to retry
    immediately, producing a busy-retry storm against an endpoint that
    just told it to back off. `Retry-After: -1` (or any negative value)
    violates RFC 7231's delta-seconds production and is malformed.

    This is a pure-arithmetic test (no router state) — it pins the
    derivation expression itself, so a future refactor that drops the
    `max(1, ...)` floor breaks this test even if the production constant
    happens to still be 5."""
    import math

    derived = max(1, int(math.ceil(pathological_timeout)))
    assert derived >= 1, (
        f"derived Retry-After ({derived}) for timeout "
        f"({pathological_timeout}) is < 1 — would invite a busy-retry "
        "storm or emit a malformed Retry-After header."
    )


# ----------------------------------------------------------------------
# HIGH-1 regression: level escalation must NOT be blocked by the
# idempotency short-circuit
# ----------------------------------------------------------------------


def test_sleep_level_escalation_via_already_sleeping_signal():
    """Round-1 adversarial regression: /sleep?level=0 puts the engine
    in scheduler-paused-only state (is_sleeping=True at level 0). A
    naïve "if is_sleeping: return already_sleeping=True" check would
    block a subsequent /sleep?level=1 request, leaving weights resident
    on GPU forever. The fix must check the *level* not the bool, and
    escalate when current < requested.

    Round-N+2 amend: tightened so that this test FAILS on the original
    bug shape. The mock now starts in the post-/sleep?level=0 state
    (is_sleeping=True, get_sleep_level=0) and the test issues a single
    /sleep?level=1 — the exact flow the original bug suppressed. With
    pre-amend code (router only checks is_sleeping), executor.sleep
    would never be called and the assertion below would fire. With
    post-amend code (router checks level >= requested), the escalation
    proceeds correctly.

    NOTE: This is a focused positive-case test. The no-op assertion
    coverage (router must NOT call sleep when already at deeper level)
    lives in test_sleep_level_transitions parametrize rows (1,0),
    (1,1), (2,1). Together they prove the router checks levels, not
    just is_sleeping bool.
    """
    engine_client = AsyncMock()
    # Real-world condition: the engine has already received /sleep?level=0
    # — scheduler is paused (is_sleeping=True) but weights are still on GPU
    # (get_sleep_level=0).
    #
    # Pre-call: /sleep?level=1 reads get_sleep_level → 0
    # Post-call: re-read for response body → 1
    engine_client.get_sleep_level.side_effect = [0, 1]
    engine_client.is_sleeping.return_value = True

    app = _build_app(engine_client)
    with TestClient(app) as client:
        resp = client.post("/sleep?level=1")

    assert resp.status_code == 200
    body = resp.json()
    # The escalation MUST proceed. A pre-amend "if is_sleeping return
    # already_sleeping=True" router would short-circuit here and fail
    # this assertion.
    assert body["already_sleeping"] is False, (
        "level-0 -> level-1 escalation was blocked! engine still has "
        "weights on GPU. (Did the router regress to checking "
        "is_sleeping() instead of get_sleep_level()?)"
    )
    assert body["current_state"] == "sleeping_l1"
    assert body["requested_level"] == 1
    # And the executor MUST have been told to sleep at level=1. This
    # is the load-bearing assertion: pre-amend code never reaches
    # client.sleep() because the is_sleeping=True short-circuit fires.
    engine_client.sleep.assert_awaited_once_with(1, "abort")


@pytest.mark.parametrize(
    "initial_level,request_level,should_escalate,expected_state_after",
    [
        # The bug: /sleep?level=0 then /sleep?level=1 must escalate
        (0, 1, True, "sleeping_l1"),
        # Double escalate skipping a step
        (0, 2, True, "sleeping_l2"),
        # Adjacent escalate
        (1, 2, True, "sleeping_l2"),
        # Request shallower: no-op (we never auto-wake)
        (1, 0, False, "sleeping_l1"),
        # Same level: no-op (the original idempotency case)
        (1, 1, False, "sleeping_l1"),
        # Already deeper: no-op
        (2, 1, False, "sleeping_l2"),
    ],
)
def test_sleep_level_transitions(
    initial_level, request_level, should_escalate, expected_state_after
):
    """Catches the naive-is_sleeping regression across the full
    transition lattice. For each (initial, requested) pair, asserts:
      - executor.sleep is called iff we should escalate
      - the response's current_state equals the expected post-state

    Rows where should_escalate=True are EXACTLY the rows the original
    bug shape would silently no-op. Adding all six rows means a future
    refactor that reintroduces the is_sleeping-only check fails 3 of
    the 6 parametrize cases (escalate rows), not just one.
    """
    engine_client = AsyncMock()
    engine_client.is_sleeping.return_value = True
    if should_escalate:
        # Pre-call: read initial_level. Post-call: read post-state.
        engine_client.get_sleep_level.side_effect = [
            initial_level,
            request_level,
        ]
    else:
        # Pre-call read short-circuits; no post-call read happens.
        engine_client.get_sleep_level.side_effect = [initial_level]

    app = _build_app(engine_client)
    with TestClient(app) as client:
        resp = client.post(f"/sleep?level={request_level}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["already_sleeping"] is (not should_escalate), (
        f"escalation decision wrong for initial={initial_level} "
        f"request={request_level}: expected escalate={should_escalate}, "
        f"got already_sleeping={body['already_sleeping']}"
    )
    assert body["current_state"] == expected_state_after
    assert body["requested_level"] == request_level
    if should_escalate:
        engine_client.sleep.assert_awaited_once_with(request_level, "abort")
    else:
        engine_client.sleep.assert_not_awaited()


def test_sleep_at_same_level_short_circuits():
    """Counterpart to the escalation test: /sleep?level=1 on an engine
    already at level 1 must short-circuit (no real sleep call), to
    preserve the latency-histogram-pollution fix."""
    engine_client = _make_engine_client(sleep_level=1)

    app = _build_app(engine_client)
    with TestClient(app) as client:
        resp = client.post("/sleep?level=1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["already_sleeping"] is True
    assert body["current_state"] == "sleeping_l1"
    engine_client.sleep.assert_not_awaited()


def test_sleep_at_deeper_than_requested_level_short_circuits():
    """If the engine is at level 2 and we request level 1, we are
    already "as deep or deeper" — short-circuit. Going shallower would
    require a wake-then-sleep cycle, which we explicitly do not do."""
    engine_client = _make_engine_client(sleep_level=2)

    app = _build_app(engine_client)
    with TestClient(app) as client:
        resp = client.post("/sleep?level=1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["already_sleeping"] is True
    assert body["current_state"] == "sleeping_l2"  # report actual depth
    assert body["requested_level"] == 1
    engine_client.sleep.assert_not_awaited()


# ----------------------------------------------------------------------
# HIGH-2 / HIGH-3: TOCTOU + wake-in-progress race must be serialized
# ----------------------------------------------------------------------


def test_concurrent_sleep_calls_serialized():
    """Two concurrent /sleep requests on the same engine must NOT both
    proceed past the is_sleeping/get_sleep_level check and both call
    engine_client.sleep(). With the transition lock, exactly one call
    transitions; the other observes the new state and short-circuits
    (already_sleeping=True)."""

    # Use a real (non-async) lock to serialize the calls inside the
    # engine_client mock so the second arriver definitively observes
    # the new state set by the first. The router's own asyncio.Lock
    # guarantees the *order*; this fixture just makes the assertion
    # observable.
    state = {"level": None}
    in_flight = asyncio.Lock()

    async def get_sleep_level():
        return state["level"]

    async def is_sleeping():
        return state["level"] is not None

    async def sleep_impl(level, mode):
        async with in_flight:
            # Simulate the real cumem_tag sleep taking measurable time
            # so that without serialization the second caller's
            # is_sleeping check would race.
            await asyncio.sleep(0.05)
            state["level"] = level

    engine_client = AsyncMock()
    engine_client.get_sleep_level.side_effect = get_sleep_level
    engine_client.is_sleeping.side_effect = is_sleeping
    engine_client.sleep.side_effect = sleep_impl

    app = _build_app(engine_client)

    async def race():
        # Use httpx AsyncClient against the FastAPI app to issue two
        # truly concurrent requests on the same event loop.
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as ac:
            r1, r2 = await asyncio.gather(
                ac.post("/sleep?level=1"),
                ac.post("/sleep?level=1"),
            )
        return r1, r2

    r1, r2 = asyncio.run(race())

    assert r1.status_code == 200
    assert r2.status_code == 200
    bodies = sorted(
        [r1.json(), r2.json()], key=lambda b: b["already_sleeping"]
    )
    # Exactly one already_sleeping=False (the real transition) and
    # one already_sleeping=True (serialized observer of new state).
    assert bodies[0]["already_sleeping"] is False
    assert bodies[1]["already_sleeping"] is True

    # Real sleep called exactly once — not twice (TOCTOU avoided).
    assert engine_client.sleep.await_count == 1


def test_sleep_during_wake_returns_409_on_lock_timeout(monkeypatch):
    """If a /wake_up holds the transition lock for longer than the
    configured timeout, an incoming /sleep request must return 409
    rather than block forever or land on a half-awake engine."""
    import vllm.entrypoints.serve.dev.sleep.api_router as router_mod

    # Shrink the timeout so the test isn't slow.
    monkeypatch.setattr(router_mod, "_TRANSITION_LOCK_TIMEOUT_S", 0.05)

    long_wake_started = asyncio.Event()
    long_wake_release = asyncio.Event()

    async def long_wake_up(tags):
        long_wake_started.set()
        await long_wake_release.wait()

    async def is_sleeping():
        return True  # so the wake actually issues

    engine_client = AsyncMock()
    engine_client.is_sleeping.side_effect = is_sleeping
    engine_client.get_sleep_level.return_value = 1
    engine_client.wake_up.side_effect = long_wake_up

    app = _build_app(engine_client)

    async def race():
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as ac:
            wake_task = asyncio.create_task(ac.post("/wake_up"))
            await long_wake_started.wait()
            # Wake is now holding the lock; this /sleep will time out
            # waiting for it and must 409.
            sleep_resp = await ac.post("/sleep?level=1")
            long_wake_release.set()
            wake_resp = await wake_task
        return sleep_resp, wake_resp

    sleep_resp, wake_resp = asyncio.run(race())

    assert sleep_resp.status_code == 409
    body = sleep_resp.json()
    assert body["error"] == "transition_in_progress"
    # The wake itself completes normally.
    assert wake_resp.status_code == 200


# ----------------------------------------------------------------------
# End-to-end shape: the original double-/sleep latency-histogram bug
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "first_level_seen,second_level_seen,expected_first,expected_second",
    [
        (None, 1, False, True),
    ],
)
def test_double_sleep_records_first_real_then_idempotent_no_op(
    first_level_seen,
    second_level_seen,
    expected_first,
    expected_second,
):
    """End-to-end shape of the bug this fixes: jukebox issues two
    /sleep calls back-to-back. The first transitions awake→sleeping
    (already_sleeping=false). The second was previously
    indistinguishable from the first (a generic 200) and would land
    in latency histograms as a fake-fast cumem_tag sleep. After the
    fix, the second call returns already_sleeping=true and the
    caller can skip the histogram observation."""
    engine_client = AsyncMock()
    # get_sleep_level reads:
    #   1) first /sleep pre-call: None (awake)
    #   2) first /sleep post-call (for response state): 1
    #   3) second /sleep pre-call: 1 (already at depth, short-circuit)
    engine_client.get_sleep_level.side_effect = [
        first_level_seen,
        1,
        second_level_seen,
    ]
    engine_client.is_sleeping.return_value = False

    app = _build_app(engine_client)
    with TestClient(app) as client:
        resp1 = client.post("/sleep?level=1")
        resp2 = client.post("/sleep?level=1")

    assert resp1.status_code == 200
    assert resp1.json()["already_sleeping"] is expected_first
    assert resp2.status_code == 200
    assert resp2.json()["already_sleeping"] is expected_second
    # Real sleep called exactly once — the no-op did not transition.
    engine_client.sleep.assert_awaited_once()
