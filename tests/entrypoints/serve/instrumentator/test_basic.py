# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from http import HTTPStatus
from unittest.mock import AsyncMock, Mock

import openai
import pytest
import pytest_asyncio
import requests
from fastapi import Request

from tests.utils import RemoteOpenAIServer
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.version import __version__ as VLLM_VERSION

MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def server_args(request: pytest.FixtureRequest) -> list[str]:
    """Provide extra arguments to the server via indirect parametrization

    Usage:

    >>> @pytest.mark.parametrize(
    >>>     "server_args",
    >>>     [
    >>>         ["--max-model-len", "10100"],
    >>>         [
    >>>             "--model=NousResearch/Hermes-3-Llama-3.1-70B",
    >>>             "--enable-auto-tool-choice",
    >>>         ],
    >>>     ],
    >>>     indirect=True,
    >>> )
    >>> def test_foo(server, client):
    >>>     ...

    This will run `test_foo` twice with servers with:
    - `--max-model-len 10100`
    - `--model=NousResearch/Hermes-3-Llama-3.1-70B --enable-auto-tool-choice`.

    """
    if not hasattr(request, "param"):
        return []

    val = request.param

    if isinstance(val, str):
        return [val]

    return request.param


@pytest.fixture(scope="module")
def server(server_args):
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        "--max-num-seqs",
        "128",
        *server_args,
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_show_version(server: RemoteOpenAIServer):
    response = requests.get(server.url_for("version"))
    response.raise_for_status()

    assert response.json() == {"version": VLLM_VERSION}


@pytest.mark.asyncio
async def test_check_health(server: RemoteOpenAIServer):
    response = requests.get(server.url_for("health"))

    assert response.status_code == HTTPStatus.OK


@pytest.mark.parametrize(
    "server_args",
    [
        pytest.param(["--max-model-len", "10100"]),
    ],
    indirect=True,
)
@pytest.mark.asyncio
async def test_request_cancellation(server: RemoteOpenAIServer):
    # clunky test: send an ungodly amount of load in with short timeouts
    # then ensure that it still responds quickly afterwards

    chat_input = [{"role": "user", "content": "Write a long story"}]
    client = server.get_async_client(timeout=0.5)
    tasks = []
    # Request about 2 million tokens
    for _ in range(200):
        task = asyncio.create_task(
            client.chat.completions.create(
                messages=chat_input,
                model=MODEL_NAME,
                max_tokens=10000,
                extra_body={"min_tokens": 10000},
                temperature=0.0,
            )
        )
        tasks.append(task)

    done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    # Make sure all requests were sent to the server and timed out
    # (We don't want to hide other errors like 400s that would invalidate this
    # test)
    assert len(pending) == 0
    for d in done:
        with pytest.raises(openai.APITimeoutError):
            d.result()

    # If the server had not cancelled all the other requests, then it would not
    # be able to respond to this one within the timeout
    client = server.get_async_client(timeout=5)
    response = await client.chat.completions.create(
        messages=chat_input, model=MODEL_NAME, max_tokens=10, temperature=0.0
    )

    assert len(response.choices) == 1


@pytest.mark.asyncio
async def test_request_wrong_content_type(server: RemoteOpenAIServer):
    chat_input = [{"role": "user", "content": "Write a long story"}]
    client = server.get_async_client()

    with pytest.raises(openai.APIStatusError):
        await client.chat.completions.create(
            messages=chat_input,
            model=MODEL_NAME,
            max_tokens=10000,
            extra_headers={"Content-Type": "application/x-www-form-urlencoded"},
        )


@pytest.mark.parametrize(
    "server_args",
    [pytest.param(["--enable-server-load-tracking"], id="enable-server-load-tracking")],
    indirect=True,
)
@pytest.mark.asyncio
async def test_server_load(server: RemoteOpenAIServer):
    # Check initial server load
    response = requests.get(server.url_for("load"))
    assert response.status_code == HTTPStatus.OK
    assert response.json().get("server_load") == 0

    def make_long_completion_request():
        return requests.post(
            server.url_for("v1/completions"),
            headers={"Content-Type": "application/json"},
            json={
                "prompt": "Give me a long story",
                "max_tokens": 1000,
                "temperature": 0,
            },
        )

    # Start the completion request in a background thread.
    completion_future = asyncio.create_task(
        asyncio.to_thread(make_long_completion_request)
    )

    # Give a short delay to ensure the request has started.
    await asyncio.sleep(0.1)

    # Check server load while the completion request is running.
    response = requests.get(server.url_for("load"))
    assert response.status_code == HTTPStatus.OK
    assert response.json().get("server_load") == 1

    # Wait for the completion request to finish.
    await completion_future
    await asyncio.sleep(0.1)

    # Check server load after the completion request has finished.
    response = requests.get(server.url_for("load"))
    assert response.status_code == HTTPStatus.OK
    assert response.json().get("server_load") == 0


@pytest.mark.asyncio
async def test_health_check_engine_dead_error():
    # Import the health function directly to test it in isolation
    from vllm.entrypoints.serve.instrumentator.health import health

    # Create a mock request that simulates what FastAPI would provide
    mock_request = Mock(spec=Request)
    mock_app_state = Mock()
    mock_engine_client = AsyncMock()
    mock_engine_client.check_health.side_effect = EngineDeadError()
    mock_app_state.engine_client = mock_engine_client
    mock_request.app.state = mock_app_state

    # Test the health function directly with our mocked request
    # This simulates what would happen if the engine dies
    response = await health(mock_request)

    # Assert that it returns 503 Service Unavailable
    assert response.status_code == 503


def _make_decode_request(running: int, last_token_age: float | None) -> Mock:
    """Build a mock Request whose engine_client.get_decode_liveness()
    returns ``(running, last_token_age)``."""
    mock_request = Mock(spec=Request)
    mock_app_state = Mock()
    mock_engine_client = AsyncMock()
    mock_engine_client.get_decode_liveness.return_value = (running, last_token_age)
    mock_app_state.engine_client = mock_engine_client
    mock_request.app.state = mock_app_state
    return mock_request


@pytest.mark.asyncio
async def test_health_decode_ok_when_progressing():
    """Running > 0 + recent token => 200 ok."""
    from vllm.entrypoints.serve.instrumentator.health import health_decode

    response = await health_decode(_make_decode_request(running=3, last_token_age=0.5))
    assert response.status_code == 200
    import json

    body = json.loads(response.body)
    assert body["status"] == "ok"
    assert body["running"] == 3
    assert body["last_token_age_seconds"] == 0.5
    assert body["stall_threshold_seconds"] > 0


@pytest.mark.asyncio
async def test_health_decode_idle_when_no_running():
    """running == 0 is idle, not stalled, even if last_token_age is huge."""
    from vllm.entrypoints.serve.instrumentator.health import health_decode

    response = await health_decode(_make_decode_request(running=0, last_token_age=9999.0))
    assert response.status_code == 200
    import json

    body = json.loads(response.body)
    assert body["status"] == "idle"
    assert body["running"] == 0


@pytest.mark.asyncio
async def test_health_decode_idle_when_never_decoded():
    """No token ever emitted (cold start) => idle, 200."""
    from vllm.entrypoints.serve.instrumentator.health import health_decode

    response = await health_decode(_make_decode_request(running=0, last_token_age=None))
    assert response.status_code == 200
    import json

    body = json.loads(response.body)
    assert body["status"] == "idle"
    assert body["last_token_age_seconds"] is None


@pytest.mark.asyncio
async def test_health_decode_idle_when_long_prefill_in_flight():
    """Running > 0 but no token EVER emitted (e.g. long prefill on the first
    request the engine has ever seen) is reported as idle, not stalled —
    we have no reference point to call it stalled."""
    from vllm.entrypoints.serve.instrumentator.health import health_decode

    response = await health_decode(_make_decode_request(running=1, last_token_age=None))
    assert response.status_code == 200
    import json

    body = json.loads(response.body)
    assert body["status"] == "idle"
    assert body["running"] == 1


@pytest.mark.asyncio
async def test_health_decode_stalled_when_running_and_old_token(monkeypatch):
    """Running > 0 + last token older than threshold => 503 stalled."""
    import vllm.envs as envs

    # Tighten the threshold for the test so we don't need to mock 60+ seconds.
    # Use setenv (not setattr) so we exercise the same lookup path the route
    # uses at runtime, and so the override is robust against enable_envs_cache()
    # being toggled. We also clear the cache (if present) to make the next read
    # see the new value.
    monkeypatch.setenv("VLLM_DECODE_LIVENESS_STALL_SECONDS", "1.0")
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()
    from vllm.entrypoints.serve.instrumentator.health import health_decode

    response = await health_decode(_make_decode_request(running=2, last_token_age=5.0))
    assert response.status_code == 503
    import json

    body = json.loads(response.body)
    assert body["status"] == "stalled"
    assert body["running"] == 2
    assert body["last_token_age_seconds"] == 5.0
    assert body["stall_threshold_seconds"] == 1.0


@pytest.mark.asyncio
async def test_health_decode_handles_client_failure():
    """If get_decode_liveness() raises, the endpoint returns 503 (treating
    "engine can't tell us its liveness" as stalled) rather than 500."""
    from vllm.entrypoints.serve.instrumentator.health import health_decode

    mock_request = Mock(spec=Request)
    mock_app_state = Mock()
    mock_engine_client = AsyncMock()
    mock_engine_client.get_decode_liveness.side_effect = RuntimeError("kaboom")
    mock_app_state.engine_client = mock_engine_client
    mock_request.app.state = mock_app_state

    response = await health_decode(mock_request)
    assert response.status_code == 503
    import json

    body = json.loads(response.body)
    assert body["status"] == "stalled"
    assert "kaboom" in body["error"]


def _make_bare_manager(engine_indexes=(0,)):
    """Build a StatLoggerManager via __new__ with the per-engine state
    initialised but no loggers attached. Used by direct unit tests
    that do not need a real VllmConfig."""
    from vllm.v1.metrics.loggers import StatLoggerManager

    manager = StatLoggerManager.__new__(StatLoggerManager)
    manager.engine_indexes = list(engine_indexes)
    manager.stat_loggers = []
    manager._last_token_emit_time_by_engine = {}
    manager._last_num_running_reqs_by_engine = {}
    manager._last_prefill_activity_time_by_engine = {}
    manager._last_token_emit_time = None
    manager._last_num_running_reqs = 0
    return manager


def test_stat_logger_manager_tracks_decode_liveness():
    """Direct unit test of the StatLoggerManager bookkeeping — exercises
    record() updates and the get_decode_liveness() snapshot without
    needing to spin up a full engine."""
    from vllm.v1.metrics.stats import IterationStats, SchedulerStats

    manager = _make_bare_manager()

    # Fresh manager: no decode observed, no running, no prefill.
    running, token_age, prefill_age = manager.get_decode_liveness()
    assert running == 0
    assert token_age is None
    assert prefill_age is None

    # Record a step with running=3, no generated tokens, AND prefill compute
    # this iteration (e.g. an all-prefill step). The new prefill tracker
    # should pick this up; decode tracker should remain None.
    sched = SchedulerStats()
    sched.num_running_reqs = 3
    iter_stats = IterationStats()
    iter_stats.num_generation_tokens = 0
    iter_stats.num_prompt_tokens = 4096
    manager.record(scheduler_stats=sched, iteration_stats=iter_stats)
    running, token_age, prefill_age = manager.get_decode_liveness()
    assert running == 3
    assert token_age is None  # still no token ever observed
    assert prefill_age is not None
    assert prefill_age < 1.0

    # Record a step with a decoded token.
    iter_stats2 = IterationStats()
    iter_stats2.num_generation_tokens = 1
    iter_stats2.num_prompt_tokens = 0
    manager.record(scheduler_stats=sched, iteration_stats=iter_stats2)
    running, token_age, prefill_age = manager.get_decode_liveness()
    assert running == 3
    assert token_age is not None
    assert token_age >= 0.0
    assert token_age < 1.0  # the token was "just now"

    # Drain: next step reports running=0. The counter should decay (we
    # overwrite, not max), otherwise /health/decode would forever think
    # there is work in flight.
    sched_drained = SchedulerStats()
    sched_drained.num_running_reqs = 0
    iter_stats3 = IterationStats()
    iter_stats3.num_generation_tokens = 0
    iter_stats3.num_prompt_tokens = 0
    manager.record(scheduler_stats=sched_drained, iteration_stats=iter_stats3)
    running, _, _ = manager.get_decode_liveness()
    assert running == 0


@pytest.mark.asyncio
async def test_health_decode_503_when_engine_errored():
    """/health/decode must agree with /health when the engine is dead.

    Without an explicit check_health() guard, /health would return 503 but
    /health/decode could return a stale 200 ("ok" / "idle") sourced from the
    last successful record() call — masking engine death from orchestrators
    that probe the new endpoint instead of (or in addition to) /health.
    """
    from vllm.entrypoints.serve.instrumentator.health import health_decode

    mock_request = Mock(spec=Request)
    mock_app_state = Mock()
    mock_engine_client = AsyncMock()
    mock_engine_client.check_health.side_effect = EngineDeadError()
    # Even though check_health raises first, leave a plausible "happy" return
    # on get_decode_liveness() so we know the 503 came from the dead-engine
    # guard rather than the liveness branch.
    mock_engine_client.get_decode_liveness.return_value = (1, 0.1)
    mock_app_state.engine_client = mock_engine_client
    mock_request.app.state = mock_app_state

    response = await health_decode(mock_request)
    assert response.status_code == 503
    import json

    body = json.loads(response.body)
    assert body["status"] == "stalled"
    assert "error" in body
    # And the liveness method must NOT have been consulted — we shed early
    # rather than relying on stale per-step bookkeeping from a dead engine.
    mock_engine_client.get_decode_liveness.assert_not_called()


def test_stat_logger_manager_carry_forward_across_recreate():
    """Models the elastic-EP scale-up path: when AsyncLLM rebuilds the
    StatLoggerManager, the decode-liveness bookkeeping must be carried
    forward — otherwise an in-progress stall is silently masked because
    the fresh manager reports as never-decoded.
    """
    from vllm.v1.metrics.stats import IterationStats, SchedulerStats

    old_mgr = _make_bare_manager()

    # Drive a successful decoding step into the old manager.
    sched = SchedulerStats()
    sched.num_running_reqs = 5
    iter_stats = IterationStats()
    iter_stats.num_generation_tokens = 1
    iter_stats.num_prompt_tokens = 0
    old_mgr.record(scheduler_stats=sched, iteration_stats=iter_stats)
    old_running, old_age, _ = old_mgr.get_decode_liveness()
    assert old_running == 5
    assert old_age is not None

    # Simulate AsyncLLM.scale_elastic_ep() rebuilding the manager and
    # carrying forward the bookkeeping (scalar shim path; the existing
    # carry-forward in scale_elastic_ep moves the scalar attributes).
    new_mgr = _make_bare_manager(engine_indexes=(0, 1))
    new_mgr._last_token_emit_time = old_mgr._last_token_emit_time
    new_mgr._last_num_running_reqs = old_mgr._last_num_running_reqs

    new_running, new_age, _ = new_mgr.get_decode_liveness()
    assert new_running == 5
    assert new_age is not None
    # A stall present before the scale-up remains visible until the next
    # successful record() advances the timestamp on the new manager.


@pytest.mark.asyncio
async def test_async_llm_get_decode_liveness_delegates_to_logger_manager():
    """AsyncLLM.get_decode_liveness() must delegate to its logger_manager
    when one exists, and return (0, None) — the safe "idle" sentinel —
    when stat-logging is disabled (no logger_manager). The latter is what
    keeps the endpoint healthy when the operator runs with --disable-log-stats.
    """
    from vllm.v1.engine.async_llm import AsyncLLM

    # Bypass __init__ — we only need to exercise the delegation logic.
    inst = AsyncLLM.__new__(AsyncLLM)

    # No logger_manager => (0, None, None) idle sentinel.
    inst.logger_manager = None
    assert await inst.get_decode_liveness() == (0, None, None)

    # logger_manager present => delegate.
    fake_mgr = Mock()
    fake_mgr.get_decode_liveness.return_value = (7, 2.5, 0.1)
    inst.logger_manager = fake_mgr
    running, token_age, prefill_age = await inst.get_decode_liveness()
    assert running == 7
    assert token_age == 2.5
    assert prefill_age == 0.1
    fake_mgr.get_decode_liveness.assert_called_once()


@pytest.mark.asyncio
async def test_health_decode_prefilling_when_long_prefill_after_decode(monkeypatch):
    """HIGH-(a) regression: long prefill (>decode_threshold) AFTER the engine
    has already decoded a token must NOT trip 503. The endpoint returns 200
    with status="prefilling" when prefill activity is recent even though the
    last decoded token is older than the decode-stall threshold.
    """
    import vllm.envs as envs

    monkeypatch.setenv("VLLM_DECODE_LIVENESS_STALL_SECONDS", "1.0")
    monkeypatch.setenv("VLLM_PREFILL_LIVENESS_STALL_SECONDS", "120.0")
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()
    from vllm.entrypoints.serve.instrumentator.health import health_decode

    # last decoded token is 5s old (> 1.0s decode threshold) but prefill
    # activity is 0.5s old (< 120s prefill threshold) — engine is busy
    # prefilling a long prompt, not stalled.
    response = await health_decode(
        _make_decode_request(running=1, last_token_age=5.0, last_prefill_age=0.5)
    )
    assert response.status_code == 200
    import json

    body = json.loads(response.body)
    assert body["status"] == "prefilling"
    assert body["running"] == 1
    assert body["last_token_age_seconds"] == 5.0
    assert body["last_prefill_age_seconds"] == 0.5


@pytest.mark.asyncio
async def test_health_decode_stalled_when_prefill_also_stale(monkeypatch):
    """HIGH-(a) negative case: when BOTH the decode threshold AND the
    prefill threshold are exceeded, the engine is genuinely stalled and
    the endpoint must still return 503. Prefill recency is a get-out-of-
    jail card only when prefill is actually recent.
    """
    import vllm.envs as envs

    monkeypatch.setenv("VLLM_DECODE_LIVENESS_STALL_SECONDS", "1.0")
    monkeypatch.setenv("VLLM_PREFILL_LIVENESS_STALL_SECONDS", "2.0")
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()
    from vllm.entrypoints.serve.instrumentator.health import health_decode

    response = await health_decode(
        _make_decode_request(running=1, last_token_age=10.0, last_prefill_age=10.0)
    )
    assert response.status_code == 503
    import json

    body = json.loads(response.body)
    assert body["status"] == "stalled"


@pytest.mark.asyncio
async def test_health_decode_stalled_when_prefill_never_observed(monkeypatch):
    """HIGH-(a) edge case: decode threshold exceeded and the engine has
    never been observed prefilling (prefill_age is None). Pre-existing
    stall semantics apply — return 503.
    """
    import vllm.envs as envs

    monkeypatch.setenv("VLLM_DECODE_LIVENESS_STALL_SECONDS", "1.0")
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()
    from vllm.entrypoints.serve.instrumentator.health import health_decode

    response = await health_decode(
        _make_decode_request(running=1, last_token_age=10.0, last_prefill_age=None)
    )
    assert response.status_code == 503
    import json

    body = json.loads(response.body)
    assert body["status"] == "stalled"


def test_stat_logger_manager_dp_partial_stall_surfaces():
    """HIGH-(b) regression: in a data-parallel deployment, a single
    stalled shard must surface even when sibling shards are healthy.
    Before the fix, the scalar _last_num_running_reqs was overwritten
    by whichever engine called record() last — a healthy shard finishing
    its work and reporting num_running_reqs=0 would mask a sibling
    shard with running>0 stuck on a stalled step.
    """
    import time

    from vllm.v1.metrics.stats import IterationStats, SchedulerStats

    manager = _make_bare_manager(engine_indexes=(0, 1))

    # Engine 0 is stalled: it recorded a step with running=2 a long time
    # ago, then never recorded again. We simulate this by directly
    # populating the dict (calling record() would advance time.monotonic()
    # to "now", which would defeat the test).
    stale_time = time.monotonic() - 9999.0
    manager._last_num_running_reqs_by_engine[0] = 2
    manager._last_token_emit_time_by_engine[0] = stale_time
    # Engine 1 is healthy: it just recorded a step with running=0 (drained)
    # and a fresh decoded token.
    sched_healthy = SchedulerStats()
    sched_healthy.num_running_reqs = 0
    iter_healthy = IterationStats()
    iter_healthy.num_generation_tokens = 1
    iter_healthy.num_prompt_tokens = 0
    manager.record(
        scheduler_stats=sched_healthy,
        iteration_stats=iter_healthy,
        engine_idx=1,
    )

    running, token_age, _ = manager.get_decode_liveness()
    # Max running across engines: engine 0 has 2, engine 1 has 0 -> 2.
    assert running == 2, (
        f"DP partial stall should surface running=2 from engine 0, got {running}"
    )
    # Max token age: engine 0 9999s wins over engine 1 ~0s.
    assert token_age is not None
    assert token_age > 100.0, (
        f"DP partial stall should surface engine 0 old token age, got {token_age}"
    )


def test_stat_logger_manager_dp_per_engine_record_independence():
    """HIGH-(b) positive case: per-engine bookkeeping in record() does NOT
    cross-contaminate. A record() call for engine 0 must not change the
    state for engine 1 and vice-versa.
    """
    from vllm.v1.metrics.stats import IterationStats, SchedulerStats

    manager = _make_bare_manager(engine_indexes=(0, 1))

    # Engine 0 records running=5, a decoded token, and prefill compute.
    sched0 = SchedulerStats()
    sched0.num_running_reqs = 5
    iter0 = IterationStats()
    iter0.num_generation_tokens = 1
    iter0.num_prompt_tokens = 100
    manager.record(scheduler_stats=sched0, iteration_stats=iter0, engine_idx=0)

    # Engine 1 records running=0, no token, no prefill.
    sched1 = SchedulerStats()
    sched1.num_running_reqs = 0
    iter1 = IterationStats()
    iter1.num_generation_tokens = 0
    iter1.num_prompt_tokens = 0
    manager.record(scheduler_stats=sched1, iteration_stats=iter1, engine_idx=1)

    assert manager._last_num_running_reqs_by_engine[0] == 5
    assert manager._last_num_running_reqs_by_engine[1] == 0
    assert manager._last_token_emit_time_by_engine.get(0) is not None
    assert manager._last_token_emit_time_by_engine.get(1) is None
    assert manager._last_prefill_activity_time_by_engine.get(0) is not None
    assert manager._last_prefill_activity_time_by_engine.get(1) is None


@pytest.mark.asyncio
async def test_health_decode_503_on_non_engine_dead_check_health_failure():
    """MED: check_health() raising any non-EngineDeadError exception must
    also surface as 503 (with the error in the body) rather than leaking
    a 500 to the orchestrator. Without this branch, an unexpected engine
    failure would crash the route and the watchdog would have no signal.
    """
    from vllm.entrypoints.serve.instrumentator.health import health_decode

    mock_request = Mock(spec=Request)
    mock_app_state = Mock()
    mock_engine_client = AsyncMock()
    mock_engine_client.check_health.side_effect = RuntimeError("engine wedged")
    # If we reach get_decode_liveness, the test should fail loudly.
    mock_engine_client.get_decode_liveness.return_value = (1, 0.1, 0.1)
    mock_app_state.engine_client = mock_engine_client
    mock_request.app.state = mock_app_state

    response = await health_decode(mock_request)
    assert response.status_code == 503
    import json

    body = json.loads(response.body)
    assert body["status"] == "stalled"
    assert "engine wedged" in body["error"]
    mock_engine_client.get_decode_liveness.assert_not_called()

