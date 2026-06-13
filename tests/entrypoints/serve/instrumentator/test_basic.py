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
    monkeypatch.setattr(envs, "VLLM_DECODE_LIVENESS_STALL_SECONDS", 1.0)
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


def test_stat_logger_manager_tracks_decode_liveness():
    """Direct unit test of the StatLoggerManager bookkeeping — exercises
    record() updates and the get_decode_liveness() snapshot without
    needing to spin up a full engine."""
    from unittest.mock import MagicMock

    from vllm.v1.metrics.loggers import StatLoggerManager
    from vllm.v1.metrics.stats import IterationStats, SchedulerStats

    # Build a manager with no default loggers (skip Prometheus) so we don't
    # need a real VllmConfig — record() will just no-op on the loggers list.
    vllm_config = MagicMock()
    manager = StatLoggerManager.__new__(StatLoggerManager)
    manager.engine_indexes = [0]
    manager.stat_loggers = []
    manager._last_token_emit_time = None
    manager._last_num_running_reqs = 0
    del vllm_config

    # Fresh manager: no decode observed, no running.
    running, age = manager.get_decode_liveness()
    assert running == 0
    assert age is None

    # Record a step with running=3 but 0 tokens (e.g. all-prefill step).
    sched = SchedulerStats()
    sched.num_running_reqs = 3
    iter_stats = IterationStats()
    iter_stats.num_generation_tokens = 0
    manager.record(scheduler_stats=sched, iteration_stats=iter_stats)
    running, age = manager.get_decode_liveness()
    assert running == 3
    assert age is None  # still no token ever observed

    # Record a step with a decoded token.
    iter_stats2 = IterationStats()
    iter_stats2.num_generation_tokens = 1
    manager.record(scheduler_stats=sched, iteration_stats=iter_stats2)
    running, age = manager.get_decode_liveness()
    assert running == 3
    assert age is not None
    assert age >= 0.0
    assert age < 1.0  # the token was "just now"

    # Drain: next step reports running=0. The counter should decay (we
    # overwrite, not max), otherwise /health/decode would forever think
    # there's work in flight.
    sched_drained = SchedulerStats()
    sched_drained.num_running_reqs = 0
    iter_stats3 = IterationStats()
    iter_stats3.num_generation_tokens = 0
    manager.record(scheduler_stats=sched_drained, iteration_stats=iter_stats3)
    running, _ = manager.get_decode_liveness()
    assert running == 0

