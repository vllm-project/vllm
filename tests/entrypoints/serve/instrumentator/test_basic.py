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


def _make_decode_request(
    inflight: int,
    last_progress_age: float | None,
    oldest_unprogressed_age: float | None,
) -> Mock:
    """Build a mock Request whose engine_client.get_decode_liveness()
    returns ``(inflight, last_progress_age, oldest_unprogressed_age)``."""
    mock_request = Mock(spec=Request)
    mock_app_state = Mock()
    mock_engine_client = AsyncMock()
    mock_engine_client.get_decode_liveness.return_value = (
        inflight,
        last_progress_age,
        oldest_unprogressed_age,
    )
    mock_app_state.engine_client = mock_engine_client
    mock_request.app.state = mock_app_state
    return mock_request


@pytest.mark.asyncio
async def test_health_decode_ok_when_progressing():
    """inflight > 0 + recent progress => 200 ok."""
    from vllm.entrypoints.serve.instrumentator.health import health_decode

    response = await health_decode(
        _make_decode_request(
            inflight=3, last_progress_age=0.5, oldest_unprogressed_age=None
        )
    )
    assert response.status_code == 200
    import json

    body = json.loads(response.body)
    assert body["status"] == "ok"
    assert body["inflight"] == 3
    assert body["last_progress_age_seconds"] == 0.5
    assert body["stall_threshold_seconds"] > 0


@pytest.mark.asyncio
async def test_health_decode_idle_when_no_inflight():
    """inflight == 0 is idle, not stalled, even with a huge last_progress_age."""
    from vllm.entrypoints.serve.instrumentator.health import health_decode

    response = await health_decode(
        _make_decode_request(
            inflight=0, last_progress_age=9999.0, oldest_unprogressed_age=None
        )
    )
    assert response.status_code == 200
    import json

    body = json.loads(response.body)
    assert body["status"] == "idle"
    assert body["inflight"] == 0


@pytest.mark.asyncio
async def test_health_decode_idle_when_cold_start():
    """Nothing ever admitted (cold start) => idle, 200."""
    from vllm.entrypoints.serve.instrumentator.health import health_decode

    response = await health_decode(
        _make_decode_request(
            inflight=0, last_progress_age=None, oldest_unprogressed_age=None
        )
    )
    assert response.status_code == 200
    import json

    body = json.loads(response.body)
    assert body["status"] == "idle"
    assert body["last_progress_age_seconds"] is None


@pytest.mark.asyncio
async def test_health_decode_ok_when_long_prefill_under_threshold(monkeypatch):
    """inflight > 0, no progress yet (long prefill on the first request), but
    the admission age is UNDER the threshold => 200 ok, NOT stalled. A legit
    slow prefill must not false-positive."""
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_DECODE_LIVENESS_STALL_SECONDS", 60.0)
    from vllm.entrypoints.serve.instrumentator.health import health_decode

    response = await health_decode(
        _make_decode_request(
            inflight=1, last_progress_age=None, oldest_unprogressed_age=5.0
        )
    )
    assert response.status_code == 200
    import json

    body = json.loads(response.body)
    assert body["status"] == "ok"
    assert body["inflight"] == 1


@pytest.mark.asyncio
async def test_health_decode_stalled_when_running_and_old_progress(monkeypatch):
    """inflight > 0 + last progress older than threshold => 503 stalled
    (mid-stream stall rule a)."""
    import vllm.envs as envs

    # Tighten the threshold for the test so we don't need to mock 60+ seconds.
    monkeypatch.setattr(envs, "VLLM_DECODE_LIVENESS_STALL_SECONDS", 1.0)
    from vllm.entrypoints.serve.instrumentator.health import health_decode

    response = await health_decode(
        _make_decode_request(
            inflight=2, last_progress_age=5.0, oldest_unprogressed_age=None
        )
    )
    assert response.status_code == 503
    import json

    body = json.loads(response.body)
    assert body["status"] == "stalled"
    assert body["inflight"] == 2
    assert body["last_progress_age_seconds"] == 5.0
    assert body["stall_threshold_seconds"] == 1.0


@pytest.mark.asyncio
async def test_health_decode_stalled_when_inflight_but_never_progressed(monkeypatch):
    """The decode-step-0 deadlock case (#45094): a request is in flight,
    NO output has ever been produced for it (last_progress_age is None), and
    its admission is older than the threshold => 503 stalled (rule b).

    This is the regression the pre-fix tree gets WRONG: the old
    ``last_token_age is None`` branch returned 200 "idle" unconditionally,
    masking the deadlock. With the API-process-local admission heartbeat we
    correctly report 503 stalled."""
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_DECODE_LIVENESS_STALL_SECONDS", 60.0)
    from vllm.entrypoints.serve.instrumentator.health import health_decode

    response = await health_decode(
        _make_decode_request(
            inflight=1, last_progress_age=None, oldest_unprogressed_age=120.0
        )
    )
    assert response.status_code == 503
    import json

    body = json.loads(response.body)
    assert body["status"] == "stalled"
    assert body["inflight"] == 1
    assert body["last_progress_age_seconds"] is None
    assert body["oldest_unprogressed_admission_age_seconds"] == 120.0
    assert body["stall_threshold_seconds"] == 60.0


@pytest.mark.asyncio
async def test_health_decode_stalled_when_oldest_unprogressed_over_threshold_with_recent_other_progress(  # noqa: E501
    monkeypatch,
):
    """A subtle real case: many requests are flowing (last_progress_age is
    recent because OTHER requests keep producing tokens) but ONE admitted
    request has never produced a single output and has been waiting past the
    threshold. Rule (b) must still fire — last_progress_age alone would miss
    it."""
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_DECODE_LIVENESS_STALL_SECONDS", 60.0)
    from vllm.entrypoints.serve.instrumentator.health import health_decode

    response = await health_decode(
        _make_decode_request(
            inflight=4, last_progress_age=0.2, oldest_unprogressed_age=90.0
        )
    )
    assert response.status_code == 503
    import json

    body = json.loads(response.body)
    assert body["status"] == "stalled"


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


def test_decode_liveness_tracker_admission_advances_without_engine_output():
    """The admission heartbeat advances in the API process WITHOUT any
    record()/output delivery — proving the write is OUTSIDE the
    EngineCore-gated output_handler path. This is the core property that
    makes the step-0 deadlock detectable."""
    import time

    from vllm.v1.engine.async_llm import DecodeLivenessTracker

    tracker = DecodeLivenessTracker()

    # Cold: nothing admitted, nothing ever progressed.
    inflight, last_progress_age, oldest_unprogressed_age = tracker.snapshot()
    assert inflight == 0
    assert last_progress_age is None
    assert oldest_unprogressed_age is None

    # Admit a request. NO output is delivered (no record_progress call).
    tracker.record_admission("req-1")
    time.sleep(0.01)
    inflight, last_progress_age, oldest_unprogressed_age = tracker.snapshot()
    assert inflight == 1
    # last_progress_age stays None: the engine has produced nothing.
    assert last_progress_age is None
    # The admission age advances purely from wall-clock, with zero engine
    # cooperation.
    assert oldest_unprogressed_age is not None
    assert oldest_unprogressed_age > 0.0


def test_decode_liveness_tracker_inflight_decrements_on_finish():
    """record_finished decrements inflight and is idempotent (an abort racing
    the terminal output must not double-decrement)."""
    from vllm.v1.engine.async_llm import DecodeLivenessTracker

    tracker = DecodeLivenessTracker()
    tracker.record_admission("a")
    tracker.record_admission("b")
    assert tracker.snapshot()[0] == 2

    tracker.record_finished("a")
    assert tracker.snapshot()[0] == 1

    # Idempotent: finishing "a" again does nothing.
    tracker.record_finished("a")
    assert tracker.snapshot()[0] == 1

    tracker.record_finished("b")
    assert tracker.snapshot()[0] == 0

    # Never goes negative.
    tracker.record_finished("ghost")
    assert tracker.snapshot()[0] == 0


def test_decode_liveness_tracker_progress_resets_on_any_output():
    """record_progress() advances last_progress_age regardless of token count
    (a prefill-only chunk with num_generation_tokens == 0 still counts), and
    mark_request_progressed clears a request from the un-progressed set so
    rule (b) no longer fires for it."""
    import time

    from vllm.v1.engine.async_llm import DecodeLivenessTracker

    tracker = DecodeLivenessTracker()
    tracker.record_admission("req-1")
    time.sleep(0.01)

    # Before any output: un-progressed admission age is set, no progress yet.
    _, last_progress_age, oldest_unprogressed_age = tracker.snapshot()
    assert last_progress_age is None
    assert oldest_unprogressed_age is not None

    # A prefill-only output arrives (the caller does NOT gate this on
    # generation-token count — record_progress is called for ANY output).
    tracker.record_progress()
    tracker.mark_request_progressed("req-1")

    inflight, last_progress_age, oldest_unprogressed_age = tracker.snapshot()
    assert inflight == 1  # still in flight
    # Progress is now recent.
    assert last_progress_age is not None
    assert last_progress_age >= 0.0
    assert last_progress_age < 1.0
    # The request has progressed, so it is no longer an un-progressed admission
    # — rule (b) will not fire for it.
    assert oldest_unprogressed_age is None


def test_decode_liveness_tracker_oldest_unprogressed_is_oldest():
    """oldest_unprogressed_admission_age reflects the OLDEST still-un-progressed
    in-flight request, and ignores requests that have already progressed."""
    import time

    from vllm.v1.engine.async_llm import DecodeLivenessTracker

    tracker = DecodeLivenessTracker()
    tracker.record_admission("old")
    time.sleep(0.02)
    tracker.record_admission("new")

    # Both un-progressed: oldest is "old".
    _, _, oldest_old = tracker.snapshot()
    assert oldest_old is not None

    # "old" progresses; only "new" remains un-progressed, so the reported age
    # drops to "new"'s (younger) admission age.
    tracker.mark_request_progressed("old")
    _, _, oldest_new = tracker.snapshot()
    assert oldest_new is not None
    assert oldest_new < oldest_old

