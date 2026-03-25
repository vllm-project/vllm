# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for Responses API store backends.

Tests are organized in two groups:

1. **Backend tests** — verify get/put/overwrite/missing-key for each backend.
2. **Scenario tests** — simulate the enable_store / request.store / previous_response_id
   interaction matrix from serving.py to ensure correct behavior in every combination.
"""

import asyncio
import json

import pytest
import pytest_asyncio

from vllm.entrypoints.openai.responses.protocol import ResponsesResponse
from vllm.entrypoints.openai.responses.store.base import ResponsesStore
from vllm.entrypoints.openai.responses.store.file import FileResponsesStore
from vllm.entrypoints.openai.responses.store.memory import (
    InMemoryResponsesStore,
)


def _make_response(response_id: str, status: str = "completed") -> ResponsesResponse:
    """Create a minimal ResponsesResponse for testing."""
    return ResponsesResponse(
        id=response_id,
        created_at=1700000000,
        model="test-model",
        object="response",
        output=[],
        status=status,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        temperature=1.0,
        top_p=1.0,
        background=False,
        max_output_tokens=1024,
        service_tier="auto",
        truncation="disabled",
    )


def _make_messages(text: str = "hello") -> list[dict]:
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": text},
    ]


# ---------------------------------------------------------------------------
# Fixtures — one per backend
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def memory_store():
    return InMemoryResponsesStore()


@pytest_asyncio.fixture
async def file_store(tmp_path):
    return FileResponsesStore(str(tmp_path / "store"))


# ---------------------------------------------------------------------------
# Parametrized tests that run against every backend
# ---------------------------------------------------------------------------


@pytest.fixture(params=["memory", "file"])
def store(request, tmp_path) -> ResponsesStore:
    if request.param == "memory":
        return InMemoryResponsesStore()
    if request.param == "file":
        return FileResponsesStore(str(tmp_path / "store"))
    raise ValueError(request.param)


@pytest.mark.asyncio
async def test_get_missing_response(store: ResponsesStore):
    result = await store.get_response("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_get_missing_messages(store: ResponsesStore):
    result = await store.get_messages("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_put_and_get_response(store: ResponsesStore):
    resp = _make_response("resp_001")
    await store.put_response("resp_001", resp)

    retrieved = await store.get_response("resp_001")
    assert retrieved is not None
    assert retrieved.id == "resp_001"
    assert retrieved.status == "completed"


@pytest.mark.asyncio
async def test_put_and_get_messages(store: ResponsesStore):
    msgs = _make_messages("hi there")
    await store.put_messages("resp_002", msgs)

    retrieved = await store.get_messages("resp_002")
    assert retrieved is not None
    assert len(retrieved) == 2
    assert retrieved[1]["content"] == "hi there"


@pytest.mark.asyncio
async def test_overwrite_response(store: ResponsesStore):
    resp1 = _make_response("resp_003", status="in_progress")
    await store.put_response("resp_003", resp1)

    resp2 = _make_response("resp_003", status="completed")
    await store.put_response("resp_003", resp2)

    retrieved = await store.get_response("resp_003")
    assert retrieved is not None
    assert retrieved.status == "completed"


@pytest.mark.asyncio
async def test_overwrite_messages(store: ResponsesStore):
    await store.put_messages("resp_004", _make_messages("first"))
    await store.put_messages("resp_004", _make_messages("second"))

    retrieved = await store.get_messages("resp_004")
    assert retrieved is not None
    assert retrieved[1]["content"] == "second"


@pytest.mark.asyncio
async def test_multiple_keys_independent(store: ResponsesStore):
    await store.put_response("a", _make_response("a"))
    await store.put_response("b", _make_response("b"))

    assert (await store.get_response("a")).id == "a"
    assert (await store.get_response("b")).id == "b"
    assert await store.get_response("c") is None


# ---------------------------------------------------------------------------
# File-backend-specific tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_persistence_across_instances(tmp_path):
    """Verify that a second FileResponsesStore instance can read data
    written by the first."""
    path = str(tmp_path / "shared")

    store1 = FileResponsesStore(path)
    await store1.put_response("resp_x", _make_response("resp_x"))
    await store1.put_messages("resp_x", _make_messages("persisted"))

    # New instance, same directory
    store2 = FileResponsesStore(path)
    resp = await store2.get_response("resp_x")
    assert resp is not None
    assert resp.id == "resp_x"

    msgs = await store2.get_messages("resp_x")
    assert msgs is not None
    assert msgs[1]["content"] == "persisted"


@pytest.mark.asyncio
async def test_file_directory_structure(tmp_path):
    """Verify the expected directory layout is created."""
    path = tmp_path / "layout_test"
    store = FileResponsesStore(str(path))

    await store.put_response("r1", _make_response("r1"))
    await store.put_messages("r1", _make_messages())

    assert (path / "responses" / "r1.json").is_file()
    assert (path / "messages" / "r1.json").is_file()

    # Verify the JSON is valid
    data = json.loads((path / "responses" / "r1.json").read_text())
    assert data["id"] == "r1"


# ---------------------------------------------------------------------------
# Scenario tests — simulate serving.py store interaction logic
#
# These replicate the exact gating logic from OpenAIServingResponses:
#   - Line 349-356: if request.store and not enable_store → coerce to False
#   - Line 496-497: if request.store → put_messages
#   - Line 883-888: if request.store → put_response
#   - Line 360-364: previous_response_id → get_response (always reads)
# ---------------------------------------------------------------------------


def _simulate_store_coercion(enable_store: bool, request_store: bool) -> bool:
    """Replicate the request.store coercion logic from serving.py L349-356."""
    if request_store and not enable_store:
        return False
    return request_store


async def _simulate_create(
    store: ResponsesStore,
    store_lock: asyncio.Lock,
    enable_store: bool,
    request_store: bool,
    request_id: str,
    previous_response_id: str | None = None,
) -> tuple[str, ResponsesResponse | None, str | None]:
    """Simulate the create_responses flow and return
    (outcome, response_or_none, error_or_none).

    Outcomes: "ok", "prev_not_found", "completed"
    """
    # Coerce request.store (serving.py L349-356)
    effective_store = _simulate_store_coercion(enable_store, request_store)

    # Lookup previous response (serving.py L360-364)
    prev_response = None
    if previous_response_id is not None:
        async with store_lock:
            prev_response = await store.get_response(previous_response_id)
        if prev_response is None:
            return "prev_not_found", None, f"{previous_response_id} not found"

    # Store messages (serving.py L496-497)
    messages = _make_messages(f"turn for {request_id}")
    if effective_store:
        await store.put_messages(request_id, messages)

    # Build and store response (serving.py L883-888)
    response = _make_response(request_id, status="completed")
    if effective_store:
        async with store_lock:
            await store.put_response(response.id, response)

    return "completed", response, None


async def _simulate_retrieve(
    store: ResponsesStore,
    store_lock: asyncio.Lock,
    response_id: str,
) -> ResponsesResponse | None:
    """Simulate retrieve_responses (serving.py L1285-1296)."""
    async with store_lock:
        return await store.get_response(response_id)


@pytest.mark.asyncio
async def test_scenario_store_disabled_default_request():
    """enable_store=False, request.store=True (default).
    Store coerces to False. Nothing persisted. Retrieve returns None."""
    store = InMemoryResponsesStore()
    lock = asyncio.Lock()

    outcome, resp, err = await _simulate_create(
        store, lock, enable_store=False, request_store=True, request_id="r1"
    )
    assert outcome == "completed"

    # Nothing was stored
    assert await store.get_response("r1") is None
    assert await store.get_messages("r1") is None

    # Retrieve returns None (would be 404 in serving)
    assert await _simulate_retrieve(store, lock, "r1") is None


@pytest.mark.asyncio
async def test_scenario_store_disabled_prev_response_id_fails():
    """enable_store=False. Request A completes (not stored).
    Request B references A via previous_response_id → not found."""
    store = InMemoryResponsesStore()
    lock = asyncio.Lock()

    # Request A: store disabled, default store=True → coerced to False
    await _simulate_create(
        store, lock, enable_store=False, request_store=True, request_id="a"
    )

    # Request B: references A
    outcome, _, err = await _simulate_create(
        store,
        lock,
        enable_store=False,
        request_store=True,
        request_id="b",
        previous_response_id="a",
    )
    assert outcome == "prev_not_found"
    assert err is not None and "a" in err


@pytest.mark.asyncio
async def test_scenario_store_enabled_multi_turn():
    """enable_store=True, request.store=True (default).
    Request A stored. Request B references A → works.
    Request C references B → works."""
    store = InMemoryResponsesStore()
    lock = asyncio.Lock()

    # Turn 1
    outcome, _, _ = await _simulate_create(
        store, lock, enable_store=True, request_store=True, request_id="t1"
    )
    assert outcome == "completed"
    assert await store.get_response("t1") is not None
    assert await store.get_messages("t1") is not None

    # Turn 2 references Turn 1
    outcome, _, _ = await _simulate_create(
        store,
        lock,
        enable_store=True,
        request_store=True,
        request_id="t2",
        previous_response_id="t1",
    )
    assert outcome == "completed"
    assert await store.get_response("t2") is not None

    # Turn 3 references Turn 2
    outcome, _, _ = await _simulate_create(
        store,
        lock,
        enable_store=True,
        request_store=True,
        request_id="t3",
        previous_response_id="t2",
    )
    assert outcome == "completed"


@pytest.mark.asyncio
async def test_scenario_store_enabled_client_opts_out():
    """enable_store=True, but client sends store=False.
    Response not persisted. Retrieve returns None."""
    store = InMemoryResponsesStore()
    lock = asyncio.Lock()

    outcome, _, _ = await _simulate_create(
        store, lock, enable_store=True, request_store=False, request_id="r1"
    )
    assert outcome == "completed"

    # Client opted out — nothing stored
    assert await store.get_response("r1") is None
    assert await store.get_messages("r1") is None


@pytest.mark.asyncio
async def test_scenario_store_enabled_mixed_store_flags():
    """enable_store=True. Request A: store=True (stored).
    Request B: store=False, references A → works (A exists).
    Request C: references B → fails (B was not stored)."""
    store = InMemoryResponsesStore()
    lock = asyncio.Lock()

    # A: stored
    await _simulate_create(
        store, lock, enable_store=True, request_store=True, request_id="a"
    )
    assert await store.get_response("a") is not None

    # B: references A, but opts out of storing itself
    outcome, _, _ = await _simulate_create(
        store,
        lock,
        enable_store=True,
        request_store=False,
        request_id="b",
        previous_response_id="a",
    )
    assert outcome == "completed"
    assert await store.get_response("b") is None  # B not stored

    # C: references B → fails because B wasn't stored
    outcome, _, err = await _simulate_create(
        store,
        lock,
        enable_store=True,
        request_store=True,
        request_id="c",
        previous_response_id="b",
    )
    assert outcome == "prev_not_found"


@pytest.mark.asyncio
async def test_scenario_file_backend_multi_turn(tmp_path):
    """Same as multi-turn test but with file backend to verify
    persistence works end-to-end."""
    store = FileResponsesStore(str(tmp_path / "store"))
    lock = asyncio.Lock()

    # Turn 1
    await _simulate_create(
        store, lock, enable_store=True, request_store=True, request_id="t1"
    )

    # Simulate restart — new store instance, same directory
    store2 = FileResponsesStore(str(tmp_path / "store"))
    lock2 = asyncio.Lock()

    # Turn 2 on new instance references Turn 1
    outcome, _, _ = await _simulate_create(
        store2,
        lock2,
        enable_store=True,
        request_store=True,
        request_id="t2",
        previous_response_id="t1",
    )
    assert outcome == "completed"
    assert await store2.get_response("t2") is not None

    # Verify messages from turn 1 survived the "restart"
    msgs = await store2.get_messages("t1")
    assert msgs is not None
    assert msgs[1]["content"] == "turn for t1"


@pytest.mark.asyncio
async def test_scenario_cancel_unstored_response():
    """Cancelling a response that was never stored returns None (404)."""
    store = InMemoryResponsesStore()
    lock = asyncio.Lock()

    # Create with store=False
    await _simulate_create(
        store, lock, enable_store=True, request_store=False, request_id="r1"
    )

    # Try to retrieve (simulates cancel's first step)
    result = await _simulate_retrieve(store, lock, "r1")
    assert result is None


@pytest.mark.asyncio
async def test_scenario_cancel_stored_response():
    """Cancelling a stored response finds it and can mutate status."""
    store = InMemoryResponsesStore()
    lock = asyncio.Lock()

    await _simulate_create(
        store, lock, enable_store=True, request_store=True, request_id="r1"
    )

    # Simulate cancel: read, mutate, re-put
    async with lock:
        response = await store.get_response("r1")
        assert response is not None
        assert response.status == "completed"
        response.status = "cancelled"
        await store.put_response("r1", response)

    # Verify mutation persisted
    result = await store.get_response("r1")
    assert result.status == "cancelled"
