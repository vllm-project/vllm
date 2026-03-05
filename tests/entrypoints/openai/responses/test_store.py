# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the pluggable ResponseStore abstraction.

Tests focus on the business-logic methods (unless_status guard,
update_response_status transitions) and the factory's env-var loading.
Integration tests verify that serving.py actually uses the store correctly
through its public methods.
"""

from http import HTTPStatus

import pytest

from vllm.entrypoints.openai.responses.store import (
    InMemoryResponseStore,
    ResponseStore,
    create_response_store,
)


def _make_response(response_id: str = "resp_1", status: str = "completed"):
    """Create a minimal ResponsesResponse via model_construct."""
    from vllm.entrypoints.openai.responses.protocol import ResponsesResponse

    return ResponsesResponse.model_construct(id=response_id, status=status)


# ---------------------------------------------------------------------------
# put_response unless_status guard — the cancelled-response protection
# ---------------------------------------------------------------------------


class TestPutResponseUnlessStatus:
    """Tests for the unless_status guard on put_response.

    In production this prevents a completed response from overwriting
    a cancelled one (serving.py finalize path).
    """

    @pytest.mark.asyncio
    async def test_skips_write_when_existing_status_matches(self):
        store = InMemoryResponseStore()
        await store.put_response("r1", _make_response("r1", "cancelled"))

        new = _make_response("r1", "completed")
        ok = await store.put_response("r1", new, unless_status="cancelled")

        assert ok is False
        assert (await store.get_response("r1")).status == "cancelled"

    @pytest.mark.asyncio
    async def test_allows_write_when_existing_status_differs(self):
        store = InMemoryResponseStore()
        await store.put_response("r1", _make_response("r1", "queued"))

        new = _make_response("r1", "completed")
        ok = await store.put_response("r1", new, unless_status="cancelled")

        assert ok is True
        assert (await store.get_response("r1")).status == "completed"

    @pytest.mark.asyncio
    async def test_allows_write_when_nothing_stored_yet(self):
        store = InMemoryResponseStore()
        new = _make_response("r1", "completed")
        ok = await store.put_response("r1", new, unless_status="cancelled")

        assert ok is True
        assert await store.get_response("r1") is new


# ---------------------------------------------------------------------------
# update_response_status — atomic status transitions
# ---------------------------------------------------------------------------


class TestUpdateResponseStatus:
    """Tests for the atomic status transition method.

    In production this is used by:
    - _run_background_request: queued/in_progress -> failed
    - cancel_responses: queued/in_progress -> cancelled
    """

    @pytest.mark.asyncio
    async def test_transitions_when_current_status_is_allowed(self):
        """The actual production path: status IS in allowed set."""
        store = InMemoryResponseStore()
        await store.put_response("r1", _make_response("r1", "queued"))

        result = await store.update_response_status(
            "r1", "failed", allowed_current_statuses={"queued", "in_progress"}
        )

        assert result is not None
        assert result.status == "failed"
        # Verify the stored object was mutated (not a copy)
        assert (await store.get_response("r1")).status == "failed"

    @pytest.mark.asyncio
    async def test_rejects_when_current_status_not_in_allowed_set(self):
        """e.g. trying to fail a completed response — should be a no-op."""
        store = InMemoryResponseStore()
        await store.put_response("r1", _make_response("r1", "completed"))

        result = await store.update_response_status(
            "r1", "failed", allowed_current_statuses={"queued", "in_progress"}
        )

        assert result is None
        assert (await store.get_response("r1")).status == "completed"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        store = InMemoryResponseStore()
        result = await store.update_response_status("ghost", "failed")
        assert result is None

    @pytest.mark.asyncio
    async def test_unconditional_update_when_no_allowed_set(self):
        """Without allowed_current_statuses, any transition succeeds."""
        store = InMemoryResponseStore()
        await store.put_response("r1", _make_response("r1", "completed"))

        result = await store.update_response_status("r1", "cancelled")

        assert result is not None
        assert result.status == "cancelled"


# ---------------------------------------------------------------------------
# Factory — env var loading
# ---------------------------------------------------------------------------


class TestFactory:
    def test_default_returns_in_memory(self, monkeypatch):
        monkeypatch.delenv("VLLM_RESPONSES_STORE_BACKEND", raising=False)
        import vllm.envs as envs_mod

        envs_mod._ENVS_INITIALIZED = False
        store = create_response_store()
        assert isinstance(store, InMemoryResponseStore)

    def test_custom_backend_loaded_via_qualname(self, monkeypatch):
        monkeypatch.setenv(
            "VLLM_RESPONSES_STORE_BACKEND",
            "vllm.entrypoints.openai.responses.store.InMemoryResponseStore",
        )
        import vllm.envs as envs_mod

        envs_mod._ENVS_INITIALIZED = False
        store = create_response_store()
        assert isinstance(store, InMemoryResponseStore)

    def test_non_subclass_raises_type_error(self, monkeypatch):
        monkeypatch.setenv("VLLM_RESPONSES_STORE_BACKEND", "builtins.int")
        import vllm.envs as envs_mod

        envs_mod._ENVS_INITIALIZED = False
        with pytest.raises(TypeError, match="not a ResponseStore subclass"):
            create_response_store()

    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            ResponseStore()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Integration: serving.py cancel_responses uses the store correctly
# ---------------------------------------------------------------------------


def _make_minimal_serving(enable_store: bool = True):
    """Minimal OpenAIServingResponses with a real InMemoryResponseStore."""
    from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

    obj = OpenAIServingResponses.__new__(OpenAIServingResponses)
    obj.enable_store = enable_store
    obj.store = InMemoryResponseStore()
    obj.background_tasks = {}
    obj.log_error_stack = False
    return obj


class TestServingCancelIntegration:
    """Verify cancel_responses interacts with the store correctly.

    These tests exercise the refactored cancel_responses code that replaced
    the old get-check-mutate-under-lock pattern with
    store.update_response_status.
    """

    @pytest.mark.asyncio
    async def test_cancel_queued_response_succeeds(self):
        from vllm.entrypoints.openai.responses.protocol import ResponsesResponse

        serving = _make_minimal_serving()
        resp = _make_response("resp_bg", status="queued")
        await serving.store.put_response("resp_bg", resp)

        result = await serving.cancel_responses("resp_bg")

        assert isinstance(result, ResponsesResponse)
        assert result.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_completed_response_returns_error(self):
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse

        serving = _make_minimal_serving()
        resp = _make_response("resp_done", status="completed")
        await serving.store.put_response("resp_done", resp)

        result = await serving.cancel_responses("resp_done")

        assert isinstance(result, ErrorResponse)
        assert "Cannot cancel" in result.error.message

    @pytest.mark.asyncio
    async def test_cancel_unknown_id_returns_404(self):
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse

        serving = _make_minimal_serving()
        result = await serving.cancel_responses("resp_ghost")

        assert isinstance(result, ErrorResponse)
        assert result.error.code == HTTPStatus.NOT_FOUND


class TestServingRetrieveIntegration:
    """Verify retrieve_responses reads from the store correctly."""

    @pytest.mark.asyncio
    async def test_retrieve_stored_response(self):
        from vllm.entrypoints.openai.responses.protocol import ResponsesResponse

        serving = _make_minimal_serving()
        resp = _make_response("resp_1", status="completed")
        await serving.store.put_response("resp_1", resp)

        result = await serving.retrieve_responses("resp_1", None, False)

        assert isinstance(result, ResponsesResponse)
        assert result is resp

    @pytest.mark.asyncio
    async def test_retrieve_unknown_id_returns_404(self):
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse

        serving = _make_minimal_serving()
        result = await serving.retrieve_responses("resp_ghost", None, False)

        assert isinstance(result, ErrorResponse)
        assert result.error.code == HTTPStatus.NOT_FOUND
