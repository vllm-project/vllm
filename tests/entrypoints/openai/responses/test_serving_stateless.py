# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for stateless multi-turn Responses API (RFC #26934 + @grs).

Covers:
- Protocol-level validation (pydantic model constraints on ResponsesRequest)
- State carrier injection helpers (_build_state_carrier / _extract_state_from_response)
- Error paths when enable_store=False (retrieve, cancel, previous_response_id)
- Error path when previous_response lacks a state carrier (no include= on prior turn)
- background=True rejected when previous_response is set
- utils.py skipping of state-carrier items
- construct_input_messages contract (prev_msg prepended before new turn)
"""

from http import HTTPStatus
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Signing key fixture — deterministic per test run
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def set_signing_key(monkeypatch):
    monkeypatch.setenv("VLLM_RESPONSES_STATE_SIGNING_KEY", "cc" * 32)
    import vllm.entrypoints.openai.responses.state as state_mod

    state_mod._SIGNING_KEY = None
    yield
    state_mod._SIGNING_KEY = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_serving(enable_store: bool = False):
    """Return an OpenAIServingResponses instance with only the attributes
    exercised by the methods under test.  We bypass __init__ via __new__ and
    set exactly what is needed.

    Attributes touched by retrieve_responses / cancel_responses:
      enable_store, response_store, response_store_lock, background_tasks,
      log_error_stack (for create_error_response from base class)
    Attributes touched by create_responses (up to store check):
      models (for _check_model), engine_client (for errored check)
    """
    import asyncio

    from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

    obj = OpenAIServingResponses.__new__(OpenAIServingResponses)
    obj.enable_store = enable_store
    obj.response_store = {}
    obj.response_store_lock = asyncio.Lock()
    obj.background_tasks = {}
    obj.log_error_stack = False
    # _check_model needs models.is_base_model to return True (model found)
    obj.models = MagicMock()
    obj.models.is_base_model.return_value = True
    # _validate_create_responses_input checks this before hitting the store guard
    obj.use_harmony = False
    return obj


# ---------------------------------------------------------------------------
# Protocol validation — pydantic model constraints
# ---------------------------------------------------------------------------


class TestProtocolValidation:
    def test_previous_response_and_id_mutually_exclusive(self):
        from vllm.entrypoints.openai.responses.protocol import (
            ResponsesRequest,
            ResponsesResponse,
        )

        # model_construct bypasses field validation so we get a typed instance
        # without needing all required fields — Pydantic accepts it for type checks.
        fake_prev = ResponsesResponse.model_construct(id="resp_fake")
        with pytest.raises(Exception, match="Cannot set both"):
            ResponsesRequest(
                model="test",
                input="hello",
                previous_response_id="resp_abc",
                previous_response=fake_prev,
            )

    def test_previous_response_silently_clears_store(self):
        from vllm.entrypoints.openai.responses.protocol import (
            ResponsesRequest,
            ResponsesResponse,
        )

        fake_prev = ResponsesResponse.model_construct(id="resp_fake")
        req = ResponsesRequest(
            model="test",
            input="hello",
            store=True,
            previous_response=fake_prev,
        )
        assert req.store is False

    def test_background_with_previous_response_raises(self):
        """background=True + previous_response must be rejected.

        The mode='before' validator allows background=True when store=True
        (the default).  Our mode='after' validator must then catch that
        previous_response was also set and raise — otherwise the request
        would produce an unretrievable background response (store gets
        silently cleared to False, but background remains True).
        """
        from vllm.entrypoints.openai.responses.protocol import (
            ResponsesRequest,
            ResponsesResponse,
        )

        fake_prev = ResponsesResponse.model_construct(id="resp_fake")
        with pytest.raises(Exception, match="background"):
            ResponsesRequest(
                model="test",
                input="hello",
                background=True,
                previous_response=fake_prev,
            )


# ---------------------------------------------------------------------------
# State carrier round-trip via serving helpers
# (thin wrappers over state.py — verified here to confirm the integration)
# ---------------------------------------------------------------------------


class TestStateCarrierHelpers:
    def test_build_and_extract_roundtrip(self):
        from openai.types.responses.response_reasoning_item import ResponseReasoningItem
        from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

        serving = _make_minimal_serving()
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

        carrier = OpenAIServingResponses._build_state_carrier(serving, messages, "test_req_id_12345")

        assert isinstance(carrier, ResponseReasoningItem)
        assert carrier.type == "reasoning"
        assert carrier.status == "completed"
        assert carrier.encrypted_content.startswith("vllm:1:")

        mock_response = MagicMock()
        mock_response.output = [carrier]

        recovered = OpenAIServingResponses._extract_state_from_response(serving, mock_response)
        assert recovered == messages

    def test_extract_returns_none_when_no_carrier(self):
        from openai.types.responses import ResponseOutputMessage, ResponseOutputText
        from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

        serving = _make_minimal_serving()

        text = ResponseOutputText(type="output_text", text="hi", annotations=[])
        msg = ResponseOutputMessage(
            id="msg_1", type="message", role="assistant",
            content=[text], status="completed",
        )
        mock_response = MagicMock()
        mock_response.output = [msg]

        assert OpenAIServingResponses._extract_state_from_response(serving, mock_response) is None

    def test_extract_raises_on_tampered_carrier(self):
        from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

        serving = _make_minimal_serving()
        carrier = OpenAIServingResponses._build_state_carrier(
            serving, [{"role": "user", "content": "hi"}], "req"
        )

        parts = carrier.encrypted_content.split(":", 3)
        parts[3] = "0" * 64
        carrier.encrypted_content = ":".join(parts)

        mock_response = MagicMock()
        mock_response.output = [carrier]

        with pytest.raises(ValueError, match="HMAC verification failed"):
            OpenAIServingResponses._extract_state_from_response(serving, mock_response)


# ---------------------------------------------------------------------------
# Error paths when enable_store=False
# ---------------------------------------------------------------------------


class TestStorelessErrorPaths:
    @pytest.mark.asyncio
    async def test_retrieve_without_store_returns_501(self):
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

        serving = _make_minimal_serving(enable_store=False)
        result = await OpenAIServingResponses.retrieve_responses(
            serving, "resp_123", None, False
        )

        assert isinstance(result, ErrorResponse)
        assert result.error.code == HTTPStatus.NOT_IMPLEMENTED
        assert "VLLM_ENABLE_RESPONSES_API_STORE" in result.error.message

    @pytest.mark.asyncio
    async def test_cancel_without_store_unknown_id_returns_404(self):
        """With store off and no matching background task, cancel returns 404."""
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

        serving = _make_minimal_serving(enable_store=False)
        result = await OpenAIServingResponses.cancel_responses(serving, "resp_123")

        assert isinstance(result, ErrorResponse)
        assert result.error.code == HTTPStatus.NOT_FOUND

    @pytest.mark.asyncio
    async def test_cancel_without_store_active_task_returns_400(self):
        """With store off but a matching in-flight task, cancel cancels it and
        returns 400 (not 404, not 501) — the task was cancelled but no stored
        response object can be returned in stateless mode.

        This is the success branch of the stateless cancel path and was the
        actual behavior added by the review fix.
        """
        import asyncio

        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

        serving = _make_minimal_serving(enable_store=False)

        # Simulate an in-flight background task that blocks until cancelled.
        async def _blocking():
            await asyncio.sleep(9999)

        task = asyncio.ensure_future(_blocking())
        # Yield once so the task starts and reaches its first await point;
        # otherwise cancel() on an unstarted task may mark it cancelled before
        # the coroutine body ever runs, but task.cancelled() is still True.
        await asyncio.sleep(0)

        serving.background_tasks["resp_inflight"] = task

        result = await OpenAIServingResponses.cancel_responses(serving, "resp_inflight")

        assert isinstance(result, ErrorResponse)
        assert result.error.code == HTTPStatus.BAD_REQUEST
        assert "stateless mode" in result.error.message
        assert task.cancelled(), "task was not cancelled"

    @pytest.mark.asyncio
    async def test_previous_response_without_carrier_returns_400(self):
        """previous_response with no state carrier + store disabled → 400.

        Regression test for P1 review finding: if a client passes
        previous_response but the prior turn was generated without
        include=['reasoning.encrypted_content'], _extract_state_from_response
        returns None.  With store disabled there is no msg_store fallback, so
        we must return a clear 400 rather than letting msg_store[id] raise
        KeyError → 500.
        """
        from openai.types.responses import ResponseOutputMessage, ResponseOutputText

        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.openai.responses.protocol import (
            ResponsesRequest,
            ResponsesResponse,
        )
        from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

        # Build a previous response whose output has NO state carrier item.
        text = ResponseOutputText(type="output_text", text="Hello!", annotations=[])
        msg = ResponseOutputMessage(
            id="msg_1", type="message", role="assistant",
            content=[text], status="completed",
        )
        prev_resp = ResponsesResponse.model_construct(
            id="resp_old", output=[msg]
        )

        serving = _make_minimal_serving(enable_store=False)
        serving.engine_client = MagicMock()
        serving.engine_client.errored = False

        req = ResponsesRequest(
            model="test",
            input="What is my name?",
            store=False,
            previous_response=prev_resp,
        )

        result = await OpenAIServingResponses.create_responses(serving, req)

        assert isinstance(result, ErrorResponse)
        assert result.error.code == HTTPStatus.BAD_REQUEST
        assert "state carrier" in result.error.message

    @pytest.mark.asyncio
    async def test_previous_response_id_without_store_returns_400(self):
        """create_responses should reject previous_response_id when store is off."""
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
        from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

        serving = _make_minimal_serving(enable_store=False)
        # Minimal extra attributes that create_responses reads before the store check
        serving.engine_client = MagicMock()
        serving.engine_client.errored = False

        req = ResponsesRequest(
            model="test",
            input="hello",
            previous_response_id="resp_old",
            store=False,
        )

        result = await OpenAIServingResponses.create_responses(serving, req)

        assert isinstance(result, ErrorResponse)
        assert result.error.code == HTTPStatus.BAD_REQUEST
        assert "VLLM_ENABLE_RESPONSES_API_STORE" in result.error.message


# ---------------------------------------------------------------------------
# utils.py — state-carrier items are transparently skipped
# ---------------------------------------------------------------------------


class TestUtilsStateCarrierSkipping:
    def test_construct_chat_messages_skips_state_carrier(self):
        from openai.types.responses import ResponseOutputMessage, ResponseOutputText
        from openai.types.responses.response_reasoning_item import ResponseReasoningItem
        from vllm.entrypoints.openai.responses.state import serialize_state
        from vllm.entrypoints.openai.responses.utils import construct_chat_messages_with_tool_call

        blob = serialize_state([{"role": "user", "content": "prev"}])
        carrier = ResponseReasoningItem(
            id="rs_state_abc", type="reasoning", summary=[],
            status="completed", encrypted_content=blob,
        )
        text = ResponseOutputText(type="output_text", text="Hello!", annotations=[])
        msg = ResponseOutputMessage(
            id="msg_1", type="message", role="assistant",
            content=[text], status="completed",
        )

        result = construct_chat_messages_with_tool_call([carrier, msg])

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hello!"

    def test_single_message_from_state_carrier_is_none(self):
        from openai.types.responses.response_reasoning_item import ResponseReasoningItem
        from vllm.entrypoints.openai.responses.state import serialize_state
        from vllm.entrypoints.openai.responses.utils import _construct_single_message_from_response_item

        blob = serialize_state([{"role": "user", "content": "hi"}])
        carrier = ResponseReasoningItem(
            id="rs_state_xyz", type="reasoning", summary=[],
            status="completed", encrypted_content=blob,
        )
        assert _construct_single_message_from_response_item(carrier) is None

    def test_external_encrypted_content_still_raises(self):
        from openai.types.responses.response_reasoning_item import ResponseReasoningItem
        from vllm.entrypoints.openai.responses.utils import _construct_single_message_from_response_item

        external = ResponseReasoningItem(
            id="rs_ext", type="reasoning", summary=[],
            status="completed",
            encrypted_content="opaque-blob-not-from-vllm",
        )
        with pytest.raises(ValueError, match="not supported"):
            _construct_single_message_from_response_item(external)


# ---------------------------------------------------------------------------
# construct_input_messages with prev_messages override
# ---------------------------------------------------------------------------


class TestPrevMessagesOverride:
    def test_construct_input_messages_prepends_prev_msg(self):
        """construct_input_messages correctly prepends a deserialized history
        list (prev_msg) before the new user turn.

        This is the contract the stateless path relies on: after extracting
        message history from the encrypted_content carrier, the history is
        passed as prev_msg and must appear in order before the new input.
        """
        from vllm.entrypoints.openai.responses.utils import construct_input_messages

        prev = [
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am helpful."},
        ]
        result = construct_input_messages(
            request_instructions=None,
            request_input="What did you say?",
            prev_msg=prev,
            prev_response_output=None,
        )
        assert result[0] == {"role": "user", "content": "Who are you?"}
        assert result[1] == {"role": "assistant", "content": "I am helpful."}
        assert result[2] == {"role": "user", "content": "What did you say?"}
