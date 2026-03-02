# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for stateless multi-turn Responses API (RFC #26934 + @grs).

Covers:
- Protocol-level validation (pydantic model constraints on ResponsesRequest)
- State carrier injection helpers (_build_state_carrier / _extract_state_from_response)
- Error paths when enable_store=False (retrieve, cancel, previous_response_id)
- utils.py skipping of state-carrier items
- construct_input_messages with prev_messages
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

    def test_no_previous_response_preserves_store_true(self):
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

        req = ResponsesRequest(model="test", input="hello", store=True)
        assert req.store is True


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
    async def test_cancel_without_store_returns_501(self):
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

        serving = _make_minimal_serving(enable_store=False)
        result = await OpenAIServingResponses.cancel_responses(serving, "resp_123")

        assert isinstance(result, ErrorResponse)
        assert result.error.code == HTTPStatus.NOT_IMPLEMENTED

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
    def test_prev_messages_used_over_empty_msg_store(self):
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

    def test_full_stateless_roundtrip(self):
        """serialize → embed in carrier → extract → same messages."""
        from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses

        serving = _make_minimal_serving()
        original = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "My name is Alice"},
            {"role": "assistant", "content": "Hello, Alice!"},
        ]
        carrier = OpenAIServingResponses._build_state_carrier(serving, original, "req_abc123456789")

        mock_response = MagicMock()
        mock_response.output = [carrier]

        recovered = OpenAIServingResponses._extract_state_from_response(serving, mock_response)
        assert recovered == original
