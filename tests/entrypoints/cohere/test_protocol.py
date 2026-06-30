# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``vllm/entrypoints/cohere/protocol.py``.

The module is mostly a thin wrapper around the official ``cohere`` SDK
types plus a few local additions:

* :class:`CohereError` envelope.
* :class:`CohereChatV2Request` (model required, ``max_tokens`` non-negative).
* :class:`CohereChatV2Response` plus the usage / logprob helpers.
* The streaming event subclasses that bake a wire-format ``type``
  discriminator into ``model_dump()`` so SSE consumers can demux on it.
"""

import pytest
from pydantic import ValidationError

from vllm.entrypoints.cohere.protocol import (
    AssistantChatMessageV2,
    AssistantMessageResponse,
    CitationEndEvent,
    CitationStartEvent,
    CohereChatV2Request,
    CohereChatV2Response,
    CohereError,
    CohereLogprobItem,
    CohereUsage,
    CohereUsageBilledUnits,
    CohereUsageTokens,
    ContentDeltaEvent,
    ContentEndEvent,
    ContentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    SystemChatMessageV2,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolPlanDeltaEvent,
    UserChatMessageV2,
)

# ======================================================================
# CohereError
# ======================================================================


class TestCohereError:
    def test_message_only(self):
        err = CohereError(message="boom")
        assert err.message == "boom"
        assert err.id is None

    def test_with_id(self):
        err = CohereError(message="boom", id="req_123")
        assert err.id == "req_123"

    def test_model_dump_excludes_none(self):
        err = CohereError(message="boom")
        assert err.model_dump(exclude_none=True) == {"message": "boom"}


# ======================================================================
# CohereChatV2Request
# ======================================================================


class TestCohereChatV2Request:
    def test_minimal_required_fields(self):
        req = CohereChatV2Request(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
        assert req.model == "m"
        assert req.stream is False
        assert req.max_tokens is None
        assert req.tools is None
        assert req.documents is None
        assert req.kv_transfer_params is None
        assert req.chat_template_kwargs is None

    def test_empty_model_rejected(self):
        with pytest.raises(ValidationError, match="model is required"):
            CohereChatV2Request(model="", messages=[{"role": "user", "content": "hi"}])

    def test_negative_max_tokens_rejected(self):
        with pytest.raises(ValidationError, match="non-negative"):
            CohereChatV2Request(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=-1,
            )

    def test_zero_max_tokens_allowed(self):
        # Zero is allowed (the docs allow 0 -> return prompt only).
        req = CohereChatV2Request(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=0,
        )
        assert req.max_tokens == 0

    def test_full_schema(self):
        """Sanity-check that every field listed in the model accepts a value."""
        req = CohereChatV2Request(
            model="m",
            stream=True,
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "hello",
                    "tool_plan": "p",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "description": "d",
                        "parameters": {},
                    },
                }
            ],
            strict_tools=True,
            tool_choice="REQUIRED",
            documents=["plain", {"id": "d1", "data": {"text": "t"}}],
            citation_options={"mode": "accurate"},
            response_format={"type": "json_object"},
            safety_mode="CONTEXTUAL",
            max_tokens=128,
            stop_sequences=["</s>"],
            temperature=0.5,
            seed=42,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            k=50,
            p=0.95,
            logprobs=True,
            thinking={"type": "enabled", "token_budget": 1024},
            priority=1,
            kv_transfer_params={"x": 1},
            chat_template_kwargs={"y": 2},
        )
        # Discriminated-union messages resolve to the SDK variants.
        assert isinstance(req.messages[0], SystemChatMessageV2)
        assert isinstance(req.messages[1], UserChatMessageV2)
        assert isinstance(req.messages[2], AssistantChatMessageV2)
        assert req.thinking.type == "enabled"
        assert req.thinking.token_budget == 1024
        assert req.citation_options.mode == "accurate"
        assert req.response_format.type == "json_object"
        assert req.safety_mode == "CONTEXTUAL"
        assert req.tool_choice == "REQUIRED"
        assert req.tools[0].function.name == "f"
        # Documents accept str OR Document.
        assert isinstance(req.documents[0], str)
        assert req.documents[1].id == "d1"

    def test_invalid_tool_choice_rejected(self):
        with pytest.raises(ValidationError):
            CohereChatV2Request(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                tool_choice="ANY",  # not REQUIRED/NONE
            )

    def test_unknown_field_ignored(self):
        # Pydantic by default ignores extra fields. Make sure that contract
        # holds so clients sending forward-compatible kwargs don't 422.
        req = CohereChatV2Request(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            some_future_field="x",  # type: ignore[arg-type]
        )
        assert req.model == "m"


# ======================================================================
# Usage / Logprobs
# ======================================================================


class TestUsage:
    def test_billed_units_optional(self):
        u = CohereUsageBilledUnits()
        assert u.input_tokens is None
        assert u.output_tokens is None

    def test_tokens_serialization(self):
        u = CohereUsageTokens(input_tokens=10, output_tokens=5)
        assert u.model_dump() == {"input_tokens": 10.0, "output_tokens": 5.0}

    def test_usage_envelope(self):
        u = CohereUsage(
            billed_units=CohereUsageBilledUnits(input_tokens=10, output_tokens=5),
            tokens=CohereUsageTokens(input_tokens=10, output_tokens=5),
            cached_tokens=3,
        )
        assert u.cached_tokens == 3
        assert u.billed_units.input_tokens == 10

    def test_logprob_item(self):
        lp = CohereLogprobItem(text="hi", token_ids=[1, 2], logprobs=[-0.1, -0.2])
        assert lp.token_ids == [1, 2]
        assert lp.logprobs == [-0.1, -0.2]


# ======================================================================
# CohereChatV2Response
# ======================================================================


class TestCohereChatV2Response:
    def test_minimal(self):
        msg = AssistantMessageResponse(content=[{"type": "text", "text": "hi"}])
        resp = CohereChatV2Response(
            id="r1",
            finish_reason="COMPLETE",
            message=msg,
        )
        assert resp.id == "r1"
        assert resp.usage is None
        assert resp.kv_transfer_params is None

    def test_invalid_finish_reason_rejected(self):
        msg = AssistantMessageResponse(content=[{"type": "text", "text": "hi"}])
        with pytest.raises(ValidationError):
            CohereChatV2Response(
                id="r1",
                finish_reason="NOT_A_REASON",  # type: ignore[arg-type]
                message=msg,
            )


# ======================================================================
# Streaming event ``type`` discriminator baked into model_dump()
# ======================================================================


class TestStreamingEventTypeField:
    """Each event subclass adds a ``type: Literal[...]`` field with a
    default so ``model_dump()`` always emits the wire-format discriminator
    (the parent SDK classes don't declare ``type`` as a Pydantic field).
    """

    @pytest.mark.parametrize(
        "cls, expected_type, kwargs",
        [
            (
                MessageStartEvent,
                "message-start",
                {"id": "a", "delta": {"message": {"role": "assistant"}}},
            ),
            (
                ContentStartEvent,
                "content-start",
                {
                    "index": 0,
                    "delta": {"message": {"content": {"type": "text", "text": ""}}},
                },
            ),
            (
                ContentDeltaEvent,
                "content-delta",
                {
                    "index": 0,
                    "delta": {"message": {"content": {"text": "hi"}}},
                },
            ),
            (ContentEndEvent, "content-end", {"index": 0}),
            (
                ToolPlanDeltaEvent,
                "tool-plan-delta",
                {"delta": {"message": {"tool_plan": "thinking"}}},
            ),
            (
                ToolCallStartEvent,
                "tool-call-start",
                {
                    "index": 0,
                    "delta": {
                        "message": {
                            "tool_calls": {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "f", "arguments": ""},
                            }
                        }
                    },
                },
            ),
            (
                ToolCallDeltaEvent,
                "tool-call-delta",
                {
                    "index": 0,
                    "delta": {
                        "message": {"tool_calls": {"function": {"arguments": "{}"}}}
                    },
                },
            ),
            (ToolCallEndEvent, "tool-call-end", {"index": 0}),
            (
                CitationStartEvent,
                "citation-start",
                {
                    "index": 0,
                    "delta": {
                        "message": {
                            "citations": {"start": 0, "end": 5, "text": "hello"}
                        }
                    },
                },
            ),
            (CitationEndEvent, "citation-end", {"index": 0}),
            (
                MessageEndEvent,
                "message-end",
                {"id": "a", "delta": {"finish_reason": "COMPLETE"}},
            ),
        ],
    )
    def test_type_field_default(self, cls, expected_type, kwargs):
        ev = cls(**kwargs)
        # type field is auto-populated from the Literal default.
        assert ev.type == expected_type
        # The discriminator must be present in the serialized payload so
        # clients reading the stream can demux on it.
        dumped = ev.model_dump(exclude_none=True)
        assert dumped["type"] == expected_type
        # Same in JSON form (what ``_emit`` serializes).
        assert f'"type":"{expected_type}"' in ev.model_dump_json(exclude_none=True)

    def test_type_field_cannot_be_overridden_to_wrong_value(self):
        # Literal types reject any value other than the bake-in default.
        with pytest.raises(ValidationError):
            MessageStartEvent(
                id="a",
                delta={"message": {"role": "assistant"}},
                type="other",  # type: ignore[arg-type]
            )
