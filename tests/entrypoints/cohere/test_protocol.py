# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``vllm/entrypoints/cohere/protocol.py``.

The module is mostly a thin wrapper around the official ``cohere`` SDK
types plus a few local additions:

* :class:`CohereError` envelope.
* :class:`CohereChatV2Request` (model and messages required,
  ``max_tokens`` non-negative).
* :class:`CohereChatV2Response` plus the usage / logprob helpers.
* The streaming event subclasses that bake a wire-format ``type``
  discriminator into ``model_dump()`` so SSE consumers can demux on it.
"""

import pytest
from pydantic import ValidationError

import vllm.envs as envs
from vllm.entrypoints.cohere.protocol import (
    AssistantMessageResponse,
    CitationEndEvent,
    CitationStartEvent,
    CohereChatV2Request,
    CohereChatV2Response,
    CohereError,
    CohereUsageTokens,
    ContentDeltaEvent,
    ContentEndEvent,
    ContentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolPlanDeltaEvent,
)
from vllm.exceptions import VLLMValidationError

# ======================================================================
# CohereError
# ======================================================================


class TestCohereError:
    def test_model_dump_excludes_none(self):
        err = CohereError(message="boom")
        assert err.model_dump(exclude_none=True) == {"message": "boom"}


# ======================================================================
# CohereChatV2Request
# ======================================================================


class TestCohereChatV2Request:
    def test_empty_model_rejected(self):
        with pytest.raises(VLLMValidationError, match="model is required") as exc_info:
            CohereChatV2Request(model="", messages=[{"role": "user", "content": "hi"}])
        assert exc_info.value.parameter == "model"
        assert exc_info.value.value is None

    def test_negative_max_tokens_rejected(self):
        with pytest.raises(VLLMValidationError, match="non-negative") as exc_info:
            CohereChatV2Request(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=-1,
            )
        assert exc_info.value.parameter == "max_tokens"
        assert exc_info.value.value == -1

    def test_empty_messages_rejected(self):
        with pytest.raises(
            VLLMValidationError, match="messages must contain at least one message"
        ) as exc_info:
            CohereChatV2Request(model="m", messages=[])
        assert exc_info.value.parameter == "messages"
        assert exc_info.value.value is None

    def test_zero_max_tokens_allowed(self):
        # Zero is allowed (the docs allow 0 -> return prompt only).
        req = CohereChatV2Request(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=0,
        )
        assert req.max_tokens == 0

    def test_invalid_tool_choice_rejected(self):
        with pytest.raises(ValidationError):
            CohereChatV2Request(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                tool_choice="ANY",  # not REQUIRED/NONE
            )

    def test_stop_sequences_bounded_by_env_limit(self):
        messages = [{"role": "user", "content": "hi"}]
        limit = envs.VLLM_MAX_STOP_STRINGS

        req = CohereChatV2Request(
            model="m",
            messages=messages,
            stop_sequences=[f"s{i}" for i in range(limit)],
        )
        assert len(req.stop_sequences) == limit

        with pytest.raises(ValidationError):
            CohereChatV2Request(
                model="m",
                messages=messages,
                stop_sequences=[f"s{i}" for i in range(limit + 1)],
            )


# ======================================================================
# Usage / Logprobs
# ======================================================================


class TestUsage:
    def test_tokens_serialization(self):
        u = CohereUsageTokens(input_tokens=10, output_tokens=5)
        assert u.model_dump() == {"input_tokens": 10.0, "output_tokens": 5.0}


# ======================================================================
# CohereChatV2Response
# ======================================================================


class TestCohereChatV2Response:
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
