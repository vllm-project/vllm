# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify that ``system_fingerprint`` is omitted (not serialised as null)
from non-streaming responses when the server is started with
``--fingerprint-mode none``.

Regression test for https://github.com/vllm-project/vllm/issues/57376
"""

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    DeltaMessage,
    UsageInfo,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionResponse,
    CompletionResponseChoice,
)

# -- Helpers --


def _make_chat_response(fingerprint):
    return ChatCompletionResponse(
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=DeltaMessage(role="assistant", content="Hi"),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        system_fingerprint=fingerprint,
    )


def _make_completion_response(fingerprint):
    return CompletionResponse(
        model="test-model",
        choices=[
            CompletionResponseChoice(
                index=0,
                text="Hi",
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        system_fingerprint=fingerprint,
    )


# -- Chat Completion Tests --


class TestChatCompletionFingerprintNone:
    def test_key_absent_when_fingerprint_is_none(self):
        resp = _make_chat_response(fingerprint=None)
        dumped = resp.model_dump(
            exclude={"system_fingerprint"} if resp.system_fingerprint is None else None
        )
        assert "system_fingerprint" not in dumped

    def test_key_present_when_fingerprint_is_set(self):
        resp = _make_chat_response(fingerprint="vllm-abc123")
        dumped = resp.model_dump(
            exclude={"system_fingerprint"} if resp.system_fingerprint is None else None
        )
        assert "system_fingerprint" in dumped
        assert dumped["system_fingerprint"] == "vllm-abc123"

    def test_other_fields_unaffected_when_fingerprint_excluded(self):
        resp = _make_chat_response(fingerprint=None)
        dumped = resp.model_dump(
            exclude={"system_fingerprint"} if resp.system_fingerprint is None else None
        )
        assert dumped["model"] == "test-model"
        assert len(dumped["choices"]) == 1
        assert dumped["usage"]["total_tokens"] == 2


# -- Completion Tests --


class TestCompletionFingerprintNone:
    def test_key_absent_when_fingerprint_is_none(self):
        resp = _make_completion_response(fingerprint=None)
        dumped = resp.model_dump(
            exclude={"system_fingerprint"} if resp.system_fingerprint is None else None
        )
        assert "system_fingerprint" not in dumped

    def test_key_present_when_fingerprint_is_set(self):
        resp = _make_completion_response(fingerprint="vllm-abc123")
        dumped = resp.model_dump(
            exclude={"system_fingerprint"} if resp.system_fingerprint is None else None
        )
        assert "system_fingerprint" in dumped
        assert dumped["system_fingerprint"] == "vllm-abc123"

    def test_other_fields_unaffected_when_fingerprint_excluded(self):
        resp = _make_completion_response(fingerprint=None)
        dumped = resp.model_dump(
            exclude={"system_fingerprint"} if resp.system_fingerprint is None else None
        )
        assert dumped["model"] == "test-model"
        assert len(dumped["choices"]) == 1
        assert dumped["usage"]["total_tokens"] == 2
