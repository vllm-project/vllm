# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backward-compat shim: responses mirror ``reasoning`` into the deprecated
``reasoning_content`` field for clients that only read the latter."""

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatMessage
from vllm.entrypoints.openai.engine.protocol import DeltaMessage

pytestmark = pytest.mark.skip_global_cleanup


def test_delta_message_mirrors_reasoning_into_reasoning_content():
    delta = DeltaMessage(reasoning="thinking out loud")

    dumped = json.loads(delta.model_dump_json(exclude_unset=True))

    assert dumped["reasoning"] == "thinking out loud"
    assert dumped["reasoning_content"] == "thinking out loud"


def test_delta_message_without_reasoning_omits_reasoning_content():
    delta = DeltaMessage(content="hello")

    dumped = json.loads(delta.model_dump_json(exclude_unset=True))

    assert "reasoning" not in dumped
    assert "reasoning_content" not in dumped


def test_chat_message_mirrors_reasoning_into_reasoning_content():
    message = ChatMessage(role="assistant", content="answer", reasoning="why")

    dumped = json.loads(message.model_dump_json(exclude_unset=True))

    assert dumped["reasoning"] == "why"
    assert dumped["reasoning_content"] == "why"


def test_chat_message_without_reasoning_omits_reasoning_content():
    message = ChatMessage(role="assistant", content="answer")

    dumped = json.loads(message.model_dump_json(exclude_unset=True))

    assert "reasoning" not in dumped
    assert "reasoning_content" not in dumped
