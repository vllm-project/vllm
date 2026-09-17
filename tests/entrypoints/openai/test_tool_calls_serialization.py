# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for tool_calls Iterable → list materialisation.

Regression tests for https://github.com/vllm-project/vllm/issues/34792.

Setting VLLM_LOGGING_LEVEL=debug caused tool calling to break for Mistral
models because:
  1. The OpenAI Python SDK types tool_calls as Iterable[...] in
     ChatCompletionAssistantMessageParam.
  2. Pydantic v2, when validating from Python objects (not from raw JSON),
     wraps Iterable fields in a one-shot lazy iterator.
  3. Debug logging called model_dump_json() which consumed that iterator.
  4. The Mistral tokenizer then saw empty tool_calls and raised
     "ValueError: Unexpected tool call id ...".
"""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest


def _make_tool_call(tc_id: str, name: str, args: str) -> dict:
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def _make_request(messages: list) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=messages,
    )


def test_tool_calls_list_preserved_after_model_dump():
    """tool_calls in assistant messages must be readable after model_dump_json.

    When the request is built from Python dicts (as in the Anthropic → OpenAI
    conversion path), Pydantic v2 previously wrapped the Iterable tool_calls
    in a one-shot iterator.  model_dump_json() consumed it, leaving subsequent
    readers (e.g. the Mistral tokenizer) with an empty sequence.
    """
    tool_call = _make_tool_call("call_abc123", "get_weather", '{"city": "Paris"}')
    messages = [
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": '{"temperature": 20}',
        },
    ]

    req = _make_request(messages)

    # Simulate debug logging: serialize the model (this was the trigger)
    _ = req.model_dump_json()

    # The assistant message must still have accessible tool_calls afterwards
    assistant_msg = req.messages[1]
    assert isinstance(assistant_msg, dict)
    tool_calls = assistant_msg.get("tool_calls")
    assert tool_calls is not None, "tool_calls must not be None after model_dump_json"
    assert isinstance(tool_calls, list), "tool_calls must be a list"
    assert len(tool_calls) > 0, "tool_calls must not be empty after model_dump_json"


def test_tool_calls_from_generator_are_materialised():
    """tool_calls passed as a generator must be converted to list on validation."""
    tool_call = _make_tool_call("call_gen1", "search", '{"query": "vllm"}')

    def tool_calls_gen():
        yield tool_call

    messages = [
        {"role": "user", "content": "Search for vllm"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls_gen(),  # one-shot generator
        },
    ]

    req = _make_request(messages)
    assistant_msg = req.messages[1]
    assert isinstance(assistant_msg, dict)

    # Iterate twice — must not raise or return empty on second pass
    tool_calls_first = list(assistant_msg.get("tool_calls", []))
    tool_calls_second = list(assistant_msg.get("tool_calls", []))

    assert len(tool_calls_first) == 1, "First read must return the tool call"
    assert len(tool_calls_second) == 1, "Second read must also return the tool call"


def test_tool_calls_list_passthrough():
    """tool_calls already provided as a list must remain a list."""
    tool_call = _make_tool_call("call_list1", "calculate", '{"expr": "2+2"}')
    messages = [
        {"role": "user", "content": "Calculate 2+2"},
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
    ]

    req = _make_request(messages)
    assistant_msg = req.messages[1]
    assert isinstance(assistant_msg, dict)
    assert isinstance(assistant_msg.get("tool_calls"), list)


def test_messages_without_tool_calls_unaffected():
    """Messages without tool_calls must be handled correctly."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    req = _make_request(messages)
    # None of the messages should have tool_calls injected
    for msg in req.messages:
        assert isinstance(msg, dict)
        assert msg.get("tool_calls") is None or msg.get("tool_calls") == []


@pytest.mark.parametrize("num_tool_calls", [1, 3])
def test_multiple_tool_calls_materialised(num_tool_calls: int):
    """Multiple tool calls in a single message are all preserved."""
    tool_calls = [
        _make_tool_call(f"call_{i}", f"func_{i}", f'{{"arg": {i}}}')
        for i in range(num_tool_calls)
    ]
    messages = [
        {"role": "user", "content": "Do things"},
        {"role": "assistant", "content": None, "tool_calls": iter(tool_calls)},
    ]

    req = _make_request(messages)
    assistant_msg = req.messages[1]
    assert isinstance(assistant_msg, dict)

    result_tool_calls = assistant_msg.get("tool_calls")
    assert isinstance(result_tool_calls, list)
    assert len(result_tool_calls) == num_tool_calls

    # Verify after model_dump_json too
    _ = req.model_dump_json()
    assert len(assistant_msg.get("tool_calls", [])) == num_tool_calls
