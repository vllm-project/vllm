# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Coder-parser-specific tests.

Tests that exercise behaviour shared with the XML parser live in
``tests/tool_parsers/test_qwen3_xml_coder_shared.py``.  Only tests that
depend on Coder-only API (e.g. ``is_tool_call_started``) or on Coder-only
streaming behaviour (e.g. character-by-character chunking) belong here.
"""

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def qwen3_tool_parser(qwen3_tokenizer):
    return Qwen3CoderToolParser(qwen3_tokenizer, tools=None)


def test_extract_tool_calls_streaming_split_tag(qwen3_tool_parser):
    """``<tool_call>`` arrives split across two deltas (``<tool`` then
    ``_call>``).  ``is_tool_call_started`` must flip to ``True`` once the
    full tag exists in ``current_text``, and the partial tag must not leak
    into ``DeltaMessage.content``.

    This relies on the Coder parser's ``is_tool_call_started`` attribute,
    which has no equivalent on the XML parser.
    """
    request = ChatCompletionRequest(model=MODEL, messages=[])

    prev_text_1 = "I will use a tool."
    delta_text_1 = "<tool"
    curr_text_1 = prev_text_1 + delta_text_1

    msg1 = qwen3_tool_parser.extract_tool_calls_streaming(
        previous_text=prev_text_1,
        current_text=curr_text_1,
        delta_text=delta_text_1,
        previous_token_ids=[1, 2, 3],
        current_token_ids=[1, 2, 3, 4],
        delta_token_ids=[4],
        request=request,
    )

    prev_text_2 = curr_text_1
    delta_text_2 = "_call>"
    curr_text_2 = prev_text_2 + delta_text_2

    msg2 = qwen3_tool_parser.extract_tool_calls_streaming(
        previous_text=prev_text_2,
        current_text=curr_text_2,
        delta_text=delta_text_2,
        previous_token_ids=[1, 2, 3, 4],
        current_token_ids=[1, 2, 3, 4, 5],
        delta_token_ids=[5],
        request=request,
    )

    assert qwen3_tool_parser.is_tool_call_started is True

    if msg1 and msg1.content:
        assert "<tool" not in msg1.content
    if msg2 and msg2.content:
        assert "_call>" not in msg2.content


def test_extract_tool_calls_streaming_various_chunk_sizes(
    qwen3_tokenizer,
):
    """Coder streaming must reconstruct arguments correctly even when the
    deltas arrive a single character at a time.

    The XML parser's SAX-based streaming cannot tolerate ``chunk_size=1``
    by design (an XML tag is not parseable until ``>`` arrives), so this
    robustness test stays Coder-only.
    """
    request = ChatCompletionRequest(model="test", messages=[])

    template_text = """<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>"""

    for chunk_size in [1, 3, 15, len(template_text)]:
        parser = Qwen3CoderToolParser(qwen3_tokenizer, tools=None)

        tool_states = {}
        current_text = ""
        previous_text = ""
        ptr = 0

        while ptr < len(template_text):
            delta = template_text[ptr:ptr + chunk_size]
            previous_text = current_text
            current_text += delta
            ptr += chunk_size

            delta_message = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=request,
            )

            if delta_message and delta_message.tool_calls:
                for tool_call in delta_message.tool_calls:
                    idx = tool_call.index
                    if idx not in tool_states:
                        tool_states[idx] = {
                            "id": None,
                            "name": None,
                            "arguments": "",
                            "type": None,
                        }
                    if tool_call.id:
                        tool_states[idx]["id"] = tool_call.id
                    if tool_call.type:
                        tool_states[idx]["type"] = tool_call.type
                    if tool_call.function:
                        if tool_call.function.name:
                            tool_states[idx]["name"] = tool_call.function.name
                        if tool_call.function.arguments is not None:
                            tool_states[idx]["arguments"] += (
                                tool_call.function.arguments
                            )

        assert 0 in tool_states, f"chunk_size={chunk_size}"
        assert tool_states[0]["name"] == "example_function_name"
        args = json.loads(tool_states[0]["arguments"])
        assert args["example_parameter_1"] == "value_1"
        assert args["example_parameter_2"] == (
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines"
        )
