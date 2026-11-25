# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)


def test_delta_tool_call_serialization_omit_null():
    """Verify null fields in DeltaToolCall and DeltaFunctionCall are omitted."""
    # First chunk: includes id, type, index, and initial function call
    tc1 = DeltaToolCall(
        id="call_abc123",
        type="function",
        index=0,
        function=DeltaFunctionCall(name="get_weather", arguments=""),
    )
    msg1 = DeltaMessage(role="assistant", tool_calls=[tc1])
    dump1 = json.loads(msg1.model_dump_json())

    assert dump1["tool_calls"][0]["id"] == "call_abc123"
    assert dump1["tool_calls"][0]["type"] == "function"
    assert dump1["tool_calls"][0]["index"] == 0
    assert dump1["tool_calls"][0]["function"] == {
        "name": "get_weather",
        "arguments": "",
    }

    # Subsequent chunk: id and type are None (should be omitted, not null)
    tc2 = DeltaToolCall(
        index=0,
        function=DeltaFunctionCall(arguments='{"location": "Tokyo"}'),
    )
    msg2 = DeltaMessage(tool_calls=[tc2])
    dump2 = json.loads(msg2.model_dump_json())

    # id and type must not be in the serialized delta dictionary
    assert "id" not in dump2["tool_calls"][0]
    assert "type" not in dump2["tool_calls"][0]
    assert dump2["tool_calls"][0]["index"] == 0
    assert "name" not in dump2["tool_calls"][0]["function"]
    assert dump2["tool_calls"][0]["function"]["arguments"] == '{"location": "Tokyo"}'
