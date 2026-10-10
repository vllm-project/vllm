# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)


def test_delta_tool_call_serialization_omit_null():
    """Verify that null fields in DeltaToolCall and DeltaFunctionCall are omitted

    when serialized to JSON, maintaining OpenAI streaming spec compatibility.
    """
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


@pytest.mark.parametrize("exc_cls", [RuntimeError, AttributeError])
def test_dist_cleanup_empty_cache_resilience(mocker, exc_cls):
    """Verify that cleanup_dist_env_and_memory executes empty_host_cache

    even if empty_cache raises RuntimeError or AttributeError.
    """
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
    from vllm.platforms import current_platform

    mocker.patch.object(current_platform, "is_cpu", return_value=False)
    mock_empty_cache = mocker.patch(
        "torch.accelerator.empty_cache",
        side_effect=exc_cls("Accelerator unavailable"),
    )
    mock_empty_host = mocker.patch("torch.accelerator.empty_host_cache")

    cleanup_dist_env_and_memory(shutdown_ray=False)

    mock_empty_cache.assert_called_once()
    mock_empty_host.assert_called_once()
