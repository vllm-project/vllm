import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from openai.types.responses import ResponseFunctionToolCall
import vllm.entrypoints.openai.responses.context as context

def test_parsable_context_retry():
    context.envs.VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY = True
    ctx = object.__new__(context.ParsableContext)
    ctx.called_tools = set()
    ctx.available_tools = ["browser", "container", "python"]
    
    session = SimpleNamespace(call_tool=AsyncMock())
    call = ResponseFunctionToolCall(
        id="fc_test",
        call_id="call_test",
        type="function_call",
        name="web_search_preview",
        arguments="{bad",
    )
    
    loop = asyncio.get_event_loop()
    
    # Test search
    res = loop.run_until_complete(ctx.call_search_tool(session, call))
    assert len(res) == 1
    assert res[0].type == "function_call_output"
    assert res[0].call_id == "call_test"
    assert "Error parsing tool arguments as JSON" in res[0].output
    
    # Test container
    res = loop.run_until_complete(ctx.call_container_tool(session, call))
    assert len(res) == 1
    assert res[0].type == "function_call_output"
    assert res[0].call_id == "call_test"
    assert "Error parsing tool arguments as JSON" in res[0].output
    
    # Test python
    res = loop.run_until_complete(ctx.call_python_tool(session, call))
    assert len(res) == 1
    assert res[0].type == "function_call_output"
    assert res[0].call_id == "call_test"
    assert "Error parsing tool arguments as JSON" in res[0].output

if __name__ == "__main__":
    test_parsable_context_retry()
    print("Passed")
