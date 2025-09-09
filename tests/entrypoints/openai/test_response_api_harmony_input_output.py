# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import httpx
import pytest
import pytest_asyncio
from openai import OpenAI

from ...utils import RemoteOpenAIServer

MODEL_NAME = "openai/gpt-oss-20b"


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(scope="module")
def server(monkeypatch_module: pytest.MonkeyPatch):
    args = ["--enforce-eager", "--tool-server", "demo"]

    with monkeypatch_module.context() as m:
        m.setenv("VLLM_ENABLE_RESPONSES_API_STORE", "1")
        m.setenv("VLLM_RESPONSES_API_ENABLE_HARMONY_MESSAGES_OUTPUT", "1")
        with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


async def send_harmony_request(server, data: dict) -> dict:
    """Helper function to send requests with harmony messages using HTTP."""
    async with httpx.AsyncClient(timeout=120.0) as http_client:
        response = await http_client.post(
            f"{server.url_root}/v1/responses",
            json=data,
            headers={"Authorization": f"Bearer {server.DUMMY_API_KEY}"})

        response.raise_for_status()
        return response.json()


class HarmonyResponse:
    """Helper class to make HTTP response look like OpenAI client response."""

    def __init__(self, data: dict):
        self.status = data["status"]
        self.input_harmony_messages = data.get("input_harmony_messages", [])
        self.output_harmony_messages = data.get("output_harmony_messages", [])
        self.id = data.get("id")


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_harmony_message_deserialization(client: OpenAI, model_name: str,
                                               server):
    """Test that harmony messages can be properly deserialized from JSON."""

    # Create some harmony messages manually (as they would come from a client)
    previous_harmony_messages = [{
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": "What is the capital of France?"
        }],
        "channel":
        None,
        "recipient":
        None,
        "content_type":
        None
    }, {
        "role":
        "assistant",
        "content": [{
            "type": "text",
            "text": "The capital of France is Paris."
        }],
        "channel":
        None,
        "recipient":
        None,
        "content_type":
        None
    }]

    # Use direct HTTP request since OpenAI client doesn't support custom params
    response_json = await send_harmony_request(
        server, {
            "model": model_name,
            "input": "Tell me more about that city.",
            "instructions": "Use the previous conversation context.",
            "previous_response_harmony_messages": previous_harmony_messages
        })

    response = HarmonyResponse(response_json)

    assert response is not None
    assert response.status == "completed"

    # Verify the response includes both the previous and new messages
    all_messages = (response.input_harmony_messages +
                    response.output_harmony_messages)

    # Verify that all messages have proper serialization
    all_messages = (response.input_harmony_messages +
                    response.output_harmony_messages)
    for msg in all_messages:
        assert "role" in msg
        assert "content" in msg
        assert isinstance(msg["content"], list)

        # Ensure content is not empty objects
        for content_item in msg["content"]:
            assert isinstance(content_item, dict)
            assert len(content_item) > 0  # Should not be empty {}


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_harmony_message_round_trip(client: OpenAI, model_name: str,
                                          server):
    """Test full round-trip: get harmony messages from response, send back."""

    # First request using standard OpenAI client
    response1 = await client.responses.create(
        model=model_name,
        input="What is 2 + 2?",
        instructions="Provide a simple answer.")

    assert response1 is not None
    assert response1.status == "completed"

    # Extract harmony messages from first response
    first_input_messages = response1.input_harmony_messages
    first_output_messages = response1.output_harmony_messages

    # Combine all messages from first conversation
    all_first_messages = first_input_messages + first_output_messages

    # Second request using harmony messages from first response - use HTTP
    response2_json = await send_harmony_request(
        server, {
            "model": model_name,
            "input": "Now what is 3 + 3?",
            "instructions": "Continue the math conversation.",
            "previous_response_harmony_messages": all_first_messages
        })

    response2 = HarmonyResponse(response2_json)

    assert response2 is not None
    assert response2.status == "completed"

    # Verify that second response contains more messages (original + new)
    second_input_messages = response2.input_harmony_messages
    second_output_messages = response2.output_harmony_messages

    # Should have at least the messages from the first conversation plus new
    assert len(second_input_messages) > len(first_input_messages)

    # Verify all messages in the full conversation have proper content
    all_second_messages = second_input_messages + second_output_messages
    text_message_count = 0

    for msg in all_second_messages:
        assert "role" in msg
        assert "content" in msg

        for content_item in msg["content"]:
            if content_item.get("type") == "text":
                assert "text" in content_item
                assert len(content_item["text"].strip()) > 0
                text_message_count += 1

    # Should have at least some text messages in the conversation
    assert text_message_count > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_harmony_message_context_continuation(client: OpenAI,
                                                    model_name: str, server):
    """Test that harmony messages provide proper context continuation."""

    # First establish context with a specific topic
    response1_json = await send_harmony_request(
        server, {
            "model": model_name,
            "input":
            "I'm planning a trip to Tokyo. What's the best time to visit?",
            "instructions": "Provide travel advice."
        })

    response1 = HarmonyResponse(response1_json)
    assert response1.status == "completed"

    # Get all messages from the first conversation
    all_messages = (response1.input_harmony_messages +
                    response1.output_harmony_messages)

    # Continue the conversation with a follow-up question
    response2_json = await send_harmony_request(
        server, {
            "model": model_name,
            "input": "What about food recommendations for that city?",
            "instructions": "Continue helping with travel planning.",
            "previous_response_harmony_messages": all_messages
        })

    response2 = HarmonyResponse(response2_json)
    assert response2.status == "completed"

    # Verify context is maintained - should have more messages now
    assert len(response2.input_harmony_messages) > len(
        response1.input_harmony_messages)

    # The conversation should contain references to the original topic
    all_content = []
    for msg in (response2.input_harmony_messages +
                response2.output_harmony_messages):
        for content_item in msg["content"]:
            if content_item.get("type") == "text" and "text" in content_item:
                all_content.append(content_item["text"].lower())

    # Should contain references to the original context (Tokyo/trip)
    conversation_text = " ".join(all_content)
    assert ("tokyo" in conversation_text or "trip" in conversation_text
            or "travel" in conversation_text)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_harmony_message_empty_list(client: OpenAI, model_name: str,
                                          server):
    """Test that empty harmony messages list works properly."""

    response_json = await send_harmony_request(
        server,
        {
            "model": model_name,
            "input": "What's 5 + 5?",
            "instructions": "Answer the math question.",
            "previous_response_harmony_messages": []  # Empty list
        })

    response = HarmonyResponse(response_json)
    assert response.status == "completed"
    assert len(response.input_harmony_messages) > 0
    assert len(response.output_harmony_messages) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_harmony_message_none_parameter(client: OpenAI, model_name: str,
                                              server):
    """Test that None harmony messages parameter works (same as omitting)."""

    response_json = await send_harmony_request(
        server, {
            "model": model_name,
            "input": "What's 7 + 8?",
            "instructions": "Answer the math question.",
            "previous_response_harmony_messages": None
        })

    response = HarmonyResponse(response_json)
    assert response.status == "completed"
    assert len(response.input_harmony_messages) > 0
    assert len(response.output_harmony_messages) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_harmony_message_validation_error(client: OpenAI,
                                                model_name: str, server):
    """Test that malformed harmony messages produce validation errors."""

    # Test with invalid harmony message structure
    invalid_harmony_messages = [{
        "role": "user",
        # Missing required "content" field
        "channel": None,
        "recipient": None,
        "content_type": None
    }]

    async with httpx.AsyncClient(timeout=120.0) as http_client:
        response = await http_client.post(
            f"{server.url_root}/v1/responses",
            json={
                "model": model_name,
                "input": "Hello",
                "previous_response_harmony_messages": invalid_harmony_messages
            },
            headers={"Authorization": f"Bearer {server.DUMMY_API_KEY}"})

        # Should get an error (4xx or 5xx status code due to missing content)
        assert response.status_code >= 400


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_harmony_message_chain_conversation(client: OpenAI,
                                                  model_name: str, server):
    """Test chaining multiple requests with harmony messages."""

    # Start a conversation
    response1_json = await send_harmony_request(
        server, {
            "model": model_name,
            "input": "My favorite color is blue.",
            "instructions": "Remember this information."
        })

    response1 = HarmonyResponse(response1_json)
    assert response1.status == "completed"

    # Continue with context from first response
    messages_after_1 = (response1.input_harmony_messages +
                        response1.output_harmony_messages)

    response2_json = await send_harmony_request(
        server, {
            "model": model_name,
            "input": "What's my favorite color?",
            "instructions": "Use the previous context.",
            "previous_response_harmony_messages": messages_after_1
        })

    response2 = HarmonyResponse(response2_json)
    assert response2.status == "completed"

    # Continue with context from second response
    messages_after_2 = (response2.input_harmony_messages +
                        response2.output_harmony_messages)

    response3_json = await send_harmony_request(
        server, {
            "model": model_name,
            "input": "What about my favorite number? It's 42.",
            "instructions": "Remember this new information too.",
            "previous_response_harmony_messages": messages_after_2
        })

    response3 = HarmonyResponse(response3_json)
    assert response3.status == "completed"

    # Final request should have context from all previous messages
    messages_after_3 = (response3.input_harmony_messages +
                        response3.output_harmony_messages)

    response4_json = await send_harmony_request(
        server, {
            "model": model_name,
            "input": "What are my favorite color and number?",
            "instructions": "Recall both pieces of information.",
            "previous_response_harmony_messages": messages_after_3
        })

    response4 = HarmonyResponse(response4_json)
    assert response4.status == "completed"

    # Verify the conversation has grown with each interaction
    assert len(response4.input_harmony_messages) > len(
        response3.input_harmony_messages)
    assert len(response3.input_harmony_messages) > len(
        response2.input_harmony_messages)
    assert len(response2.input_harmony_messages) > len(
        response1.input_harmony_messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_harmony_message_with_python_tool(client: OpenAI,
                                                model_name: str, server):
    """Test harmony messages with Python tool usage and context preservation."""
    # First request that should trigger Python tool usage using proper tool spec
    response1_json = await send_harmony_request(
        server, {
            "model": model_name,
            "input": "Calculate the square root of 144 using Python code.",
            "tools": [{
                "type": "code_interpreter",
                "container": {
                    "type": "auto"
                }
            }]
        })

    response1 = HarmonyResponse(response1_json)
    assert response1.status == "completed"

    # Verify we have harmony messages from the first response
    assert len(response1.input_harmony_messages) > 0
    assert len(response1.output_harmony_messages) > 0

    # Look for tool usage in the messages
    all_messages_1 = (response1.input_harmony_messages +
                      response1.output_harmony_messages)

    # Check if any message contains tool calls or tool results
    has_tool_content = False
    for msg in all_messages_1:
        if "content" in msg and isinstance(msg["content"], list):
            for content_item in msg["content"]:
                if (content_item.get("type")
                        in ["tool_call", "tool_result", "code_interpreter"]
                        or ("text" in content_item and
                            ("python" in content_item["text"].lower()
                             or "calculate" in content_item["text"].lower()))):
                    has_tool_content = True
                    break
        if has_tool_content:
            break

    # Continue conversation with tool context
    response2_json = await send_harmony_request(
        server, {
            "model": model_name,
            "input": "Now calculate the square of that result.",
            "instructions": "Use the result from the previous calculation.",
            "previous_response_harmony_messages": all_messages_1,
            "tools": [{
                "type": "code_interpreter",
                "container": {
                    "type": "auto"
                }
            }]
        })

    response2 = HarmonyResponse(response2_json)
    assert response2.status == "completed"

    # Verify second response has more messages than first (accumulated context)
    assert len(response2.input_harmony_messages) > len(
        response1.input_harmony_messages)

    # Verify all messages maintain proper structure
    all_messages_2 = (response2.input_harmony_messages +
                      response2.output_harmony_messages)

    for msg in all_messages_2:
        assert "role" in msg
        assert "content" in msg
        assert isinstance(msg["content"], list)

        for content_item in msg["content"]:
            assert isinstance(content_item, dict)
            assert len(content_item) > 0  # Should not be empty

    # Verify we have more total messages in the second response
    assert len(all_messages_2) > len(all_messages_1)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_harmony_tools_serialization_chaining(client: OpenAI,
                                                    model_name: str, server):
    """Test that tools in system/developer content are properly serialized
    and deserialized when chaining requests."""
    # Create harmony messages with system content containing tools
    system_content_with_tools = {
        "role":
        "system",
        "content": [{
            "type": "system_content",
            "model_identity":
            "You are a helpful AI assistant with access to tools.",
            "reasoning_effort": "Medium",
            "knowledge_cutoff": "2024-06",
            "tools": {
                "python": {
                    "name":
                    "python",
                    "description":
                    "Execute Python code",
                    "tools": [{
                        "name": "execute_python",
                        "description": "Execute Python code in a sandbox",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": "Python code to execute"
                                }
                            },
                            "required": ["code"]
                        }
                    }]
                },
                "browser": {
                    "name":
                    "browser",
                    "description":
                    "Browse the web",
                    "tools": [{
                        "name": "search_web",
                        "description": "Search the web for information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query"
                                }
                            },
                            "required": ["query"]
                        }
                    }]
                }
            }
        }]
    }

    # Developer content with tools
    developer_content_with_tools = {
        "role":
        "developer",
        "content": [{
            "type": "developer_content",
            "instructions": "Use the available tools to help the user.",
            "tools": {
                "functions": {
                    "name":
                    "functions",
                    "description":
                    "Custom functions available to the assistant",
                    "tools": [{
                        "name": "calculate_area",
                        "description": "Calculate area of geometric shapes",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "shape": {
                                    "type": "string",
                                    "enum":
                                    ["circle", "rectangle", "triangle"]
                                },
                                "dimensions": {
                                    "type": "object",
                                    "description": "Shape-specific dimensions"
                                }
                            },
                            "required": ["shape", "dimensions"]
                        }
                    }]
                }
            }
        }]
    }

    # First request with harmony messages containing tools
    initial_harmony_messages = [
        system_content_with_tools, developer_content_with_tools, {
            "role":
            "user",
            "content": [{
                "type":
                "text",
                "text":
                "I need to calculate the area of a circle with radius 5."
            }]
        }
    ]

    response1_json = await send_harmony_request(
        server, {
            "model": model_name,
            "input": "Please help me with this calculation.",
            "instructions": "Use the available tools as needed.",
            "previous_response_harmony_messages": initial_harmony_messages
        })

    response1 = HarmonyResponse(response1_json)
    assert response1.status == "completed"

    # Extract harmony messages from first response
    all_messages_1 = (response1.input_harmony_messages +
                      response1.output_harmony_messages)

    # Verify tools are preserved in system content
    system_tools_found = False
    developer_tools_found = False

    for msg in all_messages_1:
        if msg.get("role") == "system" and "content" in msg:
            for content_item in msg["content"]:
                if (content_item.get("type") == "system_content"
                        and "tools" in content_item):
                    tools = content_item["tools"]
                    assert isinstance(tools, dict)
                    # Verify python and browser tools are present
                    if "python" in tools and "browser" in tools:
                        system_tools_found = True
                        # Verify structure is intact
                        python_tool = tools["python"]
                        assert isinstance(python_tool, dict)
                        assert python_tool.get("name") == "python"
                        assert "tools" in python_tool
                        assert len(python_tool["tools"]) > 0

        if msg.get("role") == "developer" and "content" in msg:
            for content_item in msg["content"]:
                if (content_item.get("type") == "developer_content"
                        and "tools" in content_item):
                    tools = content_item["tools"]
                    assert isinstance(tools, dict)
                    # Verify functions tool is present
                    if "functions" in tools:
                        developer_tools_found = True
                        functions_tool = tools["functions"]
                        assert isinstance(functions_tool, dict)
                        assert functions_tool.get("name") == "functions"
                        assert "tools" in functions_tool
                        assert len(functions_tool["tools"]) > 0

    # At least one should be found (tools are preserved)
    assert system_tools_found or developer_tools_found, (
        "Tools should be preserved in serialized harmony messages")

    # Second request using messages from first response (should maintain tools)
    response2_json = await send_harmony_request(
        server, {
            "model": model_name,
            "input":
            "Now calculate the area of a rectangle with width 3 and height 4.",
            "instructions": "Continue using the available tools.",
            "previous_response_harmony_messages": all_messages_1
        })

    response2 = HarmonyResponse(response2_json)
    assert response2.status == "completed"

    # Extract harmony messages from second response
    all_messages_2 = (response2.input_harmony_messages +
                      response2.output_harmony_messages)

    # Verify tools are still preserved after second round-trip
    system_tools_found_2 = False
    developer_tools_found_2 = False

    for msg in all_messages_2:
        if msg.get("role") == "system" and "content" in msg:
            for content_item in msg["content"]:
                if (content_item.get("type") == "system_content"
                        and "tools" in content_item):
                    tools = content_item["tools"]
                    if isinstance(tools, dict) and "python" in tools:
                        system_tools_found_2 = True

        if msg.get("role") == "developer" and "content" in msg:
            for content_item in msg["content"]:
                if (content_item.get("type") == "developer_content"
                        and "tools" in content_item):
                    tools = content_item["tools"]
                    if isinstance(tools, dict) and "functions" in tools:
                        developer_tools_found_2 = True

    # Tools should still be preserved after chaining
    assert system_tools_found_2 or developer_tools_found_2, (
        "Tools should be preserved after chaining requests")

    # Third request - test longer chain to ensure robustness
    response3_json = await send_harmony_request(
        server, {
            "model":
            model_name,
            "input": ("Finally, calculate the area of a triangle with "
                      "base 6 and height 8."),
            "instructions":
            "Use the calculation tools for this final request.",
            "previous_response_harmony_messages":
            all_messages_2
        })

    response3 = HarmonyResponse(response3_json)
    assert response3.status == "completed"

    # Verify conversation has grown and tools are maintained
    all_messages_3 = (response3.input_harmony_messages +
                      response3.output_harmony_messages)

    assert len(all_messages_3) > len(all_messages_2) > len(all_messages_1)

    # Final verification that tools are still intact
    final_tools_found = False
    for msg in all_messages_3:
        if "content" in msg and isinstance(msg["content"], list):
            for content_item in msg["content"]:
                if (content_item.get("type")
                        in ["system_content", "developer_content"]
                        and "tools" in content_item):
                    tools = content_item["tools"]
                    if isinstance(tools, dict) and len(tools) > 0:
                        final_tools_found = True
                        # Verify at least one tool has proper structure
                        for tool_name, tool_config in tools.items():
                            assert isinstance(tool_config, dict)
                            assert "name" in tool_config
                            assert tool_config["name"] == tool_name
                            break
                        break
        if final_tools_found:
            break

    assert final_tools_found, (
        "Tools should be preserved through the entire chain")
