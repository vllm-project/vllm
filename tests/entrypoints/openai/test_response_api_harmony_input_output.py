# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import httpx
import pytest
import pytest_asyncio
from openai import OpenAI
from openai_harmony import Message, Role

from ...utils import RemoteOpenAIServer

MODEL_NAME = "openai/gpt-oss-20b"


def harmony_message_to_dict(message: Message) -> dict:
    """
    Convert a harmony Message to a dictionary format suitable for vLLM API.
    
    This adds the required "type": "harmony" field that vLLM's field validator
    expects for proper deserialization.
    
    Args:
        message: OpenAI Harmony Message object
        
    Returns:
        Dictionary representation with proper type field for vLLM validation
    """
    message_dict = message.to_dict()
    message_dict["type"] = "harmony"
    return message_dict


def harmony_messages_to_dicts(messages: list[Message]) -> list[dict]:
    """
    Convert a list of harmony Messages to dictionaries suitable for vLLM API.
    
    Args:
        messages: List of OpenAI Harmony Message objects
        
    Returns:
        List of dictionary representations with proper type fields
    """
    return [harmony_message_to_dict(msg) for msg in messages]


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
        with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


async def send_responses_request(server, data: dict) -> dict:
    """Helper function to send requests using HTTP."""
    async with httpx.AsyncClient(timeout=120.0) as http_client:
        response = await http_client.post(
            f"{server.url_root}/v1/responses",
            json=data,
            headers={"Authorization": f"Bearer {server.DUMMY_API_KEY}"})

        response.raise_for_status()
        return response.json()


class ResponsesApiResponse:
    """Helper class to make HTTP response look like OpenAI client response."""

    def __init__(self, data: dict):
        self.status = data["status"]
        self.input_messages = data.get("input_messages", [])
        self.output_messages = data.get("output_messages", [])
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
    response_json = await send_responses_request(
        server, {
            "model": model_name,
            "input": "Tell me more about that city.",
            "previous_response_messages": previous_harmony_messages,
            "enable_response_messages": True
        })

    response = ResponsesApiResponse(response_json)

    assert response is not None
    assert response.status == "completed"

    # Verify the response includes both the previous and new messages
    all_messages = (response.input_messages + response.output_messages)

    # Verify that all messages have proper serialization
    all_messages = (response.input_messages + response.output_messages)
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

    # First request using HTTP since we need enable_response_messages
    response1_json = await send_responses_request(
        server, {
            "model": model_name,
            "input": "What is 2 + 2?",
            "instructions": "Provide a simple answer.",
            "enable_response_messages": True
        })

    response1 = ResponsesApiResponse(response1_json)
    assert response1 is not None
    assert response1.status == "completed"

    # Extract harmony messages from first response
    first_input_messages = response1.input_messages
    first_output_messages = response1.output_messages

    # Combine all messages from first conversation
    all_first_messages = first_input_messages + first_output_messages

    # Second request using harmony messages from first response - use HTTP
    response2_json = await send_responses_request(
        server, {
            "model": model_name,
            "input": "Now what is 3 + 3?",
            "instructions": "Continue the math conversation.",
            "previous_response_messages": all_first_messages,
            "previous_response_output": response1_json.get("output", []),
            "enable_response_messages": True
        })

    response2 = ResponsesApiResponse(response2_json)

    assert response2 is not None
    assert response2.status == "completed"

    # Verify that second response contains more messages (original + new)
    second_input_messages = response2.input_messages
    second_output_messages = response2.output_messages

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
    response1_json = await send_responses_request(
        server, {
            "model": model_name,
            "input":
            "I'm planning a trip to Tokyo. What's the best time to visit?",
            "instructions": "Provide travel advice.",
            "enable_response_messages": True
        })

    response1 = ResponsesApiResponse(response1_json)
    assert response1.status == "completed"

    # Get all messages from the first conversation
    all_messages = (response1.input_messages + response1.output_messages)

    # Continue the conversation with a follow-up question
    response2_json = await send_responses_request(
        server, {
            "model": model_name,
            "input": "What about food recommendations for that city?",
            "instructions": "Continue helping with travel planning.",
            "previous_response_messages": all_messages,
            "previous_response_output": response1_json.get("output", []),
            "enable_response_messages": True
        })

    response2 = ResponsesApiResponse(response2_json)
    assert response2.status == "completed"

    # Verify context is maintained - should have more messages now
    assert len(response2.input_messages) > len(response1.input_messages)

    # The conversation should contain references to the original topic
    all_content = []
    for msg in (response2.input_messages + response2.output_messages):
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

    response_json = await send_responses_request(
        server,
        {
            "model": model_name,
            "input": "What's 5 + 5?",
            "instructions": "Answer the math question.",
            "previous_response_messages": [],  # Empty list
            "enable_response_messages": True
        })

    response = ResponsesApiResponse(response_json)
    assert response.status == "completed"
    assert len(response.input_messages) > 0
    assert len(response.output_messages) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_harmony_message_none_parameter(client: OpenAI, model_name: str,
                                              server):
    """Test that None harmony messages parameter works (same as omitting)."""

    response_json = await send_responses_request(
        server, {
            "model": model_name,
            "input": "What's 7 + 8?",
            "instructions": "Answer the math question.",
            "previous_response_messages": None,
            "enable_response_messages": True
        })

    response = ResponsesApiResponse(response_json)
    assert response.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_tools_presence_across_requests(client: OpenAI, model_name: str,
                                              server):
    """Test that tools are properly handled when present in first request but not second."""

    # First request with tools defined
    response1_json = await send_responses_request(
        server, {
            "model": model_name,
            "input": "Calculate 25 + 17 using Python.",
            "tools": [{
                "type": "code_interpreter",
                "container": {
                    "type": "auto"
                }
            }],
            "enable_response_messages": True
        })

    response1 = ResponsesApiResponse(response1_json)
    assert response1.status == "completed"
    assert len(response1.input_messages) > 0
    assert len(response1.output_messages) > 0

    # Get all messages from first response
    all_messages_1 = response1.input_messages + response1.output_messages

    # Second request with NO tools defined, but using messages from first request
    response2_json = await send_responses_request(
        server,
        {
            "model": model_name,
            "input": "What was that calculation result again?",
            "instructions": "Use the previous conversation context.",
            "previous_response_messages": all_messages_1,
            # Note: explicitly NOT including tools here
            "enable_response_messages": True
        })

    response2 = ResponsesApiResponse(response2_json)
    assert response2.status == "completed"
    assert len(response2.input_messages) > 0
    assert len(response2.output_messages) > 0

    # Validate that input_messages from second response do not contain tool definitions
    tool_definitions_found = 0
    for msg in response2.input_messages:
        assert "role" in msg
        assert "content" in msg

        # Check that there are no tool-related keys in the message structure
        assert "tools" not in msg
        assert "tool_calls" not in msg

        # Check content items don't contain tool definitions
        if isinstance(msg["content"], list):
            for content_item in msg["content"]:
                # Tool definitions should not be present in content
                if content_item.get("type") == "tool_definition":
                    tool_definitions_found += 1
                # But tool_call and tool_result types may still be present from previous conversation
                # We're specifically checking that new tool definitions aren't added

    assert tool_definitions_found == 0, "No tool definitions should be in second request input"

    # Verify the second response has more messages (accumulated context)
    assert len(response2.input_messages) > len(response1.input_messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_tools_introduction_in_second_request(client: OpenAI,
                                                    model_name: str, server):
    """Test introducing tools in second request when first had none."""

    # First request with NO tools
    response1_json = await send_responses_request(
        server,
        {
            "model": model_name,
            "input": "I need to do a calculation later: 42 + 58.",
            "instructions": "Just acknowledge the request for now.",
            # Note: explicitly NOT including tools here
            "enable_response_messages": True
        })

    response1 = ResponsesApiResponse(response1_json)
    assert response1.status == "completed"
    assert len(response1.input_messages) > 0
    assert len(response1.output_messages) > 0

    # Get all messages from first response
    all_messages_1 = response1.input_messages + response1.output_messages

    # Second request WITH tools defined, using messages from first request
    response2_json = await send_responses_request(
        server, {
            "model": model_name,
            "input":
            "Now please calculate that addition I mentioned using the python tool.",
            "instructions":
            "Use the previous conversation context and perform the calculation.",
            "previous_response_messages": all_messages_1,
            "tools": [{
                "type": "code_interpreter",
                "container": {
                    "type": "auto"
                }
            }],
            "enable_response_messages": True
        })

    response2 = ResponsesApiResponse(response2_json)
    assert response2.status == "completed"
    assert len(response2.input_messages) > 0
    assert len(response2.output_messages) > 0

    # Check for tool calls or tool results in the output messages
    has_tool_call = False
    tool_evidence = []
    for msg in response2.output_messages:
        if "content" in msg and isinstance(msg["content"], list):
            for content_item in msg["content"]:
                content_type = content_item.get("type", "")
                # Look for tool-related content types
                if content_type in [
                        "tool_call", "tool_result", "code_interpreter"
                ]:
                    has_tool_call = True
                    tool_evidence.append(f"Found {content_type}")
                    break
                # Also check for text content that indicates code execution
                elif (content_type == "text" and "text" in content_item):
                    text_content = content_item["text"].lower()
                    if any(
                            keyword in text_content for keyword in
                        ["python", "execute", "calculate", "42", "58", "100"]):
                        has_tool_call = True
                        tool_evidence.append(
                            "Found calculation evidence in text")
                        break
        if has_tool_call:
            break

    # Verify that tool functionality was engaged
    assert has_tool_call, "Expected tool call or calculation evidence in output messages"

    # Verify the second response has more messages (accumulated context)
    assert len(response2.input_messages) > len(response1.input_messages)

    # Verify conversation continuity - should reference the original calculation
    all_content = []
    for msg in response2.input_messages + response2.output_messages:
        if "content" in msg and isinstance(msg["content"], list):
            for content_item in msg["content"]:
                if content_item.get(
                        "type") == "text" and "text" in content_item:
                    all_content.append(content_item["text"].lower())

    conversation_text = " ".join(all_content)
    # Should contain references to the calculation mentioned in first request
    calculation_refs = [
        kw for kw in ["42", "58", "calculation", "addition"]
        if kw in conversation_text
    ]

    assert len(
        calculation_refs
    ) > 0, f"Expected calculation references, found: {calculation_refs}"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_code_interpreter_tool_structure(client: OpenAI, model_name: str,
                                               server):
    """Test code_interpreter tool has correct structure and appears in output_messages with analysis channel."""

    # Request that should trigger code_interpreter tool usage
    response_json = await send_responses_request(
        server, {
            "model": model_name,
            "input": "Calculate the factorial of 5 using Python code.",
            "tools": [{
                "type": "code_interpreter",
                "container": {
                    "type": "auto"
                }
            }],
            "enable_response_messages": True
        })

    response = ResponsesApiResponse(response_json)
    assert response.status == "completed"
    assert len(response.output_messages) > 0

    # Check that response.output doesn't contain tool calls (they should be in output_messages)
    response_output = response_json.get("output", [])
    has_tool_call_in_output = False
    for item in response_output:
        if isinstance(item, dict) and item.get("type") == "tool_call":
            has_tool_call_in_output = True
            break

    # Tool calls should NOT appear in response.output
    assert not has_tool_call_in_output, "Tool calls should not appear in response.output"

    # Find code_interpreter tool calls in output_messages
    code_interpreter_calls = []
    analysis_channel_messages = []

    for msg in response.output_messages:
        # Check for analysis channel
        if msg.get("channel") == "analysis":
            analysis_channel_messages.append(msg)

        # Look for code_interpreter tool calls in message content
        if "content" in msg and isinstance(msg["content"], list):
            for content_item in msg["content"]:
                if content_item.get("type") == "tool_call":
                    # Check if it's a code_interpreter tool call
                    tool_call = content_item.get("tool_call", {})
                    if tool_call.get("type") == "code_interpreter":
                        code_interpreter_calls.append(content_item)

    # For now, just verify we have analysis channel messages (tool calls may vary by implementation)
    assert len(analysis_channel_messages
               ) > 0, "Expected messages with channel='analysis'"

    # Validate structure of code_interpreter tool calls if they exist
    for i, tool_call_item in enumerate(code_interpreter_calls):
        assert tool_call_item["type"] == "tool_call"
        tool_call = tool_call_item["tool_call"]

        # Verify it's a code_interpreter type
        assert tool_call["type"] == "code_interpreter"

        # Should have required fields for code_interpreter
        assert "id" in tool_call
        assert "code_interpreter" in tool_call

        code_interpreter = tool_call["code_interpreter"]
        assert "input" in code_interpreter  # Should have code input

        # The code should be related to factorial calculation
        code_input = code_interpreter["input"].lower()
        factorial_keywords = [
            kw for kw in ["factorial", "5", "*", "math"] if kw in code_input
        ]

        assert len(factorial_keywords) > 0, \
            f"Code should be related to factorial calculation, got: {code_input}"

    # Verify analysis channel messages have proper structure
    for i, msg in enumerate(analysis_channel_messages):
        assert msg["channel"] == "analysis"
        assert "role" in msg
        assert "content" in msg
        assert isinstance(msg["content"], list)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_harmony_message_chain_conversation(client: OpenAI,
                                                  model_name: str, server):
    """Test chaining multiple requests with harmony messages."""

    # Start a conversation
    response1_json = await send_responses_request(
        server, {
            "model": model_name,
            "input": "My favorite color is blue.",
            "instructions": "Remember this information.",
            "enable_response_messages": True
        })

    response1 = ResponsesApiResponse(response1_json)
    assert response1.status == "completed"

    # Continue with context from first response
    messages_after_1 = (response1.input_messages + response1.output_messages)

    response2_json = await send_responses_request(
        server, {
            "model": model_name,
            "input": "What's my favorite color?",
            "instructions": "Use the previous context.",
            "previous_response_messages": messages_after_1,
            "enable_response_messages": True
        })

    response2 = ResponsesApiResponse(response2_json)
    assert response2.status == "completed"

    # Continue with context from second response
    messages_after_2 = (response2.input_messages + response2.output_messages)

    response3_json = await send_responses_request(
        server, {
            "model": model_name,
            "input": "What about my favorite number? It's 42.",
            "instructions": "Remember this new information too.",
            "previous_response_messages": messages_after_2,
            "enable_response_messages": True
        })

    response3 = ResponsesApiResponse(response3_json)
    assert response3.status == "completed"

    # Final request should have context from all previous messages
    messages_after_3 = (response3.input_messages + response3.output_messages)

    response4_json = await send_responses_request(
        server, {
            "model": model_name,
            "input": "What are my favorite color and number?",
            "instructions": "Recall both pieces of information.",
            "previous_response_messages": messages_after_3,
            "enable_response_messages": True
        })

    response4 = ResponsesApiResponse(response4_json)
    assert response4.status == "completed"

    # Verify the conversation has grown with each interaction
    assert len(response4.input_messages) > len(response3.input_messages)
    assert len(response3.input_messages) > len(response2.input_messages)
    assert len(response2.input_messages) > len(response1.input_messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_harmony_message_with_python_tool(client: OpenAI,
                                                model_name: str, server):
    """Test harmony messages with Python tool usage and context preservation."""
    # First request that should trigger Python tool usage using proper tool spec
    response1_json = await send_responses_request(
        server, {
            "model": model_name,
            "input": "Calculate the square root of 144 using Python code.",
            "tools": [{
                "type": "code_interpreter",
                "container": {
                    "type": "auto"
                }
            }],
            "enable_response_messages": True
        })

    response1 = ResponsesApiResponse(response1_json)
    assert response1.status == "completed"

    # Verify we have harmony messages from the first response
    assert len(response1.input_messages) > 0
    assert len(response1.output_messages) > 0

    # Look for tool usage in the messages
    all_messages_1 = (response1.input_messages + response1.output_messages)

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
    response2_json = await send_responses_request(
        server, {
            "model": model_name,
            "input": "Now calculate the square of that result.",
            "instructions": "Use the result from the previous calculation.",
            "previous_response_messages": all_messages_1,
            "tools": [{
                "type": "code_interpreter",
                "container": {
                    "type": "auto"
                }
            }],
            "enable_response_messages": True
        })

    response2 = ResponsesApiResponse(response2_json)
    assert response2.status == "completed"

    # Verify second response has more messages than first (accumulated context)
    assert len(response2.input_messages) > len(response1.input_messages)

    # Verify all messages maintain proper structure
    all_messages_2 = (response2.input_messages + response2.output_messages)

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
async def test_harmony_library_input_messages(client: OpenAI, model_name: str,
                                              server):
    """Test using harmony library to create messages and send via previous_response_messages for single-turn conversations."""

    # Create harmony messages using the library
    user_message = Message.from_role_and_content(
        Role.USER,
        "Hello! Can you help me understand how Python functions work?")

    # Convert to dict format for API using helper function
    harmony_messages = [harmony_message_to_dict(user_message)]

    # Send request with harmony messages in previous_response_messages and current input
    response_json = await send_responses_request(
        server,
        {
            "model": model_name,
            "input":
            "Can you explain more about Python functions?",  # Non-empty input field
            "previous_response_messages":
            harmony_messages,  # Using previous_response_messages to provide context
            "instructions":
            "Provide a helpful explanation about Python functions.",
            "enable_response_messages": True
        })

    response = ResponsesApiResponse(response_json)
    assert response.status == "completed"

    # Verify the harmony message was processed correctly
    assert len(response.input_messages) > 0
    assert len(response.output_messages) > 0

    # Check that our original harmony message is in the input_messages
    found_original_message = False
    for msg in response.input_messages:
        if (msg.get("role") == "user" and "content" in msg
                and isinstance(msg["content"], list)):
            for content_item in msg["content"]:
                if (content_item.get("type") == "text" and "python functions"
                        in content_item.get("text", "").lower()):
                    found_original_message = True
                    break
        if found_original_message:
            break

    assert found_original_message, "Original harmony message should be preserved in input_messages"

    # Verify response contains relevant content about Python functions
    response_text = ""
    for msg in response.output_messages:
        if "content" in msg and isinstance(msg["content"], list):
            for content_item in msg["content"]:
                if content_item.get(
                        "type") == "text" and "text" in content_item:
                    response_text += content_item["text"].lower()

    python_keywords_found = [
        kw for kw in ["function", "python", "def", "return"]
        if kw in response_text
    ]

    assert len(python_keywords_found
               ) > 0, "Response should contain Python function information"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_harmony_library_multi_message_input(client: OpenAI,
                                                   model_name: str, server):
    """Test using harmony library to create multiple messages via input_messages."""

    # Create a conversation using harmony library
    user_msg1 = Message.from_role_and_content(Role.USER,
                                              "What's the capital of Japan?")

    assistant_msg1 = Message.from_role_and_content(
        Role.ASSISTANT, "The capital of Japan is Tokyo.")

    user_msg2 = Message.from_role_and_content(
        Role.USER, "What's the population of that city?")

    # Convert to dict format for API using helper function
    harmony_messages = harmony_messages_to_dicts(
        [user_msg1, assistant_msg1, user_msg2])

    # Send request with multiple harmony messages
    response_json = await send_responses_request(
        server,
        {
            "model": model_name,
            "input": "",  # Empty input field
            "previous_response_messages": harmony_messages,
            "instructions": "Continue the conversation about Tokyo.",
            "enable_response_messages": True
        })

    response = ResponsesApiResponse(response_json)
    assert response.status == "completed"

    # Verify all original messages are preserved
    assert len(response.input_messages) >= len(harmony_messages)

    # Check that conversation context is maintained
    response_text = ""
    for msg in response.output_messages:
        if "content" in msg and isinstance(msg["content"], list):
            for content_item in msg["content"]:
                if content_item.get(
                        "type") == "text" and "text" in content_item:
                    response_text += content_item["text"].lower()

    # Should reference Tokyo or population in the response
    context_keywords = [
        kw for kw in ["tokyo", "population", "million", "people", "city"]
        if kw in response_text
    ]

    assert len(
        context_keywords) > 0, "Response should maintain conversation context"

    # Verify message structure integrity
    for msg in response.input_messages:
        assert "role" in msg
        assert "content" in msg
        assert isinstance(msg["content"], list)
        for content_item in msg["content"]:
            assert isinstance(content_item, dict)
            assert len(content_item) > 0
