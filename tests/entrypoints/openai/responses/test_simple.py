# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import pytest_asyncio
from openai import OpenAI

from ....utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-8B"


@pytest.fixture(scope="module")
def server():
    args = ["--reasoning-parser", "qwen3", "--max_model_len", "5000"]
    env_dict = dict(
        VLLM_ENABLE_RESPONSES_API_STORE="1",
        # uncomment for tool calling
        # PYTHON_EXECUTION_BACKEND="dangerously_use_uv",
    )

    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=env_dict) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_basic(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="What is 13 * 24?",
    )
    assert response is not None
    print("response: ", response)
    assert response.status == "completed"
    assert response.incomplete_details is None


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_enable_response_messages(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="Hello?",
        extra_body={"enable_response_messages": True},
    )
    assert response.status == "completed"
    assert response.input_messages[0]["type"] == "raw_message_tokens"
    assert type(response.input_messages[0]["message"]) is str
    assert len(response.input_messages[0]["message"]) > 10
    assert type(response.input_messages[0]["tokens"][0]) is int
    assert type(response.output_messages[0]["message"]) is str
    assert len(response.output_messages[0]["message"]) > 10
    assert type(response.output_messages[0]["tokens"][0]) is int


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_reasoning_item(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input=[
            {"type": "message", "content": "Hello.", "role": "user"},
            {
                "type": "reasoning",
                "id": "lol",
                "content": [
                    {
                        "type": "reasoning_text",
                        "text": "We need to respond: greeting.",
                    }
                ],
                "summary": [],
            },
        ],
        temperature=0.0,
    )
    assert response is not None
    assert response.status == "completed"
    # make sure we get a reasoning and text output
    assert response.output[0].type == "reasoning"
    assert response.output[1].type == "message"
    assert type(response.output[1].content[0].text) is str


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_streaming_output_consistency(client: OpenAI, model_name: str):
    """Test that streaming delta text matches the final response output_text.

    This test verifies that when using streaming mode:
    1. The concatenated text from all 'response.output_text.delta' events
    2. Matches the 'output_text' in the final 'response.completed' event
    """
    response = await client.responses.create(
        model=model_name,
        input="Say hello in one sentence.",
        stream=True,
    )

    events = []
    async for event in response:
        events.append(event)

    assert len(events) > 0

    # Concatenate all delta text from streaming events
    streaming_text = "".join(
        event.delta for event in events if event.type == "response.output_text.delta"
    )

    # Get the final response from the last event
    response_completed_event = events[-1]
    assert response_completed_event.type == "response.completed"
    assert response_completed_event.response.status == "completed"

    # Get output_text from the final response
    final_output_text = response_completed_event.response.output_text

    # Verify final response has output
    assert len(response_completed_event.response.output) > 0

    # Verify streaming text matches final output_text
    assert streaming_text == final_output_text, (
        f"Streaming text does not match final output_text.\n"
        f"Streaming: {streaming_text!r}\n"
        f"Final: {final_output_text!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_max_tokens(client: OpenAI, model_name: str):
    response = await client.responses.create(
        model=model_name,
        input="What is the first paragraph of Moby Dick?",
        reasoning={"effort": "low"},
        max_output_tokens=30,
    )
    assert response is not None
    assert response.status == "incomplete"
    assert response.incomplete_details.reason == "max_output_tokens"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_extra_sampling_params(client: OpenAI, model_name: str):
    """Test that extra sampling parameters are accepted and work."""
    # Test with multiple sampling parameters - just verify they're accepted
    response = await client.responses.create(
        model=model_name,
        input="Write a short sentence",
        max_output_tokens=50,
        temperature=0.7,
        top_p=0.9,
        extra_body={
            "top_k": 40,
            "repetition_penalty": 1.2,
            "seed": 42,
        },
    )

    # Verify request succeeded and parameters were accepted
    assert response.status in ["completed", "incomplete"]
    assert len(response.output) > 0
    assert response.output[0].content[0].text  # Has text output
