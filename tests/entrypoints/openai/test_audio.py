from typing import Dict, List

import openai
import pytest
import pytest_asyncio

from vllm.assets.audio import AudioAsset
from vllm.multimodal.utils import encode_audio_base64, fetch_audio

from ...utils import RemoteOpenAIServer

MODEL_NAME = "fixie-ai/ultravox-v0_3"
TEST_AUDIO_URLS = [
    AudioAsset("winning_call").url,
]


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "5",
        "--enforce-eager",
        "--trust-remote-code",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.fixture(scope="session")
def base64_encoded_audio() -> Dict[str, str]:
    return {
        audio_url: encode_audio_base64(*fetch_audio(audio_url))
        for audio_url in TEST_AUDIO_URLS
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("audio_url", TEST_AUDIO_URLS)
async def test_single_chat_session_audio(client: openai.AsyncOpenAI,
                                         model_name: str, audio_url: str):
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {
                    "url": audio_url
                }
            },
            {
                "type": "text",
                "text": "What's happening in this audio?"
            },
        ],
    }]

    # test single completion
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        logprobs=True,
        temperature=0.0,
        top_logprobs=5)
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]
    assert choice.finish_reason == "length"
    assert chat_completion.usage == openai.types.CompletionUsage(
        completion_tokens=10, prompt_tokens=202, total_tokens=212)

    message = choice.message
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 10
    assert message.role == "assistant"
    messages.append({"role": "assistant", "content": message.content})

    # test multi-turn dialogue
    messages.append({"role": "user", "content": "express your result in json"})
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
    )
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("audio_url", TEST_AUDIO_URLS)
async def test_single_chat_session_audio_base64encoded(
        client: openai.AsyncOpenAI, model_name: str, audio_url: str,
        base64_encoded_audio: Dict[str, str]):

    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {
                    "url":
                    f"data:audio/wav;base64,{base64_encoded_audio[audio_url]}"
                }
            },
            {
                "type": "text",
                "text": "What's happening in this audio?"
            },
        ],
    }]

    # test single completion
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        logprobs=True,
        temperature=0.0,
        top_logprobs=5)
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]
    assert choice.finish_reason == "length"
    assert chat_completion.usage == openai.types.CompletionUsage(
        completion_tokens=10, prompt_tokens=202, total_tokens=212)

    message = choice.message
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 10
    assert message.role == "assistant"
    messages.append({"role": "assistant", "content": message.content})

    # test multi-turn dialogue
    messages.append({"role": "user", "content": "express your result in json"})
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
    )
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("audio_url", TEST_AUDIO_URLS)
async def test_single_chat_session_input_audio(
        client: openai.AsyncOpenAI, model_name: str, audio_url: str,
        base64_encoded_audio: Dict[str, str]):
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64_encoded_audio[audio_url],
                    "format": "wav"
                }
            },
            {
                "type": "text",
                "text": "What's happening in this audio?"
            },
        ],
    }]

    # test single completion
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        logprobs=True,
        top_logprobs=5)
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]
    assert choice.finish_reason == "length"
    assert chat_completion.usage == openai.types.CompletionUsage(
        completion_tokens=10, prompt_tokens=202, total_tokens=212)

    message = choice.message
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 10
    assert message.role == "assistant"
    messages.append({"role": "assistant", "content": message.content})

    # test multi-turn dialogue
    messages.append({"role": "user", "content": "express your result in json"})
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
    )
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("audio_url", TEST_AUDIO_URLS)
async def test_chat_streaming_audio(client: openai.AsyncOpenAI,
                                    model_name: str, audio_url: str):
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {
                    "url": audio_url
                }
            },
            {
                "type": "text",
                "text": "What's happening in this audio?"
            },
        ],
    }]

    # test single completion
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
    )
    output = chat_completion.choices[0].message.content
    stop_reason = chat_completion.choices[0].finish_reason

    # test streaming
    stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        stream=True,
    )
    chunks: List[str] = []
    finish_reason_count = 0
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.role:
            assert delta.role == "assistant"
        if delta.content:
            chunks.append(delta.content)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
    # finish reason should only return in last block
    assert finish_reason_count == 1
    assert chunk.choices[0].finish_reason == stop_reason
    assert delta.content
    assert "".join(chunks) == output


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("audio_url", TEST_AUDIO_URLS)
async def test_chat_streaming_input_audio(client: openai.AsyncOpenAI,
                                          model_name: str, audio_url: str,
                                          base64_encoded_audio: Dict[str,
                                                                     str]):
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64_encoded_audio[audio_url],
                    "format": "wav"
                }
            },
            {
                "type": "text",
                "text": "What's happening in this audio?"
            },
        ],
    }]

    # test single completion
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
    )
    output = chat_completion.choices[0].message.content
    stop_reason = chat_completion.choices[0].finish_reason

    # test streaming
    stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        stream=True,
    )
    chunks: List[str] = []
    finish_reason_count = 0
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.role:
            assert delta.role == "assistant"
        if delta.content:
            chunks.append(delta.content)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
    # finish reason should only return in last block
    assert finish_reason_count == 1
    assert chunk.choices[0].finish_reason == stop_reason
    assert delta.content
    assert "".join(chunks) == output


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("audio_url", TEST_AUDIO_URLS)
async def test_multi_audio_input(client: openai.AsyncOpenAI, model_name: str,
                                 audio_url: str,
                                 base64_encoded_audio: Dict[str, str]):

    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {
                    "url": audio_url
                }
            },
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64_encoded_audio[audio_url],
                    "format": "wav"
                }
            },
            {
                "type": "text",
                "text": "What's happening in this audio?"
            },
        ],
    }]

    with pytest.raises(openai.BadRequestError):  # test multi-audio input
        await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=10,
            temperature=0.0,
        )

    # the server should still work afterwards
    completion = await client.completions.create(
        model=model_name,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
    )
    completion = completion.choices[0].text
    assert completion is not None and len(completion) >= 0
