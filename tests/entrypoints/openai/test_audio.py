from typing import Dict, List

import openai
import pytest

from vllm.assets.audio import AudioAsset
from vllm.multimodal.utils import encode_audio_base64, fetch_audio

from ...utils import RemoteOpenAIServer

MODEL_NAME = "fixie-ai/ultravox-v0_3"
TEST_AUDIO_URLS = [
    AudioAsset("winning_call").url,
    AudioAsset("mary_had_lamb").url
]


@pytest.fixture(scope="module")
def server():
    args = [
        "--dtype", "bfloat16", "--max-model-len", "4096", "--enforce-eager",
        "--limit-mm-per-prompt", f"audio={len(TEST_AUDIO_URLS)}"
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server):
    return server.get_async_client()


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
    chat_completion = await client.chat.completions.create(model=model_name,
                                                           messages=messages,
                                                           max_tokens=10,
                                                           logprobs=True,
                                                           top_logprobs=5)
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]
    assert choice.finish_reason == "length"
    assert chat_completion.usage.completion_tokens == 10

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
        max_tokens=10,
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
    chat_completion = await client.chat.completions.create(model=model_name,
                                                           messages=messages,
                                                           max_tokens=10,
                                                           logprobs=True,
                                                           top_logprobs=5)
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]
    assert choice.finish_reason == "length"
    assert chat_completion.usage.completion_tokens == 10

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
        max_tokens=10,
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
        max_tokens=10,
        temperature=0.0,
    )
    output = chat_completion.choices[0].message.content
    stop_reason = chat_completion.choices[0].finish_reason

    # test streaming
    stream = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
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
async def test_multiple_audio_urls(client: openai.AsyncOpenAI,
                                   model_name: str):
    messages = [{
        "role":
        "user",
        "content": [
            *({
                "type": "audio_url",
                "audio_url": {
                    "url": audio_url
                }
            } for audio_url in TEST_AUDIO_URLS),
            {
                "type": "text",
                "text": "What sport and what nursery rhyme are referenced?"
            },
        ],
    }]

    # test single completion
    chat_completion = await client.chat.completions.create(model=model_name,
                                                           messages=messages,
                                                           max_tokens=10,
                                                           logprobs=True,
                                                           temperature=0.0,
                                                           top_logprobs=5)
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]
    assert choice.finish_reason == "length"
    assert chat_completion.usage.completion_tokens == 10

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
        max_tokens=10,
    )
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 0
