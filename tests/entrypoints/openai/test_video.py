# SPDX-License-Identifier: Apache-2.0

import openai
import pytest
import pytest_asyncio

from vllm.multimodal.utils import encode_video_base64, fetch_video

from ...utils import RemoteOpenAIServer

MODEL_NAME = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
MAXIMUM_VIDEOS = 4

TEST_VIDEO_URLS = [
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4",
]


@pytest.fixture(scope="module")
def server():
    args = [
        "--task",
        "generate",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "2",
        "--enforce-eager",
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        f"video={MAXIMUM_VIDEOS}",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.fixture(scope="session")
def base64_encoded_video() -> dict[str, str]:
    return {
        video_url: encode_video_base64(fetch_video(video_url))
        for video_url in TEST_VIDEO_URLS
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("video_url", TEST_VIDEO_URLS)
async def test_single_chat_session_video(client: openai.AsyncOpenAI,
                                         model_name: str, video_url: str):
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {
                    "url": video_url
                }
            },
            {
                "type": "text",
                "text": "What's in this video?"
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
        completion_tokens=10, prompt_tokens=6299, total_tokens=6309)

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
@pytest.mark.parametrize("video_url", TEST_VIDEO_URLS)
async def test_single_chat_session_video_beamsearch(client: openai.AsyncOpenAI,
                                                    model_name: str,
                                                    video_url: str):
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {
                    "url": video_url
                }
            },
            {
                "type": "text",
                "text": "What's in this video?"
            },
        ],
    }]

    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        n=2,
        max_completion_tokens=10,
        logprobs=True,
        top_logprobs=5,
        extra_body=dict(use_beam_search=True))
    assert len(chat_completion.choices) == 2
    assert chat_completion.choices[
        0].message.content != chat_completion.choices[1].message.content


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("video_url", TEST_VIDEO_URLS)
async def test_single_chat_session_video_base64encoded(
        client: openai.AsyncOpenAI, model_name: str, video_url: str,
        base64_encoded_video: dict[str, str]):

    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {
                    "url":
                    f"data:video/jpeg;base64,{base64_encoded_video[video_url]}"
                }
            },
            {
                "type": "text",
                "text": "What's in this video?"
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
        completion_tokens=10, prompt_tokens=6299, total_tokens=6309)

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
@pytest.mark.parametrize("video_url", TEST_VIDEO_URLS)
async def test_single_chat_session_video_base64encoded_beamsearch(
        client: openai.AsyncOpenAI, model_name: str, video_url: str,
        base64_encoded_video: dict[str, str]):

    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {
                    "url":
                    f"data:video/jpeg;base64,{base64_encoded_video[video_url]}"
                }
            },
            {
                "type": "text",
                "text": "What's in this video?"
            },
        ],
    }]
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        n=2,
        max_completion_tokens=10,
        extra_body=dict(use_beam_search=True))
    assert len(chat_completion.choices) == 2
    assert chat_completion.choices[
        0].message.content != chat_completion.choices[1].message.content


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("video_url", TEST_VIDEO_URLS)
async def test_chat_streaming_video(client: openai.AsyncOpenAI,
                                    model_name: str, video_url: str):
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {
                    "url": video_url
                }
            },
            {
                "type": "text",
                "text": "What's in this video?"
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
    chunks: list[str] = []
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
@pytest.mark.parametrize(
    "video_urls",
    [TEST_VIDEO_URLS[:i] for i in range(2, len(TEST_VIDEO_URLS))])
async def test_multi_video_input(client: openai.AsyncOpenAI, model_name: str,
                                 video_urls: list[str]):

    messages = [{
        "role":
        "user",
        "content": [
            *({
                "type": "video_url",
                "video_url": {
                    "url": video_url
                }
            } for video_url in video_urls),
            {
                "type": "text",
                "text": "What's in this video?"
            },
        ],
    }]

    if len(video_urls) > MAXIMUM_VIDEOS:
        with pytest.raises(openai.BadRequestError):  # test multi-video input
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
    else:
        chat_completion = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=10,
            temperature=0.0,
        )
        message = chat_completion.choices[0].message
        assert message.content is not None and len(message.content) >= 0
