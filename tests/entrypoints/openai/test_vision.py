from typing import Dict, List

import openai
import pytest
import pytest_asyncio

from vllm.multimodal.utils import encode_image_base64, fetch_image

from ...utils import RemoteOpenAIServer

MODEL_NAME = "microsoft/Phi-3.5-vision-instruct"
MAXIMUM_IMAGES = 2

# Test different image extensions (JPG/PNG) and formats (gray/RGB/RGBA)
TEST_IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Venn_diagram_rgb.svg/1280px-Venn_diagram_rgb.svg.png",
    "https://upload.wikimedia.org/wikipedia/commons/0/0b/RGBA_comp.png",
]


@pytest.fixture(scope="module")
def server():
    args = [
        "--task",
        "generate",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "5",
        "--enforce-eager",
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        f"image={MAXIMUM_IMAGES}",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.fixture(scope="session")
def base64_encoded_image() -> Dict[str, str]:
    return {
        image_url: encode_image_base64(fetch_image(image_url))
        for image_url in TEST_IMAGE_URLS
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("image_url", TEST_IMAGE_URLS)
async def test_single_chat_session_image(client: openai.AsyncOpenAI,
                                         model_name: str, image_url: str):
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            },
            {
                "type": "text",
                "text": "What's in this image?"
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
        completion_tokens=10, prompt_tokens=772, total_tokens=782)

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
@pytest.mark.parametrize("image_url", TEST_IMAGE_URLS)
async def test_single_chat_session_image_beamsearch(client: openai.AsyncOpenAI,
                                                    model_name: str,
                                                    image_url: str):
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            },
            {
                "type": "text",
                "text": "What's in this image?"
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
@pytest.mark.parametrize("image_url", TEST_IMAGE_URLS)
async def test_single_chat_session_image_base64encoded(
        client: openai.AsyncOpenAI, model_name: str, image_url: str,
        base64_encoded_image: Dict[str, str]):

    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url":
                    f"data:image/jpeg;base64,{base64_encoded_image[image_url]}"
                }
            },
            {
                "type": "text",
                "text": "What's in this image?"
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
        completion_tokens=10, prompt_tokens=772, total_tokens=782)

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
@pytest.mark.parametrize("image_url", TEST_IMAGE_URLS)
async def test_single_chat_session_image_base64encoded_beamsearch(
        client: openai.AsyncOpenAI, model_name: str, image_url: str,
        base64_encoded_image: Dict[str, str]):

    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url":
                    f"data:image/jpeg;base64,{base64_encoded_image[image_url]}"
                }
            },
            {
                "type": "text",
                "text": "What's in this image?"
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
@pytest.mark.parametrize("image_url", TEST_IMAGE_URLS)
async def test_chat_streaming_image(client: openai.AsyncOpenAI,
                                    model_name: str, image_url: str):
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            },
            {
                "type": "text",
                "text": "What's in this image?"
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
@pytest.mark.parametrize(
    "image_urls",
    [TEST_IMAGE_URLS[:i] for i in range(2, len(TEST_IMAGE_URLS))])
async def test_multi_image_input(client: openai.AsyncOpenAI, model_name: str,
                                 image_urls: List[str]):

    messages = [{
        "role":
        "user",
        "content": [
            *({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            } for image_url in image_urls),
            {
                "type": "text",
                "text": "What's in this image?"
            },
        ],
    }]

    if len(image_urls) > MAXIMUM_IMAGES:
        with pytest.raises(openai.BadRequestError):  # test multi-image input
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
