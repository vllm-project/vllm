# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import openai
import pytest
import pytest_asyncio
from transformers import AutoProcessor

from vllm.multimodal.utils import encode_image_base64, fetch_image

from ...utils import RemoteOpenAIServer

MODEL_NAME = "microsoft/Phi-3.5-vision-instruct"
MAXIMUM_IMAGES = 2

# Test different image extensions (JPG/PNG) and formats (gray/RGB/RGBA)
TEST_IMAGE_ASSETS = [
    "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",  # "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    "Grayscale_8bits_palette_sample_image.png",  # "https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png",
    "1280px-Venn_diagram_rgb.svg.png",  # "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Venn_diagram_rgb.svg/1280px-Venn_diagram_rgb.svg.png",
    "RGBA_comp.png",  # "https://upload.wikimedia.org/wikipedia/commons/0/0b/RGBA_comp.png",
]

EXPECTED_MM_BEAM_SEARCH_RES = [
    [
        "The image shows a wooden boardwalk leading through a",
        "The image shows a wooden boardwalk extending into a",
    ],
    [
        "The image shows two parrots perched on",
        "The image shows two birds perched on a cur",
    ],
    [
        "The image shows a Venn diagram with three over",
        "This image shows a Venn diagram with three over",
    ],
    [
        "This image displays a gradient of colors ranging from",
        "This image displays a gradient of colors forming a spectrum",
    ],
]


@pytest.fixture(scope="module")
def server():
    args = [
        "--runner",
        "generate",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "5",
        "--enforce-eager",
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        json.dumps({"image": MAXIMUM_IMAGES}),
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.fixture(scope="session")
def base64_encoded_image(local_asset_server) -> dict[str, str]:
    return {
        image_asset:
        encode_image_base64(local_asset_server.get_image_asset(image_asset))
        for image_asset in TEST_IMAGE_ASSETS
    }


def get_hf_prompt_tokens(model_name, content, image_url):
    processor = AutoProcessor.from_pretrained(model_name,
                                              trust_remote_code=True,
                                              num_crops=4)

    placeholder = "<|image_1|>\n"
    messages = [{
        "role": "user",
        "content": f"{placeholder}{content}",
    }]
    images = [fetch_image(image_url)]

    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, images, return_tensors="pt")

    return inputs.input_ids.shape[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("image_url", TEST_IMAGE_ASSETS, indirect=True)
async def test_single_chat_session_image(client: openai.AsyncOpenAI,
                                         model_name: str, image_url: str):
    content_text = "What's in this image?"
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
                "text": content_text
            },
        ],
    }]

    max_completion_tokens = 10
    # test single completion
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        logprobs=True,
        temperature=0.0,
        top_logprobs=5)
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]
    assert choice.finish_reason == "length"
    hf_prompt_tokens = get_hf_prompt_tokens(model_name, content_text,
                                            image_url)
    assert chat_completion.usage == openai.types.CompletionUsage(
        completion_tokens=max_completion_tokens,
        prompt_tokens=hf_prompt_tokens,
        total_tokens=hf_prompt_tokens + max_completion_tokens)

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
@pytest.mark.parametrize("image_url", TEST_IMAGE_ASSETS, indirect=True)
async def test_error_on_invalid_image_url_type(client: openai.AsyncOpenAI,
                                               model_name: str,
                                               image_url: str):
    content_text = "What's in this image?"
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "image_url",
                "image_url": image_url
            },
            {
                "type": "text",
                "text": content_text
            },
        ],
    }]

    # image_url should be a dict {"url": "some url"}, not directly a string
    with pytest.raises(openai.BadRequestError):
        _ = await client.chat.completions.create(model=model_name,
                                                 messages=messages,
                                                 max_completion_tokens=10,
                                                 temperature=0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("image_url", TEST_IMAGE_ASSETS, indirect=True)
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
@pytest.mark.parametrize("raw_image_url", TEST_IMAGE_ASSETS)
@pytest.mark.parametrize("image_url", TEST_IMAGE_ASSETS, indirect=True)
async def test_single_chat_session_image_base64encoded(
        client: openai.AsyncOpenAI, model_name: str, raw_image_url: str,
        image_url: str, base64_encoded_image: dict[str, str]):

    content_text = "What's in this image?"
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url":
                    f"data:image/jpeg;base64,{base64_encoded_image[raw_image_url]}"
                }
            },
            {
                "type": "text",
                "text": content_text
            },
        ],
    }]

    max_completion_tokens = 10
    # test single completion
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        logprobs=True,
        temperature=0.0,
        top_logprobs=5)
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]
    assert choice.finish_reason == "length"
    hf_prompt_tokens = get_hf_prompt_tokens(model_name, content_text,
                                            image_url)
    assert chat_completion.usage == openai.types.CompletionUsage(
        completion_tokens=max_completion_tokens,
        prompt_tokens=hf_prompt_tokens,
        total_tokens=hf_prompt_tokens + max_completion_tokens)

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
@pytest.mark.parametrize("image_idx", list(range(len(TEST_IMAGE_ASSETS))))
async def test_single_chat_session_image_base64encoded_beamsearch(
        client: openai.AsyncOpenAI, model_name: str, image_idx: int,
        base64_encoded_image: dict[str, str]):
    # NOTE: This test also validates that we pass MM data through beam search
    raw_image_url = TEST_IMAGE_ASSETS[image_idx]
    expected_res = EXPECTED_MM_BEAM_SEARCH_RES[image_idx]

    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url":
                    f"data:image/jpeg;base64,{base64_encoded_image[raw_image_url]}"
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
        temperature=0.0,
        extra_body=dict(use_beam_search=True))
    assert len(chat_completion.choices) == 2
    for actual, expected_str in zip(chat_completion.choices, expected_res):
        assert actual.message.content == expected_str


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("image_url", TEST_IMAGE_ASSETS, indirect=True)
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
    "image_urls",
    [TEST_IMAGE_ASSETS[:i] for i in range(2, len(TEST_IMAGE_ASSETS))],
    indirect=True)
async def test_multi_image_input(client: openai.AsyncOpenAI, model_name: str,
                                 image_urls: list[str]):

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


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize(
    "image_urls",
    [TEST_IMAGE_ASSETS[:i] for i in range(2, len(TEST_IMAGE_ASSETS))],
    indirect=True)
async def test_completions_with_image(
    client: openai.AsyncOpenAI,
    model_name: str,
    image_urls: list[str],
):
    for image_url in image_urls:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role":
                    "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            }
                        },
                    ],
                },
            ],
            model=model_name,
        )
        assert chat_completion.choices[0].message.content is not None
        assert isinstance(chat_completion.choices[0].message.content, str)
        assert len(chat_completion.choices[0].message.content) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize(
    "image_urls",
    [TEST_IMAGE_ASSETS[:i] for i in range(2, len(TEST_IMAGE_ASSETS))],
    indirect=True)
async def test_completions_with_image_with_uuid(
    client: openai.AsyncOpenAI,
    model_name: str,
    image_urls: list[str],
):
    for image_url in image_urls:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role":
                    "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                            "uuid": image_url
                        },
                    ],
                },
            ],
            model=model_name,
        )
        assert chat_completion.choices[0].message.content is not None
        assert isinstance(chat_completion.choices[0].message.content, str)
        assert len(chat_completion.choices[0].message.content) > 0

        # Second request, with empty image but the same uuid.
        chat_completion_with_empty_image = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role":
                    "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {},
                            "uuid": image_url
                        },
                    ],
                },
            ],
            model=model_name,
        )
        assert chat_completion_with_empty_image.choices[
            0].message.content is not None
        assert isinstance(
            chat_completion_with_empty_image.choices[0].message.content, str)
        assert len(
            chat_completion_with_empty_image.choices[0].message.content) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_completions_with_empty_image_with_uuid_without_cache_hit(
    client: openai.AsyncOpenAI,
    model_name: str,
):
    with pytest.raises(openai.BadRequestError):
        _ = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role":
                    "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {},
                            "uuid": "uuid_not_previously_seen"
                        },
                    ],
                },
            ],
            model=model_name,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize(
    "image_urls",
    [TEST_IMAGE_ASSETS[:i] for i in range(2, len(TEST_IMAGE_ASSETS))],
    indirect=True)
async def test_completions_with_image_with_incorrect_uuid_format(
    client: openai.AsyncOpenAI,
    model_name: str,
    image_urls: list[str],
):
    for image_url in image_urls:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role":
                    "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "incorrect_uuid_key": image_url,
                            },
                            "also_incorrect_uuid_key": image_url,
                        },
                    ],
                },
            ],
            model=model_name,
        )
        assert chat_completion.choices[0].message.content is not None
        assert isinstance(chat_completion.choices[0].message.content, str)
        assert len(chat_completion.choices[0].message.content) > 0
