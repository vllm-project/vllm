# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import openai
import pytest
import pytest_asyncio
from transformers import AutoProcessor

from vllm.multimodal.base import MediaWithBytes
from vllm.multimodal.utils import encode_image_url, fetch_image
from vllm.platforms import current_platform

from ...utils import RemoteOpenAIServer

MODEL_NAME = "microsoft/Phi-3.5-vision-instruct"
MAXIMUM_IMAGES = 2

# Test different image extensions (JPG/PNG) and formats (gray/RGB/RGBA)
TEST_IMAGE_ASSETS = [
    "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    "Grayscale_8bits_palette_sample_image.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/Grayscale_8bits_palette_sample_image.png",
    "1280px-Venn_diagram_rgb.svg.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/1280px-Venn_diagram_rgb.svg.png",
    "RGBA_comp.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/RGBA_comp.png",
]

# Required terms for beam search validation
# Each entry is a list of term groups - ALL groups must match
# Each group is a list of alternatives - at least ONE term in the group must appear
# This provides semantic validation while allowing wording variation
REQUIRED_BEAM_SEARCH_TERMS = [
    # Boardwalk image: must have "boardwalk" AND ("wooden" or "wood")
    [["boardwalk"], ["wooden", "wood"]],
    # Parrots image: must have ("parrot" or "bird") AND "two"
    [["parrot", "bird"], ["two"]],
    # Venn diagram: must have "venn" AND "diagram"
    [["venn"], ["diagram"]],
    # Gradient image: must have "gradient" AND ("color" or "spectrum")
    [["gradient"], ["color", "spectrum"]],
]


def check_output_matches_terms(content: str, term_groups: list[list[str]]) -> bool:
    """
    Check if content matches all required term groups.
    Each term group requires at least one of its terms to be present.
    All term groups must be satisfied.
    """
    content_lower = content.lower()
    for group in term_groups:
        if not any(term.lower() in content_lower for term in group):
            return False
    return True


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

    # ROCm: Increase timeouts to handle potential network delays and slower
    # video processing when downloading multiple videos from external sources
    env_overrides = {}
    if current_platform.is_rocm():
        env_overrides = {
            "VLLM_VIDEO_FETCH_TIMEOUT": "120",
            "VLLM_ENGINE_ITERATION_TIMEOUT_S": "300",
        }

    with RemoteOpenAIServer(MODEL_NAME, args, env_dict=env_overrides) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.fixture(scope="session")
def url_encoded_image(local_asset_server) -> dict[str, str]:
    return {
        image_asset: encode_image_url(local_asset_server.get_image_asset(image_asset))
        for image_asset in TEST_IMAGE_ASSETS
    }


def dummy_messages_from_image_url(
    image_urls: str | list[str],
    content_text: str = "What's in this image?",
):
    if isinstance(image_urls, str):
        image_urls = [image_urls]

    return [
        {
            "role": "user",
            "content": [
                *(
                    {"type": "image_url", "image_url": {"url": image_url}}
                    for image_url in image_urls
                ),
                {"type": "text", "text": content_text},
            ],
        }
    ]


def get_hf_prompt_tokens(model_name, content, image_url):
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True, num_crops=4
    )

    placeholder = "<|image_1|>\n"
    messages = [
        {
            "role": "user",
            "content": f"{placeholder}{content}",
        }
    ]
    image = fetch_image(image_url)
    # Unwrap MediaWithBytes if present
    if isinstance(image, MediaWithBytes):
        image = image.media
    images = [image]

    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(prompt, images, return_tensors="pt")

    return inputs.input_ids.shape[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("image_url", TEST_IMAGE_ASSETS, indirect=True)
async def test_single_chat_session_image(
    client: openai.AsyncOpenAI, model_name: str, image_url: str
):
    content_text = "What's in this image?"
    messages = dummy_messages_from_image_url(image_url, content_text)

    max_completion_tokens = 10
    # test single completion
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        logprobs=True,
        temperature=0.0,
        top_logprobs=5,
    )
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]
    assert choice.finish_reason == "length"
    hf_prompt_tokens = get_hf_prompt_tokens(model_name, content_text, image_url)
    assert chat_completion.usage == openai.types.CompletionUsage(
        completion_tokens=max_completion_tokens,
        prompt_tokens=hf_prompt_tokens,
        total_tokens=hf_prompt_tokens + max_completion_tokens,
    )

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
async def test_error_on_invalid_image_url_type(
    client: openai.AsyncOpenAI, model_name: str, image_url: str
):
    content_text = "What's in this image?"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": image_url},
                {"type": "text", "text": content_text},
            ],
        }
    ]

    # image_url should be a dict {"url": "some url"}, not directly a string
    with pytest.raises(openai.BadRequestError):
        _ = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=10,
            temperature=0.0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("image_url", TEST_IMAGE_ASSETS, indirect=True)
async def test_single_chat_session_image_beamsearch(
    client: openai.AsyncOpenAI, model_name: str, image_url: str
):
    content_text = "What's in this image?"
    messages = dummy_messages_from_image_url(image_url, content_text)

    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        n=2,
        max_completion_tokens=10,
        logprobs=True,
        top_logprobs=5,
        extra_body=dict(use_beam_search=True),
    )
    assert len(chat_completion.choices) == 2
    assert (
        chat_completion.choices[0].message.content
        != chat_completion.choices[1].message.content
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("raw_image_url", TEST_IMAGE_ASSETS)
@pytest.mark.parametrize("image_url", TEST_IMAGE_ASSETS, indirect=True)
async def test_single_chat_session_image_base64encoded(
    client: openai.AsyncOpenAI,
    model_name: str,
    raw_image_url: str,
    image_url: str,
    url_encoded_image: dict[str, str],
):
    content_text = "What's in this image?"
    messages = dummy_messages_from_image_url(
        url_encoded_image[raw_image_url],
        content_text,
    )

    max_completion_tokens = 10
    # test single completion
    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        logprobs=True,
        temperature=0.0,
        top_logprobs=5,
    )
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]
    assert choice.finish_reason == "length"
    hf_prompt_tokens = get_hf_prompt_tokens(model_name, content_text, image_url)
    assert chat_completion.usage == openai.types.CompletionUsage(
        completion_tokens=max_completion_tokens,
        prompt_tokens=hf_prompt_tokens,
        total_tokens=hf_prompt_tokens + max_completion_tokens,
    )

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
    client: openai.AsyncOpenAI,
    model_name: str,
    image_idx: int,
    url_encoded_image: dict[str, str],
):
    # NOTE: This test validates that we pass MM data through beam search
    raw_image_url = TEST_IMAGE_ASSETS[image_idx]
    required_terms = REQUIRED_BEAM_SEARCH_TERMS[image_idx]

    messages = dummy_messages_from_image_url(url_encoded_image[raw_image_url])

    chat_completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        n=2,
        max_completion_tokens=10,
        temperature=0.0,
        extra_body=dict(use_beam_search=True),
    )
    assert len(chat_completion.choices) == 2

    # Verify beam search produces two different non-empty outputs
    content_0 = chat_completion.choices[0].message.content
    content_1 = chat_completion.choices[1].message.content

    # Emit beam search outputs for debugging
    print(
        f"Beam search outputs for image {image_idx} ({raw_image_url}): "
        f"Output 0: {content_0!r}, Output 1: {content_1!r}"
    )

    assert content_0, "First beam search output should not be empty"
    assert content_1, "Second beam search output should not be empty"
    assert content_0 != content_1, "Beam search should produce different outputs"

    # Verify each output contains the required terms for this image
    for i, content in enumerate([content_0, content_1]):
        if not check_output_matches_terms(content, required_terms):
            pytest.fail(
                f"Output {i} '{content}' doesn't contain required terms. "
                f"Expected all of these term groups (at least one from each): "
                f"{required_terms}"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("image_url", TEST_IMAGE_ASSETS, indirect=True)
async def test_chat_streaming_image(
    client: openai.AsyncOpenAI, model_name: str, image_url: str
):
    messages = dummy_messages_from_image_url(image_url)

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
    indirect=True,
)
async def test_multi_image_input(
    client: openai.AsyncOpenAI, model_name: str, image_urls: list[str]
):
    messages = dummy_messages_from_image_url(image_urls)

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
    indirect=True,
)
async def test_completions_with_image(
    client: openai.AsyncOpenAI,
    model_name: str,
    image_urls: list[str],
):
    for image_url in image_urls:
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
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
    indirect=True,
)
async def test_completions_with_image_with_uuid(
    client: openai.AsyncOpenAI,
    model_name: str,
    image_urls: list[str],
):
    for image_url in image_urls:
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
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
                            "uuid": image_url,
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
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image.",
                        },
                        {"type": "image_url", "image_url": {}, "uuid": image_url},
                    ],
                },
            ],
            model=model_name,
        )
        assert chat_completion_with_empty_image.choices[0].message.content is not None
        assert isinstance(
            chat_completion_with_empty_image.choices[0].message.content, str
        )
        assert len(chat_completion_with_empty_image.choices[0].message.content) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_completions_with_empty_image_with_uuid_without_cache_hit(
    client: openai.AsyncOpenAI,
    model_name: str,
):
    with pytest.raises(openai.BadRequestError):
        _ = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {},
                            "uuid": "uuid_not_previously_seen",
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
    indirect=True,
)
async def test_completions_with_image_with_incorrect_uuid_format(
    client: openai.AsyncOpenAI,
    model_name: str,
    image_urls: list[str],
):
    for image_url in image_urls:
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
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
