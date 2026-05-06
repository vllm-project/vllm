# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer
from vllm.multimodal.utils import encode_image_url

# Use a small vision model for testing
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MAXIMUM_IMAGES = 2
# Test different image extensions (JPG/PNG) and formats (gray/RGB/RGBA)
TEST_IMAGE_ASSETS = [
    "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    "Grayscale_8bits_palette_sample_image.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/Grayscale_8bits_palette_sample_image.png",
    "1280px-Venn_diagram_rgb.svg.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/1280px-Venn_diagram_rgb.svg.png",
    "RGBA_comp.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/RGBA_comp.png",
]


@pytest.fixture(scope="module")
def default_image_server_args():
    return [
        "--enforce-eager",
        "--max-model-len",
        "6000",
        "--max-num-seqs",
        "128",
        "--limit-mm-per-prompt",
        json.dumps({"image": MAXIMUM_IMAGES}),
    ]


@pytest.fixture(scope="module")
def image_server(default_image_server_args):
    with RemoteOpenAIServer(
        MODEL_NAME,
        default_image_server_args,
        env_dict={"VLLM_ENABLE_RESPONSES_API_STORE": "1"},
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(image_server):
    async with image_server.get_async_client() as async_client:
        yield async_client


@pytest.fixture(scope="session")
def url_encoded_image(local_asset_server) -> dict[str, str]:
    return {
        image_url: encode_image_url(local_asset_server.get_image_asset(image_url))
        for image_url in TEST_IMAGE_ASSETS
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("image_url", TEST_IMAGE_ASSETS, indirect=True)
async def test_single_chat_session_image(
    client: openai.AsyncOpenAI, model_name: str, image_url: str
):
    content_text = "What's in this image?"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": image_url,
                    "detail": "auto",
                },
                {"type": "input_text", "text": content_text},
            ],
        }
    ]

    # test image url
    response = await client.responses.create(
        model=model_name,
        input=messages,
    )
    assert len(response.output_text) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("raw_image_url", TEST_IMAGE_ASSETS)
async def test_single_chat_session_image_base64encoded(
    client: openai.AsyncOpenAI,
    model_name: str,
    raw_image_url: str,
    url_encoded_image: dict[str, str],
):
    content_text = "What's in this image?"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": url_encoded_image[raw_image_url],
                    "detail": "auto",
                },
                {"type": "input_text", "text": content_text},
            ],
        }
    ]
    # test image base64
    response = await client.responses.create(
        model=model_name,
        input=messages,
    )
    assert len(response.output_text) > 0


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
    messages = [
        {
            "role": "user",
            "content": [
                *(
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": "auto",
                    }
                    for image_url in image_urls
                ),
                {"type": "input_text", "text": "What's in this image?"},
            ],
        }
    ]

    if len(image_urls) > MAXIMUM_IMAGES:
        with pytest.raises(openai.BadRequestError):  # test multi-image input
            await client.responses.create(
                model=model_name,
                input=messages,
            )
        # the server should still work afterwards
        response = await client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "user",
                    "content": "What's the weather like in Paris today?",
                }
            ],
        )
        assert len(response.output_text) > 0
    else:
        response = await client.responses.create(
            model=model_name,
            input=messages,
        )
        assert len(response.output_text) > 0
