# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
from PIL import Image

from tests.conftest import ImageTestAssets
from tests.utils import RemoteOpenAIServer
from vllm.multimodal.utils import encode_image_base64

# any model with a chat template defined in tokenizer_config should work here
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


@pytest.fixture(scope="module")
def default_server_args():
    return [
        # use half precision for speed and memory savings in CI environment
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
        "--limit-mm-per-prompt",
        "{\"image\": 1}",
    ]


@pytest.fixture(scope="module")
def server(default_server_args):
    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


def pil_image_to_data_url(image: Image.Image) -> str:
    image_base64 = encode_image_base64(image)
    return f"data:image/jpeg;base64,{image_base64}"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_completions_with_image(
    client: openai.AsyncOpenAI,
    model_name: str,
    image_assets: ImageTestAssets,
):
    # Test case: Single image embeds input
    image = image_assets[0].pil_image
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
                            "url": pil_image_to_data_url(image),
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
async def test_completions_with_image_with_uuid(
    client: openai.AsyncOpenAI,
    model_name: str,
    image_assets: ImageTestAssets,
):
    # Test case: Single image embeds input
    image = image_assets[0].pil_image
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
                            "url": pil_image_to_data_url(image),
                        },
                        "uuid": "1234"
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
async def test_completions_with_image_with_incorrect_uuid_format(
    client: openai.AsyncOpenAI,
    model_name: str,
    image_assets: ImageTestAssets,
):
    # Test case: Single image embeds input
    image = image_assets[0].pil_image
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
                            "url": pil_image_to_data_url(image),
                            "incorrect_uuid_key": "1234",
                        },
                        "also_incorrect_uuid_key": "1234",
                    },
                ],
            },
        ],
        model=model_name,
    )
    assert chat_completion.choices[0].message.content is not None
    assert isinstance(chat_completion.choices[0].message.content, str)
    assert len(chat_completion.choices[0].message.content) > 0
