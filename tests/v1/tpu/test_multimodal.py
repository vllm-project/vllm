# SPDX-License-Identifier: Apache-2.0
import base64
from time import time

import openai
import pytest
import requests

from vllm import envs

from ...utils import RemoteOpenAIServer

if not envs.VLLM_USE_V1:
    pytest.skip(
        "Skipping V1 tests. Rerun with `VLLM_USE_V1=1` to test.",
        allow_module_level=True,
    )


def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')

    return result


@pytest.mark.asyncio
async def test_encoder_compilation(monkeypatch):
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    # model_name = "bczhou/TinyLLaVA-1.5B"
    server_args = [
        "--max-model-len", "4096", "--max-num-seqs", "16",
        "--max-num-batched-tokens", "512"
    ]
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    # with tempfile.TemporaryDirectory() as temp_dir:
    #     monkeypatch.setenv("VLLM_XLA_CACHE_PATH", temp_dir)
    # # Server will pre-compile on first startup.
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client: openai.AsyncOpenAI = remote_server.get_async_client()
        image_base64 = encode_base64_content_from_url(image_url)
        req_body = dict(messages=[{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                },
            ],
        }],
                        model=model_name,
                        max_completion_tokens=48)
        s = time()
        chat_completion_from_base64 = await client.chat.completions.create(
            **req_body)
        run1 = time() - s
        print("RUN1", run1)
        s = time()
        chat_completion_from_base64 = await client.chat.completions.create(
            **req_body)
        run2 = time() - s
        print("RUN2", run2)
        result = chat_completion_from_base64.choices[0].message.content
        assert result
