# SPDX-License-Identifier: Apache-2.0
import tempfile
from time import time

import openai
import pytest

from vllm import envs
from vllm.multimodal.utils import encode_image_base64, fetch_image
from vllm.platforms import current_platform

from ...entrypoints.openai.test_vision import TEST_IMAGE_URLS
from ...utils import RemoteOpenAIServer

if not envs.VLLM_USE_V1:
    pytest.skip(
        "Skipping V1 tests. Rerun with `VLLM_USE_V1=1` to test.",
        allow_module_level=True,
    )


@pytest.fixture(scope="session")
def base64_encoded_image() -> dict[str, str]:
    return {
        image_url: encode_image_base64(fetch_image(image_url))
        for image_url in TEST_IMAGE_URLS
    }


@pytest.mark.asyncio
@pytest.mark.skipif(not current_platform.is_tpu(),
                    reason="This test needs a TPU")
async def test_encoder_compilation(monkeypatch,
                                   base64_encoded_image: dict[str, str]):

    def whats_in_this_image_msg(b64):
        return [{
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
                        "url": f"data:image/jpeg;base64,{b64}"
                    },
                },
            ],
        }]

    model_name = "llava-hf/llava-1.5-7b-hf"
    server_args = [
        "--max-model-len", "4096", "--max-num-seqs", "16",
        "--trust-remote-code", "--max-num-batched-tokens", "128",
        "--chat-template", "examples/template_llava.jinja"
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        monkeypatch.setenv("VLLM_XLA_CACHE_PATH", temp_dir)
        # Server will pre-compile on first startup (takes a long time).
        with RemoteOpenAIServer(model_name, server_args,
                                max_wait_seconds=600) as remote_server:
            client: openai.AsyncOpenAI = remote_server.get_async_client()
            image_base64 = base64_encoded_image[TEST_IMAGE_URLS[0]]

            s = time()
            chat_completion_from_base64 = await client.chat.completions.create(
                model=model_name,
                messages=whats_in_this_image_msg(image_base64),
                max_completion_tokens=16,
                temperature=0.0)
            # TODO first run is still slow due to pre/post
            run1 = time() - s

            # Other requests now should be much faster
            for image_url in TEST_IMAGE_URLS:
                image_base64 = base64_encoded_image[image_url]
                s = time()
                chat_completion_from_base64 = await client.chat.completions\
                    .create(
                    model=model_name,
                    messages=whats_in_this_image_msg(image_base64),
                    max_completion_tokens=24,
                    temperature=0.0)
                run_i = time() - s
                result = chat_completion_from_base64.choices[0].message.content
                assert result
                assert run1 * 0.1 > run_i
