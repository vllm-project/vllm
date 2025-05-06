# SPDX-License-Identifier: Apache-2.0

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
@pytest.mark.parametrize("model_name", ["llava-hf/llava-1.5-7b-hf"])
async def test_basic_vision(model_name: str, base64_encoded_image: dict[str,
                                                                        str]):

    pytest.skip("Skip this test until it's fixed.")

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

    server_args = [
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "16",
        "--gpu-memory-utilization",
        "0.95",
        "--trust-remote-code",
        "--max-num-batched-tokens",
        "576",
        # NOTE: max-num-batched-tokens>=mm_item_size
        "--disable_chunked_mm_input",
        "--chat-template",
        "examples/template_llava.jinja"
    ]

    # Server will pre-compile on first startup (takes a long time).
    with RemoteOpenAIServer(model_name, server_args,
                            max_wait_seconds=600) as remote_server:
        client: openai.AsyncOpenAI = remote_server.get_async_client()

        # Other requests now should be much faster
        for image_url in TEST_IMAGE_URLS:
            image_base64 = base64_encoded_image[image_url]
            chat_completion_from_base64 = await client.chat.completions\
                .create(
                model=model_name,
                messages=whats_in_this_image_msg(image_base64),
                max_completion_tokens=24,
                temperature=0.0)
            result = chat_completion_from_base64
            assert result
            choice = result.choices[0]
            assert choice.finish_reason == "length"

            message = choice.message
            message = result.choices[0].message
            assert message.content is not None and len(message.content) >= 10
            assert message.role == "assistant"
