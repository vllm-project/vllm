# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref

import pytest

from tests.entrypoints.multimodal.conftest import TEST_IMAGE_ASSETS
from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory


@pytest.fixture(scope="function")
def vision_llm():
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        max_model_len=4096,
        max_num_seqs=5,
        enforce_eager=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 2},
        seed=0,
    )

    yield weakref.proxy(llm)

    del llm

    cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    "image_urls", [[TEST_IMAGE_ASSETS[0], TEST_IMAGE_ASSETS[1]]], indirect=True
)
def test_chat_multi_image(vision_llm, image_urls: list[str]):
    messages = [
        {
            "role": "user",
            "content": [
                *(
                    {"type": "image_url", "image_url": {"url": image_url}}
                    for image_url in image_urls
                ),
                {"type": "text", "text": "What's in this image?"},
            ],
        }
    ]
    outputs = vision_llm.chat(messages)
    assert len(outputs) >= 0
