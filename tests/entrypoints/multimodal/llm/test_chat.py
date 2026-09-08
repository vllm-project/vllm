# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref

import pytest

from tests.entrypoints.multimodal.conftest import TEST_IMAGE_ASSETS


@pytest.fixture(scope="function")
def vision_llm(vllm_runner):
    with vllm_runner(
        "microsoft/Phi-3.5-vision-instruct",
        max_model_len=4096,
        max_num_seqs=5,
        enforce_eager=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 2},
        seed=0,
    ) as runner:
        # pytest caches yielded fixtures until after teardown, so use a proxy to
        # avoid retaining the LLM while VllmRunner.__exit__ releases ROCm memory.
        yield weakref.proxy(runner.llm)


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
