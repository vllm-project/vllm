# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import LLM
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from ..openai.test_vision import TEST_IMAGE_ASSETS


def _make_messages(image_url: str) -> list[ChatCompletionMessageParam]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ]


@pytest.mark.parametrize("image_urls", [TEST_IMAGE_ASSETS[:2]])
def test_mm_cache_stats(image_urls):
    llm = LLM(
        model="HuggingFaceTB/SmolVLM-256M-Instruct",
        max_model_len=4096,
        max_num_seqs=5,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 2},
    )
    engine = llm.llm_engine

    # In case the previous test failed, we still need to reset the cache
    # (which is shared across tests)
    engine.reset_mm_cache()
    engine.processor.stat_cache()

    llm.chat(_make_messages(image_urls[0]))

    cache_stats = engine.processor.stat_cache()
    assert cache_stats and cache_stats.queries == 1

    llm.chat(_make_messages(image_urls[1]))

    cache_stats = engine.processor.stat_cache()
    assert cache_stats and cache_stats.queries == 2

    llm.chat(_make_messages(image_urls[0]))

    cache_stats = engine.processor.stat_cache()
    assert cache_stats and cache_stats.queries == 2

    engine.reset_mm_cache()

    cache_stats = engine.processor.stat_cache()
    assert cache_stats and cache_stats.queries == 0
