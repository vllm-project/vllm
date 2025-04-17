# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm import LLM
from vllm.engine.llm_engine import LLMEngine as V0LLMEngine
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine

from ..openai.test_vision import TEST_IMAGE_URLS


def _make_messages(image_url: str) -> list[ChatCompletionMessageParam]:
    return [{
        "role":
        "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                },
            },
        ],
    }]


@pytest.mark.parametrize("image_urls",
                         [[TEST_IMAGE_URLS[0], TEST_IMAGE_URLS[1]]])
@pytest.mark.parametrize("use_v1", [True, False])
def test_mm_cache_stats(
    image_urls: list[str],
    use_v1: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1" if use_v1 else "0")

        llm = LLM(
            model="HuggingFaceTB/SmolVLM-256M-Instruct",
            max_model_len=4096,
            max_num_seqs=5,
            enforce_eager=True,
            limit_mm_per_prompt={"image": 2},
        )
        engine = llm.llm_engine
        if isinstance(engine, V0LLMEngine):
            mm_registry = engine.input_preprocessor.mm_registry
        elif isinstance(engine, V1LLMEngine):
            mm_registry = engine.processor.mm_registry

        # In case the previous test failed, we still need to reset the cache
        # (which is shared across tests)
        engine.reset_mm_cache()
        mm_registry.make_processor_cache_stats()

        llm.chat(_make_messages(image_urls[0]))

        cache_stats = mm_registry.make_processor_cache_stats()
        assert cache_stats.size_items == 1

        llm.chat(_make_messages(image_urls[1]))

        cache_stats = mm_registry.make_processor_cache_stats()
        assert cache_stats.size_items == 2

        llm.chat(_make_messages(image_urls[0]))

        cache_stats = mm_registry.make_processor_cache_stats()
        assert cache_stats.size_items == 2

        engine.reset_mm_cache()

        cache_stats = mm_registry.make_processor_cache_stats()
        assert cache_stats.size_items == 0
        assert cache_stats.reset is True
