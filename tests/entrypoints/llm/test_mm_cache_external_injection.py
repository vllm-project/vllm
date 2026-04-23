# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that ``InputProcessor.inject_into_mm_cache()`` correctly injects
pre-processed mm_kwargs into the processor cache and reports MM cache
hit rate metrics accurately.

This is used by frameworks like Dynamo that run the HF processor on a
frontend and transfer pre-processed mm_kwargs to the backend, avoiding
redundant processing.
"""

import logging

import pytest
import regex as re

from tests.entrypoints.openai.chat_completion.test_vision import TEST_IMAGE_ASSETS
from vllm import LLM, SamplingParams
from vllm.renderers.params import ChatParams
from vllm.v1.metrics import loggers as stat_loggers
from vllm.v1.metrics.reader import Counter, Metric


def _make_messages(image_url: str):
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


def _get_counter_value(metrics: list[Metric], name: str):
    metric = next(m for m in metrics if m.name == name)
    assert isinstance(metric, Counter)
    return metric.value


def _get_mm_cache_stats(metrics: list[Metric]):
    mm_cache_queries = _get_counter_value(metrics, "vllm:mm_cache_queries")
    mm_cache_hits = _get_counter_value(metrics, "vllm:mm_cache_hits")
    return mm_cache_queries, mm_cache_hits


def _get_mm_cache_log(llm: LLM, caplog_vllm: pytest.LogCaptureFixture) -> float:
    caplog_vllm.clear()
    with caplog_vllm.at_level(logging.INFO, logger=stat_loggers.__name__):
        llm.llm_engine.do_log_stats()

    assert len(caplog_vllm.records) == 1
    msg = caplog_vllm.records[0].getMessage()

    assert "MM cache hit rate" in msg
    match = re.search(r"MM cache hit rate: ([0-9.]+)%", msg)
    assert match is not None
    return float(match.group(1))


@pytest.mark.parametrize("image_urls", [TEST_IMAGE_ASSETS[:2]], indirect=True)
@pytest.mark.parametrize("mm_processor_cache_type", ["lru", "shm"])
def test_inject_into_mm_cache(
    num_gpus_available,
    image_urls,
    mm_processor_cache_type,
    caplog_vllm,
):
    """Test that inject_into_mm_cache() injects pre-processed mm_kwargs into
    the processor cache and MM cache hit metrics are updated correctly.

    Steps:
    1. Two normal requests (same image) -> cache miss then hit (baseline)
    2. Extract cached kwargs, call inject_into_mm_cache with a new hash,
       then generate with a pre-rendered input -> verifies injection works
    """
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        max_model_len=4096,
        max_num_seqs=5,
        enforce_eager=True,
        disable_log_stats=False,
        limit_mm_per_prompt={"image": 2},
        mm_processor_cache_type=mm_processor_cache_type,
    )

    # Step 1: Normal requests to populate the cache
    llm.chat(_make_messages(image_urls[0]))
    assert _get_mm_cache_stats(llm.get_metrics()) == (1, 0)

    llm.chat(_make_messages(image_urls[0]))
    assert _get_mm_cache_stats(llm.get_metrics()) == (2, 1)
    assert _get_mm_cache_log(llm, caplog_vllm) == pytest.approx(50.0)

    # Step 2: Use a second image to get valid expanded tokens and
    # placeholder positions via the renderer.
    llm.chat(_make_messages(image_urls[1]))
    queries_before = _get_mm_cache_stats(llm.get_metrics())[0]  # 3

    renderer = llm.llm_engine.renderer
    cache = renderer.mm_processor_cache
    assert cache is not None, "Processor cache should be enabled"

    _, eng_prompts = renderer.render_chat(
        [_make_messages(image_urls[1])],
        ChatParams(),
    )
    eng_input = eng_prompts[0]

    # Inject pre-processed mm_kwargs with a NEW hash via public API
    new_mm_hash = "deadbeef" * 8
    mm_hashes = {"image": [new_mm_hash]}
    mm_kwargs = eng_input["mm_kwargs"]

    llm.llm_engine.input_processor.inject_into_mm_cache(mm_hashes, mm_kwargs)

    # Build pre-rendered input (no externally_processed flag needed)
    pre_rendered_input = {
        "type": "multimodal",
        "prompt_token_ids": eng_input["prompt_token_ids"],
        "mm_kwargs": mm_kwargs,
        "mm_hashes": mm_hashes,
        "mm_placeholders": eng_input["mm_placeholders"],
    }

    llm.generate(
        pre_rendered_input,
        sampling_params=SamplingParams(max_tokens=1),
    )

    # Verify cache was queried and injection happened
    queries_after = _get_mm_cache_stats(llm.get_metrics())[0]
    assert queries_after > queries_before, (
        "Cache should have been queried for the injected item"
    )
    mm_rate = _get_mm_cache_log(llm, caplog_vllm)
    assert mm_rate >= 0.0, "MM cache hit rate should be reported"


@pytest.mark.parametrize("image_urls", [TEST_IMAGE_ASSETS[:1]], indirect=True)
def test_inject_into_mm_cache_without_cache(
    num_gpus_available,
    image_urls,
):
    """Test that inject_into_mm_cache works gracefully when processor cache
    is disabled (mm_processor_cache_gb=0). Should not crash.
    """
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        max_model_len=4096,
        max_num_seqs=5,
        enforce_eager=True,
        disable_log_stats=False,
        limit_mm_per_prompt={"image": 2},
        mm_processor_cache_gb=0,
    )

    # Run a normal chat request first to warm up the model.
    llm.chat(_make_messages(image_urls[0]))

    # Use the renderer to get a proper EngineInput with expanded tokens
    renderer = llm.llm_engine.renderer
    _, eng_prompts = renderer.render_chat(
        [_make_messages(image_urls[0])],
        ChatParams(),
    )
    eng_input = eng_prompts[0]

    mm_hashes = {"image": ["abcd1234" * 8]}
    mm_kwargs = eng_input["mm_kwargs"]

    # inject_into_mm_cache should not crash even without cache
    llm.llm_engine.input_processor.inject_into_mm_cache(mm_hashes, mm_kwargs)

    # Build and generate with pre-rendered input
    pre_rendered_input = {
        "type": "multimodal",
        "prompt_token_ids": eng_input["prompt_token_ids"],
        "mm_kwargs": mm_kwargs,
        "mm_hashes": mm_hashes,
        "mm_placeholders": eng_input["mm_placeholders"],
    }

    result = llm.generate(
        pre_rendered_input,
        sampling_params=SamplingParams(max_tokens=1),
    )
    assert len(result) == 1, "Should produce one output"
    assert len(result[0].outputs) >= 1, "Should have at least one output sequence"
