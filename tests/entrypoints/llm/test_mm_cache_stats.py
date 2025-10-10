# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

import pytest
import regex as re

from vllm import LLM
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.v1.metrics import loggers as stat_loggers
from vllm.v1.metrics.reader import Counter, Metric

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
def test_mm_cache_stats(
    num_gpus_available,
    image_urls,
    mm_processor_cache_type,
    caplog_vllm,
):
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        max_model_len=4096,
        max_num_seqs=5,
        enforce_eager=True,
        mm_processor_cache_type=mm_processor_cache_type,
        disable_log_stats=False,
        limit_mm_per_prompt={"image": 2},
    )

    llm.chat(_make_messages(image_urls[0]))
    assert _get_mm_cache_stats(llm.get_metrics()) == (1, 0)
    assert _get_mm_cache_log(llm, caplog_vllm) == pytest.approx(0.0)

    llm.chat(_make_messages(image_urls[1]))
    assert _get_mm_cache_stats(llm.get_metrics()) == (2, 0)
    assert _get_mm_cache_log(llm, caplog_vllm) == pytest.approx(0.0)

    llm.chat(_make_messages(image_urls[0]))
    assert _get_mm_cache_stats(llm.get_metrics()) == (3, 1)
    assert _get_mm_cache_log(llm, caplog_vllm) == pytest.approx(33.3)

    # NOTE: This only resets hit rate stats in CachingMetrics
    # The raw queries and hits counts remain unaffected
    llm.reset_mm_cache()

    llm.chat(_make_messages(image_urls[0]))
    assert _get_mm_cache_stats(llm.get_metrics()) == (4, 1)
    assert _get_mm_cache_log(llm, caplog_vllm) == pytest.approx(0.0)

    llm.chat(_make_messages(image_urls[1]))
    assert _get_mm_cache_stats(llm.get_metrics()) == (5, 1)
    assert _get_mm_cache_log(llm, caplog_vllm) == pytest.approx(0.0)
