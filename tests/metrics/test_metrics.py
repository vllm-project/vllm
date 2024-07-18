from typing import List

import pytest
from prometheus_client import REGISTRY

from vllm import EngineArgs, LLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

MODELS = [
    "facebook/opt-125m",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [128])
def test_metric_counter_prompt_tokens(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    vllm_model = vllm_runner(model,
                             dtype=dtype,
                             disable_log_stats=False,
                             gpu_memory_utilization=0.4)
    tokenizer = vllm_model.model.get_tokenizer()
    prompt_token_counts = [len(tokenizer.encode(p)) for p in example_prompts]
    # This test needs at least 2 prompts in a batch of different lengths to
    # verify their token count is correct despite padding.
    assert len(example_prompts) > 1, "at least 2 prompts are required"
    assert prompt_token_counts[0] != prompt_token_counts[1], (
        "prompts of different lengths are required")
    vllm_prompt_token_count = sum(prompt_token_counts)

    _ = vllm_model.generate_greedy(example_prompts, max_tokens)
    stat_logger = vllm_model.model.llm_engine.stat_logger
    metric_count = stat_logger.metrics.counter_prompt_tokens.labels(
        **stat_logger.labels)._value.get()

    assert vllm_prompt_token_count == metric_count, (
        f"prompt token count: {vllm_prompt_token_count!r}\n"
        f"metric: {metric_count!r}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [128])
def test_metric_counter_generation_tokens(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    vllm_model = vllm_runner(model,
                             dtype=dtype,
                             disable_log_stats=False,
                             gpu_memory_utilization=0.4)
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    tokenizer = vllm_model.model.get_tokenizer()
    stat_logger = vllm_model.model.llm_engine.stat_logger
    metric_count = stat_logger.metrics.counter_generation_tokens.labels(
        **stat_logger.labels)._value.get()
    vllm_generation_count = 0
    for i in range(len(example_prompts)):
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        prompt_ids = tokenizer.encode(example_prompts[i])
        # vllm_output_ids contains both prompt tokens and generation tokens.
        # We're interested only in the count of the generation tokens.
        vllm_generation_count += len(vllm_output_ids) - len(prompt_ids)

    assert vllm_generation_count == metric_count, (
        f"generation token count: {vllm_generation_count!r}\n"
        f"metric: {metric_count!r}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize(
    "served_model_name",
    [None, [], ["ModelName0"], ["ModelName0", "ModelName1", "ModelName2"]])
def test_metric_set_tag_model_name(vllm_runner, model: str, dtype: str,
                                   served_model_name: List[str]) -> None:
    vllm_model = vllm_runner(model,
                             dtype=dtype,
                             disable_log_stats=False,
                             gpu_memory_utilization=0.3,
                             served_model_name=served_model_name)
    stat_logger = vllm_model.model.llm_engine.stat_logger
    metrics_tag_content = stat_logger.labels["model_name"]

    del vllm_model

    if served_model_name is None or served_model_name == []:
        assert metrics_tag_content == model, (
            f"Metrics tag model_name is wrong! expect: {model!r}\n"
            f"actual: {metrics_tag_content!r}")
    else:
        assert metrics_tag_content == served_model_name[0], (
            f"Metrics tag model_name is wrong! expect: "
            f"{served_model_name[0]!r}\n"
            f"actual: {metrics_tag_content!r}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [4])
@pytest.mark.parametrize("disable_log_stats", [True, False])
@pytest.mark.asyncio
async def test_async_engine_log_metrics_regression(
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    disable_log_stats: bool,
) -> None:
    """
    Regression test ensuring async engine generates metrics
    when disable_log_stats=False
    (see: https://github.com/vllm-project/vllm/pull/4150#pullrequestreview-2008176678)
    """
    engine_args = AsyncEngineArgs(model=model,
                                  dtype=dtype,
                                  disable_log_stats=disable_log_stats)
    async_engine = AsyncLLMEngine.from_engine_args(engine_args)
    for i, prompt in enumerate(example_prompts):
        results = async_engine.generate(
            prompt,
            SamplingParams(max_tokens=max_tokens),
            f"request-id-{i}",
        )
        # Exhaust the async iterator to make the async engine work
        async for _ in results:
            pass

    assert_metrics(async_engine.engine, disable_log_stats,
                   len(example_prompts))


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [4])
@pytest.mark.parametrize("disable_log_stats", [True, False])
def test_engine_log_metrics_regression(
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    disable_log_stats: bool,
) -> None:
    engine_args = EngineArgs(model=model,
                             dtype=dtype,
                             disable_log_stats=disable_log_stats)
    engine = LLMEngine.from_engine_args(engine_args)
    for i, prompt in enumerate(example_prompts):
        engine.add_request(
            f"request-id-{i}",
            prompt,
            SamplingParams(max_tokens=max_tokens),
        )
    while engine.has_unfinished_requests():
        engine.step()

    assert_metrics(engine, disable_log_stats, len(example_prompts))


def assert_metrics(engine: LLMEngine, disable_log_stats: bool,
                   num_requests: int) -> None:
    if disable_log_stats:
        with pytest.raises(AttributeError):
            _ = engine.stat_logger
    else:
        assert (engine.stat_logger
                is not None), "engine.stat_logger should be set"
        # Ensure the count bucket of request-level histogram metrics matches
        # the number of requests as a simple sanity check to ensure metrics are
        # generated
        labels = {'model_name': engine.model_config.model}
        request_histogram_metrics = [
            "vllm:e2e_request_latency_seconds",
            "vllm:request_prompt_tokens",
            "vllm:request_generation_tokens",
            "vllm:request_params_best_of",
            "vllm:request_params_n",
        ]
        for metric_name in request_histogram_metrics:
            metric_value = REGISTRY.get_sample_value(f"{metric_name}_count",
                                                     labels)
            assert (
                metric_value == num_requests), "Metrics should be collected"
