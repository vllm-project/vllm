# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import ray
from prometheus_client import REGISTRY

import vllm.envs as envs
from vllm import EngineArgs, LLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.metrics import RayPrometheusStatLogger
from vllm.sampling_params import SamplingParams
from vllm.test_utils import MODEL_WEIGHTS_S3_BUCKET


@pytest.fixture(scope="function", autouse=True)
def use_v0_only(monkeypatch):
    """
    This module tests V0 internals, so set VLLM_USE_V1=0.
    """
    monkeypatch.setenv('VLLM_USE_V1', '0')


MODELS = [
    "distilbert/distilgpt2",
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
    with vllm_runner(model,
                     dtype=dtype,
                     disable_log_stats=False,
                     gpu_memory_utilization=0.4) as vllm_model:
        tokenizer = vllm_model.llm.get_tokenizer()
        prompt_token_counts = [
            len(tokenizer.encode(p)) for p in example_prompts
        ]
        # This test needs at least 2 prompts in a batch of different lengths to
        # verify their token count is correct despite padding.
        assert len(example_prompts) > 1, "at least 2 prompts are required"
        assert prompt_token_counts[0] != prompt_token_counts[1], (
            "prompts of different lengths are required")
        vllm_prompt_token_count = sum(prompt_token_counts)

        _ = vllm_model.generate_greedy(example_prompts, max_tokens)
        stat_logger = vllm_model.llm.llm_engine.stat_loggers['prometheus']
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
    with vllm_runner(model,
                     dtype=dtype,
                     disable_log_stats=False,
                     gpu_memory_utilization=0.4) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
        tokenizer = vllm_model.llm.get_tokenizer()
        stat_logger = vllm_model.llm.llm_engine.stat_loggers['prometheus']
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
@pytest.mark.parametrize("max_tokens", [128, 129])
@pytest.mark.parametrize("disable_async_output_proc", [True, False])
def test_metric_counter_generation_tokens_multi_step(
    vllm_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    disable_async_output_proc: bool,
) -> None:
    num_scheduler_steps = 8
    with vllm_runner(
            model,
            disable_log_stats=False,
            gpu_memory_utilization=0.4,
            num_scheduler_steps=num_scheduler_steps,
            disable_async_output_proc=disable_async_output_proc,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
        tokenizer = vllm_model.llm.get_tokenizer()
        stat_logger = vllm_model.llm.llm_engine.stat_loggers['prometheus']
        metric_count = stat_logger.metrics.counter_generation_tokens.labels(
            **stat_logger.labels)._value.get()
        vllm_generation_count = 0
        for i in range(len(example_prompts)):
            vllm_output_ids, vllm_output_str = vllm_outputs[i]
            prompt_ids = tokenizer.encode(example_prompts[i])
            # vllm_output_ids contains both prompt tokens and generation tokens.
            # We're interested only in the count of the generation tokens.
            vllm_generation_count += len(vllm_output_ids) - len(prompt_ids)

    # The multi-step scheduling will continue to execute forward even when
    # encountering EOS, leading to slightly imprecise metrics.
    assert abs(vllm_generation_count - metric_count) <\
        len(example_prompts) * num_scheduler_steps, \
        (f"generation token count: {vllm_generation_count!r}\n"
         f"metric: {metric_count!r}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize(
    "served_model_name",
    [None, [], ["ModelName0"], ["ModelName0", "ModelName1", "ModelName2"]])
def test_metric_set_tag_model_name(vllm_runner, model: str, dtype: str,
                                   served_model_name: list[str]) -> None:
    with vllm_runner(model,
                     dtype=dtype,
                     disable_log_stats=False,
                     gpu_memory_utilization=0.3,
                     served_model_name=served_model_name) as vllm_model:
        stat_logger = vllm_model.llm.llm_engine.stat_loggers['prometheus']
        metrics_tag_content = stat_logger.labels["model_name"]

    if envs.VLLM_CI_USE_S3:
        model = f"{MODEL_WEIGHTS_S3_BUCKET}/{model}"
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
    engine_args = AsyncEngineArgs(
        model=model,
        dtype=dtype,
        disable_log_stats=disable_log_stats,
    )
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

    assert_metrics(model, async_engine.engine, disable_log_stats,
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
    engine_args = EngineArgs(
        model=model,
        dtype=dtype,
        disable_log_stats=disable_log_stats,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    for i, prompt in enumerate(example_prompts):
        engine.add_request(
            f"request-id-{i}",
            prompt,
            SamplingParams(max_tokens=max_tokens),
        )
    while engine.has_unfinished_requests():
        engine.step()

    if envs.VLLM_CI_USE_S3:
        model = f"{MODEL_WEIGHTS_S3_BUCKET}/{model}"
    assert_metrics(model, engine, disable_log_stats, len(example_prompts))


def assert_metrics(model: str, engine: LLMEngine, disable_log_stats: bool,
                   num_requests: int) -> None:
    if disable_log_stats:
        with pytest.raises(AttributeError):
            _ = engine.stat_loggers
    else:
        assert (engine.stat_loggers
                is not None), "engine.stat_loggers should be set"
        # Ensure the count bucket of request-level histogram metrics matches
        # the number of requests as a simple sanity check to ensure metrics are
        # generated
        labels = {'model_name': model}
        request_histogram_metrics = [
            "vllm:e2e_request_latency_seconds",
            "vllm:request_prompt_tokens",
            "vllm:request_generation_tokens",
            "vllm:request_params_n",
            "vllm:request_params_max_tokens",
        ]
        for metric_name in request_histogram_metrics:
            metric_value = REGISTRY.get_sample_value(f"{metric_name}_count",
                                                     labels)
            assert (
                metric_value == num_requests), "Metrics should be collected"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [16])
def test_engine_log_metrics_ray(
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    # This test is quite weak - it only checks that we can use
    # RayPrometheusStatLogger without exceptions.
    # Checking whether the metrics are actually emitted is unfortunately
    # non-trivial.

    # We have to run in a Ray task for Ray metrics to be emitted correctly
    @ray.remote(num_gpus=1)
    def _inner():

        class _RayPrometheusStatLogger(RayPrometheusStatLogger):

            def __init__(self, *args, **kwargs):
                self._i = 0
                super().__init__(*args, **kwargs)

            def log(self, *args, **kwargs):
                self._i += 1
                return super().log(*args, **kwargs)

        engine_args = EngineArgs(
            model=model,
            dtype=dtype,
            disable_log_stats=False,
        )
        engine = LLMEngine.from_engine_args(engine_args)
        logger = _RayPrometheusStatLogger(
            local_interval=0.5,
            labels=dict(model_name=engine.model_config.served_model_name),
            vllm_config=engine.vllm_config)
        engine.add_logger("ray", logger)
        for i, prompt in enumerate(example_prompts):
            engine.add_request(
                f"request-id-{i}",
                prompt,
                SamplingParams(max_tokens=max_tokens),
            )
        while engine.has_unfinished_requests():
            engine.step()
        assert logger._i > 0, ".log must be called at least once"

    ray.get(_inner.remote())
