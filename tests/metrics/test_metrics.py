import time
from typing import List

import pytest
import ray
from prometheus_client import REGISTRY

from vllm import EngineArgs, LLMEngine
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.metrics import RayPrometheusStatLogger
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
    with vllm_runner(model,
                     dtype=dtype,
                     disable_log_stats=False,
                     gpu_memory_utilization=0.4) as vllm_model:
        tokenizer = vllm_model.model.get_tokenizer()
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
        stat_logger = vllm_model.model.llm_engine.stat_loggers['prometheus']
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
        tokenizer = vllm_model.model.get_tokenizer()
        stat_logger = vllm_model.model.llm_engine.stat_loggers['prometheus']
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
        tokenizer = vllm_model.model.get_tokenizer()
        stat_logger = vllm_model.model.llm_engine.stat_loggers['prometheus']
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
                                   served_model_name: List[str]) -> None:
    with vllm_runner(model,
                     dtype=dtype,
                     disable_log_stats=False,
                     gpu_memory_utilization=0.3,
                     served_model_name=served_model_name) as vllm_model:
        stat_logger = vllm_model.model.llm_engine.stat_loggers['prometheus']
        metrics_tag_content = stat_logger.labels["model_name"]

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


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [10])
def test_metric_spec_decode(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    k = 5

    with vllm_runner(
            model,
            dtype=dtype,
            disable_log_stats=False,
            gpu_memory_utilization=0.4,
            speculative_model=model,
            num_speculative_tokens=k,
    ) as vllm_model:

        # Force log interval to be 0 to catch all metrics.
        stat_logger = vllm_model.model.llm_engine.stat_loggers['prometheus']
        stat_logger.local_interval = 0

        # Note that the purpose of this test is to verify spec decode
        # metrics instead of functional correctness, so the expected values
        # are intended to be loose.
        metric_name_to_expected_fn = {
            "gauge_spec_decode_draft_acceptance_rate": lambda v: 0 <= v <= 1,
            "gauge_spec_decode_efficiency": lambda v: 0 <= v <= 1,
            "counter_spec_decode_num_accepted_tokens": lambda v: 0 <= v <= k,
            "counter_spec_decode_num_draft_tokens": lambda v: v == k,
            "counter_spec_decode_num_emitted_tokens":
            lambda v: 0 <= v <= k + 1,
        }

        # Use one request to better inspect the metrics.
        prompts = example_prompts[:1]

        _ = vllm_model.generate_greedy(prompts, max_tokens)
        for metric_name, is_expected in metric_name_to_expected_fn.items():
            metric_val = getattr(
                stat_logger.metrics,
                metric_name).labels(**stat_logger.labels)._value.get()
            assert is_expected(metric_val), (
                f"the value of metric {metric_name} ({metric_val}) "
                "does not meet expectation")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [10])
@pytest.mark.parametrize("log_interval", [1, 3, 5, 7])
def test_metric_spec_decode_interval(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    log_interval: int,
) -> None:
    k = 5

    engine_args = EngineArgs(model=model,
                             dtype=dtype,
                             disable_log_stats=False,
                             gpu_memory_utilization=0.4,
                             speculative_model=model,
                             num_speculative_tokens=k,
                             enforce_eager=True)

    engine = LLMEngine.from_engine_args(engine_args)

    try:

        engine.add_request(
            "request-id-0",
            example_prompts[0],
            SamplingParams(max_tokens=max_tokens),
        )

        # set log internal
        stat_logger = engine.stat_loggers['prometheus']
        stat_logger.local_interval = log_interval

        # prefill
        engine.step()

        # wait for 5 seconds to ensure that spec decode metrics
        # get triggered in first decode step
        time.sleep(5)

        # first decode step should trigger async collection of metrics
        engine.step()

        # wait one second to allow H2D transfer to finish
        time.sleep(1)

        # second decode step should now be able to collect the spec
        # decode stats and the request should also be finished
        engine.step()

        # must have finisehd now
        assert not engine.has_unfinished_requests()

        # wait to ensure logging occurs
        time.sleep(log_interval)

        # force logging
        engine.step()

        # Note that the purpose of this test is to verify spec decode
        # metrics instead of functional correctness, so the expected values
        # are intended to be loose.
        metric_name_to_expected_fn = {
            "gauge_spec_decode_draft_acceptance_rate": lambda v: 0 <= v <= 1,
            "gauge_spec_decode_efficiency": lambda v: 0 <= v <= 1,
            "counter_spec_decode_num_accepted_tokens": lambda v: 0 <= v <= k,
            "counter_spec_decode_num_draft_tokens": lambda v: v == k,
            "counter_spec_decode_num_emitted_tokens":
            lambda v: 0 <= v <= k + 1,
        }

        for metric_name, is_expected in metric_name_to_expected_fn.items():
            metric_val = getattr(
                stat_logger.metrics,
                metric_name).labels(**stat_logger.labels)._value.get()
            assert is_expected(metric_val), (
                f"the value of metric {metric_name} ({metric_val}) "
                "does not meet expectation")

    finally:
        del engine
        cleanup_dist_env_and_memory()


def assert_metrics(engine: LLMEngine, disable_log_stats: bool,
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
        labels = {'model_name': engine.model_config.model}
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
            max_model_len=engine.model_config.max_model_len)
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
