# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import ray

from vllm.config.model import ModelDType
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncEngineArgs, AsyncLLM
from vllm.v1.metrics.ray_wrappers import RayPrometheusMetric, RayPrometheusStatLogger

MODELS = [
    "distilbert/distilgpt2",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [16])
def test_engine_log_metrics_ray(
    example_prompts,
    model: str,
    dtype: ModelDType,
    max_tokens: int,
) -> None:
    """Simple smoke test, verifying this can be used without exceptions.
    Need to start a Ray cluster in order to verify outputs."""

    @ray.remote(num_gpus=1)
    class EngineTestActor:
        async def run(self):
            engine_args = AsyncEngineArgs(
                model=model, dtype=dtype, disable_log_stats=False, enforce_eager=True
            )

            engine = AsyncLLM.from_engine_args(
                engine_args, stat_loggers=[RayPrometheusStatLogger]
            )

            for i, prompt in enumerate(example_prompts):
                results = engine.generate(
                    request_id=f"request-id-{i}",
                    prompt=prompt,
                    sampling_params=SamplingParams(max_tokens=max_tokens),
                )

                async for _ in results:
                    pass

    # Create the actor and call the async method
    actor = EngineTestActor.remote()  # type: ignore[attr-defined]
    ray.get(actor.run.remote())


def test_sanitized_opentelemetry_name():
    """Test the metric name sanitization logic for Ray."""

    # Only a-z, A-Z, 0-9, _, test valid characters are preserved
    valid_name = "valid_metric_123_abcDEF"
    assert (
        RayPrometheusMetric._get_sanitized_opentelemetry_name(valid_name) == valid_name
    )

    # Test dash, dot, are replaced
    name_with_dash_dot = "metric-name.test"
    expected = "metric_name_test"
    assert (
        RayPrometheusMetric._get_sanitized_opentelemetry_name(name_with_dash_dot)
        == expected
    )

    # Test colon is replaced with underscore
    name_with_colon = "metric:name"
    expected = "metric_name"
    assert (
        RayPrometheusMetric._get_sanitized_opentelemetry_name(name_with_colon)
        == expected
    )

    # Test multiple invalid characters are replaced
    name_with_invalid = "metric:name@with#special%chars"
    expected = "metric_name_with_special_chars"
    assert (
        RayPrometheusMetric._get_sanitized_opentelemetry_name(name_with_invalid)
        == expected
    )

    # Test mixed valid and invalid characters
    complex_name = "vllm:engine_stats/time.latency_ms-99p"
    expected = "vllm_engine_stats_time_latency_ms_99p"
    assert (
        RayPrometheusMetric._get_sanitized_opentelemetry_name(complex_name) == expected
    )

    # Test empty string
    assert RayPrometheusMetric._get_sanitized_opentelemetry_name("") == ""
