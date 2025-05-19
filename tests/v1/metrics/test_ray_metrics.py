# SPDX-License-Identifier: Apache-2.0
import pytest
import ray

from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncEngineArgs, AsyncLLM
from vllm.v1.metrics.ray_wrappers import RayPrometheusStatLogger


@pytest.fixture(scope="function", autouse=True)
def use_v1_only(monkeypatch):
    """
    The change relies on V1 APIs, so set VLLM_USE_V1=1.
    """
    monkeypatch.setenv('VLLM_USE_V1', '1')


MODELS = [
    "distilbert/distilgpt2",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [16])
def test_engine_log_metrics_ray(
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    """ Simple smoke test, verifying this can be used without exceptions.
    Need to start a Ray cluster in order to verify outputs."""

    @ray.remote(num_gpus=1)
    class EngineTestActor:

        async def run(self):
            engine_args = AsyncEngineArgs(
                model=model,
                dtype=dtype,
                disable_log_stats=False,
            )

            engine = AsyncLLM.from_engine_args(
                engine_args, stat_loggers=[RayPrometheusStatLogger])

            for i, prompt in enumerate(example_prompts):
                engine.generate(
                    request_id=f"request-id-{i}",
                    prompt=prompt,
                    sampling_params=SamplingParams(max_tokens=max_tokens),
                )

    # Create the actor and call the async method
    actor = EngineTestActor.remote()  # type: ignore[attr-defined]
    ray.get(actor.run.remote())
