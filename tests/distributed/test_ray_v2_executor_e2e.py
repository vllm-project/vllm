# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Orchestration-level integration tests for RayExecutorV2.
"""

import gc
import os

import ray

from tests.distributed.ray_v2_utils import enable_ray_v2_backend  # noqa: F401
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor

MODEL = "facebook/opt-125m"


def _get_env_var(worker, name):
    """Called on RayWorkerProc workers via collective_rpc."""
    return os.environ.get(name)


@ray.remote(num_cpus=0)
class AsyncLLMActor:
    async def __init__(self):
        self.engine: AsyncLLM

    async def start(self, pg, bundle_indices=None, ray_runtime_env=None):
        os.environ["VLLM_USE_RAY_V2_EXECUTOR_BACKEND"] = "1"
        # VLLM_ALLOW_INSECURE_SERIALIZATION is needed so collective_rpc can
        # pickle _get_env_var over the AsyncLLM -> EngineCore ZMQ boundary.
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        if bundle_indices is not None:
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = bundle_indices
        else:
            os.environ.pop("VLLM_RAY_BUNDLE_INDICES", None)

        engine_args = AsyncEngineArgs(
            model=MODEL,
            tensor_parallel_size=2,
            distributed_executor_backend="ray",
            enforce_eager=True,
            max_model_len=256,
            gpu_memory_utilization=0.8,
        )
        vllm_config = engine_args.create_engine_config()
        vllm_config.parallel_config.placement_group = pg
        if ray_runtime_env is not None:
            vllm_config.parallel_config.ray_runtime_env = ray_runtime_env

        executor_class = Executor.get_class(vllm_config)
        self.engine = AsyncLLM(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=False,
            log_requests=False,
        )

    async def generate(self, prompt):
        params = SamplingParams(max_tokens=16)
        result = None
        async for output in self.engine.generate(
            prompt, params, request_id="test_request_id"
        ):
            result = output
        assert result is not None
        return result.outputs[0].text

    async def get_worker_env(self, name):
        results = await self.engine.collective_rpc(
            _get_env_var,
            timeout=10,
            args=(name,),
        )
        return results

    async def shutdown(self):
        if engine := getattr(self, "engine", None):
            engine.shutdown()
            del self.engine
            gc.collect()


def test_multi_replicas():
    """Two actors each run AsyncLLM with TP=2 via RayExecutorV2.

    Actor 1 starts first and claims 80% of GPU memory.  Without lazy
    RayWorkerProc init, actor 2 lands on the *same* two GPUs and fails
    because there is not enough free memory.
    """
    ray.init(ignore_reinit_error=True)

    pg1 = ray.util.placement_group([{"GPU": 1, "CPU": 1}] * 2, strategy="PACK")
    pg2 = ray.util.placement_group([{"GPU": 1, "CPU": 1}] * 2, strategy="PACK")
    ray.get([pg1.ready(), pg2.ready()])

    actor1 = AsyncLLMActor.remote()  # type: ignore[attr-defined]
    actor2 = AsyncLLMActor.remote()  # type: ignore[attr-defined]

    ray.get(actor1.start.remote(pg1))
    ray.get(actor2.start.remote(pg2))

    out1, out2 = ray.get(
        [
            actor1.generate.remote("Hello world"),
            actor2.generate.remote("Hello world"),
        ]
    )
    assert len(out1) > 0
    assert len(out2) > 0


def test_multi_replicas_with_bundle_indices():
    """Two actors share one 4-GPU placement group with out-of-order
    bundle indices: actor 1 gets bundles [2,1], actor 2 gets [0,3].
    """
    ray.init(ignore_reinit_error=True)

    pg = ray.util.placement_group([{"GPU": 1, "CPU": 1}] * 4, strategy="PACK")
    ray.get(pg.ready())

    actor1 = AsyncLLMActor.remote()  # type: ignore[attr-defined]
    actor2 = AsyncLLMActor.remote()  # type: ignore[attr-defined]

    ray.get(actor1.start.remote(pg, bundle_indices="2,1"))
    ray.get(actor2.start.remote(pg, bundle_indices="0,3"))

    out1, out2 = ray.get(
        [
            actor1.generate.remote("Hello world"),
            actor2.generate.remote("Hello world"),
        ]
    )
    assert len(out1) > 0
    assert len(out2) > 0


def test_env_var_and_runtime_env_propagation():
    """
    Verify env vars (NCCL_, HF_) and parallel_config.ray_runtime_env
    propagate to RayWorkerProc actors.
    """
    sentinel_vars = {
        "NCCL_DEBUG": "INFO",
        "HF_TOKEN": "test_sentinel_token",
    }
    for k, v in sentinel_vars.items():
        os.environ[k] = v

    try:
        ray.init(ignore_reinit_error=True)

        pg = ray.util.placement_group([{"GPU": 1, "CPU": 1}] * 2, strategy="PACK")
        ray.get(pg.ready())

        ray_runtime_env = {
            "env_vars": {"RAY_RUNTIME_ENV_MARKER": "ray_runtime_env"},
        }

        actor = AsyncLLMActor.remote()  # type: ignore[attr-defined]
        ray.get(actor.start.remote(pg, ray_runtime_env=ray_runtime_env))

        for name, expected in sentinel_vars.items():
            results = ray.get(actor.get_worker_env.remote(name))
            for val in results:
                assert val == expected

        results = ray.get(actor.get_worker_env.remote("RAY_RUNTIME_ENV_MARKER"))
        for val in results:
            assert val == "ray_runtime_env"

    finally:
        for k in sentinel_vars:
            os.environ.pop(k, None)
