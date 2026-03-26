# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Orchestration-level integration tests for RayExecutorV2.
"""

import gc
import os
import pathlib

import ray

from tests.distributed.ray_v2_utils import enable_ray_v2_backend  # noqa: F401

MODEL = "facebook/opt-125m"


def _get_env_var(worker, name):
    return os.environ.get(name)


def _ray_init():
    """Start Ray with the project root on workers' PYTHONPATH.

    Without this, workers cannot unpickle actor classes defined in the
    ``tests`` package, causing FunctionActorManager to fall back to
    TemporaryActor which drops async method signatures."""
    project_root = str(pathlib.Path(__file__).resolve().parents[2])
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PYTHONPATH": project_root}},
    )


class _AsyncLLMActor:
    def start(self, pg, bundle_indices=None, ray_runtime_env=None):
        os.environ["VLLM_USE_RAY_V2_EXECUTOR_BACKEND"] = "1"
        # Needed so collective_rpc can pickle _get_env_var over the
        # AsyncLLM -> EngineCore ZMQ boundary.
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        if bundle_indices is not None:
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = bundle_indices
        else:
            os.environ.pop("VLLM_RAY_BUNDLE_INDICES", None)

        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.v1.executor.abstract import Executor

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
        from vllm.sampling_params import SamplingParams

        params = SamplingParams(max_tokens=16)
        result = None
        async for output in self.engine.generate(
            prompt, params, request_id="test_request_id"
        ):
            result = output
        assert result is not None
        return result.outputs[0].text

    async def generate_and_get_worker_envs(self, prompt, env_names):
        from vllm.sampling_params import SamplingParams

        params = SamplingParams(max_tokens=16)
        result = None
        async for output in self.engine.generate(
            prompt, params, request_id="test_request_id"
        ):
            result = output
        assert result is not None
        text = result.outputs[0].text

        env_results = {}
        for name in env_names:
            vals = await self.engine.collective_rpc(
                _get_env_var, timeout=10, args=(name,)
            )
            env_results[name] = vals
        return text, env_results

    def shutdown(self):
        if engine := getattr(self, "engine", None):
            engine.shutdown()
            del self.engine
            gc.collect()


AsyncLLMActor = ray.remote(num_cpus=0, max_concurrency=1)(_AsyncLLMActor)


def test_multi_replicas():
    _ray_init()

    pg1 = ray.util.placement_group([{"GPU": 1, "CPU": 1}] * 2, strategy="PACK")
    pg2 = ray.util.placement_group([{"GPU": 1, "CPU": 1}] * 2, strategy="PACK")
    ray.get([pg1.ready(), pg2.ready()])

    actor1 = AsyncLLMActor.remote()
    actor2 = AsyncLLMActor.remote()

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
    _ray_init()

    pg = ray.util.placement_group([{"GPU": 1, "CPU": 1}] * 4, strategy="PACK")
    ray.get(pg.ready())

    actor1 = AsyncLLMActor.remote()
    actor2 = AsyncLLMActor.remote()

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
        _ray_init()

        pg = ray.util.placement_group([{"GPU": 1, "CPU": 1}] * 2, strategy="PACK")
        ray.get(pg.ready())

        ray_runtime_env = {
            "env_vars": {"RAY_RUNTIME_ENV_TEST": "ray_runtime_env"},
        }

        actor = AsyncLLMActor.remote()
        ray.get(actor.start.remote(pg, ray_runtime_env=ray_runtime_env))

        all_env_names = list(sentinel_vars) + ["RAY_RUNTIME_ENV_TEST"]
        text, env_results = ray.get(
            actor.generate_and_get_worker_envs.remote("Hello world", all_env_names)
        )
        assert len(text) > 0

        for name, expected in sentinel_vars.items():
            for val in env_results[name]:
                assert val == expected

        for val in env_results["RAY_RUNTIME_ENV_TEST"]:
            assert val == "ray_runtime_env"

    finally:
        for k in sentinel_vars:
            os.environ.pop(k, None)
