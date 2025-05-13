# SPDX-License-Identifier: Apache-2.0
"""Test that various errors are handled properly."""

import pytest
import ray

from vllm.engine.arg_utils import EngineArgs
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.ray_distributed_executor import RayDistributedExecutor


def test_failed_ray_worker(monkeypatch: pytest.MonkeyPatch):
    # even for TP=1 and PP=1,
    # if users specify ray, we will use ray.
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_USE_RAY_SPMD_WORKER", "1")

        engine_args = EngineArgs(model="distilbert/distilgpt2",
                                 enforce_eager=True)
        vllm_config = engine_args.create_engine_config()

        engine_core = EngineCore(vllm_config=vllm_config,
                                 executor_class=RayDistributedExecutor,
                                 log_stats=False)

        assert engine_core.model_executor.uses_ray

        engine_core.model_executor.check_health()

        for worker in engine_core.model_executor.workers:
            ray.kill(worker)

        with pytest.raises(RuntimeError):
            engine_core.model_executor.check_health()
