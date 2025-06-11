# SPDX-License-Identifier: Apache-2.0
from typing import Callable

import pytest
import os

from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.utils import get_distributed_init_method, get_open_port, get_ip

from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import UniProcExecutor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker
from vllm.worker.worker_base import WorkerWrapperBase


@pytest.fixture(autouse=True)
def allow_insecure_serialization(monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

@pytest.fixture(autouse=True)
def cleanup():
    cleanup_dist_env_and_memory(shutdown_ray=True)


@pytest.fixture(autouse=True)
def tensorizer_config():
    config = TensorizerConfig(tensorizer_uri="vllm")
    return config


def assert_from_collective_rpc(engine: LLM,
                               closure: Callable,
                               closure_kwargs: dict):
    res = engine.collective_rpc(method=closure, kwargs=closure_kwargs)
    return all(res)


# This is an object pulled from tests/v1/engine/test_engine_core.py
# Modified to strip the `load_model` method from its `_init_executor`
# method. It's purely used as a dummy utility to run methods that test
# Tensorizer functionality
class DummyExecutor(UniProcExecutor):

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """
        self.driver_worker = WorkerWrapperBase(
            vllm_config=self.vllm_config,
            rpc_rank=0
            )
        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port()
        )
        local_rank = 0
        # set local rank as the device index if specified
        device_info = self.vllm_config.device_config.device.__str__().split(
            ":"
        )
        if len(device_info) > 1:
            local_rank = int(device_info[1])
        rank = 0
        is_driver_worker = True
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")

    @property
    def max_concurrent_batches(self) -> int:
        return 2

    def shutdown(self):
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)