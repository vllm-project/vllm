# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import Future
from typing import List, Type, Union

import torch
import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.executor.executor_base import ExecutorBase
from vllm.executor.uniproc_executor import (  # noqa
    ExecutorWithExternalLauncher as ExecutorWithExternalLauncherV0)
from vllm.executor.uniproc_executor import (  # noqa
    UniProcExecutor as UniProcExecutorV0)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput


class Executor(ExecutorBase):
    """
    Abstract class for v1 executors, mainly define some methods for v1.
    For methods shared by v0 and v1, define them in ExecutorBase"""

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> Type["Executor"]:
        executor_class: Type[Executor]
        parallel_config = vllm_config.parallel_config
        distributed_executor_backend = (
            parallel_config.distributed_executor_backend)
        # distributed_executor_backend must be set in VllmConfig.__post_init__
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, ExecutorBase):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"ExecutorBase. Got {distributed_executor_backend}.")
            executor_class = distributed_executor_backend
        elif distributed_executor_backend == "ray":
            from vllm.v1.executor.ray_distributed_executor import (  # noqa
                RayDistributedExecutor)
            executor_class = RayDistributedExecutor
        elif distributed_executor_backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor
            executor_class = MultiprocExecutor
        elif distributed_executor_backend == "uni":
            executor_class = UniProcExecutor
        elif distributed_executor_backend == "external_launcher":
            # TODO: make v1 scheduling deterministic
            # to support external launcher
            executor_class = ExecutorWithExternalLauncher
        else:
            raise ValueError("Unknown distributed executor backend: "
                             f"{distributed_executor_backend}")
        return executor_class

    def initialize_from_config(self,
                               kv_cache_configs: List[KVCacheConfig]) -> None:
        """
        Initialize the KV caches and begin the model execution loop of the
        underlying workers.
        """
        self.collective_rpc("initialize_from_config",
                            args=(kv_cache_configs, ))
        self.collective_rpc("compile_or_warm_up_model")

    def determine_available_memory(self) -> int:  # in bytes
        output = self.collective_rpc("determine_available_memory")
        # Since we use a shared centralized controller, we take the minimum
        # memory size across all workers to make sure all the memory
        # operators can be applied to all workers.
        return min(output)

    def get_kv_cache_specs(self) -> List[KVCacheSpec]:
        output = self.collective_rpc("get_kv_cache_spec")
        return output

    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        output = self.collective_rpc("execute_model",
                                     args=(scheduler_output, ))
        return output[0]

    @property
    def max_concurrent_batches(self) -> int:
        return 1

    def profile(self, is_start: bool = True):
        self.collective_rpc("profile", args=(is_start, ))


class UniProcExecutor(UniProcExecutorV0, Executor):
    pass


class ExecutorWithExternalLauncher(ExecutorWithExternalLauncherV0, Executor):

    def determine_available_memory(self) -> int:  # in bytes
        # same as determine_num_available_blocks in v0,
        # we need to get the min across all ranks.
        memory = super().determine_available_memory()
        from vllm.distributed.parallel_state import get_world_group
        cpu_group = get_world_group().cpu_group
        memory_tensor = torch.tensor([memory], device="cpu", dtype=torch.int64)
        dist.all_reduce(memory_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        return memory_tensor.item()
