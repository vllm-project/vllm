from typing import Type

from vllm.config import VllmConfig
from vllm.executor.executor_base import ExecutorBase
from vllm.executor.ray_distributed_executor import RayDistributedExecutor
from vllm.executor.uniproc_executor import (ExecutorWithExternalLauncher,
                                            UniProcExecutor)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput


class Executor(ExecutorBase):
    """Abstract class for v1 executors, mainly define some new methods."""

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> Type["Executor"]:
        executor_class: Type[Executor]
        distributed_executor_backend = (
            vllm_config.parallel_config.distributed_executor_backend)
        if distributed_executor_backend == "ray":
            executor_class = RayDistributedExecutorV1
        elif distributed_executor_backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor
            executor_class = MultiprocExecutor
        elif distributed_executor_backend == "uni":
            executor_class = UniprocExecutorV1
        elif distributed_executor_backend == "external_launcher":
            # TODO: make v1 scheduling deterministic
            # to support external launcher
            executor_class = ExecutorWithExternalLauncherV1
        return executor_class

    def initialize(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the KV caches and begin the model execution loop of the
        underlying workers.
        """
        self.collective_rpc("initialize_cache", args=(kv_cache_config, ))
        self.collective_rpc("compile_or_warm_up_model")

    def determine_available_memory(self) -> int:  # in bytes
        output = self.collective_rpc("determine_available_memory")
        # Since we use a shared centralized controller, we take the minimum
        # memory size across all workers to make sure all the memory
        # operators can be applied to all workers.
        return min(output)

    def get_kv_cache_spec(self) -> KVCacheSpec:
        output = self.collective_rpc("get_kv_cache_spec")
        for x in output:
            assert x == output[0]
        return output[0]

    def execute_model(
        self,
        scheduler_output,
    ) -> ModelRunnerOutput:
        output = self.collective_rpc("execute_model",
                                     args=(scheduler_output, ))
        return output[0]

    def profile(self, is_start: bool = True):
        self.collective_rpc("profile", args=(is_start, ))


class UniprocExecutorV1(UniProcExecutor, Executor):
    pass


class ExecutorWithExternalLauncherV1(ExecutorWithExternalLauncher, Executor):
    pass


class RayDistributedExecutorV1(RayDistributedExecutor, Executor):
    pass
