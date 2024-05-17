import asyncio
from abc import abstractmethod
from typing import Any, Awaitable, List, Optional, Set, Tuple, Union

from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.executor.gpu_executor import GPUExecutor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import ExecuteModelRequest, SamplerOutput

logger = init_logger(__name__)


class DistributedGPUExecutor(GPUExecutor):
    """Abstract superclass of multi-GPU executor implementations."""

    def __init__(self, *args, **kwargs):
        # This is non-None when the execute model loop is running
        # in the parallel workers
        self.parallel_worker_tasks: Optional[Union[Any, Awaitable[Any]]] = None
        # Updated by implementations that require additional args to be passed
        # to the _run_workers execute_model call
        self.extra_execute_model_run_workers_kwargs = {}

        super().__init__(*args, **kwargs)

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks.

        This invokes `determine_num_available_blocks` on each worker and takes
        the min of the results, guaranteeing that the selected cache sizes are
        compatible with all workers.

        Returns:
            - tuple[num_gpu_blocks, num_cpu_blocks]
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_workers("determine_num_available_blocks", )

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)

        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache in all workers.
        """

        # NOTE: We log here to avoid multiple logs when number of workers is
        # greater than one. We could log in the engine, but not all executors
        # have GPUs.
        logger.info("# GPU blocks: %d, # CPU blocks: %d", num_gpu_blocks,
                    num_cpu_blocks)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._run_workers("initialize_cache",
                          num_gpu_blocks=num_gpu_blocks,
                          num_cpu_blocks=num_cpu_blocks)

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        if self.parallel_worker_tasks is None:
            self.parallel_worker_tasks = self._run_workers(
                "start_worker_execution_loop",
                remote_workers_only_async=True,
                **self.extra_execute_model_run_workers_kwargs)

        # Only the driver worker returns the sampling results.
        return self._driver_execute_model(execute_model_req)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self._run_workers(
            "add_lora",
            lora_request=lora_request,
        )

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self._run_workers(
            "remove_lora",
            lora_id=lora_id,
        )

    def list_loras(self) -> Set[int]:
        return self._run_workers("list_loras")

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self._run_workers("save_sharded_state",
                          path=path,
                          pattern=pattern,
                          max_size=max_size)

    @abstractmethod
    def _driver_execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        """Run execute_model in the driver worker."""
        raise NotImplementedError

    @abstractmethod
    def _run_workers(
        self,
        method: str,
        *args,
        remote_workers_only_async: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        raise NotImplementedError


class DistributedGPUExecutorAsync(DistributedGPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        if self.parallel_worker_tasks is None:
            # Start model execution loop running in the parallel workers
            self.parallel_worker_tasks = asyncio.create_task(
                self._start_worker_execution_loop())

        # Only the driver worker returns the sampling results.
        return await self._driver_execute_model_async(execute_model_req)

    async def stop_remote_worker_execution_loop_async(self) -> None:
        if self.parallel_worker_tasks is None:
            return

        await self._driver_execute_model_async()
        parallel_worker_tasks = self.parallel_worker_tasks
        self.parallel_worker_tasks = None
        # Ensure that workers exit model loop cleanly
        # (this will raise otherwise)
        await parallel_worker_tasks

    @abstractmethod
    async def _driver_execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        raise NotImplementedError

    @abstractmethod
    async def _start_worker_execution_loop(self):
        raise NotImplementedError
