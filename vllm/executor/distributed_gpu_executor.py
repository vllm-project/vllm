from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.executor.gpu_executor import GPUExecutor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import SamplerOutput

logger = init_logger(__name__)


class DistributedGPUExecutor(GPUExecutor):
    """Abstract superclass of multi-GPU executor implementations."""

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

    def execute_model(self, *args, **kwargs) -> List[SamplerOutput]:
        all_outputs = self._run_workers("execute_model",
                                        driver_args=args,
                                        driver_kwargs=kwargs)

        # Only the driver worker returns the sampling results.
        return all_outputs[0]

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

    @abstractmethod
    def _run_workers(
        self,
        method: str,
        *args,
        driver_args: Optional[Tuple[Any, ...]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        raise NotImplementedError


class DistributedGPUExecutorAsync(DistributedGPUExecutor, ExecutorAsyncBase):

    @abstractmethod
    async def _run_workers_async(
        self,
        method: str,
        *args,
        driver_args: Optional[Tuple[Any, ...]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        raise NotImplementedError

    async def execute_model_async(self, *args,
                                  **kwargs) -> List[SamplerOutput]:
        all_outputs = await self._run_workers_async("execute_model",
                                                    driver_args=args,
                                                    driver_kwargs=kwargs)

        # Only the driver worker returns the sampling results.
        return all_outputs[0]
