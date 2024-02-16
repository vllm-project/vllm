from abc import abstractmethod
from typing import Any, Dict, Optional, Set, Tuple

from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.executor.utils import check_block_size_valid
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import SamplerOutput

logger = init_logger(__name__)


class MultiGPUExecutor(ExecutorBase):
    """Abstract superclass of multi-GPU executor implementations."""

    def _init_driver_worker_and_model(self, rank: int, local_rank: int,
                                      distributed_init_method: str):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker

        # Initialize the driver worker with the Worker class.
        self.driver_worker = Worker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            vision_language_config=self.vision_language_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=True,
        )
        # FIXME(woosuk): We are not properly initializing pynccl when
        # we have multiple nodes.
        self._run_workers("init_device")
        self._run_workers("load_model",
                          max_concurrent_workers=self.parallel_config.
                          max_parallel_loading_workers)

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.
        More details can be found in the
        :meth:`~vllm.worker.worker.Worker.profile_num_available_blocks` method
        from class :class:`~vllm.worker.Worker`.

        Afterwards, as there may be multiple workers,
        we take the minimum number of blocks across all workers
        to ensure this can be applied to all of them.

        Finally, the engine will initialize the KV cache
        with the calculated number of blocks.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_workers(
            "profile_num_available_blocks",
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
            cache_dtype=self.cache_config.cache_dtype,
        )

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)

        if self.cache_config.forced_num_gpu_blocks is not None:
            forced_num_gpu_blocks = self.cache_config.forced_num_gpu_blocks
            logger.info(f"Replacing profiled {num_gpu_blocks=} with "
                        f"{forced_num_gpu_blocks=}")
            num_gpu_blocks = forced_num_gpu_blocks

        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        check_block_size_valid(num_gpu_blocks, self.cache_config.block_size,
                               self.model_config.max_model_len)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        self._run_workers("init_cache_engine", cache_config=self.cache_config)
        # Warm up the model. This includes capturing the model into CUDA graph
        # if enforce_eager is False.
        self._run_workers("warm_up_model")

    def execute_model(self, *args, **kwargs) -> SamplerOutput:
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


class MultiGPUExecutorAsync(MultiGPUExecutor, ExecutorAsyncBase):

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

    async def execute_model_async(self, *args, **kwargs) -> SamplerOutput:
        all_outputs = await self._run_workers_async("execute_model",
                                                    driver_args=args,
                                                    driver_kwargs=kwargs)

        # Only the driver worker returns the sampling results.
        return all_outputs[0]
