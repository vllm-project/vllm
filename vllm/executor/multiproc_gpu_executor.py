import asyncio
import os
from functools import partial
from typing import Any, Dict, Optional, Tuple

from vllm.engine.local_worker_utils import (LocalWorkerVllm, ResultHandler,
                                            WorkerMonitor)
from vllm.executor.multi_gpu_executor import (MultiGPUExecutor,
                                              MultiGPUExecutorAsync)
from vllm.logger import init_logger
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async, set_cuda_visible_devices)

logger = init_logger(__name__)


def _create_worker(*args, **kwargs):
    # Import within worker process to avoid CUDA init issues
    from vllm.worker.worker import Worker
    return Worker(*args, **kwargs)


class MultiProcGPUExecutor(MultiGPUExecutor):
    """Python multiprocessing-based multi-GPU executor"""

    def _init_executor(self) -> None:
        assert (
            not self.speculative_config
        ), "Speculative decoding not yet supported for MultiProcGPU backend."

        # Create the parallel GPU workers.
        world_size = self.parallel_config.tensor_parallel_size

        # Set CUDA_VISIBLE_DEVICES for the driver, inherited by workers
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            set_cuda_visible_devices(range(world_size))

        from torch.cuda import device_count
        assert world_size <= device_count(), (
            "please set tensor_parallel_size to less than max local gpu count")

        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())

        if world_size == 1:
            self.workers = []
        else:
            result_handler = ResultHandler()
            self.workers = [
                LocalWorkerVllm(
                    result_handler,
                    partial(
                        _create_worker,
                        model_config=self.model_config,
                        parallel_config=self.parallel_config,
                        scheduler_config=self.scheduler_config,
                        device_config=self.device_config,
                        cache_config=self.cache_config,
                        local_rank=rank,
                        rank=rank,
                        distributed_init_method=distributed_init_method,
                        lora_config=self.lora_config,
                        vision_language_config=self.vision_language_config,
                        tensorizer_config=self.tensorizer_config,
                    )) for rank in range(1, world_size)
            ]

            for worker in self.workers:
                worker.start()

            self.worker_monitor = WorkerMonitor(self.workers, result_handler)
            result_handler.start()
            self.worker_monitor.start()

        self._init_driver_worker_and_model(0, 0, distributed_init_method)

    def shutdown(self):
        if (worker_monitor := getattr(self, "worker_monitor",
                                      None)) is not None:
            worker_monitor.close()

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

        if max_concurrent_workers:
            raise NotImplementedError(
                "max_concurrent_workers is not supported yet.")

        # Start the workers first.
        worker_outputs = [
            worker.execute_method(method, *args, **kwargs)
            for worker in self.workers
        ]

        if driver_args is None:
            driver_args = args
        if driver_kwargs is None:
            driver_kwargs = kwargs

        # Start the driver worker after all the ray workers.
        driver_worker_method = getattr(self.driver_worker, method)
        driver_worker_output = driver_worker_method(*driver_args,
                                                    **driver_kwargs)

        # Get the results of the workers.
        return [driver_worker_output
                ] + [output.get() for output in worker_outputs]

    def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        if not self.worker_monitor.is_alive():
            raise RuntimeError("Worker processes are not running")


class MultiProcGPUExecutorAsync(MultiProcGPUExecutor, MultiGPUExecutorAsync):

    async def _run_workers_async(
        self,
        method: str,
        *args,
        driver_args: Optional[Tuple[Any, ...]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        if driver_args is None:
            driver_args = args
        if driver_kwargs is None:
            driver_kwargs = kwargs

        driver_executor = make_async(getattr(self.driver_worker, method))

        # Run all the workers asynchronously.
        coros = [driver_executor(*driver_args, **driver_kwargs)] + [
            worker.execute_method_async(method, *args, **kwargs)
            for worker in self.workers
        ]

        return await asyncio.gather(*coros)
