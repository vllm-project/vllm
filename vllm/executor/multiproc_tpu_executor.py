import asyncio
import os
from functools import partial
from typing import Any, Awaitable, Dict, List, Optional, Tuple, Union

from vllm import envs
from vllm.executor.tpu_executor import TPUExecutor
from vllm.executor.multiproc_worker_utils import (ProcessWorkerWrapper,
                                                  ResultHandler, WorkerMonitor)
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.utils import (get_distributed_init_method, get_open_port,
                        get_vllm_instance_id, make_async)

logger = init_logger(__name__)


class MultiprocessingTPUExecutor(TPUExecutor):
    """Python multiprocessing-based multi-chip TPU executor"""

    def __init__(self, *args, **kwargs):
        # This is non-None when the execute model loop is running
        # in the parallel workers. It's a coroutine in the AsyncLLMEngine case.
        self.parallel_worker_tasks: Optional[Union[Any, Awaitable[Any]]] = None
        # Updated by implementations that require additional args to be passed
        # to the _run_workers execute_model call
        self.extra_execute_model_run_workers_kwargs: Dict[str, Any] = {}

        super().__init__(*args, **kwargs)

    def _init_executor(self) -> None:
        # Create the parallel TPU workers.
        world_size = self.parallel_config.tensor_parallel_size

        # Ensure that VLLM_INSTANCE_ID is set, to be inherited by workers
        os.environ["VLLM_INSTANCE_ID"] = get_vllm_instance_id()

        # Disable torch async compiling which won't work with daemonic processes
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        # TODO(woosuk)
        # assert world_size <= cuda_device_count_stateless(), (
        #     "please set tensor_parallel_size to less than max local gpu count")

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # 127.0.0.1 for communication.
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port())

        if world_size == 1:
            self.workers = []
            self.worker_monitor = None
        else:
            result_handler = ResultHandler()
            self.workers = [
                ProcessWorkerWrapper(
                    result_handler,
                    partial(
                        self._create_worker,
                        rank=rank,
                        local_rank=rank,
                        distributed_init_method=distributed_init_method,
                    )) for rank in range(1, world_size)
            ]

            self.worker_monitor = WorkerMonitor(self.workers, result_handler)
            result_handler.start()
            self.worker_monitor.start()

        self.driver_worker = self._create_worker(
            distributed_init_method=distributed_init_method)
        self._run_workers("init_device")
        self._run_workers("load_model",
                          max_concurrent_workers=self.parallel_config.
                          max_parallel_loading_workers)

    def shutdown(self):
        if (worker_monitor := getattr(self, "worker_monitor",
                                      None)) is not None:
            worker_monitor.close()

    def _driver_execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        """Run execute_model in the driver worker.

        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        return self.driver_worker.execute_model(
            execute_model_req=execute_model_req)

    def _run_workers(
        self,
        method: str,
        *args,
        async_run_remote_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers.

        Args:
            async_run_remote_workers_only: If True the method will be run only
                in the remote workers, not the driver worker. It will also be
                run asynchronously and return a list of futures rather than
                blocking on the results.
        """

        if max_concurrent_workers:
            raise NotImplementedError(
                "max_concurrent_workers is not supported yet.")

        # Start the workers first.
        worker_outputs = [
            worker.execute_method(method, *args, **kwargs)
            for worker in self.workers
        ]

        if async_run_remote_workers_only:
            # Just return futures
            return worker_outputs

        driver_worker_method = getattr(self.driver_worker, method)
        driver_worker_output = driver_worker_method(*args, **kwargs)

        # Get the results of the workers.
        return [driver_worker_output
                ] + [output.get() for output in worker_outputs]

    def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        if self.worker_monitor is not None and not self.worker_monitor.is_alive(
        ):
            raise RuntimeError("Worker processes are not running")

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        for result in parallel_worker_tasks:
            result.get()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        num_blocks = self._run_workers("determine_num_available_blocks", )
        num_tpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)
        return num_tpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        logger.info("# TPU blocks: %d, # CPU blocks: %d", num_gpu_blocks,
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
                async_run_remote_workers_only=True,
                **self.extra_execute_model_run_workers_kwargs)

        # Only the driver worker returns the sampling results.
        return self._driver_execute_model(execute_model_req)

    def stop_remote_worker_execution_loop(self) -> None:
        if self.parallel_worker_tasks is None:
            return

        self._driver_execute_model()
        parallel_worker_tasks = self.parallel_worker_tasks
        self.parallel_worker_tasks = None
        # Ensure that workers exit model loop cleanly
        # (this will raise otherwise)
        self._wait_for_tasks_completion(parallel_worker_tasks)
