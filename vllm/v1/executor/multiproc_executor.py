import atexit
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Tuple

from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.executor.multiproc_worker_utils import (
    set_multiprocessing_worker_envs)
from vllm.logger import init_logger
from vllm.utils import get_distributed_init_method, get_open_port
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_worker import WorkerProc, WorkerProcHandle

logger = init_logger(__name__)


class MultiprocExecutor:

    def __init__(self, vllm_config: VllmConfig) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        atexit.register(self.shutdown)

        self.vllm_config = vllm_config
        self.parallel_config = vllm_config.parallel_config

        world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        assert world_size == tensor_parallel_size, (
            f"world_size ({world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}). "
            f"Pipeline parallelism is not yet implemented in v1")

        # Set multiprocessing envs that are common to V0 and V1
        set_multiprocessing_worker_envs(self.parallel_config)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # 127.0.0.1 for communication.
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port())

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        self.worker_request_mq = MessageQueue(world_size, world_size)
        scheduler_output_handle = self.worker_request_mq.export_handle()

        # Create workers
        self.workers: List[WorkerProcHandle] = []
        for rank in range(world_size):
            worker = WorkerProc.make_worker_process(vllm_config, rank, rank,
                                                    distributed_init_method,
                                                    scheduler_output_handle)
            self.workers.append(worker)

    def initialize(self, num_gpu_blocks: int) -> None:
        """
        Initialize the KV caches and begin the model execution loop of the
        underlying workers.
        """
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(w.initialize, num_gpu_blocks)
                for w in self.workers
            ]
            success_vals = [f.result()
                            for f in futures]  # Wait for all to complete

        if not all(success_vals):
            raise RuntimeError("Worker initialization failed.")

        self.worker_request_mq.wait_until_ready()
        for w in self.workers:
            w.worker_response_mq.wait_until_ready()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """
        Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(w.determine_num_available_blocks)
                for w in self.workers
            ]
            num_blocks = [f.result()
                          for f in futures]  # Wait for all to complete

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)

        return num_gpu_blocks, num_cpu_blocks

    def collective_rpc(self, fn: str, timeout: Optional[float], *args,
                       **kwargs) -> Any:
        self.worker_request_mq.enqueue((fn, args, kwargs))

        try:
            responses = []
            for w in self.workers:
                status, result = w.worker_response_mq.dequeue(timeout=timeout)

                if status != WorkerProc.ResponseStatus.SUCCESS:
                    if isinstance(result, Exception):
                        raise result
                    else:
                        raise RuntimeError("Worker failed")

                responses.append(result)

            return responses[0]
        except TimeoutError:
            raise TimeoutError(f"RPC call to {fn} timed out.") from None
        except Exception as e:
            # Re-raise any other exceptions
            raise e

    def execute_model(
        self,
        scheduler_output,
    ) -> ModelRunnerOutput:
        model_output = self.collective_rpc("execute_model", None,
                                           scheduler_output)
        return model_output

    def profile(self, is_start=True):
        raise NotImplementedError

    def _ensure_worker_termination(self):
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""

        def wait_for_termination(procs, timeout):
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        # Send SIGTERM if still running
        active_procs = [w.proc for w in self.workers if w.proc.is_alive()]
        self.workers = None
        for p in active_procs:
            p.terminate()
        if wait_for_termination(active_procs, 4):
            return

        # Send SIGKILL if still running
        active_procs = [p for p in active_procs if p.is_alive()]
        for p in active_procs:
            p.kill()

    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if (hasattr(self, 'workers') and self.workers is not None):
            for w in self.workers:  #TODO: not sure if needed
                w.worker_response_mq = None
            self._ensure_worker_termination()

        self.worker_request_mq = None

    def check_health(self) -> None:
        self.collective_rpc("check_health", 10)
        return
