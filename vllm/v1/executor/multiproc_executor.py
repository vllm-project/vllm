import atexit
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import torch

from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger
from vllm.triton_utils import maybe_set_triton_cache_manager
from vllm.utils import (cuda_is_initialized, get_distributed_init_method,
                        get_open_port, get_vllm_instance_id)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_worker import WorkerProc, WorkerProcHandle

logger = init_logger(__name__)


class MultiprocExecutor:

    def __init__(self, vllm_config: VllmConfig) -> None:
        # Call self.shutdown at exix to clean up
        # and ensure workers will be terminated.
        atexit.register(self.shutdown)

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        assert world_size == tensor_parallel_size, (
            f"world_size ({world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}) -- pipeline "
            f"parallelism is not yet implemented in v1")

        if (cuda_is_initialized()
                and os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn"):
            logger.warning("CUDA was previously initialized. We must use "
                           "the `spawn` multiprocessing start method. Setting "
                           "VLLM_WORKER_MULTIPROC_METHOD to 'spawn'.")
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Ensure that VLLM_INSTANCE_ID is set, to be inherited by workers
        os.environ["VLLM_INSTANCE_ID"] = get_vllm_instance_id()

        # Disable torch async compiling which won't work with daemonic processes
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        # Configure thread parallelism if OMP_NUM_THREADS isn't set
        #
        # Helps to avoid CPU contention. The default of spawning a thread per
        # core combined with multiprocessing for each GPU can have a negative
        # impact on performance. The contention is amplified when running in a
        # container where CPU limits can cause throttling.
        default_omp_num_threads = 1
        if "OMP_NUM_THREADS" not in os.environ and (
                current_parallelism :=
                torch.get_num_threads()) > default_omp_num_threads:
            logger.warning(
                "Reducing Torch parallelism from %d threads to %d to avoid "
                "unnecessary CPU contention. Set OMP_NUM_THREADS in the "
                "external environment to tune this value as needed.",
                current_parallelism, default_omp_num_threads)
            os.environ["OMP_NUM_THREADS"] = str(default_omp_num_threads)
            torch.set_num_threads(default_omp_num_threads)

        # workaround for https://github.com/vllm-project/vllm/issues/6103
        if world_size > 1:
            maybe_set_triton_cache_manager()

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

        model_output_mq_handle = self.workers[0].model_output_mq_handle
        self.model_output_mq = MessageQueue.create_from_handle(
            model_output_mq_handle, 0)

    def _run_on_workers(self, fn: str, *args) -> List:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(getattr(type(w), fn), w, *args)
                for w in self.workers
            ]
            result = [f.result() for f in futures]  # Wait for all to complete
        return result

    def initialize(self, num_gpu_blocks: int) -> None:
        """
        Initialize the KV caches and begin the model execution loop of the
        underlying workers.
        """
        success_vals = self._run_on_workers('initialize', num_gpu_blocks)
        if not all(success_vals):
            raise RuntimeError("Worker initialization failed.")

        self.worker_request_mq.wait_until_ready()
        self.model_output_mq.wait_until_ready()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """
        Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_on_workers('determine_num_available_blocks')

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)

        return num_gpu_blocks, num_cpu_blocks

    def execute_model(
        self,
        scheduler_output,
    ) -> ModelRunnerOutput:
        self.worker_request_mq.enqueue(scheduler_output)
        model_output = self.model_output_mq.dequeue()
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

        # Wait for workers to terminate gracefully
        active_procs = [w.proc for w in self.workers if w.proc.is_alive()]
        self.workers = None
        if wait_for_termination(active_procs, 5):
            return

        # Send SIGTERM if still running
        active_procs = [p for p in active_procs if p.is_alive()]
        for p in active_procs:
            p.terminate()
        if wait_for_termination(active_procs, 5):
            return

        # Send SIGKILL if still running
        active_procs = [p for p in active_procs if p.is_alive()]
        for p in active_procs:
            p.kill()
        if not wait_for_termination(active_procs, 5):
            raise RuntimeError("Failed to terminate worker processes")

    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if (hasattr(self, 'workers') and self.workers is not None):
            self._ensure_worker_termination()

        self.model_output_mq = None
        self.worker_request_mq = None

    def check_health(self) -> None:
        # MultiprocExecutor will always be healthy as long as
        # it's running.
        return
