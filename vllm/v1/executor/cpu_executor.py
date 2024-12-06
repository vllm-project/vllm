import os
from functools import partial
from typing import Any, Awaitable, List, Optional, Dict, Tuple, Union


from vllm.config import VllmConfig
from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.executor.multiproc_worker_utils import (ProcessWorkerWrapper,
                                                  ResultHandler, WorkerMonitor)
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.utils import (get_distributed_init_method, get_open_port,
                        get_vllm_instance_id, make_async,
                        enable_trace_function_call_for_thread,
                        resolve_obj_by_qualname, update_environment_variables)

from vllm.v1.executor.abstract import Executor
from vllm.v1.worker.cpu_worker import CPUWorkerV1
logger = init_logger(__name__)


class CPUExecutor(Executor):

    uses_ray: bool = False

    def __init__(self, vllm_config: VllmConfig) -> None:
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
        assert self.device_config.device_type == "cpu"
        # Reminder: Please update docs/source/serving/compatibility_matrix.rst
        # If the feature combo become valid
        assert self.lora_config is None, "cpu backend doesn't support LoRA"

        #
        # Environment variables for CPU executor
        #

        # Ensure that VLLM_INSTANCE_ID is set, to be inherited by workers
        os.environ["VLLM_INSTANCE_ID"] = get_vllm_instance_id()

        # Disable torch async compiling which won't work with daemonic processes
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        # Intel OpenMP setting
        ld_prealod_str = os.getenv("LD_PRELOAD", "")
        if "libiomp5.so" in ld_prealod_str:
            # The time(milliseconds) that a thread should wait after
            # completing the execution of a parallel region, before sleeping.
            os.environ['KMP_BLOCKTIME'] = "1"
            # Prevents the CPU to run into low performance state
            os.environ['KMP_TPAUSE'] = "0"
            # Provides fine granularity parallelism
            os.environ['KMP_FORKJOIN_BARRIER_PATTERN'] = "dist,dist"
            os.environ['KMP_PLAIN_BARRIER_PATTERN'] = "dist,dist"
            os.environ['KMP_REDUCTION_BARRIER_PATTERN'] = "dist,dist"

        # To hint IPEX uses shared memory based AllReduce
        os.environ["LOCAL_WORLD_SIZE"] = str(
            self.parallel_config.tensor_parallel_size)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # 127.0.0.1 for communication.
        ip = "127.0.0.1"
        port = get_open_port()
        self.distributed_init_method = get_distributed_init_method(ip, port)

        is_async = isinstance(self, CPUExecutorAsync)

        world_size = self.parallel_config.tensor_parallel_size
        result_handler = ResultHandler()
        self.parallel_worker_tasks: Optional[Union[Any, Awaitable[Any]]] = None
        self.workers = []

        if is_async:
            self.workers = [
                ProcessWorkerWrapper(
                    result_handler,
                    partial(
                        self._create_worker,
                        rank=rank,
                        local_rank=rank,
                    )) for rank in range(0, world_size)
            ]
            self.driver_worker = self.workers[0]
            self.workers = self.workers[1:]
            self.driver_method_invoker = _async_driver_method_invoker
        else:
            self.driver_worker = self._create_worker()
            self.driver_method_invoker = _driver_method_invoker

            if world_size != 1:
                self.workers = [
                    ProcessWorkerWrapper(
                        result_handler,
                        partial(
                            self._create_worker,
                            rank=rank,
                            local_rank=rank,
                        )) for rank in range(1, world_size)
                ]

        self.worker_monitor = None
        if world_size != 1 or is_async:
            if is_async:
                async_worker_list = self.workers + [self.driver_worker]
            else:
                async_worker_list = self.workers
            self.worker_monitor = WorkerMonitor(async_worker_list,
                                                result_handler)
            result_handler.start()
            self.worker_monitor.start()

        self._run_workers("initialize")
        self._run_workers("load_model")

    def _create_worker(
            self,
            local_rank: int = 0,
            rank: int = 0,
    ):

        wrapper = WorkerWrapperBaseV1(vllm_config=self.vllm_config)

        assert self.distributed_init_method is not None

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=self.distributed_init_method,
            # kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=rank == 0,
        )
        wrapper.init_worker(**kwargs)

        return wrapper.worker

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

        driver_worker_output = self.driver_method_invoker(
            self.driver_worker, method, *args, **kwargs)

        # Get the results of the workers.
        return [driver_worker_output
                ] + [output.get() for output in worker_outputs]

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        return self.driver_method_invoker(self.driver_worker,
                                          "determine_num_available_blocks")

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int = 0) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # NOTE: We log here to avoid multiple logs when number of workers is
        # greater than one. We could log in the engine, but not all executors
        # have GPUs.
        # NOTE: `cpu block` for CPU backend is located on CPU memory but is
        # referred as `gpu block`. Because we want to reuse the existing block
        # management procedure.
        logger.info("# CPU blocks: %d", num_gpu_blocks)

        self._run_workers("initialize_cache",
                          num_gpu_blocks=num_gpu_blocks,
                          num_cpu_blocks=num_cpu_blocks)

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:

        output = self.driver_method_invoker(self.driver_worker,
                                            "execute_model", execute_model_req)
        return output

    def stop_remote_worker_execution_loop(self) -> None:
        if self.parallel_worker_tasks is None:
            return
        """
        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        self.driver_method_invoker(self.driver_worker, "execute_model", None)
        parallel_worker_tasks = self.parallel_worker_tasks
        self.parallel_worker_tasks = None
        # Ensure that workers exit model loop cleanly
        # (this will raise otherwise)
        self._wait_for_tasks_completion(parallel_worker_tasks)

    def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        if self.worker_monitor is not None and not self.worker_monitor.is_alive(
        ):
            raise RuntimeError("Worker processes are not running")

    def shutdown(self):
        if (worker_monitor := getattr(self, "worker_monitor",
                                      None)) is not None:
            worker_monitor.close()

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        for result in parallel_worker_tasks:
            result.get()


class CPUExecutorAsync(CPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        output = await make_async(self.execute_model
                                  )(execute_model_req=execute_model_req, )
        return output

    async def check_health_async(self) -> None:
        self.check_health()


def _driver_method_invoker(driver, method: str, *args, **kwargs):
    return getattr(driver, method)(*args, **kwargs)


def _async_driver_method_invoker(driver, method: str, *args, **kwargs):
    return driver.execute_method(method, *args, **kwargs).get()

class WorkerWrapperBaseV1:
    """
    The whole point of this class is to lazily initialize the worker.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    def __init__(
            self,
            vllm_config: VllmConfig,
    ) -> None:
        self.vllm_config = vllm_config
        trust_remote_code = vllm_config.model_config.trust_remote_code
        self.worker: Optional[CPUWorkerV1] = None
        if trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

    @staticmethod
    def update_environment_variables(envs: Dict[str, str]) -> None:
        key = 'CUDA_VISIBLE_DEVICES'
        if key in envs and key in os.environ:
            # overwriting CUDA_VISIBLE_DEVICES is desired behavior
            # suppress the warning in `update_environment_variables`
            del os.environ[key]
        update_environment_variables(envs)

    def init_worker(self, *args, **kwargs):
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        enable_trace_function_call_for_thread()

        # see https://github.com/NVIDIA/nccl/issues/1234
        os.environ['NCCL_CUMEM_ENABLE'] = '0'

        from vllm.plugins import load_general_plugins
        load_general_plugins()

        worker_class = resolve_obj_by_qualname("vllm.v1.worker.cpu_worker.CPUWorkerV1")
        self.worker = worker_class(*args, **kwargs)
        assert self.worker is not None

    def execute_method(self, method, *args, **kwargs):
        try:
            target = self if self.worker is None else self.worker
            executor = getattr(target, method)
            return executor(*args, **kwargs)
        except Exception as e:
            # if the driver worker also execute methods,
            # exceptions in the rest worker may cause deadlock in rpc like ray
            # see https://github.com/vllm-project/vllm/issues/3455
            # print the error and inform the user to solve the error
            msg = (f"Error executing method {method}. "
                   "This might cause deadlock in distributed execution.")
            logger.exception(msg)
            raise e

    def __getattr__(self, attr):
        return getattr(self.worker, attr)
