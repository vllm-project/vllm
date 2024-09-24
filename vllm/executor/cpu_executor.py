import os
from functools import partial
from typing import Any, Awaitable, List, Optional, Set, Tuple, Union

import torch

import vllm.envs as envs
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.executor.multiproc_worker_utils import (ProcessWorkerWrapper,
                                                  ResultHandler, WorkerMonitor)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import ExecuteModelRequest
from vllm.utils import (GiB_bytes, get_distributed_init_method, get_open_port,
                        get_vllm_instance_id, make_async)
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class CPUExecutor(ExecutorBase):

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert self.device_config.device_type == "cpu"
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

        self.model_config = _verify_and_get_model_config(self.model_config)
        self.cache_config = _verify_and_get_cache_config(self.cache_config)
        self.scheduler_config = _verify_and_get_scheduler_config(
            self.scheduler_config)
        self.parallel_config = _verify_and_get_parallel_config(
            self.parallel_config)

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

        if world_size != 1 or is_async:
            if is_async:
                async_worker_list = self.workers + [self.driver_worker]
            else:
                async_worker_list = self.workers
            self.worker_monitor = WorkerMonitor(async_worker_list,
                                                result_handler)
            result_handler.start()
            self.worker_monitor.start()

        self._run_workers("init_device")
        self._run_workers("load_model")

    def _create_worker(
        self,
        local_rank: int = 0,
        rank: int = 0,
    ):
        worker_module_name = "vllm.worker.cpu_worker"
        worker_class_name = "CPUWorker"

        wrapper = WorkerWrapperBase(
            worker_module_name=worker_module_name,
            worker_class_name=worker_class_name,
        )

        assert self.distributed_init_method is not None

        kwargs = dict(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            load_config=self.load_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=self.distributed_init_method,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            prompt_adapter_config=self.prompt_adapter_config,
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
                         num_cpu_blocks: int) -> None:
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
        if (self.parallel_config.tensor_parallel_size > 1
                and self.parallel_worker_tasks is None):
            self.parallel_worker_tasks = self._run_workers(
                "start_worker_execution_loop",
                async_run_remote_workers_only=True,
            )
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

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return all(self._run_workers("add_lora", lora_request))

    def remove_lora(self, lora_id: int) -> bool:
        return all(self._run_workers("remove_lora", lora_id))

    def pin_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self._run_workers(
            "pin_lora",
            lora_id=lora_id,
        ))

    def list_loras(self) -> Set[int]:
        return self.driver_method_invoker(self.driver_worker, "list_loras")

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        return all(
            self._run_workers(
                "add_prompt_adapter",
                prompt_adapter_request,
            ))

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return all(
            self._run_workers(
                "remove_prompt_adapter",
                prompt_adapter_id,
            ))

    def list_prompt_adapters(self) -> Set[int]:
        return self.driver_method_invoker(self.driver_worker,
                                          "list_prompt_adapters")

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return all(self._run_workers(
            "pin_prompt_adapter",
            prompt_adapter_id,
        ))

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

    def start_profile(self) -> None:
        self.driver_method_invoker(self.driver_worker, "start_profile")

    def stop_profile(self) -> None:
        self.driver_method_invoker(self.driver_worker, "stop_profile")


class CPUExecutorAsync(CPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        output = await make_async(self.execute_model
                                  )(execute_model_req=execute_model_req, )
        return output

    async def check_health_async(self) -> None:
        self.check_health()


def _verify_and_get_model_config(config: ModelConfig) -> ModelConfig:
    if config.dtype == torch.float16:
        logger.warning("float16 is not supported on CPU, casting to bfloat16.")
        config.dtype = torch.bfloat16
    if not config.enforce_eager:
        logger.warning(
            "CUDA graph is not supported on CPU, fallback to the eager "
            "mode.")
        config.enforce_eager = True
    return config


def _verify_and_get_scheduler_config(
        config: SchedulerConfig) -> SchedulerConfig:
    if config.chunked_prefill_enabled:
        logger.warning("Chunked prefill is not supported on CPU, disable it.")
        config.chunked_prefill_enabled = False

    return config


def _verify_and_get_cache_config(config: CacheConfig) -> CacheConfig:
    if config.enable_prefix_caching:
        logger.warning("Prefix caching is not supported on CPU, disable it.")
        config.enable_prefix_caching = False

    kv_cache_space = envs.VLLM_CPU_KVCACHE_SPACE

    if kv_cache_space >= 0:
        if kv_cache_space == 0:
            config.cpu_kvcache_space_bytes = 4 * GiB_bytes  # type: ignore
            logger.warning("Environment variable VLLM_CPU_KVCACHE_SPACE (GB) "
                           "for CPU backend is not set, using 4 by default.")
        else:
            config.cpu_kvcache_space_bytes = kv_cache_space * GiB_bytes  # type: ignore
    else:
        raise RuntimeError(
            "Invalid environment variable VLLM_CPU_KVCACHE_SPACE"
            f" {kv_cache_space}, expect a positive integer value.")

    return config


def _verify_and_get_parallel_config(config: ParallelConfig) -> ParallelConfig:
    if (config.distributed_executor_backend is not None
            and config.distributed_executor_backend != "mp"):
        logger.warning(
            "%s is not supported on CPU, fallback to mp distributed executor "
            "backend.", config.distributed_executor_backend)
        config.distributed_executor_backend = "mp"
    return config


def _driver_method_invoker(driver, method: str, *args, **kwargs):
    return getattr(driver, method)(*args, **kwargs)


def _async_driver_method_invoker(driver, method: str, *args, **kwargs):
    return driver.execute_method(method, *args, **kwargs).get()
