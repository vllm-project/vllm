import os
import copy
from typing import Dict, List, Optional, Tuple, Set

import torch
import torch.distributed

import vllm.envs as envs
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig
from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async)
from vllm.worker.cpu_worker import CPUWorker

logger = init_logger(__name__)


class CPUExecutor(ExecutorBase):

    def _init_executor(self) -> None:
        assert self.device_config.device_type == "cpu"
        assert self.lora_config is None, "cpu backend doesn't support LoRA"
        self.model_config = _verify_and_get_model_config(self.model_config)
        self.cache_config = _verify_and_get_cache_config(self.cache_config)
        self.scheduler_config = _verify_and_get_scheduler_config(
            self.scheduler_config)

        self.children_workers : List[CPUWorker] = []
        self.children_loops = []

        # Instantiate the worker and load the model to CPU.
        ip = get_ip()
        port = get_open_port()
        self.ip_port = ip + "_" + str(port)
        self.distributed_init_method = get_distributed_init_method(ip, port)
        if self.parallel_config.tensor_parallel_size > 1:
            self._init_ray_workers()
        else:
            self._init_worker()
    
    def _init_worker(self):
        self.driver_worker = CPUWorker(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            load_config=self.load_config,
            local_rank=0,
            rank=0,
            distributed_init_method=self.distributed_init_method,
            ip_port=self.ip_port,
            lora_config=self.lora_config,
            vision_language_config=self.vision_language_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=True,
        )
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def _init_ray_workers(self):
        assert self.parallel_config.tensor_parallel_size > 1
        
        import ray
        from ray.util.scheduling_strategies import (
            PlacementGroupSchedulingStrategy
        )

        #FIXME: Specify cluster addr explicitly.
        ray.init(address="auto", ignore_reinit_error=True)
        threads_num_per_node = torch.get_num_threads()
        nodes = ray.nodes()

        for node in nodes:
            node_thread_num = node["Resources"]["CPU"]
            if threads_num_per_node != node_thread_num:
                raise RuntimeError(
                    f"Number of OMP threads {node_thread_num} in child node "
                    f"doesn't match with the number {threads_num_per_node} in "
                    "driver worker.") 

        if self.parallel_config.placement_group is None:
            placement_group_specs = (
                [{"CPU": threads_num_per_node}] * \
                (self.parallel_config.world_size - 1))
            placement_group = ray.util.placement_group(
                placement_group_specs, strategy="STRICT_SPREAD")
            ray.get(placement_group.ready(), timeout=1800)
            self.parallel_config.placement_group = placement_group
            bundle_offset = 0
        else:
            placement_group = self.parallel_config.placement_group
            bundle_offset = 1

        
        model_config = copy.deepcopy(self.model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        device_config = copy.deepcopy(self.device_config)
        lora_config = copy.deepcopy(self.lora_config)
        cache_config = copy.deepcopy(self.cache_config)
        load_config = copy.deepcopy(self.load_config)
        distributed_init_method = copy.deepcopy(self.distributed_init_method)
        ip_port = copy.deepcopy(self.ip_port)
        for bundle_id in range(0, parallel_config.world_size - 1):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id + bundle_offset,
            )

            child_worker = ray.remote(
                num_cpus=threads_num_per_node,
                scheduling_strategy=scheduling_strategy
            )(CPUWorker).remote(
                model_config=None,
                parallel_config=parallel_config,
                scheduler_config=scheduler_config,
                device_config=device_config,
                cache_config=cache_config,
                load_config=load_config,
                local_rank=bundle_id + 1,
                rank=bundle_id + 1,
                distributed_init_method=distributed_init_method,
                ip_port = ip_port,
                lora_config=lora_config,
                kv_cache_dtype=cache_config.cache_dtype,
                is_driver_worker=False, 
                trust_remote_code=model_config.trust_remote_code, 
            ) # type: ignore
            self.children_workers.append(child_worker)
            ray.get(child_worker.init_runner.remote(model_config))
        
        task_handlers = []
        for child in self.children_workers:
            task_handlers.append(child.init_device.remote())
            task_handlers.append(child.load_model.remote())
        
        self._init_worker()

        # Initialize SHM CCL
        for child in self.children_workers:
            task_handlers.append(child.init_shm_manager.remote())
        ray.get(task_handlers)
        self.driver_worker.init_shm_manager()

        for child in self.children_workers:
            task_handlers.append(child.join_shm_manager.remote())
        ray.get(task_handlers)
        self.driver_worker.join_shm_manager()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """

        if self.parallel_config.tensor_parallel_size > 1:
            for child in self.children_workers:
                child.warming_up_model.remote()
        self.driver_worker.warming_up_model()

        return self.driver_worker.determine_num_available_blocks()

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

        if self.parallel_config.tensor_parallel_size > 1:
            task_handlers = []
            for child in self.children_workers:
                task_handlers.append(
                    child.initialize_cache.remote(num_gpu_blocks, num_cpu_blocks)
                )

        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

        if self.parallel_config.tensor_parallel_size > 1:
            import ray
            ray.get(task_handlers)

            # FIXME: For now, we suppose workers are ready after the 
            # cache initialization.
            self.children_loops = []
            for worker in self.children_workers:
                self.children_loops.append(worker.child_loop.remote())

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        output = self.driver_worker.execute_model(execute_model_req)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.driver_worker.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.driver_worker.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.driver_worker.list_loras()

    def check_health(self) -> None:
        # CPUExecutor will always be healthy as long as
        # it's running.
        return


class CPUExecutorAsync(CPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        output = await make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req, )
        return output

    async def check_health_async(self) -> None:
        # CPUExecutor will always be healthy as long as
        # it's running.
        return


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
    _GB = 1 << 30
    if config.enable_prefix_caching:
        logger.warning("Prefix caching is not supported on CPU, disable it.")
        config.enable_prefix_caching = False

    kv_cache_space = envs.VLLM_CPU_KVCACHE_SPACE

    if kv_cache_space >= 0:
        if kv_cache_space == 0:
            config.cpu_kvcache_space_bytes = 4 * _GB  # type: ignore
            logger.warning("Environment variable VLLM_CPU_KVCACHE_SPACE (GB) "
                           "for CPU backend is not set, using 4 by default.")
        else:
            config.cpu_kvcache_space_bytes = kv_cache_space * _GB  # type: ignore
    else:
        raise RuntimeError(
            "Invalid environment variable VLLM_CPU_KVCACHE_SPACE"
            f" {kv_cache_space}, expect a positive integer value.")

    return config
