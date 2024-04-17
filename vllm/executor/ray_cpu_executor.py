from typing import Optional

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VisionLanguageConfig)
from vllm.engine.ray_utils import RayWorkerVllm
from vllm.executor.ray_executor_base import (RayExecutorAsyncBase,
                                             RayExecutorBase)


class RayCPUExecutor(RayExecutorBase):

    def _init_worker_ray(
        self,
        ray_worker: RayWorkerVllm,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        vision_language_config: Optional[VisionLanguageConfig] = None,
    ) -> None:
        from vllm.worker.cpu_worker import CPUWorker

        ray_worker.init_worker.remote(lambda: CPUWorker(
            model_config=model_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            cache_config=cache_config,
            load_config=load_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            lora_config=lora_config,
        ))

    def _create_driver_worker(self, ray_driver: RayWorkerVllm,
                              distributed_init_method: str,
                              driver_local_rank: int, driver_rank: int):
        from vllm.worker.cpu_worker import CPUWorker
        return CPUWorker(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            load_config=self.load_config,
            local_rank=driver_local_rank,
            rank=driver_rank,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            is_driver_worker=True,
        )


class RayCPUExecutorAsync(RayCPUExecutor, RayExecutorAsyncBase):
    pass
