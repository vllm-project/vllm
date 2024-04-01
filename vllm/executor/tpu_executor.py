import os
from typing import Dict, List, Optional

from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VisionLanguageConfig)
from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.executor.utils import check_block_size_valid
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.utils import make_async

logger = init_logger(__name__)


class TPUExecutor(ExecutorBase):

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        vision_language_config: Optional[VisionLanguageConfig],
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        assert lora_config is None, "LoRA is not supported for TPU backend."
        self.vision_language_config = vision_language_config

        # Instantiate the worker and load the model to the device.
        self._init_worker()
        # Profile the memory usage and initialize the cache.
        self._init_cache()

    def _init_worker(self):
        os.environ["PJRT_DEVICE"] = "TPU"
        from vllm.worker.tpu_worker import TPUWorker

        assert self.parallel_config.world_size == 1, (
            "TPUExecutor currently only supports a single TPU chip.")
        self.driver_worker = TPUWorker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
        )
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache.

        The engine first profiles the existing memory usage.
        Then, it allocates the remaining memory for KV blocks.

        .. tip::
            You may limit the usage of TPU HBM by adjusting the
            `gpu_memory_utilization` parameter.
        """
        # Get the maximum number of blocks that can be allocated on TPU.
        num_tpu_blocks = self.driver_worker.profile_num_available_blocks(
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cache_dtype=self.cache_config.cache_dtype,
        )
        logger.info(f"# TPU blocks: {num_tpu_blocks}")

        check_block_size_valid(num_tpu_blocks, self.cache_config.block_size,
                               self.model_config.max_model_len)
        self.cache_config.num_gpu_blocks = num_tpu_blocks
        self.cache_config.num_cpu_blocks = 0

        # Allocate the KV cache.
        self.driver_worker.allocate_kv_cache(self.cache_config)
        # Warm up the model.
        self.driver_worker.warm_up_model()

    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        output = self.driver_worker.execute_model(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
        )
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError("LoRA is not implemented for TPU backend.")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not implemented for TPU backend.")

    def list_loras(self) -> List[int]:
        raise NotImplementedError("LoRA is not implemented for TPU backend.")

    def check_health(self) -> None:
        # TPUExecutor will always be healthy as long as it's running.
        return


class TPUExecutorAsync(TPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        output = await make_async(self.driver_worker.execute_model)(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy)
        return output

    async def check_health_async(self) -> None:
        # TPUExecutor will always be healthy as long as it's running.
        return
