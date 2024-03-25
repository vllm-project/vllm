from typing import Dict, List, Optional

from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.executor.executor_base import ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import SamplerOutput, SequenceGroupMetadata

logger = init_logger(__name__)


class NeuronExecutor(ExecutorBase):

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        assert lora_config is None, "LoRA is not supported for Neuron backend."
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config

        # Set the number of GPU blocks to be the same as the maximum number of
        # sequences that can be processed in a single batch. This is equivalent
        # to schedule without PagedAttention.
        self.cache_config.num_gpu_blocks = self.scheduler_config.max_num_seqs
        self.cache_config.num_cpu_blocks = 0

        # Instantiate the worker and load the model to the device.
        self._init_worker()

    def _init_worker(self):
        from vllm.worker.neuron_worker import NeuronWorker

        self.driver_worker = NeuronWorker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
        )
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def execute_model(self,
                      seq_group_metadata_list: List[SequenceGroupMetadata],
                      blocks_to_swap_in: Dict[int, int],
                      blocks_to_swap_out: Dict[int, int],
                      blocks_to_copy: Dict[int, List[int]]) -> SamplerOutput:
        assert (blocks_to_swap_in == {} and blocks_to_swap_out == {}
                and blocks_to_copy == {}), (
                    "Cache operations are not supported for Neuron backend.")

        output = self.driver_worker.execute_model(
            seq_group_metadata_list=seq_group_metadata_list)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError(
            "LoRA is not implemented for neuron backend.")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError(
            "LoRA is not implemented for neuron backend.")

    def list_loras(self) -> List[int]:
        raise NotImplementedError(
            "LoRA is not implemented for neuron backend.")

    def check_health(self) -> None:
        # NeuronExecutor will always be healthy as long as
        # it's running.
        return
