from typing import Dict, List, Set, Tuple

from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.utils import make_async

logger = init_logger(__name__)


class NeuronExecutor(ExecutorBase):

    def _init_executor(self) -> None:
        assert (self.lora_config is
                None), "LoRA is not supported for Neuron backend."
        assert (not self.speculative_config
                ), "Speculative decoding not yet supported for Neuron backend."

        # Instantiate the worker and load the model to the device.
        self._init_worker()

    def _init_worker(self):
        from vllm.worker.neuron_worker import NeuronWorker

        self.driver_worker = NeuronWorker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
            self.cache_config,
        )
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        return self.driver_worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def execute_model(self,
                      seq_group_metadata_list: List[SequenceGroupMetadata],
                      blocks_to_swap_in: Dict[int, int],
                      blocks_to_swap_out: Dict[int, int],
                      blocks_to_copy: Dict[int, List[int]],
                      num_lookahead_slots: int) -> List[SamplerOutput]:
        assert (blocks_to_swap_in == {} and blocks_to_swap_out == {}
                and blocks_to_copy == {}), (
                    "Cache operations are not supported for Neuron backend.")
        assert num_lookahead_slots == 0, (
            "lookahead not supported for Neuron backend.")

        output = self.driver_worker.execute_model(
            seq_group_metadata_list=seq_group_metadata_list)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.driver_worker.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.driver_worker.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.driver_worker.list_loras()

    def check_health(self) -> None:
        # NeuronExecutor will always be healthy as long as
        # it's running.
        return


class NeuronExecutorAsync(NeuronExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        num_lookahead_slots: int,
    ) -> List[SamplerOutput]:
        output = await make_async(self.driver_worker.execute_model)(
            seq_group_metadata_list=seq_group_metadata_list, )
        return output

    async def check_health_async(self) -> None:
        # NeuronExecutor will always be healthy as long as
        # it's running.
        return
