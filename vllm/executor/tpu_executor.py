from typing import Dict, List, Set, Tuple

from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import (ExecuteModelRequest, SamplerOutput, SequenceGroupMetadata)
from vllm.utils import make_async

logger = init_logger(__name__)


class TPUExecutor(ExecutorBase):

    def _init_executor(self) -> None:
        assert not self.speculative_config, (
            "Speculative decoding not yet supported for TPU backend")
        # Instantiate the worker and load the model to the device.
        self._init_worker()

    def _init_worker(self):
        from vllm.worker.tpu_worker import TPUWorker

        assert self.parallel_config.world_size == 1, (
            "TPUExecutor currently only supports a single TPU chip.")
        self.driver_worker = TPUWorker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
            self.cache_config,
            self.vision_language_config,
        )
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def initialize_cache(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        """Initialize the KV cache by invoking the underlying worker."""
        # NOTE: This is logged in the executor because there can be >1 worker
        # with other executors. We could log in the engine level, but work
        # remains to abstract away the device for non-GPU configurations.
        logger.info(f"# TPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")
        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        return self.driver_worker.determine_num_available_blocks()

    def execute_model(
        self, execute_model_req: ExecuteModelRequest
    ) -> List[SamplerOutput]:
        output = self.driver_worker.execute_model(execute_model_req)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError("LoRA is not implemented for TPU backend.")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not implemented for TPU backend.")

    def list_loras(self) -> Set[int]:
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
