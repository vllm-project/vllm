from typing import Any, Dict, List, Set, Tuple

from vllm.executor.executor_base import ExecutorBase, ExecutorAsyncBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.utils import make_async

logger = init_logger(__name__)


class TTExecutor(ExecutorBase):
    
    uses_ray: bool = False
    
    def _init_executor(self) -> None:
        assert not self.scheduler_config.chunked_prefill_enabled, (
            "Chunked prefill is not yet supported for TT backend")
        assert not self.speculative_config, (
            "Speculative decoding is not yet supported for TT backend")
        assert self.parallel_config.tensor_parallel_size == self.parallel_config.pipeline_parallel_size == 1, (
            "TTExecutor does not support distributed execution")
        
        # Instantiate the worker and load the model to the device.
        self.driver_worker = self._create_worker()
        self.driver_worker.init_device()
        self.driver_worker.load_model()
        
    def _get_worker_kwargs(
        self,
    ) -> Dict[str, Any]:
        """Return worker init args """
        return dict(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            load_config=self.load_config,
            is_driver_worker=True,
        )

    def _create_worker(
        self,
    ):
        from vllm.worker.tt_worker import TTWorker
        worker = TTWorker(**self._get_worker_kwargs())
        return worker
    
    def execute_model(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[SamplerOutput]:
        output = self.driver_worker.execute_model(execute_model_req)
        return output
    
    def check_health(self) -> None:
        return  # Always healthy
    
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker."""
        return self.driver_worker.determine_num_available_blocks()
    
    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        logger.info("# TT blocks: %d, # CPU blocks: %d", num_gpu_blocks,
                    num_cpu_blocks)
        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
    
    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError(
            "LoRA is currently not supported by the TT backend.")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError(
            "LoRA is currently not supported by the TT backend.")

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError(
            "LoRA is currently not supported by the TT backend.")

    def list_loras(self) -> Set[int]:
        raise NotImplementedError(
            "LoRA is currently not supported by the TT backend.")

    def add_prompt_adapter(self, prompt_adapter_request) -> bool:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the TT backend.")

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the TT backend.")

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the TT backend.")

    def list_prompt_adapters(self) -> Set[int]:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the TT backend.")
   
class TTExecutorAsync(TTExecutor, ExecutorAsyncBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> SamplerOutput:
        output = await make_async(self.driver_worker.execute_model
                                  )(execute_model_req)
        return output
