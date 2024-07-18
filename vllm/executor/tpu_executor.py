from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async)

logger = init_logger(__name__)


class TPUExecutor(ExecutorBase):

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert not self.scheduler_config.chunked_prefill_enabled, (
            "Chunked prefill is not yet supported for TPU backend")
        assert not self.speculative_config, (
            "Speculative decoding is not yet supported for TPU backend")
        if self.model_config.dtype in (torch.float16, torch.float32):
            logger.warning(
                "The TPU backend currently does not support %s. "
                "Using bfloat16 instead.", self.model_config.dtype)
            self.model_config.dtype = torch.bfloat16

        # Instantiate the worker and load the model to the device.
        self.driver_worker = self._create_worker()
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def _get_worker_kwargs(
        self,
        local_rank: int = 0,
        rank: int = 0,
        distributed_init_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return worker init args for a given rank."""
        if distributed_init_method is None:
            distributed_init_method = get_distributed_init_method(
                get_ip(), get_open_port())
        return dict(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            load_config=self.load_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            multimodal_config=self.multimodal_config,
            is_driver_worker=rank == 0,
        )

    def _create_worker(
        self,
        local_rank: int = 0,
        rank: int = 0,
        distributed_init_method: Optional[str] = None,
    ):
        from vllm.worker.tpu_worker import TPUWorker

        worker = TPUWorker(**self._get_worker_kwargs(local_rank, rank,
                                                     distributed_init_method))
        return worker

    def initialize_cache(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        """Initialize the KV cache by invoking the underlying worker."""
        # NOTE: This is logged in the executor because there can be >1 worker
        # with other executors. We could log in the engine level, but work
        # remains to abstract away the device for non-GPU configurations.
        logger.info("# TPU blocks: %d, # CPU blocks: %d", num_gpu_blocks,
                    num_cpu_blocks)
        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker."""
        return self.driver_worker.determine_num_available_blocks()

    def execute_model(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[SamplerOutput]:
        output = self.driver_worker.execute_model(execute_model_req)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError(
            "LoRA is currently not supported by the TPU backend.")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError(
            "LoRA is currently not supported by the TPU backend.")

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError(
            "LoRA is currently not supported by the TPU backend.")

    def list_loras(self) -> Set[int]:
        raise NotImplementedError(
            "LoRA is currently not supported by the TPU backend.")

    def add_prompt_adapter(self, prompt_adapter_request) -> bool:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the TPU backend.")

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the TPU backend.")

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the TPU backend.")

    def list_prompt_adapters(self) -> Set[int]:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the TPU backend.")

    def check_health(self) -> None:
        # TPUExecutor will always be healthy as long as it's running.
        return


class TPUExecutorAsync(TPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        sexecute_model_req: ExecuteModelRequest,
    ) -> SamplerOutput:
        output = await make_async(self.driver_worker.execute_model
                                  )(sexecute_model_req)
        return output
