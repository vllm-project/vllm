from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import os

from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs_v2 import ModelRunnerOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async)
from vllm.worker.worker_v2 import Worker

logger = init_logger(__name__)


class GPUExecutor(ExecutorBase):

    uses_ray: bool = False

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """
        assert self.parallel_config.world_size == 1, (
            "GPUExecutor only supports single GPU.")

        self.driver_worker = self._create_worker()
        self.driver_worker.initialize()
        self.driver_worker.load_model()

    def _create_worker(
            self,
            local_rank: int = 0,
            rank: int = 0,
            distributed_init_method: Optional[str] = None) -> Worker:
        """Return worker init args for a given rank."""
        # see https://github.com/NVIDIA/nccl/issues/1234
        os.environ['NCCL_CUMEM_ENABLE'] = '0'

        if distributed_init_method is None:
            distributed_init_method = get_distributed_init_method(
                get_ip(), get_open_port())
        return Worker(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            load_config=self.load_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            speculative_config=self.speculative_config,
            prompt_adapter_config=self.prompt_adapter_config,
            is_driver_worker=(not self.parallel_config)
            or (rank % self.parallel_config.tensor_parallel_size == 0),
            observability_config=self.observability_config,
        )

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        return self.driver_worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # NOTE: This is logged in the executor because there can be >1 worker
        # with other executors. We could log in the engine level, but work
        # remains to abstract away the device for non-GPU configurations.
        logger.info("# GPU blocks: %d, # CPU blocks: %d", num_gpu_blocks,
                    num_cpu_blocks)

        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def execute_model(
        self, scheduler_output,
    ) -> Optional[List[Union[ModelRunnerOutput]]]:
        output = self.driver_worker.execute_model(scheduler_output)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.driver_worker.list_loras()

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        assert prompt_adapter_request.prompt_adapter_id > 0, \
            "prompt_adapter_id must be greater than 0."
        return self.driver_worker.add_prompt_adapter(prompt_adapter_request)

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        assert prompt_adapter_id > 0, \
            "prompt_adapter_id must be greater than 0."
        return self.driver_worker.remove_prompt_adapter(prompt_adapter_id)

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        assert prompt_adapter_id > 0, \
                "prompt_adapter_id must be greater than 0."
        return self.driver_worker.pin_prompt_adapter(prompt_adapter_id)

    def list_prompt_adapters(self) -> Set[int]:
        return self.driver_worker.list_prompt_adapters()

    def check_health(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return

    def start_profile(self) -> None:
        self.driver_worker.start_profile()

    def stop_profile(self) -> None:
        self.driver_worker.stop_profile()


class GPUExecutorAsync(GPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        scheduler_output,
    ) -> List[Union[ModelRunnerOutput]]:
        output = await make_async(self.driver_worker.execute_model
                                  )(scheduler_output=scheduler_output)
        return output
