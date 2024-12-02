from typing import List, Set, Tuple

import openvino as ov

import vllm.envs as envs
from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.platforms import current_platform
from vllm.sequence import ExecuteModelRequest
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async)
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class OpenVINOExecutor(ExecutorBase):

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert self.device_config.device_type == "openvino"
        assert self.lora_config is None, "OpenVINO backend doesn't support LoRA"
        assert current_platform.is_openvino_cpu() or \
            current_platform.is_openvino_gpu(), \
            "OpenVINO backend supports only CPU and GPU devices"

        # Instantiate the worker and load the model to CPU.
        self._init_worker()

    def _init_worker(self):

        wrapper = WorkerWrapperBase(vllm_config=self.vllm_config)

        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        wrapper.init_worker(
            ov_core=ov.Core(),
            vllm_config=self.vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=True,
        )
        self.driver_worker = wrapper.worker
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        return self.driver_worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache by invoking the underlying worker."""
        # NOTE: We log here to avoid multiple logs when number of workers is
        # greater than one. We could log in the engine, but not all executors
        # have GPUs.
        # NOTE: In case of a CPU device, `cpu block` for OpenVINO backend
        # is located on CPU memory but is referred as `gpu block`.
        # Because we want to reuse the existing block management procedure.
        device_blocks = num_gpu_blocks
        swap_blocks = num_cpu_blocks
        logger.info("OpenVINO %s: # device blocks: %d; # swap blocks: %d",
                    envs.VLLM_OPENVINO_DEVICE, device_blocks, swap_blocks)
        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        output = self.driver_worker.execute_model(execute_model_req)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.driver_worker.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.driver_worker.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self.driver_worker.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.driver_worker.list_loras()

    def add_prompt_adapter(self, prompt_adapter_request) -> bool:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the OPENVINO backend.")

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the OPENVINO backend.")

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the OPENVINO backend.")

    def list_prompt_adapters(self) -> Set[int]:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the OPENVINO backend.")

    def check_health(self) -> None:
        # OpenVINOExecutor will always be healthy as long as
        # it's running.
        return


class OpenVINOExecutorAsync(OpenVINOExecutor, ExecutorAsyncBase):

    async def execute_model_async(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        output = await make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req, )
        return output

    async def check_health_async(self) -> None:
        # OpenVINOExecutor will always be healthy as long as
        # it's running.
        return
