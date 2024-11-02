from typing import List, Set, Tuple

from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async)

logger = init_logger(__name__)


class NeuronExecutor(ExecutorBase):

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert (self.lora_config is
                None), "LoRA is not supported for Neuron backend."
        assert (not self.speculative_config
                ), "Speculative decoding not yet supported for Neuron backend."

        # Instantiate the worker and load the model to the device.
        self._init_worker()

    def _init_worker(self):
        from vllm.worker.neuron_worker import NeuronWorker
        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        self.driver_worker = NeuronWorker(
            vllm_config=self.vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method)
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

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        assert (not execute_model_req.blocks_to_swap_in
                and not execute_model_req.blocks_to_swap_out
                and not execute_model_req.blocks_to_copy), (
                    "Cache operations are not supported for Neuron backend.")
        assert execute_model_req.num_lookahead_slots == 0, (
            "lookahead not supported for Neuron backend.")

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
            "Soft prompt is currently not supported by the Neuron backend.")

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the Neuron backend.")

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the Neuron backend.")

    def list_prompt_adapters(self) -> Set[int]:
        raise NotImplementedError(
            "Soft prompt is currently not supported by the Neuron backend.")

    def check_health(self) -> None:
        # NeuronExecutor will always be healthy as long as
        # it's running.
        return


class NeuronExecutorAsync(NeuronExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[SamplerOutput]:
        output = await make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req, )
        return output

    async def check_health_async(self) -> None:
        # NeuronExecutor will always be healthy as long as
        # it's running.
        return
