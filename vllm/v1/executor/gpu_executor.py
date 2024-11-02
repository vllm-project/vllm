import os
from typing import Optional, Tuple

from vllm.config import EngineConfig
from vllm.logger import init_logger
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


class GPUExecutor:

    def __init__(self, vllm_config: EngineConfig) -> None:
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        self.worker = self._create_worker()
        self.worker.initialize()
        self.worker.load_model()

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
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
        )

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        return self.worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # NOTE: This is logged in the executor because there can be >1 worker
        # with other executors. We could log in the engine level, but work
        # remains to abstract away the device for non-GPU configurations.
        logger.info("# GPU blocks: %d", num_gpu_blocks)
        self.worker.initialize_cache(num_gpu_blocks)
        self.worker.compile_or_warm_up_model()

    def execute_model(
        self,
        scheduler_output,
    ) -> ModelRunnerOutput:
        output = self.worker.execute_model(scheduler_output)
        return output

    def check_health(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return
