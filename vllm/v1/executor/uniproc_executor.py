import os
from typing import Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.v1.executor.abstract import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


class UniprocExecutor(Executor):

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
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

        self.worker: Worker = self._create_worker()
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

    def determine_available_memory(self) -> int:
        """Determine the available memory (in bytes) for KV cache by invoking 
        the underlying worker.
        """
        return self.worker.determine_available_memory()

    def get_kv_cache_spec(self) -> KVCacheSpec:
        """Get all kv cache needed by the model by invoking the underlying
        worker.
        """
        return self.worker.get_kv_cache_spec()

    def initialize(self, kv_cache_config: KVCacheConfig) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        self.worker.initialize_cache(kv_cache_config)
        self.worker.compile_or_warm_up_model()

    def execute_model(
        self,
        scheduler_output,
    ) -> ModelRunnerOutput:
        output = self.worker.execute_model(scheduler_output)
        return output

    def profile(self, is_start: bool = True):
        self.worker.profile(is_start)

    def shutdown(self):
        pass

    def check_health(self) -> None:
        # UniprocExecutor will always be healthy as long as
        # it's running.
        return
