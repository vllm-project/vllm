import os

import torch

from vllm.model_executor.utils import set_random_seed
from vllm.platforms import current_platform
from vllm.wde.core.config import (DeviceConfig, EngineConfig, LoadConfig,
                                  ModelConfig)
from vllm.wde.core.layers.attention import AttentionBackend
from vllm.wde.core.schema.execute_io import ExecuteOutput
from vllm.wde.core.worker import WorkerBase
from vllm.wde.prefill_only.config import PrefillOnlySchedulerConfig
from vllm.wde.prefill_only.runner.model_runner import ModelRunner
from vllm.wde.prefill_only.schema.execute_io import PrefillOnlyExecuteInput


class Worker(WorkerBase):

    def __init__(
        self,
        engine_config: EngineConfig,
        attn_backend: AttentionBackend,
    ) -> None:
        self.model_config: ModelConfig = engine_config.model_config
        self.scheduler_config: PrefillOnlySchedulerConfig = (
            engine_config.scheduler_config)
        self.device_config: DeviceConfig = engine_config.device_config
        self.load_config: LoadConfig = engine_config.load_config
        self.device = self.device_config.device
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        self.model_runner = ModelRunner(self.model_config,
                                        self.scheduler_config,
                                        self.device_config, self.load_config,
                                        attn_backend)

    def init_device(self) -> None:
        from vllm.wde.core.worker.utils import fix_distributed_environment

        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")

        fix_distributed_environment()

        # Set random seed.
        set_random_seed(self.model_config.seed)

    @torch.inference_mode
    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode
    def __call__(self,
                 execute_input: PrefillOnlyExecuteInput) -> ExecuteOutput:
        output = self.model_runner.execute_model(execute_input.model_input)
        return output


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = current_platform.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")
