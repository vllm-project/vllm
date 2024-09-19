from typing import Optional

import os
import torch
from vllm.platforms import current_platform
from vllm.model_executor.utils import set_random_seed

from vllm.wde.core.worker import WorkerBase
from vllm.wde.core.config import DeviceConfig, LoadConfig
from vllm.wde.encode_only.config import (ModelConfig,
                                         EncodeOnlySchedulerConfig,
                                         EncodeOnlyEngineConfig)
from vllm.wde.encode_only.runner.model_runner import ModelRunner
from vllm.wde.encode_only.schema.execute_io import EncodeOnlyExecuteInput
from vllm.wde.encode_only.layers.attention.backends.abstract import (
    EncodeOnlyAttentionBackend)
from vllm.config import ParallelConfig


class Worker(WorkerBase):

    def __init__(
        self,
        engine_config: EncodeOnlyEngineConfig,
        attn_backend: EncodeOnlyAttentionBackend,
    ) -> None:
        self.model_config: ModelConfig = engine_config.model_config
        self.scheduler_config: EncodeOnlySchedulerConfig = (
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

        self.dirty_fix_distributed_environment()

        # Set random seed.
        set_random_seed(self.model_config.seed)

    def dirty_fix_distributed_environment(self):
        from vllm.config import TokenizerPoolConfig
        from vllm.utils import get_distributed_init_method, get_open_port

        self.parallel_config = ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            worker_use_ray=False,
            max_parallel_loading_workers=None,
            disable_custom_all_reduce=False,
            tokenizer_pool_config=TokenizerPoolConfig.create_config(
                tokenizer_pool_size=0,
                tokenizer_pool_type="ray",
                tokenizer_pool_extra_config=None,
            ),
            ray_workers_use_nsight=False,
            distributed_executor_backend=None)

        ip = "127.0.0.1"
        port = get_open_port()
        distributed_init_method = get_distributed_init_method(ip, port)

        init_worker_distributed_environment(self.parallel_config, 0,
                                            distributed_init_method, 0)

    @torch.inference_mode
    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode
    def __call__(self, execute_input: EncodeOnlyExecuteInput):
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


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    from vllm.distributed import (ensure_model_parallel_initialized,
                                  init_distributed_environment,
                                  set_custom_all_reduce)
    """Initialize the distributed environment."""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)
