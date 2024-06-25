from typing import List, Set, Tuple

import openvino as ov
import openvino.properties.hint as hints
import torch

import vllm.envs as envs
from vllm.config import CacheConfig, ModelConfig
from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async)

logger = init_logger(__name__)


class OpenVINOExecutor(ExecutorBase):

    def _init_executor(self) -> None:
        assert self.device_config.device_type == "openvino"
        assert self.lora_config is None, "OpenVINO backend doesn't support LoRA"
        self.model_config = _verify_and_get_model_config(self.model_config)
        self.cache_config = _verify_and_get_cache_config(self.cache_config)

        # Instantiate the worker and load the model to CPU.
        self._init_worker()

    def _init_worker(self):
        from vllm.worker.openvino_worker import OpenVINOWorker

        assert (
            self.parallel_config.world_size == 1
        ), "OpenVINOExecutor only supports single CPU socket currently."

        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        self.driver_worker = OpenVINOWorker(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            load_config=self.load_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            vision_language_config=self.vision_language_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=True,
        )
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
        # NOTE: `cpu block` for OpenVINO backend is located on CPU memory but is
        # referred as `gpu block`. Because we want to reuse the existing block
        # management procedure.
        logger.info("# CPU blocks: %d", num_gpu_blocks)
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


def _verify_and_get_model_config(config: ModelConfig) -> ModelConfig:
    if config.dtype != torch.float32:
        logger.warning(
            f"Only float32 dtype is supported on OpenVINO, casting from {config.dtype}."  # noqa: G004, E501
        )
        config.dtype = torch.float32
    if not config.enforce_eager:
        logger.warning(
            "CUDA graph is not supported on OpenVINO backend, fallback to the "
            "eager mode.")
        config.enforce_eager = True
    return config


def _verify_and_get_cache_config(config: CacheConfig) -> CacheConfig:
    if envs.VLLM_OPENVINO_CPU_KV_CACHE_PRECISION == "u8":
        logger.info("KV cache type is overried to u8 via "
                    "VLLM_OPENVINO_CPU_KV_CACHE_PRECISION env var.")
        config.cache_dtype = ov.Type.u8
    else:
        core = ov.Core()
        inference_precision = core.get_property("CPU",
                                                hints.inference_precision)
        if inference_precision == ov.Type.bf16:
            config.cache_dtype = ov.Type.bf16
        else:
            config.cache_dtype = ov.Type.f16

    if config.block_size != 32:
        logger.info(
            f"OpenVINO optimal block size is 32, overriding currently set {config.block_size}"  # noqa: G004, E501
        )
        config.block_size = 32

    kv_cache_space = envs.VLLM_OPENVINO_KVCACHE_SPACE
    if kv_cache_space >= 0:
        _GB = 1 << 30
        if kv_cache_space == 0:
            config.openvino_kvcache_space_bytes = 4 * _GB  # type: ignore
            logger.warning(
                "Environment variable VLLM_OPENVINO_KVCACHE_SPACE (GB) "
                "for OpenVINO backend is not set, using 4 by default.")
        else:
            config.openvino_kvcache_space_bytes = kv_cache_space * _GB  # type: ignore
    else:
        raise RuntimeError(
            "Invalid environment variable VLLM_OPENVINO_KVCACHE_SPACE"
            f" {kv_cache_space}, expect a positive integer value.")

    return config
