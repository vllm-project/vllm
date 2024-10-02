from typing import List, Set, Tuple

import openvino as ov
import openvino.properties.hint as hints
import torch

import vllm.envs as envs
from vllm.config import CacheConfig, ModelConfig
from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.utils import (GiB_bytes, get_distributed_init_method, get_ip,
                        get_open_port, make_async)

logger = init_logger(__name__)


def is_openvino_cpu() -> bool:
    return "CPU" in envs.VLLM_OPENVINO_DEVICE


def is_openvino_gpu() -> bool:
    return "GPU" in envs.VLLM_OPENVINO_DEVICE


class OpenVINOExecutor(ExecutorBase):

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert self.device_config.device_type == "openvino"
        assert self.lora_config is None, "OpenVINO backend doesn't support LoRA"
        assert is_openvino_cpu() or is_openvino_gpu(), \
            "OpenVINO backend supports only CPU and GPU devices"

        self.ov_core = ov.Core()
        self.model_config = _verify_and_get_model_config(self.model_config)
        self.cache_config = _verify_and_get_cache_config(
            self.ov_core, self.cache_config)

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
            ov_core=self.ov_core,
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


def _verify_and_get_cache_config(ov_core: ov.Core,
                                 config: CacheConfig) -> CacheConfig:
    if envs.VLLM_OPENVINO_CPU_KV_CACHE_PRECISION == "u8":
        if not is_openvino_cpu():
            logger.info("VLLM_OPENVINO_CPU_KV_CACHE_PRECISION is"
                        "ignored for GPU, f16 data type will be used.")
            config.cache_dtype = ov.Type.f16
        else:
            logger.info("KV cache type is overridden to u8 via "
                        "VLLM_OPENVINO_CPU_KV_CACHE_PRECISION env var.")
            config.cache_dtype = ov.Type.u8
    else:
        if is_openvino_cpu():
            ov_device = envs.VLLM_OPENVINO_DEVICE
            inference_precision = ov_core.get_property(
                ov_device, hints.inference_precision)
            if inference_precision == ov.Type.bf16:
                config.cache_dtype = ov.Type.bf16
            else:
                config.cache_dtype = ov.Type.f16
        else:
            config.cache_dtype = ov.Type.f16

    if is_openvino_cpu():
        if config.block_size != 32:
            logger.info(
                f"OpenVINO CPU optimal block size is 32, overriding currently set {config.block_size}"  # noqa: G004, E501
            )
            config.block_size = 32
    else:
        if config.block_size != 16:
            logger.info(
                f"OpenVINO GPU optimal block size is 16, overriding currently set {config.block_size}"  # noqa: G004, E501
            )
            config.block_size = 16

    kv_cache_space = envs.VLLM_OPENVINO_KVCACHE_SPACE
    if kv_cache_space >= 0:
        if kv_cache_space == 0 and is_openvino_cpu():
            config.openvino_kvcache_space_bytes = 4 * GiB_bytes  # type: ignore
            logger.warning(
                "Environment variable VLLM_OPENVINO_KVCACHE_SPACE (GB) "
                "for OpenVINO backend is not set, using 4 by default.")
        else:
            config.openvino_kvcache_space_bytes = kv_cache_space * GiB_bytes  # type: ignore
    else:
        raise RuntimeError(
            "Invalid environment variable VLLM_OPENVINO_KVCACHE_SPACE"
            f" {kv_cache_space}, expect a positive integer value.")

    return config
