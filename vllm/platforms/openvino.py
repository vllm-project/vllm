from typing import TYPE_CHECKING, Optional

import torch

import vllm.envs as envs
from vllm.logger import init_logger

from .interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)

try:
    import openvino as ov
    import openvino.properties.hint as hints
except ImportError as e:
    logger.warning("Failed to import OpenVINO with %r", e)


class OpenVinoPlatform(Platform):
    _enum = PlatformEnum.OPENVINO
    device_name: str = "openvino"
    device_type: str = "openvino"
    dispatch_key: str = "CPU"

    @classmethod
    def get_default_attn_backend(cls, selected_backend: _Backend) -> _Backend:
        if selected_backend != _Backend.OPENVINO:
            logger.info("Cannot use %s backend on OpenVINO.", selected_backend)
        return _Backend.OPENVINO

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "openvino"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False

    @classmethod
    def inference_mode(cls):
        return torch.inference_mode(mode=True)

    @classmethod
    def is_openvino_cpu(cls) -> bool:
        return "CPU" in envs.VLLM_OPENVINO_DEVICE

    @classmethod
    def is_openvino_gpu(cls) -> bool:
        return "GPU" in envs.VLLM_OPENVINO_DEVICE

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on OpenViNO.")
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        from vllm.utils import GiB_bytes

        parallel_config = vllm_config.parallel_config
        assert (
            parallel_config.world_size == 1
        ), "OpenVINOExecutor only supports single CPU socket currently."

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = \
                "vllm.worker.openvino_worker.OpenVINOWorker"

        # check and update model config
        model_config = vllm_config.model_config
        if model_config.dtype != torch.float32:
            logger.warning(
                f"Only float32 dtype is supported on OpenVINO, casting from {model_config.dtype}."  # noqa: G004, E501
            )
            model_config.dtype = torch.float32
        if not model_config.enforce_eager:
            logger.warning(
                "CUDA graph is not supported on OpenVINO backend, fallback to "
                "the eager mode.")
            model_config.enforce_eager = True

        # check and update cache config
        ov_core = ov.Core()
        cache_config = vllm_config.cache_config
        if envs.VLLM_OPENVINO_CPU_KV_CACHE_PRECISION == "u8":
            if not OpenVinoPlatform.is_openvino_cpu():
                logger.info("VLLM_OPENVINO_CPU_KV_CACHE_PRECISION is"
                            "ignored for GPU, f16 data type will be used.")
                cache_config.cache_dtype = ov.Type.f16
            else:
                logger.info("KV cache type is overridden to u8 via "
                            "VLLM_OPENVINO_CPU_KV_CACHE_PRECISION env var.")
                cache_config.cache_dtype = ov.Type.u8
        else:
            if OpenVinoPlatform.is_openvino_cpu():
                ov_device = envs.VLLM_OPENVINO_DEVICE
                inference_precision = ov_core.get_property(
                    ov_device, hints.inference_precision)
                if inference_precision == ov.Type.bf16:
                    cache_config.cache_dtype = ov.Type.bf16
                else:
                    cache_config.cache_dtype = ov.Type.f16
            else:
                cache_config.cache_dtype = ov.Type.f16

        if OpenVinoPlatform.is_openvino_cpu():
            if cache_config.block_size != 32:
                logger.info(
                    f"OpenVINO CPU optimal block size is 32, overriding currently set {cache_config.block_size}"  # noqa: G004, E501
                )
                cache_config.block_size = 32
        else:
            if cache_config.block_size != 16:
                logger.info(
                    f"OpenVINO GPU optimal block size is 16, overriding currently set {cache_config.block_size}"  # noqa: G004, E501
                )
                cache_config.block_size = 16

        kv_cache_space = envs.VLLM_OPENVINO_KVCACHE_SPACE
        if kv_cache_space >= 0:
            if kv_cache_space == 0 and OpenVinoPlatform.is_openvino_cpu():
                cache_config.openvino_kvcache_space_bytes = 4 * GiB_bytes  # type: ignore
                logger.warning(
                    "Environment variable VLLM_OPENVINO_KVCACHE_SPACE (GB) "
                    "for OpenVINO backend is not set, using 4 by default.")
            else:
                cache_config.openvino_kvcache_space_bytes = (  # type: ignore
                    kv_cache_space * GiB_bytes)
        else:
            raise RuntimeError(
                "Invalid environment variable VLLM_OPENVINO_KVCACHE_SPACE"
                f" {kv_cache_space}, expect a positive integer value.")
