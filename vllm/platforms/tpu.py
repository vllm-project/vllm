# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Optional, Union, cast

import torch
from tpu_info import device

import vllm.envs as envs
from vllm.inputs import ProcessorInputs, PromptType
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import DEFAULT_MAX_NUM_BATCHED_TOKENS

from .interface import Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import BlockSize, ModelConfig, VllmConfig
    from vllm.pooling_params import PoolingParams
else:
    BlockSize = None
    ModelConfig = None
    VllmConfig = None
    PoolingParams = None

logger = init_logger(__name__)


class TpuPlatform(Platform):
    _enum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"
    dispatch_key: str = "XLA"
    ray_device_key: str = "TPU"
    device_control_env_var: str = "TPU_VISIBLE_CHIPS"
    simple_compile_backend: str = "openxla"

    supported_quantization: list[str] = ["tpu_int8", "compressed-tensors"]

    additional_env_vars: list[str] = [
        "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS"
    ]

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool,
                             use_mla: bool) -> str:
        if (selected_backend != _Backend.PALLAS
                and selected_backend != _Backend.PALLAS_VLLM_V1):
            logger.info("Cannot use %s backend on TPU.", selected_backend)

        if use_v1:
            logger.info("Using Pallas V1 backend.")
            return "vllm.v1.attention.backends.pallas.PallasAttentionBackend"
        else:
            logger.info("Using Pallas backend.")
            return "vllm.attention.backends.pallas.PallasAttentionBackend"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        chip_type, _ = device.get_local_chips()
        return f"TPU {chip_type.name}"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return not envs.VLLM_USE_V1

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_tpu.PunicaWrapperTPU"

    @classmethod
    def get_infinity_values(cls, dtype: torch.dtype) -> tuple[float, float]:
        return torch.finfo(dtype).min, torch.finfo(dtype).max

    @classmethod
    def can_update_inplace(cls):
        return False

    @classmethod
    def get_lora_vocab_padding_size(cls) -> int:
        return 1

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        from vllm.config import CompilationLevel

        cache_config = vllm_config.cache_config
        # For v0, the default block size is 16.
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = cast(BlockSize, 16)
        compilation_config = vllm_config.compilation_config

        # TPU only supports DYNAMO_ONCE compilation level
        if compilation_config.level != CompilationLevel.DYNAMO_ONCE:
            logger.info("[TPU] Forcing DYNAMO_ONCE compilation level")
            compilation_config.level = CompilationLevel.DYNAMO_ONCE

        if compilation_config.backend == "":
            compilation_config.backend = "openxla"

        assert vllm_config.speculative_config is None, \
            "TPU does not support speculative decoding"

        if vllm_config.model_config.dtype in (torch.float16, torch.float32):
            logger.warning(
                "The TPU backend currently does not support %s. "
                "Using bfloat16 instead.", vllm_config.model_config.dtype)
            vllm_config.model_config.dtype = torch.bfloat16

        if envs.VLLM_USE_V1:
            from vllm.v1.attention.backends.pallas import (
                PallasAttentionBackend)
            cache_config.block_size = PallasAttentionBackend.get_page_size(
                vllm_config)  # type: ignore[assignment]
            min_page_size = PallasAttentionBackend.get_min_page_size(
                vllm_config)
            if min_page_size > cache_config.block_size:
                logger.warning(
                    "Increase the page size from %s to %s to make sure there's"
                    "no SMEM OOM",
                    cache_config.block_size,
                    min_page_size,
                )
                cache_config.block_size = min_page_size  # type: ignore[assignment]

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        if parallel_config.worker_cls == "auto":
            if scheduler_config.is_multi_step:
                if envs.VLLM_USE_V1:
                    raise NotImplementedError(
                        "Multi-step scheduling is not supported (and not "
                        "needed) on vLLM V1. Please launch without "
                        "--num-scheduler-steps.")
                else:
                    parallel_config.worker_cls = \
                        "vllm.worker.multi_step_tpu_worker.MultiStepTPUWorker"
            else:
                if envs.VLLM_USE_V1:
                    parallel_config.worker_cls = \
                        "vllm.v1.worker.tpu_worker.TPUWorker"
                else:
                    parallel_config.worker_cls = \
                        "vllm.worker.tpu_worker.TPUWorker"

        assert not vllm_config.speculative_config, (
            "Speculative decoding is not yet supported for TPU backend")

        if scheduler_config.is_multimodal_model and not \
            scheduler_config.disable_chunked_mm_input:
            logger.warning("TPU does not support running Multimodal models"\
            " without setting `--disable_chunked_mm_input`. " \
            "Forcing --disable_chunked_mm_input.")
            scheduler_config.disable_chunked_mm_input = True

        if vllm_config.model_config and vllm_config.model_config.use_mla:
            logger.info(
                "MLA is enabled on a non-GPU platform; forcing chunked "
                "prefill and prefix caching to be disabled.")
            vllm_config.scheduler_config.enable_chunked_prefill = False
            vllm_config.scheduler_config.chunked_prefill_enabled = False
            vllm_config.scheduler_config.max_num_batched_tokens = max(
                vllm_config.scheduler_config.max_model_len,
                DEFAULT_MAX_NUM_BATCHED_TOKENS)

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on TPU.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.tpu_communicator.TpuCommunicator"  # noqa

    @classmethod
    def use_all_gather(cls) -> bool:
        return True

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        # V1 support on TPU is experimental
        return True

    @classmethod
    def validate_request(
        cls,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        processed_inputs: ProcessorInputs,
    ) -> None:
        """Raises if this request is unsupported on this platform"""
        if isinstance(params, SamplingParams):
            if params.guided_decoding is not None and not envs.VLLM_USE_V1:
                raise ValueError("Structured output is not supported on "
                                 f"{cls.device_name} V0.")
            if params.sampling_type == SamplingType.RANDOM_SEED:
                raise ValueError(
                    "Torch XLA does not support per-request seed.")


try:
    from tpu_commons.platforms import TpuPlatform as TpuCommonsPlatform
    TpuPlatform = TpuCommonsPlatform  # type: ignore
except ImportError:
    logger.info("tpu_commons not found, using vLLM's TpuPlatform")
    pass
