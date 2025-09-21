# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Optional, Union, cast

import torch
from tpu_info import device

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

USE_TPU_COMMONS = False


class TpuPlatform(Platform):
    _enum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"
    dispatch_key: str = "XLA"
    ray_device_key: str = "TPU"
    dist_backend: str = "gloo"
    device_control_env_var: str = "TPU_VISIBLE_CHIPS"
    simple_compile_backend: str = "openxla"

    supported_quantization: list[str] = [
        "fp8", "tpu_int8", "compressed-tensors"
    ]

    additional_env_vars: list[str] = [
        "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS"
    ]

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool, use_mla: bool,
                             has_sink) -> str:
        if (selected_backend != _Backend.PALLAS
                and selected_backend != _Backend.PALLAS_VLLM_V1):
            logger.info("Cannot use %s backend on TPU.", selected_backend)

        if not use_v1:
            raise ValueError("TPU backend only supports V1.")
        logger.info("Using Pallas V1 backend.")
        return "vllm.v1.attention.backends.pallas.PallasAttentionBackend"

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.tpu.set_device(device)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        chip_type, _ = device.get_local_chips()
        return f"TPU {chip_type.name}"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

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
        from vllm.config import CompilationLevel, CUDAGraphMode

        cache_config = vllm_config.cache_config
        # For v0, the default block size is 16.
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = cast(BlockSize, 16)
        compilation_config = vllm_config.compilation_config

        # TPU only supports DYNAMO_ONCE compilation level
        if compilation_config.level != CompilationLevel.DYNAMO_ONCE:
            logger.info("[TPU] Forcing DYNAMO_ONCE compilation level, and "
                        "disabling cudagraph.")
            compilation_config.level = CompilationLevel.DYNAMO_ONCE

        if compilation_config.cudagraph_mode is None or \
                compilation_config.cudagraph_mode.max_cudagraph_mode() \
                    != CUDAGraphMode.NONE:
            logger.info("[TPU] CUDA graph is not supported on TPU, "
                        "disabling cudagraphs.")
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        if compilation_config.backend == "":
            compilation_config.backend = "openxla"

        assert vllm_config.speculative_config is None, \
            "TPU does not support speculative decoding"

        model_config = vllm_config.model_config
        if model_config is not None and model_config.dtype in (torch.float16,
                                                               torch.float32):
            logger.warning(
                "The TPU backend currently does not support %s. "
                "Using bfloat16 instead.", model_config.dtype)
            model_config.dtype = torch.bfloat16

        from vllm.v1.attention.backends.pallas import PallasAttentionBackend
        cache_config.block_size = PallasAttentionBackend.get_page_size(
            vllm_config)  # type: ignore[assignment]

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.tpu_worker.TPUWorker"

        assert not vllm_config.speculative_config, (
            "Speculative decoding is not yet supported for TPU backend")

        if scheduler_config.is_multimodal_model and not \
                scheduler_config.disable_chunked_mm_input:
            logger.warning("TPU does not support running Multimodal models"\
            " without setting `--disable_chunked_mm_input`. " \
            "Forcing --disable_chunked_mm_input.")
            scheduler_config.disable_chunked_mm_input = True

        if model_config and model_config.use_mla:
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
        if (isinstance(params, SamplingParams)
                and params.sampling_type == SamplingType.RANDOM_SEED):
            raise ValueError("Torch XLA does not support per-request seed.")

    @classmethod
    def is_kv_cache_dtype_supported(cls, kv_cache_dtype: str,
                                    model_config: "ModelConfig") -> bool:
        return True

    @classmethod
    @torch.compile(backend="openxla")
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        torch.ops.xla.dynamo_set_buffer_donor_(dst_cache, True)
        dst_cache[dst_block_indices] = src_cache[src_block_indices].to(
            dst_cache.device)

    @classmethod
    @torch.compile(backend="openxla")
    def swap_out_blocks_to_host(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """ tpu blocks to cpu blocks"""
        torch.ops.xla.dynamo_set_buffer_donor_(src_cache, True)
        dst_cache[dst_block_indices] = src_cache[src_block_indices].cpu()

    @classmethod
    def use_sync_weight_loader(cls) -> bool:
        return True


try:
    from tpu_commons.platforms import TpuPlatform as TpuCommonsPlatform
    TpuPlatform = TpuCommonsPlatform  # type: ignore
    USE_TPU_COMMONS = True
except ImportError:
    logger.info("tpu_commons not found, using vLLM's TpuPlatform")
    pass
