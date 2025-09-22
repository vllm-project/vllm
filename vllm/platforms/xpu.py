# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TYPE_CHECKING, Optional

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils import DEFAULT_MAX_NUM_BATCHED_TOKENS

from .interface import DeviceCapability, Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
else:
    ModelConfig = None
    VllmConfig = None

logger = init_logger(__name__)


class XPUPlatform(Platform):
    _enum = PlatformEnum.XPU
    device_name: str = "xpu"
    device_type: str = "xpu"
    dispatch_key: str = "XPU"
    # Intel XPU's device key is "GPU" for Ray.
    # see https://github.com/ray-project/ray/blob/6a5eb5865eeb9ccf058a79b44f107e327e360673/python/ray/_private/accelerators/intel_gpu.py#L20 # noqa: E501
    ray_device_key: str = "GPU"
    dist_backend: str = "ccl"  # ccl | xccl
    device_control_env_var: str = "ZE_AFFINITY_MASK"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool, use_mla: bool,
                             has_sink: bool) -> str:
        use_v1 = envs.VLLM_USE_V1
        if not use_v1:
            raise ValueError("XPU backend only supports V1.")
        TRITON_ATTN_VLLM_V1 = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"  # noqa: E501
        FLASH_ATTN_V1 = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"  # noqa: E501
        if selected_backend == _Backend.TRITON_ATTN_VLLM_V1:
            logger.info_once("Using Triton backend on V1 engine.")
            return TRITON_ATTN_VLLM_V1
        elif selected_backend == _Backend.FLASH_ATTN:
            logger.info_once("Using Flash Attention backend on V1 engine.")
            return FLASH_ATTN_V1
        elif selected_backend:
            raise ValueError(
                f"Invalid attention backend for {cls.device_name}, "
                f"with use_v1: {use_v1} use_mla: {use_mla}")

        logger.info("Using Flash Attention backend on V1 engine.")
        return "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"

    @classmethod
    def is_kv_cache_dtype_supported(cls, kv_cache_dtype: str,
                                    model_config: "ModelConfig") -> bool:
        """
        Check if the kv_cache_dtype is supported.
        XPU only support fp8 kv cache with triton backend.
        """
        if envs.is_set("VLLM_ATTENTION_BACKEND") and \
            envs.VLLM_ATTENTION_BACKEND == "TRITON_ATTN_VLLM_V1":
            return kv_cache_dtype in ["fp8_e4m3", "fp8_e5m2", "fp8"]

        return False

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.xpu.set_device(device)

    @classmethod
    def get_device_capability(
        cls,
        device_id: int = 0,
    ) -> Optional[DeviceCapability]:
        # capacity format differs from cuda's and will cause unexpected
        # failure, so use None directly
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.xpu.get_device_name(device_id)

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_xpu.PunicaWrapperXPU"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.xpu.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        # in V1(or with ipex chunked prefill) block_size is 64
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 64

        # lazy import to avoid circular import
        from vllm.config import CompilationLevel, CUDAGraphMode
        compilation_config = vllm_config.compilation_config
        if compilation_config.cudagraph_mode is None or \
                compilation_config.cudagraph_mode.max_cudagraph_mode() \
                    != CUDAGraphMode.NONE:
            logger.info("[XPU] CUDA graph is not supported on XPU, disabling "
                        "cudagraphs. Fallback to cudagraph_mode=NONE")
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        if vllm_config.lora_config is not None:
            compilation_config.level = CompilationLevel.NO_COMPILATION

        # check and update parallel config
        parallel_config = vllm_config.parallel_config
        parallel_config.worker_cls = "vllm.v1.worker.xpu_worker.XPUWorker"

        if parallel_config.distributed_executor_backend is None:
            if parallel_config.world_size > 1:
                parallel_config.distributed_executor_backend = "ray"
            else:
                parallel_config.distributed_executor_backend = "uni"
        elif parallel_config.distributed_executor_backend == "mp":
            # FIXME(kunshang):
            # spawn needs calling `if __name__ == '__main__':``
            # fork is not supported for xpu start new process.
            if envs.VLLM_WORKER_MULTIPROC_METHOD != "spawn":
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
                logger.warning(
                    "Please use spawn as start method if you want to use mp.")
        elif (parallel_config.distributed_executor_backend != "ray"
              and parallel_config.distributed_executor_backend != "uni"
              and parallel_config.distributed_executor_backend
              != "external_launcher"):
            logger.warning(
                "%s is not supported on XPU, fallback to ray distributed"
                " executor backend.",
                parallel_config.distributed_executor_backend)
            parallel_config.distributed_executor_backend = "ray"

        if model_config and model_config.use_mla:
            logger.info(
                "MLA is enabled on a non-GPU platform; forcing chunked "
                "prefill and prefix caching to be disabled.")
            vllm_config.scheduler_config.enable_chunked_prefill = False
            vllm_config.scheduler_config.chunked_prefill_enabled = False
            vllm_config.scheduler_config.max_num_batched_tokens = max(
                vllm_config.scheduler_config.max_model_len,
                DEFAULT_MAX_NUM_BATCHED_TOKENS)
        from vllm.v1.attention.backends.utils import set_kv_cache_layout

        set_kv_cache_layout("NHD")
        logger.info("Setting VLLM_KV_CACHE_LAYOUT to 'NHD' for XPU; "
                    "only NHD layout is supported by XPU attention kernels.")

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def is_pin_memory_available(cls):
        return True

    @classmethod
    def get_current_memory_usage(cls,
                                 device: Optional[torch.types.Device] = None
                                 ) -> float:
        torch.xpu.reset_peak_memory_stats(device)
        return torch.xpu.max_memory_allocated(device)

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        return torch.float8_e5m2

    @classmethod
    def is_data_center_gpu(cls) -> bool:
        device_name = cls.get_device_name().lower()
        return device_name.count("data center gpu") > 0

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.xpu_communicator.XpuCommunicator"  # noqa

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        return True

    @classmethod
    def device_count(cls) -> int:
        return torch.xpu.device_count()

    @classmethod
    def check_if_supports_dtype(cls, torch_dtype: torch.dtype):
        if torch_dtype == torch.bfloat16:  # noqa: SIM102
            device_name = cls.get_device_name().lower()
            # client gpu a770
            if device_name.count("a770") > 0:
                raise ValueError(
                    "Intel Arc A770 have bfloat16 accuracy known issue. "
                    "You can use float16 instead by explicitly setting the "
                    "`dtype` flag in CLI, for example: --dtype=half.")

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from src_cache to dst_cache on XPU."""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.to(dst_cache.device)

    @classmethod
    def swap_out_blocks_to_host(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from XPU to host (CPU)."""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.cpu()
