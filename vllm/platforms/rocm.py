# SPDX-License-Identifier: Apache-2.0

import os
from functools import cache, lru_cache, wraps
from typing import TYPE_CHECKING, Optional

import torch

import vllm.envs as envs
from vllm.logger import init_logger

from .interface import DeviceCapability, Platform, PlatformEnum, _Backend

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)

try:
    from amdsmi import (AmdSmiException, amdsmi_get_gpu_asic_info,
                        amdsmi_get_processor_handles, amdsmi_init,
                        amdsmi_shut_down, amdsmi_topo_get_link_type)
except ImportError as e:
    logger.warning("Failed to import from amdsmi with %r", e)

try:
    import vllm._C  # noqa: F401
except ImportError as e:
    logger.warning("Failed to import from vllm._C with %r", e)

# import custom ops, trigger op registration
try:
    import vllm._rocm_C  # noqa: F401
except ImportError as e:
    logger.warning("Failed to import from vllm._rocm_C with %r", e)

# Models not supported by ROCm.
_ROCM_UNSUPPORTED_MODELS: list[str] = []

# Models partially supported by ROCm.
# Architecture -> Reason.
_ROCM_SWA_REASON = ("Sliding window attention (SWA) is not yet supported in "
                    "Triton flash attention. For half-precision SWA support, "
                    "please use CK flash attention by setting "
                    "`VLLM_USE_TRITON_FLASH_ATTN=0`")
_ROCM_PARTIALLY_SUPPORTED_MODELS: dict[str, str] = {
    "Qwen2ForCausalLM":
    _ROCM_SWA_REASON,
    "MistralForCausalLM":
    _ROCM_SWA_REASON,
    "MixtralForCausalLM":
    _ROCM_SWA_REASON,
    "PaliGemmaForConditionalGeneration":
    ("ROCm flash attention does not yet "
     "fully support 32-bit precision on PaliGemma"),
    "Phi3VForCausalLM":
    ("ROCm Triton flash attention may run into compilation errors due to "
     "excessive use of shared memory. If this happens, disable Triton FA "
     "by setting `VLLM_USE_TRITON_FLASH_ATTN=0`")
}
_ROCM_DEVICE_ID_NAME_MAP: dict[str, str] = {
    "0x74a0": "AMD_Instinct_MI300A",
    "0x74a1": "AMD_Instinct_MI300X",
    "0x74b5": "AMD_Instinct_MI300X",  # MI300X VF
    "0x74a5": "AMD_Instinct_MI325X",
    "0x74b9": "AMD_Instinct_MI325X",  # MI325X VF
    "0x74a9": "AMD_Instinct_MI300X_HF",
    "0x74bd": "AMD_Instinct_MI300X_HF",
}

# Prevent use of clashing `{CUDA/HIP}_VISIBLE_DEVICES``
if "HIP_VISIBLE_DEVICES" in os.environ:
    val = os.environ["HIP_VISIBLE_DEVICES"]
    if cuda_val := os.environ.get("CUDA_VISIBLE_DEVICES", None):
        assert val == cuda_val
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = val

# AMDSMI utils
# Note that NVML is not affected by `{CUDA/HIP}_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using AMDSMI is that it will not initialize CUDA


def with_amdsmi_context(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        amdsmi_init()
        try:
            return fn(*args, **kwargs)
        finally:
            amdsmi_shut_down()

    return wrapper


@cache
def on_mi250_mi300() -> bool:
    GPU_ARCH = torch.cuda.get_device_properties("cuda").gcnArchName
    return any(arch in GPU_ARCH for arch in ["gfx90a", "gfx942"])


@cache
def use_rocm_custom_paged_attention(qtype: torch.dtype, head_size: int,
                                    block_size: int, gqa_ratio: int,
                                    max_seq_len: int,
                                    sliding_window: int) -> bool:

    GPU_ARCH = torch.cuda.get_device_properties("cuda").gcnArchName
    ON_GFX9 = any(arch in GPU_ARCH for arch in ["gfx90a", "gfx942", "gfx950"])

    # rocm custom page attention not support on gfx1*
    # custom paged attn always supported on V0. On V1, requires sliding window
    # disabled due to observed numerical discrepancy.
    return (ON_GFX9 and (not envs.VLLM_USE_V1 or sliding_window == 0
                         or sliding_window == (-1, -1))
            and (qtype == torch.half or qtype == torch.bfloat16)
            and (head_size == 64 or head_size == 128)
            and (block_size == 16 or block_size == 32)
            and (gqa_ratio >= 1 and gqa_ratio <= 16) and max_seq_len <= 32768
            and (envs.VLLM_ROCM_CUSTOM_PAGED_ATTN)
            and not (envs.VLLM_ROCM_USE_AITER_PAGED_ATTN
                     and envs.VLLM_ROCM_USE_AITER))


class RocmPlatform(Platform):
    _enum = PlatformEnum.ROCM
    device_name: str = "rocm"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    # rocm shares the same device control env var as CUDA
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    supported_quantization: list[str] = [
        "awq", "gptq", "fp8", "compressed-tensors", "fbgemm_fp8", "gguf",
        "quark", "ptpc_fp8"
    ]

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1,
                             use_mla) -> str:
        if use_mla:
            from vllm.attention.backends.rocm_aiter_mla import (
                is_aiter_mla_enabled)

            if selected_backend is None:
                selected_backend = (_Backend.ROCM_AITER_MLA if
                                    is_aiter_mla_enabled() or block_size == 1
                                    else _Backend.TRITON_MLA)

            if selected_backend == _Backend.TRITON_MLA:
                if block_size != 1:
                    logger.info("Using Triton MLA backend.")
                    return "vllm.attention.backends.triton_mla.TritonMLABackend"  # noqa: E501
                else:
                    raise ValueError(
                        f" The selected backend, {selected_backend.name},"
                        f"does not support block size {block_size}.")
            elif selected_backend == _Backend.ROCM_AITER_MLA \
                or selected_backend == _Backend.ROCM_AITER_MLA_VLLM_V1:
                if block_size == 1:
                    if use_v1:
                        logger.info("Using AITER MLA backend on V1 engine.")
                        return "vllm.v1.attention.backends.mla.rocm_aiter_mla.AiterMLABackend"  # noqa: E501
                    else:
                        logger.info("Using AITER MLA backend")
                        return "vllm.attention.backends.rocm_aiter_mla.AiterMLABackend"  # noqa: E501
                else:
                    raise ValueError(
                        f" The selected backend, {selected_backend.name},"
                        f"does not support block size {block_size}."
                        "(currently only supports block size 1)")
            else:
                raise ValueError(
                    f" The selected backend, {selected_backend.name},"
                    f"is not MLA type while requested for MLA backend.")

        selected_backend = (_Backend.ROCM_FLASH if selected_backend
                            == _Backend.FLASH_ATTN else selected_backend)
        if envs.VLLM_USE_V1:
            logger.info("Using Triton Attention backend on V1 engine.")
            return ("vllm.v1.attention.backends."
                    "triton_attn.TritonAttentionBackend")
        if selected_backend == _Backend.ROCM_FLASH:
            if not cls.has_device_capability(90):
                # not Instinct series GPUs.
                logger.info("flash_attn is not supported on NAVI GPUs.")
        else:
            logger.info("%s is not supported in AMD GPUs.", selected_backend)
        logger.info("Using ROCmFlashAttention backend.")
        return "vllm.attention.backends.rocm_flash_attn.ROCmFlashAttentionBackend"  # noqa: E501

    @classmethod
    @lru_cache(maxsize=8)
    def get_device_capability(cls,
                              device_id: int = 0
                              ) -> Optional[DeviceCapability]:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @staticmethod
    @with_amdsmi_context
    def is_fully_connected(physical_device_ids: list[int]) -> bool:
        """
        Query if the set of gpus are fully connected by xgmi (1 hop)
        """
        handles = [
            amdsmi_get_processor_handles()[i] for i in physical_device_ids
        ]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        link_type = amdsmi_topo_get_link_type(
                            handle, peer_handle)
                        # type is 2 for XGMI
                        if link_type["hops"] != 1 or link_type["type"] != 2:
                            return False
                    except AmdSmiException as error:
                        logger.error("AMD 1 hop XGMI detection failed.",
                                     exc_info=error)
                        return False
        return True

    @classmethod
    @with_amdsmi_context
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = amdsmi_get_processor_handles()[physical_device_id]
        asic_info = amdsmi_get_gpu_asic_info(handle)
        device_name: str = asic_info["device_id"]
        if device_name in _ROCM_DEVICE_ID_NAME_MAP:
            return _ROCM_DEVICE_ID_NAME_MAP[device_name]
        return asic_info["market_name"]

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        if enforce_eager:
            logger.warning(
                "To see benefits of async output processing, enable CUDA "
                "graph. Since, enforce-eager is enabled, async output "
                "processor cannot be used")
            return False
        return True

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

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
                        "vllm.worker.multi_step_worker.MultiStepWorker"
            elif vllm_config.speculative_config:
                if envs.VLLM_USE_V1:
                    raise NotImplementedError(
                        "Speculative decoding is not yet supported on vLLM V1."
                    )
                else:
                    parallel_config.worker_cls = \
                        "vllm.spec_decode.spec_decode_worker.create_spec_worker"
                    parallel_config.sd_worker_cls = \
                        "vllm.worker.worker.Worker"
            else:
                if envs.VLLM_USE_V1:
                    parallel_config.worker_cls = \
                            "vllm.v1.worker.gpu_worker.Worker"
                else:
                    parallel_config.worker_cls = "vllm.worker.worker.Worker"

    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        if model_arch in _ROCM_UNSUPPORTED_MODELS:
            raise ValueError(f"Model architecture '{model_arch}' is not "
                             "supported by ROCm for now.")

        if model_arch in _ROCM_PARTIALLY_SUPPORTED_MODELS:
            msg = _ROCM_PARTIALLY_SUPPORTED_MODELS[model_arch]
            logger.warning(
                "Model architecture '%s' is partially "
                "supported by ROCm: %s", model_arch, msg)

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        super().verify_quantization(quant)
        if quant == "awq" and not envs.VLLM_USE_TRITON_AWQ:
            logger.warning(
                "Using AWQ quantization with ROCm, but VLLM_USE_TRITON_AWQ"
                " is not set, enabling VLLM_USE_TRITON_AWQ.")
        envs.VLLM_USE_TRITON_AWQ = True

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_current_memory_usage(cls,
                                 device: Optional[torch.types.Device] = None
                                 ) -> float:
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.mem_get_info(device)[1] - torch.cuda.mem_get_info(
            device)[0]

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa

    @classmethod
    def supports_mx(cls) -> bool:
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return any(gfx in gcn_arch for gfx in ["gfx95"])

    @classmethod
    def supports_fp8(cls) -> bool:
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return any(gfx in gcn_arch for gfx in ['gfx94', 'gfx95', 'gfx12'])

    @classmethod
    def is_fp8_fnuz(cls) -> bool:
        # only device 0 is checked, this assumes MI300 platforms are homogeneous
        return 'gfx94' in torch.cuda.get_device_properties(0).gcnArchName

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        if cls.is_fp8_fnuz():
            return torch.float8_e4m3fnuz
        else:
            return torch.float8_e4m3fn

    @classmethod
    def supports_v1(cls, model_config: "ModelConfig") -> bool:
        # V1 support on AMD gpus is experimental
        return True

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        # We only enable custom allreduce for MI300 series
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        supported_archs = ['gfx94', 'gfx95']
        return any(gfx in gcn_arch for gfx in supported_archs)

    @classmethod
    def get_cu_count(cls, device_id: int = 0) -> int:
        return torch.cuda.get_device_properties(
            device_id).multi_processor_count
