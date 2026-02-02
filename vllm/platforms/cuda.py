# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Code inside this file can safely assume cuda platform, e.g. importing
pynvml. However, it should not initialize cuda context.
"""

import os
from collections.abc import Callable
from functools import cache, wraps
from typing import TYPE_CHECKING, TypeVar

import torch
from typing_extensions import ParamSpec

# import custom ops, trigger op registration
import vllm._C  # noqa
from vllm.logger import init_logger
from vllm.utils.import_utils import import_pynvml
from vllm.utils.torch_utils import cuda_device_count_stateless
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interface import DeviceCapability, Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    VllmConfig = None
    CacheDType = None

logger = init_logger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

pynvml = import_pynvml()

# pytorch 2.5 uses cudnn sdpa by default, which will cause crash on some models
# see https://github.com/huggingface/diffusers/issues/9704 for details
torch.backends.cuda.enable_cudnn_sdp(False)


@cache
def _get_backend_priorities(
    use_mla: bool,
    device_capability: DeviceCapability,
) -> list[AttentionBackendEnum]:
    """Get backend priorities with lazy import to avoid circular dependency."""
    if use_mla:
        if device_capability.major == 10:
            return [
                AttentionBackendEnum.FLASHINFER_MLA,
                AttentionBackendEnum.CUTLASS_MLA,
                AttentionBackendEnum.FLASH_ATTN_MLA,
                AttentionBackendEnum.FLASHMLA,
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHMLA_SPARSE,
            ]
        else:
            return [
                AttentionBackendEnum.FLASH_ATTN_MLA,
                AttentionBackendEnum.FLASHMLA,
                AttentionBackendEnum.FLASHINFER_MLA,
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHMLA_SPARSE,
            ]
    else:
        if device_capability.major == 10:
            return [
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
            ]
        else:
            return [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
            ]


def with_nvml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        pynvml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            pynvml.nvmlShutdown()

    return wrapper


class CudaPlatformBase(Platform):
    _enum = PlatformEnum.CUDA
    device_name: str = "cuda"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    dist_backend: str = "nccl"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        if self.has_device_capability(80):
            # Ampere and Hopper or later NVIDIA GPUs.
            return [torch.bfloat16, torch.float16, torch.float32]
        if self.has_device_capability(60):
            # Pascal, Volta and Turing NVIDIA GPUs, BF16 is not supported
            return [torch.float16, torch.float32]
        # Kepler and Maxwell NVIDIA GPUs, only FP32 is supported,
        # though vLLM doesn't support these GPUs.
        return [torch.float32]

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.cuda.set_device(device)
        # With this trick we can force the device to be set eagerly
        # see https://github.com/pytorch/pytorch/issues/155668
        # for why and when it is needed
        _ = torch.zeros(1, device=device)

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        raise NotImplementedError

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_fully_connected(cls, device_ids: list[int]) -> bool:
        raise NotImplementedError

    @classmethod
    def log_warnings(cls):
        pass

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        # For MLA models, determine the backend that will be selected and set
        # block_size based on that backend's requirements. This ensures
        # consistency between the block_size set here and the backend that
        # get_attn_backend_cls will select later.
        # Note: model_config may be None during testing
        # Note: block_size is initialized in
        # HybridAttentionMambaModelConfig.verify_and_update_config
        # for models with both attention and mamba,
        # and doesn't need to be reinitialized here
        if (
            model_config is not None
            and model_config.use_mla
            and cache_config.block_size is not None
        ):
            from vllm.v1.attention.selector import AttentionSelectorConfig

            use_sparse = hasattr(model_config.hf_config, "index_topk")
            device_capability = cls.get_device_capability()

            # Only adjust block size if we can determine device capability
            if device_capability is not None:
                # Build a minimal selector config to find which backend will
                # be selected. Note: model_config.dtype is torch.dtype after init
                attn_selector_config = AttentionSelectorConfig(
                    head_size=model_config.get_head_size(),
                    dtype=model_config.dtype,  # type: ignore[arg-type]
                    kv_cache_dtype=cache_config.cache_dtype,
                    block_size=None,
                    use_mla=True,
                    has_sink=False,
                    use_sparse=use_sparse,
                    use_mm_prefix=model_config.is_mm_prefix_lm,
                )

                # Use the same selection logic as get_attn_backend_cls
                attention_config = vllm_config.attention_config
                requested_backend = (
                    attention_config.backend if attention_config else None
                )
                chosen_backend = cls.select_attention_backend(
                    selected_backend=requested_backend,
                    attn_selector_config=attn_selector_config,
                    device_capability=device_capability,
                    raise_on_invalid=False,
                )

                if chosen_backend is not None:
                    try:
                        backend_class = chosen_backend.get_class()
                        preferred_block_size = backend_class.get_preferred_block_size(
                            cache_config.block_size
                        )
                        if cache_config.block_size != preferred_block_size:
                            logger.info(
                                "Setting kv cache block size to %d for %s backend.",
                                preferred_block_size,
                                chosen_backend.name,
                            )
                            cache_config.block_size = preferred_block_size  # type: ignore[assignment]
                    except ImportError:
                        pass  # Backend selection will fail later with a better error

        scheduler_config = vllm_config.scheduler_config
        # Note: model_config may be None during testing
        if (
            model_config is not None
            and model_config.is_mm_prefix_lm
            and scheduler_config.is_multimodal_model
            and not scheduler_config.disable_chunked_mm_input
        ):
            logger.warning(
                "Forcing --disable_chunked_mm_input for models "
                "with multimodal-bidirectional attention."
            )
            scheduler_config.disable_chunked_mm_input = True

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def get_valid_backends(
        cls,
        device_capability: DeviceCapability,
        attn_selector_config: "AttentionSelectorConfig",
    ) -> tuple[
        list[tuple["AttentionBackendEnum", int]],
        dict["AttentionBackendEnum", list[str]],
    ]:
        valid_backends_priorities = []
        invalid_reasons = {}

        backend_priorities = _get_backend_priorities(
            attn_selector_config.use_mla, device_capability
        )
        for priority, backend in enumerate(backend_priorities):
            try:
                backend_class = backend.get_class()
                invalid_reasons_i = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
            except ImportError:
                invalid_reasons_i = ["ImportError"]
            if invalid_reasons_i:
                invalid_reasons[backend] = invalid_reasons_i
            else:
                valid_backends_priorities.append((backend, priority))

        return valid_backends_priorities, invalid_reasons

    @classmethod
    def select_attention_backend(
        cls,
        selected_backend: "AttentionBackendEnum | None",
        attn_selector_config: "AttentionSelectorConfig",
        device_capability: "DeviceCapability",
        raise_on_invalid: bool = True,
    ) -> "AttentionBackendEnum | None":
        """Select the best attention backend for the given configuration.

        This is the single source of truth for backend selection, used by both
        check_and_update_config (to set block_size) and get_attn_backend_cls
        (to get the backend class path).

        Args:
            selected_backend: User-specified backend, or None for auto-selection
            attn_selector_config: Configuration for attention selection
            device_capability: Device capability info
            raise_on_invalid: If True, raise ValueError when no valid backend

        Returns:
            The selected backend enum, or None if no valid backend found
            and raise_on_invalid is False
        """
        # Ensure we don't constrain by block_size during selection
        attn_selector_config = attn_selector_config._replace(block_size=None)

        # First try checking just the selected backend, if there is one.
        if selected_backend is not None:
            try:
                backend_class = selected_backend.get_class()
                validation_errors = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
            except ImportError:
                validation_errors = ["ImportError"]
            if validation_errors:
                if raise_on_invalid:
                    raise ValueError(
                        f"Selected backend {selected_backend} is not valid for "
                        f"this configuration. Reason: {validation_errors}"
                    )
                return None
            return selected_backend

        # No selected backend, so find the best valid one.
        valid_backends_priorities, invalid_reasons = cls.get_valid_backends(
            device_capability=device_capability,
            attn_selector_config=attn_selector_config,
        )

        if len(valid_backends_priorities) == 0:
            if raise_on_invalid:
                reasons_str = (
                    "{"
                    + ", ".join(
                        f"{backend.name}: [{', '.join(reasons)}]"
                        for backend, reasons in invalid_reasons.items()
                    )
                    + "}"
                )
                config_str = attn_selector_config.__repr__()
                raise ValueError(
                    f"No valid attention backend found for {cls.device_name} "
                    f"with {config_str}. Reasons: {reasons_str}."
                )
            return None

        # Select the one with the highest priority (lowest index).
        sorted_backends = sorted(valid_backends_priorities, key=lambda x: x[1])
        return sorted_backends[0][0]

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
    ) -> str:
        device_capability = cls.get_device_capability()
        assert device_capability is not None

        chosen_backend = cls.select_attention_backend(
            selected_backend=selected_backend,
            attn_selector_config=attn_selector_config,
            device_capability=device_capability,
            raise_on_invalid=True,
        )
        assert chosen_backend is not None  # raise_on_invalid=True guarantees this

        # Log the selection
        if selected_backend is not None:
            logger.info("Using %s backend.", chosen_backend)
        else:
            # Get all valid backends for logging
            valid_backends_priorities, invalid_reasons = cls.get_valid_backends(
                device_capability=device_capability,
                attn_selector_config=attn_selector_config._replace(block_size=None),
            )
            reasons_str = (
                "{"
                + ", ".join(
                    f"{backend.name}: [{', '.join(reasons)}]"
                    for backend, reasons in invalid_reasons.items()
                )
                + "}"
            )
            config_str = attn_selector_config.__repr__()
            logger.debug_once(
                f"Some attention backends are not valid for {cls.device_name} with "
                f"{config_str}. Reasons: {reasons_str}."
            )
            logger.info_once(
                "Using %s attention backend out of potential backends: %s",
                chosen_backend.name,
                tuple(b[0].name for b in valid_backends_priorities),
                scope="local",
            )

        return chosen_backend.get_path()

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list["AttentionBackendEnum"]:
        return [
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.FLASH_ATTN,
        ]

    @classmethod
    def get_vit_attn_backend(
        cls,
        head_size: int,
        dtype: torch.dtype,
        backend: "AttentionBackendEnum | None" = None,
    ) -> "AttentionBackendEnum":
        if backend is not None:
            assert backend in cls.get_supported_vit_attn_backends(), (
                f"Backend {backend} is not supported for vit attention. "
                f"Supported backends are: {cls.get_supported_vit_attn_backends()}"
            )
            logger.info_once(f"Using backend {backend} for vit attention")
            return backend

        # Try FlashAttention first
        if (cc := cls.get_device_capability()) and cc.major >= 8:
            try:
                backend_class = AttentionBackendEnum.FLASH_ATTN.get_class()
                if backend_class.supports_head_size(
                    head_size
                ) and backend_class.supports_dtype(dtype):
                    return AttentionBackendEnum.FLASH_ATTN
            except ImportError:
                pass

        return AttentionBackendEnum.TORCH_SDPA

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return (
            "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa
        )

    @classmethod
    def supports_fp8(cls) -> bool:
        return cls.has_device_capability(89)

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        return True

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    @classmethod
    def device_count(cls) -> int:
        return cuda_device_count_stateless()

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        if dtype == torch.bfloat16:  # noqa: SIM102
            if not cls.has_device_capability(80):
                capability = cls.get_device_capability()
                gpu_name = cls.get_device_name()

                if capability is None:
                    compute_str = "does not have a compute capability"
                else:
                    version_str = capability.as_version_str()
                    compute_str = f"has compute capability {version_str}"

                raise ValueError(
                    "Bfloat16 is only supported on GPUs "
                    "with compute capability of at least 8.0. "
                    f"Your {gpu_name} GPU {compute_str}. "
                    "You can use float16 instead by explicitly setting the "
                    "`dtype` flag in CLI, for example: --dtype=half."
                )

    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from src_cache to dst_cache on GPU."""
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
        """Copy blocks from GPU to host (CPU)."""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.cpu()

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True


# NVML utils
# Note that NVML is not affected by `CUDA_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using NVML is that it will not initialize CUDA
class NvmlCudaPlatform(CudaPlatformBase):
    @classmethod
    @cache
    @with_nvml_context
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        try:
            physical_device_id = cls.device_id_to_physical_device_id(device_id)
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            return DeviceCapability(major=major, minor=minor)
        except RuntimeError:
            return None

    @classmethod
    @with_nvml_context
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        try:
            return super().has_device_capability(capability, device_id)
        except RuntimeError:
            return False

    @classmethod
    @with_nvml_context
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        return cls._get_physical_device_name(physical_device_id)

    @classmethod
    @with_nvml_context
    def get_device_uuid(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return pynvml.nvmlDeviceGetUUID(handle)

    @classmethod
    @with_nvml_context
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)

    @classmethod
    @with_nvml_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """
        query if the set of gpus are fully connected by nvlink (1 hop)
        """
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                            handle,
                            peer_handle,
                            pynvml.NVML_P2P_CAPS_INDEX_NVLINK,
                        )
                        if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                            return False
                    except pynvml.NVMLError:
                        logger.exception(
                            "NVLink detection failed. This is normal if"
                            " your machine has no NVLink equipped."
                        )
                        return False
        return True

    @classmethod
    def _get_physical_device_name(cls, device_id: int = 0) -> str:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return pynvml.nvmlDeviceGetName(handle)

    @classmethod
    @with_nvml_context
    def log_warnings(cls):
        device_ids: int = pynvml.nvmlDeviceGetCount()
        if device_ids > 1:
            device_names = [cls._get_physical_device_name(i) for i in range(device_ids)]
            if (
                len(set(device_names)) > 1
                and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            ):
                logger.warning(
                    "Detected different devices in the system: %s. Please"
                    " make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to "
                    "avoid unexpected behavior.",
                    ", ".join(device_names),
                )


class NonNvmlCudaPlatform(CudaPlatformBase):
    @classmethod
    @cache
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        logger.exception(
            "NVLink detection not possible, as context support was"
            " not found. Assuming no NVLink available."
        )
        return False


# Autodetect either NVML-enabled or non-NVML platform
# based on whether NVML is available.
nvml_available = False
try:
    try:
        pynvml.nvmlInit()
        nvml_available = True
    except Exception:
        # On Jetson, NVML is not supported.
        nvml_available = False
finally:
    if nvml_available:
        pynvml.nvmlShutdown()

CudaPlatform = NvmlCudaPlatform if nvml_available else NonNvmlCudaPlatform

CudaPlatform.log_warnings()
