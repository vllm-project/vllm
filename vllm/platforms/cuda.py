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
    num_heads: int | None = None,
) -> list[AttentionBackendEnum]:
    """Get backend priorities with lazy import to avoid circular dependency."""
    if use_mla:
        if device_capability.major == 10:
            # Prefer FlashInfer at low head counts (FlashMLA uses padding)
            if num_heads is not None and num_heads <= 16:
                sparse_backends = [
                    AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
                    AttentionBackendEnum.FLASHMLA_SPARSE,
                ]
            else:
                sparse_backends = [
                    AttentionBackendEnum.FLASHMLA_SPARSE,
                    AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
                ]
            return [
                AttentionBackendEnum.FLASHINFER_MLA,
                AttentionBackendEnum.CUTLASS_MLA,
                AttentionBackendEnum.FLASH_ATTN_MLA,
                AttentionBackendEnum.FLASHMLA,
                AttentionBackendEnum.TRITON_MLA,
                *sparse_backends,
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
    ray_noset_device_env_vars: list[str] = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    ]

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
        user_specified_block_size = cache_config.block_size is not None
        if not user_specified_block_size:
            cache_config.block_size = 16

        # Ensure block_size is compatible with the attention backend.
        # Note: model_config may be None during testing.
        # Skip hybrid (attention+mamba) models — their block_size is
        # managed by HybridAttentionMambaModelConfig
        if model_config is not None and not model_config.is_hybrid:
            cls._update_block_size_for_backend(
                vllm_config,
                user_specified_block_size,
            )

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
    def _update_block_size_for_backend(
        cls,
        vllm_config: "VllmConfig",
        user_specified_block_size: bool,
    ) -> None:
        """Ensure block_size is compatible with the attention backend.

        If the user specified --block-size, the selector validates/filters
        backends by that block size (raising on incompatibility). Otherwise,
        the backend is selected unconstrained and block_size is set to the
        backend's preferred value.
        """
        from vllm.config.vllm import set_current_vllm_config
        from vllm.v1.attention.selector import AttentionSelectorConfig

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        device_capability = cls.get_device_capability()
        if device_capability is None:
            return

        use_mla = model_config.use_mla
        attn_selector_config = AttentionSelectorConfig(
            head_size=model_config.get_head_size(),
            dtype=model_config.dtype,  # type: ignore[arg-type]
            kv_cache_dtype=cache_config.cache_dtype,
            block_size=cache_config.block_size if user_specified_block_size else None,
            use_mla=use_mla,
            has_sink=False,
            use_sparse=use_mla and hasattr(model_config.hf_config, "index_topk"),
            use_mm_prefix=model_config.is_mm_prefix_lm,
        )

        user_specified_backend = vllm_config.attention_config.backend
        num_heads = model_config.get_num_attention_heads(
            vllm_config.parallel_config,
        )
        with set_current_vllm_config(vllm_config):
            chosen_backend = cls.select_attention_backend(
                selected_backend=user_specified_backend,
                attn_selector_config=attn_selector_config,
                device_capability=device_capability,
                # Don't raise here — we produce better errors below.
                raise_on_invalid=False,
                num_heads=num_heads,
            )

            # If the user's --block-size forced a non-optimal backend,
            # warn them. Only relevant when the user didn't also specify
            # --attention-backend (in which case the choice is explicit).
            if (
                chosen_backend is not None
                and user_specified_block_size
                and user_specified_backend is None
            ):
                optimal = cls.select_attention_backend(
                    selected_backend=None,
                    attn_selector_config=attn_selector_config._replace(
                        block_size=None,
                    ),
                    device_capability=device_capability,
                    raise_on_invalid=False,
                    num_heads=num_heads,
                )
                if optimal is not None and optimal != chosen_backend:
                    logger.warning(
                        "--block-size %d is not supported by the preferred "
                        "%s backend. Using %s instead, which may result "
                        "in reduced performance. Consider removing "
                        "--block-size to auto-select the optimal "
                        "block size.",
                        cache_config.block_size,
                        optimal.name,
                        chosen_backend.name,
                    )

            if chosen_backend is not None:
                if user_specified_block_size:
                    # User's block_size is compatible with the chosen
                    # backend.
                    return
                # User didn't specify --block-size, so auto-select the
                # preferred block size for the chosen backend.
                try:
                    backend_class = chosen_backend.get_class()
                except ImportError:
                    return  # Will fail later with a better error
                preferred = backend_class.get_preferred_block_size(
                    cache_config.block_size,
                )
                if cache_config.block_size != preferred:
                    logger.info(
                        "Setting kv cache block size to %d for %s backend.",
                        preferred,
                        chosen_backend.name,
                    )
                    cache_config.block_size = preferred
                return

            # No valid backend found. If the user didn't constrain the
            # selection, defer the error to get_attn_backend_cls where
            # the full config (including per-layer settings) is
            # available.
            if not user_specified_block_size:
                return

            if user_specified_backend is not None:
                # User specified --block-size and --attention-backend
                # and they are incompatible.
                try:
                    backend_class = user_specified_backend.get_class()
                    supported = backend_class.get_supported_kernel_block_sizes()
                except ImportError:
                    supported = None
                raise ValueError(
                    f"User-specified --block-size "
                    f"{cache_config.block_size} is incompatible with "
                    f"the specified --attention-backend "
                    f"{user_specified_backend.name} (supported kernel "
                    f"block sizes: {supported}). Either remove "
                    f"--block-size to auto-select, or choose a "
                    f"compatible value."
                )
            else:
                # User specified --block-size but no backend supports
                # it.
                _, invalid_reasons = cls.get_valid_backends(
                    device_capability=device_capability,
                    attn_selector_config=attn_selector_config,
                    num_heads=num_heads,
                )
                reasons_str = ", ".join(
                    f"{b.name}: [{', '.join(r)}]" for b, r in invalid_reasons.items()
                )
                raise ValueError(
                    f"No valid attention backend found for "
                    f"--block-size {cache_config.block_size}. "
                    f"Reasons: {{{reasons_str}}}. Either remove "
                    f"--block-size to auto-select, or choose a "
                    f"compatible value."
                )

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
        num_heads: int | None = None,
    ) -> tuple[
        list[tuple["AttentionBackendEnum", int]],
        dict["AttentionBackendEnum", list[str]],
    ]:
        valid_backends_priorities = []
        invalid_reasons = {}

        backend_priorities = _get_backend_priorities(
            attn_selector_config.use_mla,
            device_capability,
            num_heads,
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
        num_heads: int | None = None,
    ) -> "AttentionBackendEnum | None":
        """Select the best attention backend for the given configuration.

        Args:
            selected_backend: User-specified backend, or None for auto-selection
            attn_selector_config: Configuration for attention selection
            device_capability: Device capability info
            raise_on_invalid: If True, raise ValueError when no valid backend
            num_heads: Number of attention heads per GPU, used for backend
                priority ordering on Blackwell GPUs

        Returns:
            The selected backend enum, or None if no valid backend found
            and raise_on_invalid is False
        """
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
            num_heads=num_heads,
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
        selected_backend: "AttentionBackendEnum | None",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        device_capability = cls.get_device_capability()
        assert device_capability is not None

        chosen_backend = cls.select_attention_backend(
            selected_backend=selected_backend,
            attn_selector_config=attn_selector_config,
            num_heads=num_heads,
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
                attn_selector_config=attn_selector_config,
                num_heads=num_heads,
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
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TRITON_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
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

        cc = cls.get_device_capability()
        for vit_attn_backend in cls.get_supported_vit_attn_backends():
            if vit_attn_backend == AttentionBackendEnum.TORCH_SDPA:
                continue
            try:
                backend_class = vit_attn_backend.get_class()
                is_backend_supported = backend_class.supports_head_size(
                    head_size
                ) and backend_class.supports_dtype(dtype)
                if cc is not None:
                    is_backend_supported = (
                        is_backend_supported
                        and backend_class.supports_compute_capability(cc)
                    )
                if is_backend_supported:
                    logger.info_once(
                        f"Using backend {vit_attn_backend} for vit attention"
                    )
                    return vit_attn_backend
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

    @classmethod
    @with_nvml_context
    def get_device_numa_node(cls, device_id: int = 0) -> int | None:
        """Get NUMA node ID for a GPU device.

        Uses NVML to query which NUMA node the GPU is attached to.
        Falls back to CPU affinity-based detection if direct query fails.

        Args:
            device_id: Logical device ID (respects CUDA_VISIBLE_DEVICES)

        Returns:
            NUMA node ID, or None if it cannot be determined
        """
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)

        # Try direct NUMA node query first
        try:
            return pynvml.nvmlDeviceGetNumaNodeId(handle)
        except Exception:
            pass  # Fall through to CPU affinity method

        # Fallback: determine NUMA node from CPU affinity
        try:
            cpu_ids = cls._get_device_cpu_affinity(handle)
            if cpu_ids:
                numa_node = cls._get_numa_node_for_cpu(cpu_ids[0])
                if numa_node is not None:
                    logger.debug(
                        "Determined NUMA node %d for GPU %d via CPU affinity",
                        numa_node,
                        device_id,
                    )
                    return numa_node
        except Exception as e:
            logger.warning("Failed to get NUMA node for GPU %d: %s", device_id, e)

        return None

    @classmethod
    def _get_device_cpu_affinity(cls, handle) -> list[int]:
        """Get list of CPU IDs associated with a GPU via NVML."""
        cpu_count = os.cpu_count()
        if cpu_count is None:
            return []

        # NVML returns affinity as array of 64-bit masks
        cpu_set_size = (cpu_count + 63) // 64
        cpu_affinity_mask = pynvml.nvmlDeviceGetCpuAffinity(handle, cpu_set_size)

        # Convert bitmask to list of CPU IDs
        cpu_ids = []
        for i, mask in enumerate(cpu_affinity_mask):
            for bit in range(64):
                cpu_id = i * 64 + bit
                if cpu_id >= cpu_count:
                    break
                if mask & (1 << bit):
                    cpu_ids.append(cpu_id)
        return cpu_ids

    @classmethod
    def _get_numa_node_for_cpu(cls, cpu_id: int) -> int | None:
        """Determine which NUMA node a CPU belongs to.

        Reads from /sys/devices/system/node/ to find the NUMA topology.
        """
        from pathlib import Path

        node_path = Path("/sys/devices/system/node")
        if not node_path.exists():
            return None

        for node_dir in node_path.iterdir():
            if not node_dir.name.startswith("node"):
                continue
            try:
                node_id = int(node_dir.name[4:])  # Extract number from "nodeN"
                cpulist_file = node_dir / "cpulist"
                if cpulist_file.exists():
                    cpulist = cpulist_file.read_text().strip()
                    if cls._cpu_in_cpulist(cpu_id, cpulist):
                        return node_id
            except (ValueError, OSError):
                continue
        return None

    @classmethod
    def _cpu_in_cpulist(cls, cpu_id: int, cpulist: str) -> bool:
        """Check if a CPU ID is in a cpulist string (e.g., '0-3,8-11')."""
        for part in cpulist.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                if int(start) <= cpu_id <= int(end):
                    return True
            elif part.isdigit() and int(part) == cpu_id:
                return True
        return False

    @classmethod
    @with_nvml_context
    def get_all_device_numa_nodes(cls) -> list[int] | None:
        """Get NUMA nodes for all visible GPU devices.

        Returns:
            List of NUMA node IDs (one per GPU), or None if detection fails
        """
        try:
            device_count = cls.device_count()
            numa_nodes = []
            for device_id in range(device_count):
                numa_node = cls.get_device_numa_node(device_id)
                if numa_node is None:
                    logger.warning(
                        "Could not detect NUMA node for GPU %d, "
                        "disabling auto NUMA binding",
                        device_id,
                    )
                    return None
                numa_nodes.append(numa_node)
            return numa_nodes
        except Exception as e:
            logger.warning("Failed to get NUMA nodes for GPUs: %s", e)
            return None


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

    @classmethod
    def get_device_numa_node(cls, device_id: int = 0) -> int | None:
        """NUMA node detection not available without NVML."""
        return None

    @classmethod
    def get_all_device_numa_nodes(cls) -> list[int] | None:
        """NUMA node detection not available without NVML."""
        return None


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
