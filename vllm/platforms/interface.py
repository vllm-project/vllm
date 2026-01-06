# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import enum
import os
import platform
import random
import sys
from datetime import timedelta
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import numpy as np
import torch
from typing_extensions import deprecated

from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.logger import init_logger

if TYPE_CHECKING:
    from torch.distributed import PrefixStore, ProcessGroup

    from vllm.attention.selector import AttentionSelectorConfig
    from vllm.config import VllmConfig
    from vllm.inputs import ProcessorInputs, PromptType
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = object

logger = init_logger(__name__)


def in_wsl() -> bool:
    # Reference: https://github.com/microsoft/WSL/issues/4071
    return "microsoft" in " ".join(platform.uname()).lower()


class PlatformEnum(enum.Enum):
    CUDA = enum.auto()
    ROCM = enum.auto()
    TPU = enum.auto()
    XPU = enum.auto()
    CPU = enum.auto()
    OOT = enum.auto()
    UNSPECIFIED = enum.auto()


class CpuArchEnum(enum.Enum):
    X86 = enum.auto()
    ARM = enum.auto()
    POWERPC = enum.auto()
    S390X = enum.auto()
    RISCV = enum.auto()
    OTHER = enum.auto()
    UNKNOWN = enum.auto()


class DeviceCapability(NamedTuple):
    major: int
    minor: int

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) < (other.major, other.minor)

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) <= (other.major, other.minor)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) == (other.major, other.minor)

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) >= (other.major, other.minor)

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) > (other.major, other.minor)

    def as_version_str(self) -> str:
        return f"{self.major}.{self.minor}"

    def to_int(self) -> int:
        """
        Express device capability as an integer `<major><minor>`.

        It is assumed that the minor version is always a single digit.
        """
        assert 0 <= self.minor < 10
        return self.major * 10 + self.minor


class Platform:
    _enum: PlatformEnum
    device_name: str
    device_type: str

    # available dispatch keys:
    # check https://github.com/pytorch/pytorch/blob/313dac6c1ca0fa0cde32477509cce32089f8532a/torchgen/model.py#L134 # noqa
    # use "CPU" as a fallback for platforms not registered in PyTorch
    dispatch_key: str = "CPU"

    # available ray device keys:
    # https://github.com/ray-project/ray/blob/10ba5adadcc49c60af2c358a33bb943fb491a171/python/ray/_private/ray_constants.py#L438 # noqa
    # empty string means the device does not support ray
    ray_device_key: str = ""

    # platform-agnostic way to specify the device control environment variable,
    # .e.g. CUDA_VISIBLE_DEVICES for CUDA.
    # hint: search for "get_visible_accelerator_ids_env_var" in
    # https://github.com/ray-project/ray/tree/master/python/ray/_private/accelerators # noqa
    device_control_env_var: str = "VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER"

    # The torch.compile backend for compiling simple and
    # standalone functions. The default value is "inductor" to keep
    # the same behavior as PyTorch.
    # NOTE: for the forward part of the model, vLLM has another separate
    # compilation strategy.
    simple_compile_backend: str = "inductor"

    # The backend used for distributed communication.
    dist_backend: str = ""

    supported_quantization: list[str] = []

    additional_env_vars: list[str] = []

    _global_graph_pool: Any | None = None

    @property
    def pass_key(self) -> str:
        """Inductor config key for the PassManager custom pass"""
        return "post_grad_custom_post_pass"

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        """Returns the supported dtypes for the current platform."""
        # Be careful with the order of the dtypes. The first dtype will
        # be used as the default dtype fallback for the current platform,
        # when encountering unsupported dtypes in "auto" dtype.
        return [torch.bfloat16, torch.float16, torch.float32]

    def is_cuda(self) -> bool:
        return self._enum == PlatformEnum.CUDA

    def is_rocm(self) -> bool:
        return self._enum == PlatformEnum.ROCM

    def is_tpu(self) -> bool:
        return self._enum == PlatformEnum.TPU

    def is_xpu(self) -> bool:
        return self._enum == PlatformEnum.XPU

    def is_cpu(self) -> bool:
        return self._enum == PlatformEnum.CPU

    def is_out_of_tree(self) -> bool:
        return self._enum == PlatformEnum.OOT

    def is_unspecified(self) -> bool:
        return self._enum == PlatformEnum.UNSPECIFIED

    def get_max_output_tokens(self, prompt_len: int) -> int:
        return sys.maxsize

    def is_cuda_alike(self) -> bool:
        """Stateless version of [torch.cuda.is_available][]."""
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)

    def is_sleep_mode_available(self) -> bool:
        # TODO: Actually only mi3xx has the sleep mode support now
        # for ROCm, but currently we don't have a way to detect the
        # exact GPU model statelessly here. So we return True for
        # all ROCm platforms for now.
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)

    @classmethod
    def get_pass_manager_cls(cls) -> str:
        """
        Get the pass manager class for this platform.
        It will be registered as a custom pass under the current_platform.pass_key.
        """
        return "vllm.compilation.pass_manager.PostGradPassManager"

    @classmethod
    def get_compile_backend(cls) -> str:
        """
        Get the custom compile backend for current platform.
        """
        return cls.simple_compile_backend

    @classmethod
    def device_id_to_physical_device_id(cls, device_id: int):
        # Treat empty device control env var as unset. This is a valid
        # configuration in Ray setups where the engine is launched in
        # a CPU-only placement group located on a GPU node.
        if (
            cls.device_control_env_var in os.environ
            and os.environ[cls.device_control_env_var] != ""
        ):
            device_ids = os.environ[cls.device_control_env_var].split(",")
            physical_device_id = device_ids[device_id]
            return int(physical_device_id)
        else:
            return device_id

    @classmethod
    def import_kernels(cls) -> None:
        """Import any platform-specific C kernels."""
        try:
            import vllm._C  # noqa: F401
        except ImportError as e:
            logger.warning("Failed to import from vllm._C: %r", e)
        with contextlib.suppress(ImportError):
            import vllm._moe_C  # noqa: F401

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
    ) -> str:
        """Get the attention backend class of a device."""
        return ""

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list["AttentionBackendEnum"]:
        return [
            AttentionBackendEnum.TORCH_SDPA,
        ]

    @classmethod
    def get_vit_attn_backend(
        cls,
        head_size: int,
        dtype: torch.dtype,
        backend: Optional["AttentionBackendEnum"] = None,
    ) -> "AttentionBackendEnum":
        """
        Get the vision attention backend class of a device.

        NOTE: ViT Attention should be checked and override in the platform-specific
        implementation. we should not override this in any other places, like
        the model_executor/models/<model_name>.py.

        We check if the backend is None or not:
            1. If not, check if the backend is supported by the platform.
            2. If None, continue to the default selection logic.
        """
        if backend is not None:
            assert backend in cls.get_supported_vit_attn_backends(), (
                f"Backend {backend} is not supported for vit attention"
                f"Supported backends are: {cls.get_supported_vit_attn_backends()}"
            )
            logger.info_once(f"Using backend {backend} for vit attention")
            return backend

        logger.info_once(
            f"Using default backend {AttentionBackendEnum.TORCH_SDPA} for vit attention"
        )
        return AttentionBackendEnum.TORCH_SDPA

    @classmethod
    def get_device_capability(
        cls,
        device_id: int = 0,
    ) -> DeviceCapability | None:
        """Stateless version of [torch.cuda.get_device_capability][]."""
        return None

    @classmethod
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        """
        Test whether this platform is compatible with a device capability.

        The `capability` argument can either be:

        - A tuple `(major, minor)`.
        - An integer `<major><minor>`. (See
        [`DeviceCapability.to_int`][vllm.platforms.interface.DeviceCapability.to_int])
        """
        current_capability = cls.get_device_capability(device_id=device_id)
        if current_capability is None:
            return False

        if isinstance(capability, tuple):
            return current_capability >= capability

        return current_capability.to_int() >= capability

    @classmethod
    def is_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        """
        Test whether this platform has exactly the specified device capability.

        The `capability` argument can either be:

        - A tuple `(major, minor)`.
        - An integer `<major><minor>`. (See
        [`DeviceCapability.to_int`][vllm.platforms.interface.DeviceCapability.to_int])
        """
        current_capability = cls.get_device_capability(device_id=device_id)
        if current_capability is None:
            return False

        if isinstance(capability, tuple):
            return current_capability == capability

        return current_capability.to_int() == capability

    @classmethod
    def is_device_capability_family(
        cls,
        capability: int,
        device_id: int = 0,
    ) -> bool:
        """
        Returns True if the device capability is any <major>.x.
        Mirrors CUDA 13 'family' architecture semantics (e.g. 10.x, 11.x, 12.x).
        """
        current_capability = cls.get_device_capability(device_id=device_id)
        if current_capability is None:
            return False
        return (current_capability.to_int() // 10) == (capability // 10)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the name of a device."""
        raise NotImplementedError

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        """Get the uuid of a device, e.g. the PCI bus ID."""
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get the total memory of a device in bytes."""
        raise NotImplementedError

    @classmethod
    def inference_mode(cls):
        """A device-specific wrapper of `torch.inference_mode`.

        This wrapper is recommended because some hardware backends such as TPU
        do not support `torch.inference_mode`. In such a case, they will fall
        back to `torch.no_grad` by overriding this method.
        """
        return torch.inference_mode(mode=True)

    @classmethod
    @deprecated(
        "`seed_everything` is deprecated. It will be removed in v0.15.0 or later. "
        "Please use `vllm.utils.torch_utils.set_random_seed` instead."
    )
    def seed_everything(cls, seed: int | None = None) -> None:
        """
        Set the seed of each random module.
        `torch.manual_seed` will set seed on all devices.

        Loosely based on: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/fabric/utilities/seed.py#L20
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        raise NotImplementedError

    @classmethod
    def pre_register_and_update(
        cls, parser: FlexibleArgumentParser | None = None
    ) -> None:
        """
        Do some pre-registration or update action for the current platform.

        This function is called before global VllmConfig is initialized or cli
        arguments are parsed. It's used for out-of-tree platforms to register or
        update the configuration.

        For example, the out-of-tree quantization config can be imported and
        registered here dynamically.
        """
        pass

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """
        Check and update the configuration for the current platform.

        It can raise an exception if the configuration is not compatible with
        the current platform, or it can update the configuration to make it
        compatible with the current platform.

        The config is passed by reference, so it can be modified in place.
        """
        pass

    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        """
        Verify whether the current platform supports the specified model
        architecture.

        - This will raise an Error or Warning based on the model support on
        the current platform.
        - By default all models are considered supported.
        """
        pass

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        """
        Verify whether the quantization is supported by the current platform.
        """
        if cls.supported_quantization and quant not in cls.supported_quantization:
            raise ValueError(
                f"{quant} quantization is currently not supported in {cls.device_name}."
            )

    @classmethod
    def get_cpu_architecture(cls) -> CpuArchEnum:
        """
        Determine the CPU architecture of the current system.
        Returns CpuArchEnum indicating the architecture type.
        """
        machine = platform.machine().lower()

        if machine in ("x86_64", "amd64", "i386", "i686"):
            return CpuArchEnum.X86
        elif machine.startswith("arm") or machine.startswith("aarch"):
            return CpuArchEnum.ARM
        elif machine.startswith("ppc"):
            return CpuArchEnum.POWERPC
        elif machine == "s390x":
            return CpuArchEnum.S390X
        elif machine.startswith("riscv"):
            return CpuArchEnum.RISCV

        return CpuArchEnum.OTHER if machine else CpuArchEnum.UNKNOWN

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Checks whether pin memory is available on the current platform."""
        if in_wsl():
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning(
                "Using 'pin_memory=False' as WSL is detected. "
                "This may slow down the performance."
            )
            return False
        return True

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        """
        Return the memory usage in bytes.
        """
        raise NotImplementedError

    @classmethod
    def get_punica_wrapper(cls) -> str:
        """
        Return the punica wrapper for current platform.
        """
        raise NotImplementedError

    @classmethod
    def get_infinity_values(cls, dtype: torch.dtype) -> tuple[float, float]:
        """
        Return the platform specific values for (-inf, inf)
        """
        return float("-inf"), float("inf")

    @classmethod
    def can_update_inplace(cls) -> bool:
        """
        Checks if the platform allows inplace memory updates
        """
        return True

    @classmethod
    def get_lora_vocab_padding_size(cls) -> int:
        """
        Returns how much padding the LoRA logits need for kernels
        """
        return 256

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """
        Get device specific communicator class for distributed communication.
        """
        return "vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"  # noqa

    @classmethod
    def supports_mx(cls) -> bool:
        """
        Returns whether the current platform supports MX types.
        """
        return False

    @classmethod
    def supports_fp8(cls) -> bool:
        """
        Returns whether the current platform supports FP8 types.
        """
        return False

    @classmethod
    def is_fp8_fnuz(cls) -> bool:
        """
        Returns whether the preferred FP8 type is FNUZ on the current platform.

        There are two representations of FP8, OCP FP8 and FNUZ FP8.
        The OCP specification can be found at https://tinyurl.com/b7jvwpft.
        The FNUZ specification can be found at https://tinyurl.com/5n6hwwu5.

        AMD's MI300 and MI325 have native hardware support for FNUZ. All other
        hardware has converged on the OCP FP8 standard.
        """
        return False

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        """
        Returns the preferred FP8 type on the current platform.

        See the documentation for is_fp8_fnuz for details.
        """
        return torch.float8_e4m3fn

    @classmethod
    def use_all_gather(cls) -> bool:
        """
        Whether to use allgather in LogitsProcessor to gather the logits.
        """
        return True

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        """
        Returns if custom allreduce is supported on the current platform
        """
        return False

    @classmethod
    def opaque_attention_op(cls) -> bool:
        """
        Returns True if we register attention as one giant opaque custom op
        on the current platform
        """
        return False

    @classmethod
    def validate_request(
        cls,
        prompt: "PromptType",
        params: "SamplingParams | PoolingParams",
        processed_inputs: "ProcessorInputs",
    ) -> None:
        """Raises if this request is unsupported on this platform"""

    def __getattr__(self, key: str):
        device = getattr(torch, self.device_type, None)
        if device is not None and hasattr(device, key):
            return getattr(device, key)
        else:
            logger.warning(
                "Current platform %s does not have '%s' attribute.",
                self.device_type,
                key,
            )
            return None

    def get_global_graph_pool(self) -> Any:
        """
        Return the global graph pool for this platform.
        """
        cls = self.__class__
        if cls._global_graph_pool is None:
            cls._global_graph_pool = self.graph_pool_handle()
        return cls._global_graph_pool

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        """
        Get static graph wrapper class for static graph.
        """
        return "vllm.compilation.base_static_graph.AbstractStaticGraphWrapper"

    @classmethod
    def stateless_init_device_torch_dist_pg(
        cls,
        backend: str,
        prefix_store: "PrefixStore",
        group_rank: int,
        group_size: int,
        timeout: timedelta,
    ) -> "ProcessGroup":
        """
        Init platform-specific torch distributed process group.
        """
        raise NotImplementedError

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        """
        Check if the dtype is supported by the current platform.
        """
        raise NotImplementedError

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        """
        Returns if the hybrid kv cache is supported by the current platform.
        """
        return False

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        """
        Returns if the graph mode is supported by the current platform.
        """
        return False

    @classmethod
    def use_sync_weight_loader(cls) -> bool:
        """
        Returns if the current platform needs to sync weight loader.
        """
        return False

    @classmethod
    def make_synced_weight_loader(cls, original_weight_loader):
        """
        Wrap the original weight loader to make it synced.
        """
        if not cls.use_sync_weight_loader():
            return original_weight_loader

        def _synced_weight_loader(param, *args, **kwargs):
            out = original_weight_loader(param, *args, **kwargs)
            if param.device != torch.device("cpu"):
                torch._sync(param)
            return out

        return _synced_weight_loader

    @classmethod
    def get_nixl_supported_devices(cls) -> dict[str, tuple[str, ...]]:
        """
        Returns a mapping from device_type to a tuple of supported
        kv_buffer_device for nixl.
        """
        return {}

    @classmethod
    def get_nixl_memory_type(cls) -> str | None:
        """
        Returns the nixl memory type for the current platform.
        """
        return None

    @classmethod
    def check_max_model_len(cls, max_model_len: int) -> int:
        """
        Check max_model_len for the current platform.
        """
        return max_model_len

    @classmethod
    def set_additional_forward_context(cls, *args, **kwargs) -> dict[str, Any]:
        """
        Set some additional forward context for the current platform if needs.
        """
        return {}


class UnspecifiedPlatform(Platform):
    _enum = PlatformEnum.UNSPECIFIED
    device_type = ""
