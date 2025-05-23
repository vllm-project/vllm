# SPDX-License-Identifier: Apache-2.0
import enum
import os
import platform
import random
from platform import uname
from typing import TYPE_CHECKING, NamedTuple, Optional, Union

import numpy as np
import torch

from vllm.inputs import ProcessorInputs, PromptType
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.lora.request import LoRARequest
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.utils import FlexibleArgumentParser
else:
    ModelConfig = None
    VllmConfig = None
    LoRARequest = None
    PoolingParams = None
    SamplingParams = None
    FlexibleArgumentParser = None

logger = init_logger(__name__)


def in_wsl() -> bool:
    # Reference: https://github.com/microsoft/WSL/issues/4071
    return "microsoft" in " ".join(uname()).lower()


class _Backend(enum.Enum):
    FLASH_ATTN = enum.auto()
    FLASH_ATTN_VLLM_V1 = enum.auto()
    TRITON_ATTN_VLLM_V1 = enum.auto()
    XFORMERS = enum.auto()
    ROCM_FLASH = enum.auto()
    ROCM_AITER_MLA = enum.auto()  # Supported by V1
    ROCM_AITER_MLA_VLLM_V1 = enum.auto()
    TORCH_SDPA = enum.auto()
    FLASHINFER = enum.auto()
    TRITON_MLA = enum.auto()  # Supported by V1
    FLASHMLA = enum.auto()  # Supported by V1
    HPU_ATTN = enum.auto()
    PALLAS = enum.auto()
    PALLAS_VLLM_V1 = enum.auto()
    IPEX = enum.auto()
    BLOCK_SPARSE_FLASH_ATTN = enum.auto()
    DUAL_CHUNK_FLASH_ATTN = enum.auto()
    NO_ATTENTION = enum.auto()


class PlatformEnum(enum.Enum):
    CUDA = enum.auto()
    ROCM = enum.auto()
    TPU = enum.auto()
    HPU = enum.auto()
    XPU = enum.auto()
    CPU = enum.auto()
    NEURON = enum.auto()
    OOT = enum.auto()
    UNSPECIFIED = enum.auto()


class CpuArchEnum(enum.Enum):
    X86 = enum.auto()
    ARM = enum.auto()
    POWERPC = enum.auto()
    OTHER = enum.auto()
    UNKNOWN = enum.auto()


class DeviceCapability(NamedTuple):
    major: int
    minor: int

    def as_version_str(self) -> str:
        return f"{self.major}.{self.minor}"

    def to_int(self) -> int:
        """
        Express device capability as an integer ``<major><minor>``.

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

    supported_quantization: list[str] = []

    additional_env_vars: list[str] = []

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

    def is_hpu(self) -> bool:
        return self._enum == PlatformEnum.HPU

    def is_xpu(self) -> bool:
        return self._enum == PlatformEnum.XPU

    def is_cpu(self) -> bool:
        return self._enum == PlatformEnum.CPU

    def is_neuron(self) -> bool:
        return self._enum == PlatformEnum.NEURON

    def is_out_of_tree(self) -> bool:
        return self._enum == PlatformEnum.OOT

    def is_cuda_alike(self) -> bool:
        """Stateless version of [torch.cuda.is_available][]."""
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)

    def is_sleep_mode_available(self) -> bool:
        return self._enum == PlatformEnum.CUDA

    @classmethod
    def device_id_to_physical_device_id(cls, device_id: int):
        if cls.device_control_env_var in os.environ:
            device_ids = os.environ[cls.device_control_env_var].split(",")
            if device_ids == [""]:
                msg = (f"{cls.device_control_env_var} is set to empty string, "
                       "which means current platform support is disabled. If "
                       "you are using ray, please unset the environment "
                       f"variable `{cls.device_control_env_var}` inside the "
                       "worker/actor. Check "
                       "https://github.com/vllm-project/vllm/issues/8402 for "
                       "more information.")
                raise RuntimeError(msg)
            physical_device_id = device_ids[device_id]
            return int(physical_device_id)
        else:
            return device_id

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool,
                             use_mla: bool) -> str:
        """Get the attention backend class of a device."""
        return ""

    @classmethod
    def get_device_capability(
        cls,
        device_id: int = 0,
    ) -> Optional[DeviceCapability]:
        """Stateless version of [torch.cuda.get_device_capability][]."""
        return None

    @classmethod
    def has_device_capability(
        cls,
        capability: Union[tuple[int, int], int],
        device_id: int = 0,
    ) -> bool:
        """
        Test whether this platform is compatible with a device capability.

        The ``capability`` argument can either be:

        - A tuple ``(major, minor)``.
        - An integer ``<major><minor>``. (See {meth}`DeviceCapability.to_int`)
        """
        current_capability = cls.get_device_capability(device_id=device_id)
        if current_capability is None:
            return False

        if isinstance(capability, tuple):
            return current_capability >= capability

        return current_capability.to_int() >= capability

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
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        """
        Check if the current platform supports async output.
        """
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
    def seed_everything(cls, seed: Optional[int] = None) -> None:
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
    def pre_register_and_update(cls,
                                parser: Optional[FlexibleArgumentParser] = None
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
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
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
        if cls.supported_quantization and \
            quant not in cls.supported_quantization:
            raise ValueError(
                f"{quant} quantization is currently not supported in "
                f"{cls.device_name}.")

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

        return CpuArchEnum.OTHER if machine else CpuArchEnum.UNKNOWN

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Checks whether pin memory is available on the current platform."""
        if in_wsl():
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")
            return False
        return True

    @classmethod
    def get_current_memory_usage(cls,
                                 device: Optional[torch.types.Device] = None
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
        import vllm.envs as envs
        from vllm.config import get_current_vllm_config

        parallel_config = get_current_vllm_config().parallel_config
        return (envs.VLLM_USE_V1
                or parallel_config.distributed_executor_backend
                == "external_launcher")

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        """Returns whether the current platform can support v1 for the supplied
        model configuration.
        """
        return False

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        """
        Returns if custom allreduce is supported on the current platform
        """
        return False

    @classmethod
    def validate_request(
        cls,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        processed_inputs: ProcessorInputs,
    ) -> None:
        """Raises if this request is unsupported on this platform"""

    def __getattr__(self, key: str):
        device = getattr(torch, self.device_type, None)
        if device is not None and hasattr(device, key):
            return getattr(device, key)
        else:
            logger.warning("Current platform %s does not have '%s'" \
            " attribute.", self.device_type, key)
            return None

    @classmethod
    def get_cu_count(cls, device_id: int = 0) -> int:
        """
        Returns the total number of compute units (CU) on single GPU.
        """
        raise NotImplementedError

    @classmethod
    def get_piecewise_backend_cls(cls) -> str:
        """
        Get piecewise backend class for piecewise graph.
        """
        return "vllm.compilation.base_piecewise_backend.AbstractPiecewiseBackend"  # noqa


class UnspecifiedPlatform(Platform):
    _enum = PlatformEnum.UNSPECIFIED
    device_type = ""
