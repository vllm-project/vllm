import enum
import random
from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None


class _Backend(enum.Enum):
    FLASH_ATTN = enum.auto()
    FLASH_ATTN_VLLM_V1 = enum.auto()
    XFORMERS = enum.auto()
    ROCM_FLASH = enum.auto()
    TORCH_SDPA = enum.auto()
    OPENVINO = enum.auto()
    FLASHINFER = enum.auto()
    HPU_ATTN = enum.auto()
    PALLAS = enum.auto()
    IPEX = enum.auto()
    NO_ATTENTION = enum.auto()


class PlatformEnum(enum.Enum):
    CUDA = enum.auto()
    ROCM = enum.auto()
    TPU = enum.auto()
    HPU = enum.auto()
    XPU = enum.auto()
    CPU = enum.auto()
    NEURON = enum.auto()
    OPENVINO = enum.auto()
    UNSPECIFIED = enum.auto()


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
    device_type: str

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

    def is_openvino(self) -> bool:
        return self._enum == PlatformEnum.OPENVINO

    def is_cuda_alike(self) -> bool:
        """Stateless version of :func:`torch.cuda.is_available`."""
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)

    @classmethod
    def get_default_attn_backend(cls, selected_backend: _Backend):
        """Get the default attention backend of a device."""
        return None

    @classmethod
    def get_device_capability(
        cls,
        device_id: int = 0,
    ) -> Optional[DeviceCapability]:
        """Stateless version of :func:`torch.cuda.get_device_capability`."""
        return None

    @classmethod
    def has_device_capability(
        cls,
        capability: Union[Tuple[int, int], int],
        device_id: int = 0,
    ) -> bool:
        """
        Test whether this platform is compatible with a device capability.

        The ``capability`` argument can either be:

        - A tuple ``(major, minor)``.
        - An integer ``<major><minor>``. (See :meth:`DeviceCapability.to_int`)
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
    def seed_everything(cls, seed: int) -> None:
        """
        Set the seed of each random module.
        `torch.manual_seed` will set seed on all devices.

        Loosely based on: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/fabric/utilities/seed.py#L20
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

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


class UnspecifiedPlatform(Platform):
    _enum = PlatformEnum.UNSPECIFIED
    device_type = ""
