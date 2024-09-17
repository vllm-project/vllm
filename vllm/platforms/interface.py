import enum
from typing import NamedTuple, Optional, Tuple, overload

import torch


class PlatformEnum(enum.Enum):
    CUDA = enum.auto()
    ROCM = enum.auto()
    TPU = enum.auto()
    CPU = enum.auto()
    UNSPECIFIED = enum.auto()


class DeviceCapability(NamedTuple):
    major: int
    minor: int

    def __int__(self) -> int:
        return self.major * 10 + self.minor


class Platform:
    _enum: PlatformEnum

    def is_cuda(self) -> bool:
        return self._enum == PlatformEnum.CUDA

    def is_rocm(self) -> bool:
        return self._enum == PlatformEnum.ROCM

    def is_tpu(self) -> bool:
        return self._enum == PlatformEnum.TPU

    def is_cpu(self) -> bool:
        return self._enum == PlatformEnum.CPU

    @classmethod
    def is_cuda_alike(cls) -> bool:
        """Stateless version of :func:`torch.cuda.is_available`."""
        return False

    @classmethod
    def get_device_capability(
        cls,
        device_id: int = 0,
    ) -> Optional[DeviceCapability]:
        """Stateless version of :func:`torch.cuda.get_device_capability`."""
        return None

    @classmethod
    @overload
    def has_device_capability(
        cls,
        capability: Tuple[int, int],
        *,
        device_id: int = 0,
    ) -> bool:
        ...

    @classmethod
    @overload
    def has_device_capability(
        cls,
        *,
        major: int,
        minor: int = 0,
        device_id: int = 0,
    ) -> bool:
        ...

    @classmethod
    def has_device_capability(
        cls,
        capability: Optional[Tuple[int, int]] = None,
        *,
        major: Optional[int] = None,
        minor: int = 0,
        device_id: int = 0,
    ) -> bool:
        current_capability = cls.get_device_capability(device_id=device_id)
        if current_capability is None:
            return False

        if capability is not None:
            major, minor = capability
        else:
            assert major is not None

        return current_capability >= DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    def inference_mode(cls):
        """A device-specific wrapper of `torch.inference_mode`.

        This wrapper is recommended because some hardware backends such as TPU
        do not support `torch.inference_mode`. In such a case, they will fall
        back to `torch.no_grad` by overriding this method.
        """
        return torch.inference_mode(mode=True)


class UnspecifiedPlatform(Platform):
    _enum = PlatformEnum.UNSPECIFIED
