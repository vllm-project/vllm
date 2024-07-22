import enum
from typing import Tuple

import torch


class PlatformEnum(enum.Enum):
    CUDA = enum.auto()
    ROCM = enum.auto()
    TPU = enum.auto()
    UNSPECIFIED = enum.auto()


class Platform:
    _enum: PlatformEnum

    def is_cuda(self) -> bool:
        return self._enum == PlatformEnum.CUDA

    def is_rocm(self) -> bool:
        return self._enum == PlatformEnum.ROCM

    def is_tpu(self) -> bool:
        return self._enum == PlatformEnum.TPU

    @staticmethod
    def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
        raise NotImplementedError

    @staticmethod
    def inference_mode():
        """A device-specific wrapper of `torch.inference_mode`.

        This wrapper is recommended because some hardware backends such as TPU
        do not support `torch.inference_mode`. In such a case, they will fall
        back to `torch.no_grad` by overriding this method.
        """
        return torch.inference_mode(mode=True)


class UnspecifiedPlatform(Platform):
    _enum = PlatformEnum.UNSPECIFIED
