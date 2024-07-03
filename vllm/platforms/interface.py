import enum
from typing import Tuple


class PlatformEnum(enum.Enum):
    CUDA = enum.auto()
    ROCM = enum.auto()


class Platform:
    _enum: PlatformEnum

    def is_cuda(self) -> bool:
        return self._enum == PlatformEnum.CUDA

    def is_rocm(self) -> bool:
        return self._enum == PlatformEnum.ROCM

    @staticmethod
    def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
        raise NotImplementedError
