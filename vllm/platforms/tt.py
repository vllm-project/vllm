from typing import Tuple

from .interface import Platform, PlatformEnum


class TTPlatform(Platform):
    _enum = PlatformEnum.TT
    
    @staticmethod
    def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
        raise NotImplementedError
