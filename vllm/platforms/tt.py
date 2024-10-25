from typing import Tuple, Optional

from .interface import Platform, PlatformEnum


class TTPlatform(Platform):
    _enum = PlatformEnum.TT
    
    @staticmethod
    def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
        raise NotImplementedError
    
    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return True
