from typing import Optional

import torch

from .interface import Platform, PlatformEnum

current_platform: Optional[Platform]

if torch.version.cuda is not None:
    from .cuda import CudaPlatform
    current_platform = CudaPlatform()
elif torch.version.hip is not None:
    from .rocm import RocmPlatform
    current_platform = RocmPlatform()
else:
    current_platform = None

__all__ = ['Platform', 'PlatformEnum', 'current_platform']
