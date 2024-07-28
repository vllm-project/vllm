from typing import Optional

import torch

from vllm.utils import is_tpu

from .interface import Platform, PlatformEnum, UnspecifiedPlatform

current_platform: Optional[Platform]

if torch.version.cuda is not None:
    from .cuda import CudaPlatform
    current_platform = CudaPlatform()
elif torch.version.hip is not None:
    from .rocm import RocmPlatform
    current_platform = RocmPlatform()
elif is_tpu():
    from .tpu import TpuPlatform
    current_platform = TpuPlatform()
else:
    current_platform = UnspecifiedPlatform()

__all__ = ['Platform', 'PlatformEnum', 'current_platform']
