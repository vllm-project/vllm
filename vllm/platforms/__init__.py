import torch

from .interface import Platform, PlatformEnum

if torch.version.cuda is not None:
    from .cuda import CudaPlatform as CurrentPlatform
elif torch.version.hip is not None:
    from .rocm import RocmPlatform as CurrentPlatform
else:
    CurrentPlatform = None

__all__ = ['Platform', 'PlatformEnum', 'CurrentPlatform']
