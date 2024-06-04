from abc import abstractmethod

import torch.nn as nn

from vllm.utils import is_hip, is_cpu


class CustomOp(nn.Module):

    def forward(self, *args, **kwargs):
        if not hasattr(self, "_forward_method"):
            self._forward_method = self.dispatch_forward()
        return self._forward_method(*args, **kwargs)

    @abstractmethod
    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        # By default, we assume that HIP ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        # By default, we assume that CPU ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)

    def dispatch_forward(self):
        if is_hip():
            return self.forward_hip
        elif is_cpu():
            return self.forward_cpu
        else:
            return self.forward_cuda
