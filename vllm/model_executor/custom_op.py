import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils import is_cpu, is_hip, is_xpu


def contain_cpu_tensor(data):
    if isinstance(data, torch.Tensor):
        return data.device.type == "cpu"
    elif isinstance(data, (list, tuple)):
        return any(contain_cpu_tensor(item) for item in data)
    elif isinstance(data, dict):
        return any(contain_cpu_tensor(item) for item in data.values())

    return None


class CustomOp(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._forward_method = self.dispatch_forward()

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        """PyTorch-native implementation of the forward method.

        This method is optional. If implemented, it can be used with compilers
        such as torch.compile or PyTorch XLA. Also, it can be used for testing
        purposes.
        """
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        # By default, we assume that HIP ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        # By default, we assume that XPU ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        # By default, we assume that CPU ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs):
        # By default, we assume that TPU ops are compatible with the
        # PyTorch-native implementation.
        # NOTE(woosuk): This is a placeholder for future extensions.
        return self.forward_native(*args, **kwargs)

    def forward_gaudi(self, *args, **kwargs):
        # By default, we assume that Gaudi ops are compatible with the
        # PyTorch-native implementation.
        # NOTE(woosuk): This is a placeholder for future extensions.
        return self.forward_native(*args, **kwargs)

    def forward_dynamic(self, *args, **kwargs):
        # Dynamic patch forward by input tensor device
        cpu_device = contain_cpu_tensor(args) or contain_cpu_tensor(kwargs)
        if cpu_device:
            return self.forward_native(*args, **kwargs)
        else:
            return self.forward_cuda(*args, **kwargs)

    def dispatch_forward(self):
        # NOTE(woosuk): Here we assume that vLLM was built for only one
        # specific backend. Currently, we do not support dynamic dispatching.

        if envs.VLLM_TEST_COMPILE_NO_CUSTOM_OPS:
            return self.forward_native

        if envs.VLLM_DYNAMIC_FORWARD:
            return self.forward_dynamic

        if is_hip():
            return self.forward_hip
        elif is_cpu():
            return self.forward_cpu
        elif current_platform.is_tpu():
            return self.forward_tpu
        elif is_xpu():
            return self.forward_xpu
        else:
            return self.forward_cuda
