import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils import is_cpu, is_hip, is_xpu

# Set this flag to avoid the re-compilation due to `self`.
# NOTE(woosuk): This only works for PyTorch 2.4+.
# In PyTorch 2.5, we will not need this flag.
torch._dynamo.config.inline_inbuilt_nn_modules = True


class CustomOp(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._is_compiled = False
        self._forward_method = self.dispatch_forward()

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        """PyTorch-native implementation of the forward method.

        This method is optional. If implemented, it can be used with
        `torch.compile` to generate the custom kernel for the op.
        """
        raise NotImplementedError

    def forward_compile(self, *args, **kwargs):
        """Runs a torch-compiled version of forward_native."""
        if torch._utils.is_compiling():
            return self.forward_native(*args, **kwargs)

        if not self._is_compiled:
            options = {"cpp_wrapper": True} if is_cpu() else None
            self._forward_compiled = torch.compile(self.forward_native,
                                                   options=options)
            self._is_compiled = True
        return self._forward_compiled(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs):
        """Forward method for NVIDIA GPUs.

        By default, we use torch.compile to optimize the op. However, we can
        override this method to use a custom CUDA implementation if needed.
        """
        return self.forward_compile(*args, **kwargs)

    def forward_hip(self, *args, **kwargs):
        """Forward method for AMD GPUs.

        By default, this method is the same as forward_cuda. However, we can
        override this method to use a custom HIP implementation if needed.
        """
        return self.forward_cuda(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        """Forward method for Intel XPUs.

        By default, we use torch.compile to optimize the op. However, we can
        override this method to use a custom XPU implementation if needed.
        """
        return self.forward_compile(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        """Forward method for CPUs.

        By default, we use torch.compile to optimize the op. However, we can
        override this method to use a custom CPU implementation if needed.
        """
        return self.forward_compile(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs):
        """Forward method for TPUs.

        For TPUs, the whole model is torch-compiled instead of individual ops.
        So, we can just use the native implementation.
        """
        return self.forward_native(*args, **kwargs)

    def forward_hpu(self, *args, **kwargs):
        """Forward method for HPUs."""
        raise NotImplementedError("HPU is not supported yet.")

    def dispatch_forward(self):
        # NOTE(woosuk): Here we assume that vLLM was built for only one
        # specific backend. Currently, we do not support dynamic dispatching.

        if envs.VLLM_TEST_COMPILE_NO_CUSTOM_OPS:
            return self.forward_native

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
