import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.utils import is_cpu, is_hip, is_tpu, is_xpu


class CustomOp(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._is_compiled = False
        self._forward_method = self.dispatch_forward()

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    @staticmethod
    def forward_static(*args, **kwargs):
        """PyTorch-native implementation of the forward method.

        This method is optional. If implemented, it can be used with
        `torch.compile` to generate the custom kernel for the op.
        This function is necessary because any method that uses `self`
        triggers re-compilation in `torch.compile`.
        """
        raise NotImplementedError

    def forward_native(self, *args, **kwargs):
        """PyTorch-native implementation of the forward method.

        This method is expected to invoke the forward_static method
        with the input arguments and the attributes of `self`.
        By default, we assume that forward_static does not need any
        attributes of `self`.
        """
        return self.forward_static(*args, **kwargs)

    def forward_compile(self, *args, **kwargs):
        """Runs a torch-compiled version of forward_native.

        This method hooks into the forward_native method and compiles
        `forward_static` using `torch.compile` if it has not been compiled yet.
        """
        if not self._is_compiled and not envs.VLLM_TEST_DYNAMO_GRAPH_CAPTURE:
            self.forward_static = torch.compile(  # type: ignore
                self.forward_static)
            self._is_compiled = True
        return self.forward_native(*args, **kwargs)

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
        if not self._is_compiled and not envs.VLLM_TEST_DYNAMO_GRAPH_CAPTURE:
            self.forward_static = torch.compile(  # type: ignore
                self.forward_static,
                options={
                    "fx_graph_cache": True,
                    "cpp_wrapper": True,
                    "dce": True,
                })
            self._is_compiled = True
        return self.forward_native(*args, **kwargs)

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
        if is_hip():
            return self.forward_hip
        elif is_cpu():
            return self.forward_cpu
        elif is_tpu():
            return self.forward_tpu
        elif is_xpu():
            return self.forward_xpu
        else:
            return self.forward_cuda
