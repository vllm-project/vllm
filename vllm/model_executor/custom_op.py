from typing import Dict, Type

import torch.nn as nn

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import print_warning_once

logger = init_logger(__name__)


class CustomOp(nn.Module):
    """
    Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """

    def __init__(self):
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

    def forward_hpu(self, *args, **kwargs):
        # By default, we assume that Gaudi ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self):
        # NOTE(woosuk): Here we assume that vLLM was built for only one
        # specific backend. Currently, we do not support dynamic dispatching.
        compilation_config = get_current_vllm_config().compilation_config
        enabled = self.enabled()
        if enabled:
            compilation_config.enabled_custom_ops.update([self.__class__.name])
        else:
            compilation_config.disabled_custom_ops.update(
                [self.__class__.name])

        if not enabled:
            return self.forward_native

        if current_platform.is_rocm():
            return self.forward_hip
        elif current_platform.is_cpu():
            return self.forward_cpu
        elif current_platform.is_hpu():
            return self.forward_hpu
        elif current_platform.is_tpu():
            return self.forward_tpu
        elif current_platform.is_xpu():
            return self.forward_xpu
        else:
            return self.forward_cuda

    @classmethod
    def enabled(cls) -> bool:
        # if no name, then it was not registered
        compilation_config = get_current_vllm_config().compilation_config
        custom_ops = compilation_config.custom_ops
        if not hasattr(cls, "name"):
            print_warning_once(
                f"Custom op {cls.__name__} was not registered, "
                f"which means it won't appear in the op registry. "
                f"It will be enabled/disabled based on the global settings.")
            return CustomOp.default_on()

        enabled = f"+{cls.name}" in custom_ops
        disabled = f"-{cls.name}" in custom_ops
        assert not (enabled
                    and disabled), f"Cannot enable and disable {cls.name}"

        return (CustomOp.default_on() or enabled) and not disabled

    @staticmethod
    def default_on() -> bool:
        """
        On by default if level < CompilationLevel.PIECEWISE
        Specifying 'all' or 'none' in custom_op takes precedence.
        """
        from vllm.config import CompilationLevel
        compilation_config = get_current_vllm_config().compilation_config
        custom_ops = compilation_config.custom_ops
        count_none = custom_ops.count("none")
        count_all = custom_ops.count("all")
        return compilation_config.level < CompilationLevel.PIECEWISE and \
            not count_none > 0 or count_all > 0

    # Dictionary of all custom ops (classes, indexed by registered name).
    # To check if an op with a name is enabled, call .enabled() on the class.
    # Examples:
    # - MyOp.enabled()
    # - op_registry["my_op"].enabled()
    op_registry: Dict[str, Type['CustomOp']] = {}

    # Decorator to register custom ops.
    @classmethod
    def register(cls, name: str):

        def decorator(op_cls):
            assert name not in cls.op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            cls.op_registry[name] = op_cls
            return op_cls

        return decorator
