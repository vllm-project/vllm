# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch.nn as nn

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


class CustomOp(nn.Module):
    """
    Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """

    def __new__(cls, *args, **kwargs):
        try:
            op_name = cls.__name__
        except AttributeError:
            raise TypeError(
                f"Cannot instantiate '{cls.__name__}': its 'name' attribute "
                f"was not set, possibly because it was not decorated with "
                f"@CustomOp.register, or it's the CustomOp base class itself."
            ) from None

        if op_name not in cls.op_registry_oot:
            op_cls_to_instantiate = cls
        else:
            op_cls_to_instantiate = cls.op_registry_oot[op_name]
            logger.debug("Instantiating custom op: %s using %s", op_name,
                         str(op_cls_to_instantiate))
        return super().__new__(op_cls_to_instantiate)

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

    def forward_neuron(self, *args, **kwargs):
        # By default, we assume that Neuron ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    def forward_oot(self, *args, **kwargs):
        # By default, we assume that OOT ops are compatible with the
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
        elif current_platform.is_neuron():
            return self.forward_neuron
        elif current_platform.is_out_of_tree():
            return self.forward_oot
        else:
            return self.forward_cuda

    @classmethod
    def enabled(cls) -> bool:
        # if no name, then it was not registered
        compilation_config = get_current_vllm_config().compilation_config
        custom_ops = compilation_config.custom_ops
        if not hasattr(cls, "name"):
            logger.warning_once(
                "Custom op %s was not registered, which means it won't appear in the op registry. It will be enabled/disabled based on the global settings.",  # noqa: E501
                cls.__name__,
            )
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
    op_registry: dict[str, type['CustomOp']] = {}
    op_registry_oot: dict[str, type['CustomOp']] = {}

    # Decorator to register custom ops.
    @classmethod
    def register(cls, name: str):

        def decorator(op_cls):
            assert name not in cls.op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            cls.op_registry[name] = op_cls
            return op_cls

        return decorator

    # Decorator to register out-of-tree(oot) custom ops.
    # For OOT custom ops:
    #   if in-tree layer class is registered with an oot_custom_op layer,
    #   the oot_custom_op layer will be used instead.
    # Example:
    # - @UnquantizedFusedMoEMethod.register_oot
    #   class HPUUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod)
    # or
    # - @CustomOP.register_oot(name="UnquantizedFusedMoEMethod")
    @classmethod
    def register_oot(cls, _decorated_op_cls=None, name: Optional[str] = None):

        def decorator(op_cls):
            reg_name = name if name is not None else cls.__name__
            assert reg_name not in cls.op_registry_oot, \
                f"Duplicate op name: {reg_name}"
            op_cls.name = reg_name
            cls.op_registry_oot[reg_name] = op_cls
            return op_cls

        if _decorated_op_cls is None:
            # Called with parentheses: @CustomOP.register_oot()
            # or @CustomOP.register_oot(name="...")
            # So, _decorated_op_cls is None.
            # We return the actual decorator function.
            return decorator
        elif isinstance(_decorated_op_cls, type):  # Check if it's a class
            # Called without parentheses: @CustomOP.register_oot
            # The first argument is the class itself.
            # We call the 'decorator' function immediately with the class.
            return decorator(_decorated_op_cls)
        else:
            # Handle other unexpected cases if necessary
            raise TypeError("Decorator can only be applied to classes.")
