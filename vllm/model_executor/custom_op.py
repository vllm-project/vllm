# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import get_cached_compilation_config
from vllm.logger import init_logger
from vllm.model_executor.utils import maybe_disable_graph_partition
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Dictionary of all custom ops (classes, indexed by registered name).
# To check if an op with a name is enabled, call .enabled() on the class.
# Examples:
# - MyOp.enabled()
# - op_registry["my_op"].enabled()
op_registry: dict[str, type["CustomOp"] | type["PluggableLayer"]] = {}
op_registry_oot: dict[str, type["CustomOp"] | type["PluggableLayer"]] = {}


class PluggableLayer(nn.Module):
    """
    Base class for pluggable layers.

    A PluggableLayer is a *module-composing* abstraction: it may instantiate other
    ``torch.nn.Module`` objects as sub-layers, and its functionality depends on
    these sub-layers following a generalized invocation sequence. Also, it is stateful
    and may hold parameters or buffers.

    Unlike :class:`CustomOp`, PluggableLayer does NOT provide per-platform
    ``forward_*`` dispatch. Instead, it supports out-of-tree (OOT) replacement
    of the entire layer class at instantiation time, allowing customized
    initialization and submodule composition.
    """

    def __new__(cls, *args, **kwargs):
        try:
            layer_class_name = cls.__name__
        except AttributeError:
            raise TypeError(
                f"Cannot instantiate '{cls.__name__}': its 'name' attribute "
                f"was not set, possibly because it was not decorated with "
                f"@PluggableLayer.register, or it's the PluggableLayer itself."
            ) from None

        if layer_class_name not in op_registry_oot:
            layer_cls_to_instantiate = cls
        else:
            layer_cls_to_instantiate = op_registry_oot[layer_class_name]
            logger.debug(
                "Instantiating pluggable layer: %s using %s",
                layer_class_name,
                str(layer_cls_to_instantiate),
            )
        return super().__new__(layer_cls_to_instantiate)

    # Decorator to register pluggable layers.
    @classmethod
    def register(cls, name: str):
        def decorator(op_cls):
            assert name not in op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            op_registry[name] = op_cls
            return op_cls

        return decorator

    # Decorator to register out-of-tree(oot) pluggable layers.
    # For OOT pluggable layers:
    #   if in-tree layer class is registered with an oot_custom_layer,
    #   the oot_custom_layer will be used instead.
    @classmethod
    def register_oot(cls, _decorated_layer_cls=None, name: str | None = None):
        def decorator(layer_cls):
            reg_name = name if name is not None else cls.__name__
            assert reg_name not in op_registry_oot, f"Duplicate layer name: {reg_name}"
            layer_cls.name = reg_name
            op_registry_oot[reg_name] = layer_cls
            return layer_cls

        if _decorated_layer_cls is None:
            # Called with parentheses: @PluggableLayer.register_oot()
            # or @PluggableLayer.register_oot(name="...")
            return decorator
        elif isinstance(_decorated_layer_cls, type):  # Check if it's a class
            # Called without parentheses: @PluggableLayer.register_oot
            return decorator(_decorated_layer_cls)
        else:
            raise TypeError("Decorator can only be applied to classes.")


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

        if op_name not in op_registry_oot:
            op_cls_to_instantiate = cls
        else:
            op_cls_to_instantiate = op_registry_oot[op_name]
            logger.debug(
                "Instantiating custom op: %s using %s",
                op_name,
                str(op_cls_to_instantiate),
            )
        return super().__new__(op_cls_to_instantiate)

    def __init__(self, *, enforce_enable: bool = False, compile_native: bool = False):
        super().__init__()
        self._enforce_enable = enforce_enable
        self._forward_method = self.dispatch_forward(compile_native=compile_native)

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
        # By default, we assume that CPU ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs):
        # By default, we assume that TPU ops are compatible with the
        # PyTorch-native implementation.
        # NOTE(woosuk): This is a placeholder for future extensions.
        return self.forward_native(*args, **kwargs)

    def forward_oot(self, *args, **kwargs):
        # By default, we assume that OOT ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self, compile_native: bool):
        # NOTE(woosuk): Here we assume that vLLM was built for only one
        # specific backend. Currently, we do not support dynamic dispatching.
        compilation_config = get_cached_compilation_config()

        # NOTE(shen-shanshan): CustomOp object can be enforce enabled, e.g.,
        # enable device-specific kernels in ViT models when enabling graph
        # mode. By default, it will follow the compilation_config to determine
        # whether enable itself.
        # This enforce_enable mechanism will be removed after we adding a
        # separate compilation_config for multi-modal part.
        enabled = self._enforce_enable or self.enabled()
        if enabled:
            compilation_config.enabled_custom_ops.update([self.__class__.name])
        else:
            compilation_config.disabled_custom_ops.update([self.__class__.name])

        if not enabled:
            # Compile forward_native to avoid eager torch ops if inside
            # opaque torch custom op (e.g. fused_moe, unified_attention, etc.)
            return self.maybe_compile(self.forward_native, enable=compile_native)

        if current_platform.is_rocm():
            return self.forward_hip
        elif current_platform.is_cpu():
            return self.forward_cpu
        elif current_platform.is_tpu():
            return self.forward_tpu
        elif current_platform.is_xpu():
            return self.forward_xpu
        elif current_platform.is_out_of_tree():
            return self.forward_oot
        else:
            return self.forward_cuda

    def maybe_compile(self, fn, *, enable: bool = True):
        """
        Compile fn if compilation enabled.
        Useful for CustomOp instances called from within a torch custom op,
        meaning the forward call is hidden from the model-level torch.compile.

        NOTE: this does not enable fusion across ops, so opaque custom ops
        should still be unwrapped wherever possible.
        """
        # Do not compile if compilation disabled
        from vllm.config.compilation import CompilationMode

        if not enable:
            return fn

        # Do not compile if global compilation disabled
        compilation_config = get_cached_compilation_config()
        if compilation_config.mode == CompilationMode.NONE:
            return fn

        # If eager backend is used, do not compile either
        if compilation_config.backend == "eager":
            return fn

        # dynamic=True to avoid recompilations
        return torch.compile(
            fn,
            dynamic=True,
            backend=current_platform.simple_compile_backend,
            options=maybe_disable_graph_partition(
                current_platform.simple_compile_backend
            ),
        )

    @classmethod
    def enabled(cls) -> bool:
        # if no name, then it was not registered
        compilation_config = get_cached_compilation_config()
        custom_ops = compilation_config.custom_ops
        if not hasattr(cls, "name"):
            logger.warning_once(
                "Custom op %s was not registered, which means it won't appear "
                "in the op registry. It will be enabled/disabled based on the "
                "global settings.",
                cls.__name__,
            )
            return CustomOp.default_on()

        enabled = f"+{cls.name}" in custom_ops
        disabled = f"-{cls.name}" in custom_ops
        assert not (enabled and disabled), f"Cannot enable and disable {cls.name}"

        return (CustomOp.default_on() or enabled) and not disabled

    @staticmethod
    def default_on() -> bool:
        """
        Behavior controlled by `CompilationConfig.custom_ops`: On by default if
        'all', off by default if 'none'.
        When PyTorch Inductor is used, 'none' is the default value,
        otherwise 'all'.
        """
        compilation_config = get_cached_compilation_config()
        count_none = compilation_config.custom_ops.count("none")
        count_all = compilation_config.custom_ops.count("all")
        assert count_none + count_all == 1

        return not count_none > 0 or count_all > 0

    # Decorator to register custom ops.
    @classmethod
    def register(cls, name: str):
        def decorator(op_cls):
            assert name not in op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            op_registry[name] = op_cls
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
    def register_oot(cls, _decorated_op_cls=None, name: str | None = None):
        def decorator(op_cls):
            reg_name = name if name is not None else cls.__name__
            assert reg_name not in op_registry_oot, f"Duplicate op name: {reg_name}"
            op_cls.name = reg_name
            op_registry_oot[reg_name] = op_cls
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
