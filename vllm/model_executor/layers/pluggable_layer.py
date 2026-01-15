# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)


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
                f"@PluggableLayer.register, or it's the PluggableLayer base class itself."
            ) from None

        if layer_class_name not in cls.layer_registry_oot:
            layer_cls_to_instantiate = cls
        else:
            layer_cls_to_instantiate = cls.layer_registry_oot[layer_class_name]
            logger.debug(
                "Instantiating pluggable layer: %s using %s",
                layer_class_name,
                str(layer_cls_to_instantiate),
            )
        return super().__new__(layer_cls_to_instantiate)

    # Dictionary of all pluggable layers (classes, indexed by registered name).
    layer_registry_oot: dict[str, type["PluggableLayer"]] = {}

    # Decorator to register out-of-tree(oot) pluggable layers.
    # For OOT pluggable layers:
    #   if in-tree layer class is registered with an oot_custom_layer,
    #   the oot_custom_layer will be used instead.
    @classmethod
    def register_oot(cls, _decorated_layer_cls=None, name: str | None = None):
        def decorator(layer_cls):
            reg_name = name if name is not None else cls.__name__
            assert reg_name not in cls.layer_registry_oot, (
                f"Duplicate layer name: {reg_name}"
            )
            layer_cls.name = reg_name
            cls.layer_registry_oot[reg_name] = layer_cls
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
