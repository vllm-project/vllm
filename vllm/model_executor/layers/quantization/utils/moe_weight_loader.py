# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared utilities for online MoE weight loading and quantization.
"""

import torch
from torch.nn import Module
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.model_executor.utils import set_weight_attrs


class CopyNumelCounter(TorchDispatchMode):
    """
    Tracks total number of elements modified with `copy_`. Useful for keeping
    track of weight loading where underlying weights can be arbitrarily
    transformed (such as with `narrow`) before calling copy.
    """

    def __init__(self):
        super().__init__()
        self.copied_numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        out = func(*args, **kwargs)
        if func == torch.ops.aten.copy_.default:
            self.copied_numel += args[0].numel()
        return out


def _copy_missing_attrs(old: torch.Tensor, new: torch.Tensor) -> None:
    """Copies any attrs present in `old` but not in `new` to `new`"""
    new_attrs = set(dir(new))
    attrs_to_set = {}
    for attr in dir(old):
        if attr not in new_attrs:
            attrs_to_set[attr] = getattr(old, attr)
    set_weight_attrs(new, attrs_to_set)


class OnlineWeightLoaderMixin:
    """
    Mixin providing shared weight loading patterns for online MoE quantization.

    This mixin provides a deferred weight loading pattern that:
    1. Creates weights on meta device initially
    2. Materializes weights just-in-time when loading starts
    3. Tracks loading progress using CopyNumelCounter
    4. Calls a callback when all weights are loaded

    Classes using this mixin should call `_create_moe_weight_loader` in their
    `create_weights` method and implement `_on_all_moe_weights_loaded` to
    handle post-loading processing (e.g., quantization, kernel setup).
    """

    def _create_moe_weight_loader(
        self,
        layer: Module,
        weight_loader,
        extra_weight_attrs: dict,
    ):
        """
        Creates a patched weight loader that tracks loading progress
        and calls _on_all_moe_weights_loaded when complete.

        Args:
            layer: The MoE layer module
            weight_loader: The original weight loader function
            extra_weight_attrs: Extra attributes to set on weights

        Returns:
            A patched weight loader function
        """

        def patched_weight_loader(param, loaded_weight, *args, **kwargs):
            # Add a counter to track how many elements we have updated
            if not hasattr(layer, "_loaded_numel"):
                layer._loaded_numel = 0

                # Save the ids of original w13 and w2 so that we can
                # distinguish which one `param` should map to
                layer._w13_weight_orig_id = id(layer.w13_weight)
                layer._w2_weight_orig_id = id(layer.w2_weight)

                # When the first `loaded_weight` is about to be loaded,
                # materialize weights just-in-time
                w13_weight = torch.nn.Parameter(
                    torch.empty_like(layer.w13_weight, device=layer._load_device),
                    requires_grad=False,
                )
                set_weight_attrs(w13_weight, extra_weight_attrs)
                _copy_missing_attrs(layer.w13_weight, w13_weight)
                layer.register_parameter("w13_weight", w13_weight)

                w2_weight = torch.nn.Parameter(
                    torch.empty_like(layer.w2_weight, device=layer._load_device),
                    requires_grad=False,
                )
                set_weight_attrs(w2_weight, extra_weight_attrs)
                _copy_missing_attrs(layer.w2_weight, w2_weight)
                layer.register_parameter("w2_weight", w2_weight)
                del layer._load_device

            # Refresh the reference to `param` to reflect JIT materialization
            if id(param) == layer._w13_weight_orig_id:
                param = layer.w13_weight
            elif id(param) == layer._w2_weight_orig_id:
                param = layer.w2_weight

            # Load the current weight chunk with tracking
            copy_numel_counter = CopyNumelCounter()
            with copy_numel_counter:
                res = weight_loader(param, loaded_weight, *args, **kwargs)
            layer._loaded_numel += copy_numel_counter.copied_numel

            # If we have loaded all elements, call the completion handler
            target_loaded_numel = layer.w13_weight.numel() + layer.w2_weight.numel()
            if layer._loaded_numel == target_loaded_numel:
                self._on_all_moe_weights_loaded(layer)

                # Delete the bookkeeping
                del layer._loaded_numel
                del layer._w13_weight_orig_id
                del layer._w2_weight_orig_id
                # Prevent the usual `process_weights_after_loading` call
                layer._already_called_process_weights_after_loading = True

            return res

        return patched_weight_loader

    def _on_all_moe_weights_loaded(self, layer: Module) -> None:
        """
        Called when all MoE weights have been loaded.
        Subclasses should override this to perform quantization and kernel setup.
        Default implementation calls process_weights_after_loading.
        """
        self.process_weights_after_loading(layer)
