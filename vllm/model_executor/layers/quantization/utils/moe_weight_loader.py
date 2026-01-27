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

    Classes using this mixin should:
    - Implement `_create_scale_tensors` to create quantization-specific scales
    - Implement `process_weights_after_loading` for quantization and kernel setup
    """

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """
        Shared weight creation logic for online MoE quantization.

        Creates weights on meta device with a patched weight loader that
        tracks loading progress. Subclasses must implement
        `_create_scale_tensors` to create quantization-specific scales.
        """
        # Store layer dimensions
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype

        # Patch weight loader to track loading and trigger quantization
        weight_loader = extra_weight_attrs["weight_loader"]
        new_extra_weight_attrs = extra_weight_attrs.copy()
        new_extra_weight_attrs["weight_loader"] = self._create_moe_weight_loader(
            layer, weight_loader, extra_weight_attrs
        )
        extra_weight_attrs = new_extra_weight_attrs

        # WEIGHTS (on meta device, materialized JIT in patched_weight_loader)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                device="meta",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                device="meta",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Stash device for JIT materialization
        layer._load_device = torch.get_default_device()

        # Let subclass create quantization-specific scale parameters
        self._create_scale_parameters(
            layer,
            num_experts,
            intermediate_size_per_partition,
            hidden_size,
            extra_weight_attrs,
        )

    def _create_scale_parameters(
        self,
        layer: Module,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_size: int,
        extra_weight_attrs: dict,
    ) -> None:
        """
        Create and register scale parameters for quantization.

        Calls `_create_scale_tensors` (which subclasses must implement) to get
        the scale tensors, then registers them and sets weight attributes.
        """
        w13_weight_scale, w2_weight_scale = self._create_scale_tensors(
            layer,
            num_experts,
            intermediate_size_per_partition,
            hidden_size,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def _create_scale_tensors(
        self,
        layer: Module,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_size: int,
    ) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
        """
        Create scale tensors for quantization. Subclasses must implement.

        Args:
            layer: The MoE layer module (can be used to set extra attributes)
            num_experts: Number of experts
            intermediate_size_per_partition: Intermediate size per partition
            hidden_size: Hidden size

        Returns:
            Tuple of (w13_weight_scale, w2_weight_scale) parameters
        """
        raise NotImplementedError(
            "Subclasses must implement _create_scale_tensors"
        )

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
