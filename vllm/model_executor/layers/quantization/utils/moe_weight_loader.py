# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared utilities for online MoE weight loading and quantization.
"""

from collections.abc import Callable
from typing import Protocol

import torch
from torch.nn import Module
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.model_executor.model_loader.weight_utils import (
    initialize_single_dummy_weight,
)
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


class MoeQuantizationCallbacks(Protocol):
    """Protocol defining callbacks that MoE quantization methods must provide."""

    def create_scale_tensors(
        self,
        layer: Module,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_size: int,
    ) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
        """Create scale tensors for quantization.

        Returns:
            Tuple of (w13_weight_scale, w2_weight_scale) parameters
        """
        ...

    def get_quantized_dtype(self) -> torch.dtype:
        """Return the target dtype for quantized weights (e.g., int8, fp8)."""
        ...

    def quantize_expert(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a single expert's weight tensor.

        Args:
            weight: Weight tensor of shape [out_features, in_features]

        Returns:
            Tuple of (quantized_weight, scale) where scale shape depends on
            the quantization method (per-tensor or per-channel).
        """
        ...

    def setup_kernel(
        self,
        layer: Module,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None: ...


class MoeOnlineWeightLoader:
    """
    Handles weight loading and quantization for MoE layers.
    """

    def __init__(self, moe_quant_callbacks: MoeQuantizationCallbacks):
        self.moe_quant_callbacks = moe_quant_callbacks

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
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

        # Create quantization-specific scale parameters
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
        w13_weight_scale, w2_weight_scale = (
            self.moe_quant_callbacks.create_scale_tensors(
                layer,
                num_experts,
                intermediate_size_per_partition,
                hidden_size,
            )
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def _create_moe_weight_loader(
        self,
        layer: Module,
        weight_loader: Callable,
        extra_weight_attrs: dict,
    ) -> Callable:
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

            # If we have loaded all elements, call quantization
            target_loaded_numel = layer.w13_weight.numel() + layer.w2_weight.numel()
            if layer._loaded_numel == target_loaded_numel:
                self._on_all_weights_loaded(layer)

                # Delete the bookkeeping
                del layer._loaded_numel
                del layer._w13_weight_orig_id
                del layer._w2_weight_orig_id
                # Prevent the usual `process_weights_after_loading` call
                layer._already_called_process_weights_after_loading = True

            return res

        return patched_weight_loader

    def _on_all_weights_loaded(self, layer: Module) -> None:
        self.process_weights_after_loading(layer)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        # Handle dummy weights (--load_format dummy)
        self._materialize_dummy_weights(layer)

        # Quantize weights and setup kernel
        self.quantize_and_setup_kernel(layer)

    def quantize_and_setup_kernel(self, layer: Module) -> None:
        # Get target dtype and create output tensors
        quantized_dtype = self.moe_quant_callbacks.get_quantized_dtype()
        w13 = torch.empty_like(layer.w13_weight, dtype=quantized_dtype)
        w2 = torch.empty_like(layer.w2_weight, dtype=quantized_dtype)
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale

        # Quantize each expert
        for expert in range(layer.local_num_experts):
            w13[expert, :, :], w13_scale[expert] = (
                self.moe_quant_callbacks.quantize_expert(layer.w13_weight[expert, :, :])
            )
            w2[expert, :, :], w2_scale[expert] = (
                self.moe_quant_callbacks.quantize_expert(layer.w2_weight[expert, :, :])
            )

        # Setup kernel with quantized weights
        self.moe_quant_callbacks.setup_kernel(layer, w13, w2, w13_scale, w2_scale)

    def _materialize_dummy_weights(self, layer: Module) -> None:
        if layer.w13_weight.device == torch.device("meta"):
            w13_weight = torch.nn.Parameter(
                torch.empty_like(layer.w13_weight, device=layer._load_device),
                requires_grad=False,
            )
            set_weight_attrs(
                w13_weight, {"weight_loader": layer.w13_weight.weight_loader}
            )
            _copy_missing_attrs(layer.w13_weight, w13_weight)
            layer.register_parameter("w13_weight", w13_weight)
            initialize_single_dummy_weight(layer.w13_weight)

        if layer.w2_weight.device == torch.device("meta"):
            w2_weight = torch.nn.Parameter(
                torch.empty_like(layer.w2_weight, device=layer._load_device),
                requires_grad=False,
            )
            set_weight_attrs(
                w2_weight, {"weight_loader": layer.w2_weight.weight_loader}
            )
            _copy_missing_attrs(layer.w2_weight, w2_weight)
            layer.register_parameter("w2_weight", w2_weight)
            initialize_single_dummy_weight(layer.w2_weight)
