# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared utilities for online MoE weight loading and quantization.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch.nn import Module

from vllm.model_executor.model_loader.reload.meta import (
    CopyCounter,
    materialize_meta_tensor,
)
from vllm.model_executor.model_loader.weight_utils import (
    initialize_single_dummy_weight,
)
from vllm.model_executor.utils import set_weight_attrs


class MoeOnlineQuantizer(ABC):
    @abstractmethod
    def create_scale_tensors(
        self,
        layer: Module,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_size: int,
    ) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
        raise NotImplementedError

    @abstractmethod
    def get_quantized_dtype(self) -> torch.dtype:
        raise NotImplementedError

    @abstractmethod
    def quantize_expert(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def setup_kernel(
        self,
        layer: Module,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None:
        raise NotImplementedError


class MoeOnlineWeightQuantizer:
    """
    Handles weight loading and quantization for MoE layers.
    """

    def __init__(self, moe_quant_callbacks: MoeOnlineQuantizer):
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
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype

        # Stash device for JIT materialization
        layer._load_device = torch.get_default_device()

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

        # Create scale parameters for quantization
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

        self._create_moe_bias_params(
            layer,
            num_experts,
            intermediate_size_per_partition,
            hidden_size,
            params_dtype,
            weight_loader,
        )

        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def _create_moe_bias_params(
        self,
        layer: Module,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
    ) -> None:
        """Create bias parameters for models with biased MoE (e.g. GPT-OSS).

        Uses the original (non-patched) weight_loader since biases are regular
        parameters that don't need JIT materialization or numel tracking.
        """
        moe_config = getattr(self.moe_quant_callbacks, "moe", None)
        if moe_config is None or not moe_config.has_bias:
            return

        bias_attrs = {"weight_loader": weight_loader}
        w13_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, bias_attrs)

        w2_bias = torch.nn.Parameter(
            torch.zeros(num_experts, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, bias_attrs)

    def _create_moe_weight_loader(
        self,
        layer: Module,
        weight_loader: Callable,
        extra_weight_attrs: dict,
    ) -> Callable:
        """
        Create a patched weight loader that handles JIT materialization and
        triggers quantization when all weight shards are loaded.

        This wrapper performs three key functions:
        1. JIT Materialization: On first call, materializes w13_weight and
           w2_weight from meta device to the target device, reducing peak
           memory during model loading.
        2. Load Tracking: Tracks total elements loaded across all shards
           to detect when loading is complete.
        3. Auto-Quantization: When all shards are loaded,
           automatically call process_weights_after_loading to quantize
           weights and set up the kernel.

        Args:
            layer: The MoE layer module being loaded.
            weight_loader: The original weight loader callable to wrap.
            extra_weight_attrs: Additional attributes to set on weight tensors.

        Returns:
            A patched weight loader callable that wraps the original.
        """
        # Closure variables to track loading state without polluting layer
        load_device = layer._load_device
        loaded_numel = 0
        w13_orig_id: int | None = None
        w2_orig_id: int | None = None

        def patched_weight_loader(param, loaded_weight, *args, **kwargs):
            nonlocal loaded_numel, w13_orig_id, w2_orig_id

            if w13_orig_id is None:
                # Save the ids of original w13 and w2 so that we can
                # distinguish which one `param` should map to
                w13_orig_id = id(layer.w13_weight)
                w2_orig_id = id(layer.w2_weight)

                # Materialize weights just-in-time
                with torch.device(load_device):
                    w13_weight = torch.nn.Parameter(
                        materialize_meta_tensor(layer.w13_weight),
                        requires_grad=False,
                    )
                    set_weight_attrs(w13_weight, extra_weight_attrs)
                    layer.register_parameter("w13_weight", w13_weight)

                    w2_weight = torch.nn.Parameter(
                        materialize_meta_tensor(layer.w2_weight),
                        requires_grad=False,
                    )
                    set_weight_attrs(w2_weight, extra_weight_attrs)
                    layer.register_parameter("w2_weight", w2_weight)

            # Refresh the reference to `param` to reflect JIT materialization
            if id(param) == w13_orig_id:
                param = layer.w13_weight
            elif id(param) == w2_orig_id:
                param = layer.w2_weight

            # Load the current weight chunk with tracking
            with CopyCounter() as counter:
                res = weight_loader(param, loaded_weight, *args, **kwargs)
            loaded_numel += counter.copied_numel

            target_numel = layer.w13_weight.numel() + layer.w2_weight.numel()
            if loaded_numel == target_numel:
                self.quantize_and_setup_kernel(layer)

                # Prevent the usual `process_weights_after_loading` call
                layer._already_called_process_weights_after_loading = True

            return res

        return patched_weight_loader

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        self._materialize_dummy_weights(layer)

        self.quantize_and_setup_kernel(layer)

    def quantize_and_setup_kernel(self, layer: Module) -> None:
        quantized_dtype = self.moe_quant_callbacks.get_quantized_dtype()
        w13 = torch.empty_like(layer.w13_weight, dtype=quantized_dtype)
        w2 = torch.empty_like(layer.w2_weight, dtype=quantized_dtype)
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale

        for expert in range(layer.local_num_experts):
            w13[expert, :, :], w13_scale[expert] = (
                self.moe_quant_callbacks.quantize_expert(layer.w13_weight[expert, :, :])
            )
            w2[expert, :, :], w2_scale[expert] = (
                self.moe_quant_callbacks.quantize_expert(layer.w2_weight[expert, :, :])
            )

        self.moe_quant_callbacks.setup_kernel(layer, w13, w2, w13_scale, w2_scale)

    def _materialize_dummy_weights(self, layer: Module) -> None:
        if layer.w13_weight.device == torch.device("meta"):
            with torch.device(layer._load_device):
                w13_weight = torch.nn.Parameter(
                    materialize_meta_tensor(layer.w13_weight),
                    requires_grad=False,
                )
                set_weight_attrs(
                    w13_weight, {"weight_loader": layer.w13_weight.weight_loader}
                )
                layer.register_parameter("w13_weight", w13_weight)
            initialize_single_dummy_weight(layer.w13_weight)

        if layer.w2_weight.device == torch.device("meta"):
            with torch.device(layer._load_device):
                w2_weight = torch.nn.Parameter(
                    materialize_meta_tensor(layer.w2_weight),
                    requires_grad=False,
                )
                set_weight_attrs(
                    w2_weight, {"weight_loader": layer.w2_weight.weight_loader}
                )
                layer.register_parameter("w2_weight", w2_weight)
            initialize_single_dummy_weight(layer.w2_weight)
