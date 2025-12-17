# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LoRA layer with optional CUDA multistream parallel execution.

When enabled via VLLM_ENABLE_LORA_STREAM=1, LoRA computation runs on a
separate CUDA stream in parallel with the base layer GEMM operation.

Execution timeline (when enabled):
  Main stream: [base GEMM] ---------> [wait] -> [add result]
  Aux stream:  [LoRA: B @ (A @ x)] ---^
"""

import os
import torch
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.distributed.utils import divide
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import aux_stream, direct_register_custom_op

from .base import BaseLayerWithLoRA
from .utils import _get_lora_device

# Opt-in: disabled by default since overhead may exceed benefit for small models
_ENABLE_LORA_STREAM = os.getenv("VLLM_ENABLE_LORA_STREAM", "0") == "1"

# Counter for unique layer names
_LORA_LAYER_COUNTER = 0

# Global registry of LoRA layers by name (populated at __init__ time)
# This allows fake_impl to look up output_size during tracing
_LORA_LAYER_REGISTRY: dict[str, "BaseLinearLayerWithLoRA"] = {}


def _get_unique_layer_name() -> str:
    """Generate unique layer name for registration."""
    global _LORA_LAYER_COUNTER
    _LORA_LAYER_COUNTER += 1
    return f"lora_linear_{_LORA_LAYER_COUNTER}"


def lora_linear_parallel_apply(
    x: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """
    Custom op for parallel LoRA + base GEMM execution.

    Wraps the parallel computation to work with torch.compile.
    """
    forward_context = get_forward_context()

    # Lazy registration to forward_context if needed
    if layer_name not in forward_context.no_compile_layers:
        if layer_name not in _LORA_LAYER_REGISTRY:
            raise RuntimeError(
                f"LoRA layer '{layer_name}' not found in global registry."
            )
        forward_context.no_compile_layers[layer_name] = _LORA_LAYER_REGISTRY[layer_name]

    layer = forward_context.no_compile_layers[layer_name]
    return layer._forward_impl_parallel(x)


def lora_linear_parallel_apply_fake(
    x: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """
    Fake implementation for torch.compile tracing.

    Returns tensor with correct output shape by looking up layer from registry.
    Registry is populated at __init__ time, before tracing.
    """
    # Look up layer to get output_size (registry populated at __init__)
    if layer_name in _LORA_LAYER_REGISTRY:
        layer = _LORA_LAYER_REGISTRY[layer_name]
        output_size = layer.output_size
    else:
        # Fallback: assume same shape (shouldn't happen)
        output_size = x.shape[-1]

    # Return tensor with correct output shape
    if x.ndim == 3:
        batch, seq, _ = x.shape
        return torch.empty((batch, seq, output_size), dtype=x.dtype, device=x.device)
    else:
        return torch.empty((x.shape[0], output_size), dtype=x.dtype, device=x.device)


# Only register custom op if feature is enabled
if _ENABLE_LORA_STREAM:
    direct_register_custom_op(
        op_name="lora_linear_parallel_apply",
        op_func=lora_linear_parallel_apply,
        mutates_args=[],  # Returns new tensor, doesn't mutate
        fake_impl=lora_linear_parallel_apply_fake,
        tags=(torch.Tag.needs_fixed_stride_order,),
    )


class BaseLinearLayerWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: LinearBase):
        super().__init__()
        self.base_layer = base_layer
        self.input_size = self.base_layer.input_size
        self.tp_size = self.base_layer.tp_size
        self.tp_rank = self.base_layer.tp_rank
        self.device = _get_lora_device(self.base_layer)
        self.output_slices: tuple[int, ...]
        self.output_size: int
        self.n_slices: int

        # Use vLLM's global aux_stream for parallel execution
        self._lora_stream = aux_stream() if _ENABLE_LORA_STREAM else None

        # Generate layer name at __init__ time and register to global registry
        # This ensures output_size is available for fake_impl during tracing
        if _ENABLE_LORA_STREAM:
            self._layer_name = _get_unique_layer_name()
            _LORA_LAYER_REGISTRY[self._layer_name] = self
        else:
            self._layer_name = None

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        self.lora_config = lora_config

        if isinstance(self.base_layer, ReplicatedLinear):
            lora_a_out_size = lora_config.max_lora_rank
            lora_b_out_size = self.output_size

        elif isinstance(self.base_layer, ColumnParallelLinear):
            lora_a_out_size = (
                lora_config.max_lora_rank
                if not lora_config.fully_sharded_loras
                else divide(lora_config.max_lora_rank, self.tp_size)
            )
            lora_b_out_size = self.output_size

        elif isinstance(self.base_layer, RowParallelLinear):
            lora_a_out_size = lora_config.max_lora_rank
            lora_b_out_size = (
                self.output_size
                if not lora_config.fully_sharded_loras
                else divide(self.output_size, self.tp_size)
            )
        else:
            raise NotImplementedError

        self.lora_a_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_a_out_size,
                self.input_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            )
            for _ in range(self.n_slices)
        )
        self.lora_b_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_b_out_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
            )
            for _ in range(self.n_slices)
        )
        self.output_slices = (self.lora_b_stacked[0].shape[2],)

    def reset_lora(self, index: int):
        for s_index in range(self.n_slices):
            self.lora_a_stacked[s_index][index] = 0
            self.lora_b_stacked[s_index][index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ):
        assert isinstance(lora_a, torch.Tensor)
        assert isinstance(lora_b, torch.Tensor)
        assert (
            len(self.lora_a_stacked) == len(self.lora_b_stacked) == self.n_slices == 1
        )

        self.reset_lora(index)
        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)

        self.lora_a_stacked[0][index, 0, : lora_a.shape[0], : lora_a.shape[1]].copy_(
            lora_a, non_blocking=True
        )
        self.lora_b_stacked[0][index, 0, : lora_b.shape[0], : lora_b.shape[1]].copy_(
            lora_b, non_blocking=True
        )

    def _forward_impl_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parallel execution: LoRA on aux stream, base GEMM on main stream.
        """
        lora_stream = self._lora_stream

        # Clone input for LoRA stream
        x_clone = x.clone()
        x_clone.record_stream(lora_stream)

        # Flatten for punica wrapper
        x_flat = x_clone.flatten(0, 1) if x_clone.ndim == 3 else x_clone

        # Pre-allocate LoRA delta buffer
        lora_delta = torch.zeros(
            (x_flat.shape[0], self.output_size),
            dtype=x.dtype,
            device=x.device,
        )
        lora_delta.record_stream(lora_stream)

        # Kick off LoRA on aux stream before base GEMM starts
        lora_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(lora_stream):
            self.punica_wrapper.add_lora_linear(
                lora_delta,
                x_flat,
                self.lora_a_stacked,
                self.lora_b_stacked,
                1.0,
                self.output_slices,
            )

        # Run base GEMM on main stream (parallel with LoRA)
        # Note: We pass bias=None here since apply() handles it separately
        output = self.base_layer.quant_method.apply(self.base_layer, x, None)

        # Flatten output for merging
        output_flat = output.flatten(0, 1) if output.ndim == 3 else output

        # Wait for LoRA to complete
        torch.cuda.current_stream().wait_stream(lora_stream)

        # Merge LoRA result into output
        output_flat.add_(lora_delta)

        return output

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply base layer + LoRA with optional parallel execution.
        """
        # Decide whether to use parallel path
        use_parallel = (
            self._lora_stream is not None
            and self._layer_name is not None
            and x.is_cuda
        )

        if use_parallel:
            # Call custom op - wraps ENTIRE parallel execution
            # torch.compile will use fake_impl during tracing
            output = torch.ops.vllm.lora_linear_parallel_apply(x, self._layer_name)

            # Handle bias separately (base_layer.apply handles this normally)
            if bias is not None:
                output = output + bias
        else:
            # Sequential path (original behavior)
            output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

            # Handle batch dimension flattening
            x_flat = x
            output_flat = output
            if x.ndim == 3 and output.ndim == 3:
                output_flat = output.flatten(0, 1)
                x_flat = x.flatten(0, 1)

            lora_output = self.punica_wrapper.add_lora_linear(
                output_flat,
                x_flat,
                self.lora_a_stacked,
                self.lora_b_stacked,
                1.0,
                self.output_slices,
            )
            if not current_platform.can_update_inplace():
                if output.ndim == 3:
                    output = lora_output.view(output.shape)
                else:
                    output = lora_output

        return output

    @property
    def weight(self) -> torch.Tensor:
        if hasattr(self.base_layer, "weight"):
            return self.base_layer.weight
        elif hasattr(self.base_layer, "weight_packed"):
            return self.base_layer.weight_packed
        elif hasattr(self.base_layer, "qweight"):
            return self.base_layer.qweight
        elif hasattr(self.base_layer, "B"):
            return self.base_layer.B
        elif hasattr(self.base_layer, "W_q"):
            return self.base_layer.W_q
        else:
            raise ValueError(f"Unsupported base layer: {self.base_layer}")

    @property
    def bias(self) -> torch.Tensor | None:
        if hasattr(self.base_layer, "bias"):
            return self.base_layer.bias
        else:
            return None
