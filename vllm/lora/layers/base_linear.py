# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.distributed.utils import divide
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.utils.torch_utils import current_stream, direct_register_custom_op

from .base import BaseLayerWithLoRA
from .utils import _get_lora_device

# Global registry with stable string keys
_lora_layer_registry: dict[str, "BaseLinearLayerWithLoRA"] = {}
_layer_counter = 0


def _lora_apply_impl(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    layer_key: str,
    output_size: int,
) -> torch.Tensor:
    """Custom op implementation."""
    layer = _lora_layer_registry[layer_key]

    lora_delta = torch.zeros(
        (x.shape[0], output_size),
        dtype=x.dtype,
        device=x.device,
    )

    lora_stream = layer.punica_wrapper.lora_stream
    lora_stream.wait_stream(current_stream())
    lora_delta.record_stream(lora_stream)
    with torch.cuda.stream(lora_stream):
        output = layer.base_layer.quant_method.apply(layer.base_layer, x, bias)
        output_flat = output.flatten(0, 1) if output.ndim == 3 else output
    layer.punica_wrapper.add_lora_linear(
        lora_delta,
        x,
        layer.lora_a_stacked,
        layer.lora_b_stacked,
        1.0,
        layer.output_slices,
    )

    current_stream().wait_stream(lora_stream)
    output_flat.add_(lora_delta)

    return output_flat


def _lora_apply_fake(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    layer_key: str,
    output_size: int,
) -> torch.Tensor:
    """Fake implementation for torch.compile."""
    return torch.empty((x.shape[0], output_size), dtype=x.dtype, device=x.device)


# Register the custom op
direct_register_custom_op(
    op_name="lora_apply",
    op_func=_lora_apply_impl,
    mutates_args=[],
    fake_impl=_lora_apply_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)
lora_apply = torch.ops.vllm.lora_apply


class BaseLinearLayerWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: LinearBase):
        super().__init__()
        self.base_layer = base_layer
        self.input_size = self.base_layer.input_size
        # Ensure tp_size and tp_rank consistency with the base_layer.
        self.tp_size = self.base_layer.tp_size
        self.tp_rank = self.base_layer.tp_rank
        self.device = _get_lora_device(self.base_layer)
        self.output_slices: tuple[int, ...]
        self.output_size: int
        self.n_slices: int

        # Use stable string key
        global _layer_counter
        self._layer_key = f"lora_{_layer_counter}"
        _layer_counter += 1
        _lora_layer_registry[self._layer_key] = self

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        self.lora_config = lora_config
        #
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
        # Except for QKVParallelLinearWithLoRA and
        # MergedColumnParallelLinearWithLoRA, all other linear LoRA layers
        # store weights in a tuple of size 1. These two layers will
        # override this function.
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

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        original_shape = x.shape if x.ndim == 3 else None
        x_flat = x.flatten(0, 1) if x.ndim == 3 else x

        # Use custom op wrapper to hide record_stream from torch.compile
        output_flat = lora_apply(x_flat, bias, self._layer_key, self.output_size)
        if original_shape is not None:
            return output_flat.view(*original_shape[:-1], -1)
        return output_flat

    """

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

        original_shape = output.shape if output.ndim == 3 else None

        # In transformers backend, x and output have extra batch dimension like
        # (1, seq_len, hidden_dim), while punica expects (seq_len, hidden_dim),
        # therefore we need to flatten the batch dimensions.
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)

        lora_output: torch.Tensor | None = self.punica_wrapper.add_lora_linear(
            output, x, self.lora_a_stacked, self.lora_b_stacked, 1.0, self.output_slices
        )
        if not current_platform.can_update_inplace():
            output = lora_output

        # Reshape the flattened output back to its original shape,
        # as some MM encoders cannot handle flattened inputs.
        if original_shape is not None:
            output = output.reshape(original_shape)

        return output

    """

    @property
    def weight(self) -> torch.Tensor:
        # unquantizedLinear
        if hasattr(self.base_layer, "weight"):
            return self.base_layer.weight
        # Compressed Tensor
        elif hasattr(self.base_layer, "weight_packed"):
            return self.base_layer.weight_packed
        # GPTQ/AWQ
        elif hasattr(self.base_layer, "qweight"):
            return self.base_layer.qweight
        # marlin
        elif hasattr(self.base_layer, "B"):
            return self.base_layer.B
        else:
            raise ValueError(f"Unsupported base layer: {self.base_layer}")

    @property
    def bias(self) -> torch.Tensor | None:
        if hasattr(self.base_layer, "bias"):
            return self.base_layer.bias
        else:
            return None
