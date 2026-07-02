# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online int4 per-channel quantization for embedding layers.

This module provides int4 quantization support for large vocabulary embeddings,
reducing memory footprint from ~2 bytes per parameter (bf16) to ~0.5 bytes per
parameter (int4 packed). The quantization is performed per-channel with symmetric
scaling, and weights are packed two 4-bit values per byte.

This is particularly useful for models with very large embedding tables (e.g.,
n-gram embeddings) that would otherwise dominate the model's memory footprint.
"""

import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.utils import replace_parameter


class Int4PerChannelEmbeddingMethod(QuantizeMethodBase):
    """Online int4 per-channel quantization method for embeddings.

    Weights are packed two 4-bit values per byte and dequantized on-the-fly
    during lookup, cutting storage to ~0.5 bytes per parameter.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight_block_size = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Create weight parameters for the embedding layer.

        Note: VocabParallelEmbedding creates the weight parameter.
        This method registers the weight attributes for the quantization method.
        """
        from vllm.model_executor.parameter import set_weight_attrs

        if hasattr(layer, "weight"):
            set_weight_attrs(layer.weight, {"input_dim": 1, "output_dim": 0})
            set_weight_attrs(layer.weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Quantize weights to int4 after loading from checkpoint."""
        if getattr(layer, "_already_int4_quantized", False):
            return
        weight = layer.weight
        hidden_size = weight.shape[1]
        if hidden_size % 2 != 0:
            raise ValueError(
                f"Int4 embedding requires even embedding_dim, got {hidden_size}"
            )

        # Per-channel symmetric int4 scale: range [-7, 7].
        max_abs = weight.abs().max(dim=0, keepdim=True).values
        scale = (max_abs / 7.0).to(weight.dtype)
        q = (weight / scale).round().clamp(-7, 7).to(torch.int8)
        # Offset to unsigned 4-bit values [0, 14] and pack two per byte.
        q_uint4 = (q + 8).to(torch.uint8)
        packed = q_uint4[:, ::2] | (q_uint4[:, 1::2] << 4)

        replace_parameter(
            layer, "weight", torch.nn.Parameter(packed, requires_grad=False)
        )
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(scale, requires_grad=False),
        )
        layer._already_int4_quantized = True
        torch.accelerator.empty_cache()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply method for linear layers - not implemented for embeddings."""
        raise NotImplementedError(
            "Int4PerChannelEmbeddingMethod only supports embedding"
        )

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        """Perform embedding lookup with on-the-fly int4 dequantization."""
        # Unpack and dequantize only the selected rows.
        packed = layer.weight[input_]
        low = (packed & 0xF).to(torch.int16)
        high = (packed >> 4).to(torch.int16)
        q_uint4 = torch.stack([low, high], dim=-1).view(packed.shape[0], -1)
        q = q_uint4.to(layer.params_dtype) - 8.0
        return q * layer.weight_scale
