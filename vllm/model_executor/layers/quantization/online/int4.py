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

from typing import cast

import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase


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
        """Create weight parameter for the embedding layer.

        VocabParallelEmbedding delegates weight creation to the quant method.
        """
        from torch.nn import Parameter

        from vllm.model_executor.utils import set_weight_attrs

        if not hasattr(layer, "weight") or layer.weight is None:
            weight = Parameter(
                torch.empty(
                    sum(output_partition_sizes),
                    input_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("weight", weight)

        set_weight_attrs(layer.weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(layer.weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Quantize weights to int4 after loading from checkpoint."""
        if getattr(layer, "_already_int4_quantized", False):
            return
        # Skip layers without weight (e.g. PPMissingLayer/StageMissingLayer).
        if "weight" not in layer._parameters:
            return
        weight = cast(torch.Tensor, layer.weight)
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

        # Replace weight with packed version.
        # Save weight_loader before replacing (it lives on the tensor).
        old_weight = cast(torch.nn.Parameter, layer.weight)
        weight_loader = getattr(old_weight, "weight_loader", None)
        # Must delete from _parameters first because nn.Module.__setattr__
        # doesn't clear _parameters before calling register_parameter.
        if "weight" in layer._parameters:
            del layer._parameters["weight"]
        packed_param = torch.nn.Parameter(packed, requires_grad=False)
        if weight_loader is not None:
            from vllm.model_executor.utils import set_weight_attrs

            set_weight_attrs(packed_param, {"weight_loader": weight_loader})
        layer.register_parameter("weight", packed_param)
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
        """Compute logits via tied int4 embedding weights.

        Used when lm_head shares weights with a quantized embed_tokens.
        Unpacks and dequantizes the full weight table, then does matmul.
        """
        packed = cast(torch.Tensor, layer.weight)
        weight_scale = cast(torch.Tensor, layer.weight_scale)
        low = (packed & 0xF).to(torch.int16)
        high = (packed >> 4).to(torch.int16)
        q_uint4 = torch.stack([low, high], dim=-1).view(packed.shape[0], -1)
        weight_fp = (q_uint4.to(x.dtype) - 8.0) * weight_scale
        logits = x @ weight_fp.t()
        if bias is not None:
            logits = logits + bias
        return logits

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        """Perform embedding lookup with on-the-fly int4 dequantization."""
        # Unpack and dequantize only the selected rows.
        packed = cast(torch.Tensor, layer.weight)[input_]
        weight_scale = cast(torch.Tensor, layer.weight_scale)
        params_dtype = cast(torch.dtype, layer.params_dtype)
        low = (packed & 0xF).to(torch.int16)
        high = (packed >> 4).to(torch.int16)
        q_uint4 = torch.stack([low, high], dim=-1).view(packed.shape[0], -1)
        q = q_uint4.to(params_dtype) - 8.0
        return q * weight_scale
