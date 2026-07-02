# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Int4 per-channel quantization for vocabulary embeddings.

This module provides int4 quantization support for large vocabulary embeddings,
reducing memory footprint from ~2 bytes per parameter (bf16) to ~0.5 bytes per
parameter (int4 packed). The quantization is performed per-channel with symmetric
scaling, and weights are packed two 4-bit values per byte.

This is particularly useful for models with very large embedding tables (e.g.,
n-gram embeddings) that would otherwise dominate the model's memory footprint.
"""

import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

# Magic number constants for clarity
_INT4_QUANTIZATION_CHUNK_ROWS = 1_048_576  # ~512 MiB bf16 per chunk


class Int4PerChannelEmbeddingMethod(QuantizeMethodBase):
    """Int4 per-channel quantization method for embeddings.

    Weights are packed two 4-bit values per byte and dequantized on-the-fly
    during lookup, cutting storage to ~0.5 bytes per parameter.
    """

    def __init__(self, layer: torch.nn.Module) -> None:
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

        Note: The Int4VocabParallelEmbedding class replaces this weight with a
        packed uint8 buffer immediately after init, so this path is only used
        to satisfy the base-class contract.
        """
        weight = torch.nn.Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        from vllm.model_executor.utils import set_weight_attrs

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

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


class Int4VocabParallelEmbedding(VocabParallelEmbedding):
    """VocabParallelEmbedding with int4 per-channel quantization.

    Unlike the base class, the int4 weight is allocated directly at init time
    so the full bf16/fp16 weight is never kept on the GPU. The checkpoint weight
    is quantized on-the-fly inside the weight loader.

    This is particularly useful for very large embedding tables that would
    otherwise dominate memory usage.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = 64,
        prefix: str = "",
    ) -> None:
        if embedding_dim % 2 != 0:
            raise ValueError(
                f"Int4VocabParallelEmbedding requires even embedding_dim, "
                f"got {embedding_dim}"
            )

        # Let the base class compute TP shard metadata and allocate a temporary
        # full-precision weight. We replace it with a packed int4 buffer below.
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            params_dtype=params_dtype,
            org_num_embeddings=org_num_embeddings,
            padding_size=padding_size,
            quant_config=None,
            prefix=prefix,
        )

        device = self.weight.device
        packed_shape = (
            self.num_embeddings_per_partition,
            self.embedding_dim // 2,
        )
        packed = torch.nn.Parameter(
            torch.empty(packed_shape, dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        scale = torch.nn.Parameter(
            torch.empty(1, self.embedding_dim, dtype=self.params_dtype, device=device),
            requires_grad=False,
        )

        replace_parameter(self, "weight", packed)
        self.register_parameter("weight_scale", scale)

        # replace_parameter copies the old parameter's weight_loader onto a new
        # Parameter object, so we must install the int4 loader on self.weight.
        if hasattr(self.weight, "weight_loader"):
            delattr(self.weight, "weight_loader")
        set_weight_attrs(
            self.weight,
            {
                "weight_loader": self._int4_weight_loader,
                "input_dim": 1,
                "output_dim": 0,
            },
        )

        # Install the int4 embedding method so forward() uses the unpack path.
        self.quant_method = Int4PerChannelEmbeddingMethod(self)
        # The weight loader already quantizes the checkpoint; skip a second pass.
        self._already_int4_quantized = True

    def _pick_staging_device(self, target_device: torch.device) -> torch.device:
        """Pick an idle GPU for staging quantization chunks.

        The target GPU is usually full of model weights, so we stream chunks
        to another GPU that has free memory. Falls back to CPU if no GPU is
        available.

        NOTE: For multi-worker startup the "most free" GPU can be contended
        by several concurrent loaders, so we conservatively use CPU to avoid
        sporadic OOMs during N-gram embedding quantization.
        """
        return torch.device("cpu")

    def _int4_weight_loader(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        """Load and quantize weight to int4 in-place."""
        start_idx = self.shard_indices.org_vocab_start_index
        shard_size = self.shard_indices.org_vocab_end_index - start_idx

        if shard_size <= 0:
            param.data.fill_(0)
            self.weight_scale.data.fill_(0)
            return

        # Keep the full bf16 shard on CPU so it does not compete with the
        # model weights on the target GPU. We then stream modest chunks to an
        # idle staging GPU for fast quantization.
        loaded_shard_cpu = loaded_weight.narrow(0, start_idx, shard_size).detach().cpu()

        target_device = param.device
        staging_device = self._pick_staging_device(target_device)

        # ~512 MiB bf16 per chunk keeps staging memory modest.
        chunk_rows = _INT4_QUANTIZATION_CHUNK_ROWS

        # First pass: compute per-channel max abs.
        max_abs = torch.zeros(
            1, self.embedding_dim, dtype=torch.float32, device=staging_device
        )
        for row_start in range(0, shard_size, chunk_rows):
            row_end = min(row_start + chunk_rows, shard_size)
            chunk = loaded_shard_cpu[row_start:row_end].to(staging_device)
            chunk_max = chunk.abs().amax(dim=0, keepdim=True)
            max_abs = torch.maximum(max_abs, chunk_max)
        scale = (max_abs / 7.0).to(self.params_dtype)

        # Second pass: quantize each chunk and copy the packed result back.
        for row_start in range(0, shard_size, chunk_rows):
            row_end = min(row_start + chunk_rows, shard_size)
            chunk = loaded_shard_cpu[row_start:row_end].to(staging_device)
            q = (
                (chunk.to(torch.float32) / scale.to(torch.float32))
                .round()
                .clamp(-7, 7)
                .to(torch.int8)
            )
            q_uint4 = (q + 8).to(torch.uint8)
            packed = q_uint4[:, ::2] | (q_uint4[:, 1::2] << 4)
            param.data[row_start:row_end].copy_(packed)

        param.data[shard_size:].fill_(0)
        self.weight_scale.data.copy_(scale)
