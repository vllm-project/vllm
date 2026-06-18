# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BitNet b1.58 quantization support using BitBLAS kernels.

This implements a vLLM quantization backend for the packed ternary weight
format used by microsoft/bitnet-b1.58-2B-4T. Weights are stored as 2-bit
packed uint8 tensors (4 ternary values per byte) with per-layer scalar
weight_scale factors.

BitBLAS provides optimized BF16×INT2 GEMM kernels that operate directly
on the packed weights, giving both GPU memory savings (weights stay as
INT2 on GPU, ~8x smaller than BF16) and compute savings (reduced memory
bandwidth for weight loads).

When BitBLAS is not installed, falls back to unpack-to-bfloat16 at load
time (same memory as BF16 variant, but still loads packed checkpoints).

Reference:
  - Packing format: transformers.integrations.bitnet
  - BitBLAS: https://github.com/microsoft/BitBLAS
  - Prior art: vllm-project/vllm#17588
"""

from typing import Any

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import (
    PackedvLLMParameter,
    PerTensorScaleParameter,
)

logger = init_logger(__name__)

# Number of ternary values packed per uint8 byte (2 bits each)
_VALUES_PER_ITEM = 4

# Try to import BitBLAS for native INT2 GEMM support.
# Falls back to unpack-to-float when not available.
try:
    import bitblas
    HAS_BITBLAS = True
    logger.info("BitBLAS available — using native INT2 GEMM kernels")
except ImportError:
    HAS_BITBLAS = False
    logger.warning(
        "BitBLAS not installed — BitNet will unpack to bfloat16 at load "
        "time (same GPU memory as BF16 variant). Install with: "
        "pip install bitblas"
    )

# Pre-compiled BitBLAS matmul operators, keyed by (N, K).
# Shared across layers with the same dimensions to avoid redundant
# kernel compilation.
_BITBLAS_MATMUL_CACHE: dict[tuple[int, int], Any] = {}


def _get_bitblas_matmul(N: int, K: int) -> Any:
    """Get or create a BitBLAS Matmul operator for the given dimensions.

    BitBLAS compiles specialized CUDA kernels on first use (can take
    minutes). We cache operators by (N, K) since M is dynamic.
    """
    key = (N, K)
    if key not in _BITBLAS_MATMUL_CACHE:
        config = bitblas.MatmulConfig(
            M=1,  # Dynamic M — BitBLAS handles varying batch sizes
            N=N,
            K=K,
            A_dtype="bfloat16",
            W_dtype="int2",
            accum_dtype="float32",
            out_dtype="bfloat16",
            layout="nt",
            with_bias=False,
            group_size=None,
            with_scaling=False,
            with_zeros=False,
        )
        _BITBLAS_MATMUL_CACHE[key] = bitblas.Matmul(config=config)
        logger.info("Compiled BitBLAS kernel for N=%d, K=%d", N, K)
    return _BITBLAS_MATMUL_CACHE[key]


def _unpack_weights(
    packed: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Unpack uint8-packed ternary weights into a tensor of {-1, 0, +1}.

    Packing format (from transformers.integrations.bitnet):
      - 4 ternary values per uint8 byte, packed along dim 0 (out_features)
      - Input shape: [out_features // 4, in_features], dtype=uint8
      - Output shape: [out_features, in_features], dtype=<dtype>
      - Ternary values {-1, 0, +1} are shifted to {0, 1, 2} for packing
      - Stored as 2-bit pairs: bits[1:0], bits[3:2], bits[5:4], bits[7:6]
    """
    packed_shape = packed.shape
    original_row_dim = packed_shape[0] * _VALUES_PER_ITEM

    if len(packed_shape) == 1:
        unpacked_shape = (original_row_dim,)
    else:
        unpacked_shape = (original_row_dim, *packed_shape[1:])

    unpacked = torch.zeros(
        unpacked_shape, device=packed.device, dtype=torch.uint8
    )

    for i in range(_VALUES_PER_ITEM):
        start = i * packed_shape[0]
        end = start + packed_shape[0]
        mask = 3 << (2 * i)
        unpacked[start:end] = (packed & mask) >> (2 * i)

    return unpacked.to(dtype) - 1


class BitNetBitBLASConfig(QuantizationConfig):
    """Config class for BitNet b1.58 quantization.

    Reads the quantization_config from the model's config.json:
      {
        "quant_method": "bitnet",
        "linear_class": "autobitlinear",
        "quantization_mode": "offline"
      }
    """

    def __init__(
        self,
        quantization_mode: str = "offline",
        linear_class: str = "autobitlinear",
    ) -> None:
        super().__init__()
        self.quantization_mode = quantization_mode
        self.linear_class = linear_class

        if quantization_mode != "offline":
            raise ValueError(
                f"BitNet quantization only supports 'offline' mode, "
                f"got '{quantization_mode}'. For online mode, use the "
                f"BF16 variant (microsoft/bitnet-b1.58-2B-4T-bf16)."
            )

    def __repr__(self) -> str:
        return (
            f"BitNetBitBLASConfig(quantization_mode={self.quantization_mode}, "
            f"linear_class={self.linear_class})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "bitnet"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # BitBLAS supports V100+ (SM_70)
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BitNetBitBLASConfig":
        quantization_mode = cls.get_from_keys_or(
            config, ["quantization_mode"], default="offline"
        )
        linear_class = cls.get_from_keys_or(
            config, ["linear_class"], default="autobitlinear"
        )
        return cls(
            quantization_mode=quantization_mode,
            linear_class=linear_class,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.linear import LinearBase
        if isinstance(layer, LinearBase):
            return BitNetBitBLASLinearMethod(self)
        return None


class BitNetBitBLASLinearMethod(LinearMethodBase):
    """Linear method for BitNet b1.58 with BitBLAS native INT2 GEMM.

    Weight lifecycle:
      1. create_weights(): allocates packed uint8 qweight + scalar weight_scale
      2. process_weights_after_loading(): transforms weights for inference
         - BitBLAS path: unpack → int8 ternary → BitBLAS transform_weight()
           Weights stay as INT2 on GPU (~8x smaller than BF16)
         - Fallback path: unpack → bfloat16 (same memory as BF16 variant)
      3. apply(): runs the matmul
         - BitBLAS path: bitblas.Matmul(BF16 × INT2 → BF16)
         - Fallback path: F.linear(BF16 × BF16 → BF16)
    """

    def __init__(self, quant_config: BitNetBitBLASConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        # Packed uint8 weight: [out_features // 4, in_features]
        # The packing is along dim 0 (output), so packed_dim=0.
        packed_out = output_size_per_partition // _VALUES_PER_ITEM
        qweight = PackedvLLMParameter(
            data=torch.zeros(
                packed_out,
                input_size_per_partition,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=0,
            packed_factor=_VALUES_PER_ITEM,
            weight_loader=weight_loader,
        )

        # Per-layer scalar weight_scale.
        # For merged (QKV, gate_up) layers, each sub-projection has its
        # own scalar scale. PerTensorScaleParameter handles this by
        # storing one scale per shard.
        num_shards = len(output_partition_sizes)
        weight_scale = PerTensorScaleParameter(
            data=torch.ones(num_shards, dtype=params_dtype),
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("weight_scale", weight_scale)
        # Store partition sizes for process_weights_after_loading.
        layer.output_partition_sizes = output_partition_sizes

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Transform weights for inference after all shards are loaded."""
        device = layer.qweight.device

        # Unpack: [out//4, in] uint8 → [out, in] int8 ternary {-1, 0, +1}
        unpacked_int8 = _unpack_weights(
            layer.qweight.data, dtype=torch.int8
        )

        N, K = unpacked_int8.shape  # out_features, in_features

        if HAS_BITBLAS:
            self._process_bitblas(layer, unpacked_int8, N, K, device)
        else:
            self._process_fallback(layer, unpacked_int8, device)

    def _process_bitblas(
        self,
        layer: torch.nn.Module,
        unpacked_int8: torch.Tensor,
        N: int,
        K: int,
        device: torch.device,
    ) -> None:
        """BitBLAS path: keep weights packed as INT2 on GPU."""
        # Get or compile the BitBLAS matmul operator for this shape
        matmul_op = _get_bitblas_matmul(N, K)

        # Transform weight to BitBLAS's internal tiled INT2 format.
        # This repacks the int8 ternary into BitBLAS's optimized layout.
        weight_tiled = matmul_op.transform_weight(
            unpacked_int8.cuda()
        )

        # Store the tiled weight and operator on the layer
        layer.weight_tiled = Parameter(
            weight_tiled.to(device), requires_grad=False
        )
        layer.bitblas_matmul = matmul_op

        # Keep weight_scale as a bfloat16 tensor for post-GEMM scaling
        layer.weight_scale_float = layer.weight_scale.data.to(
            torch.bfloat16
        ).to(device)
        layer.output_partition_sizes_tensor = (
            layer.output_partition_sizes
        )

        # Clean up packed parameters
        del layer.qweight
        del layer.weight_scale
        layer._use_bitblas = True

    def _process_fallback(
        self,
        layer: torch.nn.Module,
        unpacked_int8: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Fallback path: unpack to bfloat16 (same memory as BF16 variant)."""
        dtype = torch.bfloat16
        unpacked = unpacked_int8.to(dtype)

        # Apply per-shard scales
        scales = layer.weight_scale.data.to(dtype)
        partition_sizes = layer.output_partition_sizes

        if len(scales) == 1 or all(s == scales[0] for s in scales):
            weight = unpacked * scales[0]
        else:
            weight = torch.empty_like(unpacked)
            offset = 0
            for i, size in enumerate(partition_sizes):
                weight[offset:offset + size] = (
                    unpacked[offset:offset + size] * scales[i]
                )
                offset += size

        layer.weight = Parameter(weight.to(device), requires_grad=False)

        # Clean up packed parameters
        del layer.qweight
        del layer.weight_scale
        layer._use_bitblas = False

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply BitNet linear.

        Activation quantization (act_quant) is handled by the model's
        forward() method before this is called.
        """
        if getattr(layer, "_use_bitblas", False):
            return self._apply_bitblas(layer, x, bias)
        else:
            return self._apply_fallback(layer, x, bias)

    def _apply_bitblas(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """BitBLAS path: native BF16×INT2 GEMM.

        Weights stay packed as INT2 on GPU. BitBLAS decompresses
        on-the-fly during the GEMM for bandwidth savings.
        """
        # BitBLAS matmul: [M, K] × [N, K]^T → [M, N]
        output = layer.bitblas_matmul(x, layer.weight_tiled)

        # Apply per-shard weight_scale post-GEMM.
        # For single-shard layers, this is a scalar multiply.
        # For merged layers, apply per-partition.
        scales = layer.weight_scale_float
        partition_sizes = layer.output_partition_sizes_tensor

        if len(scales) == 1:
            output = output * scales[0]
        else:
            offset = 0
            for i, size in enumerate(partition_sizes):
                output[..., offset:offset + size] *= scales[i]
                offset += size

        if bias is not None:
            output = output + bias
        return output

    def _apply_fallback(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fallback path: standard F.linear on unpacked bfloat16 weights."""
        return torch.nn.functional.linear(x, layer.weight, bias)
